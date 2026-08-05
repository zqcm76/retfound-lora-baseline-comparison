"""Phase 1 baseline models + losses.

These are deliberately *conventional* models trained from ImageNet init (ResNet34) or
from scratch (UNet). They are the comparison anchor; Phase 2 replaces the backbone with
RETFound + LoRA and reports the same metrics against these numbers (params / VRAM /
training time / accuracy).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# --------------------------------------------------------------------------- classification
class ResNet34Classifier(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True) -> None:
        super().__init__()
        weights = (
            torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = torchvision.models.resnet34(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, num_classes) logits


class BCEClsLoss(nn.Module):
    """Binary CE with an optional positive-class weight (glaucoma is the rare class)."""

    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        if pos_weight is not None:
            # registered as a buffer so it follows the module on .to(device)
            self.register_buffer("pw", torch.tensor([float(pos_weight)]))
        else:
            self.pw = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits.squeeze(1), target.float(), pos_weight=self.pw
        )


# --------------------------------------------------------------------------- segmentation
class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Vanilla UNet for optic disc/cup segmentation (input H, W must be a multiple of 16)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3, base: int = 32) -> None:
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc1 = DoubleConv(in_channels, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c[3], c[4])
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.dec4 = DoubleConv(c[4], c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = DoubleConv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)  # (B, num_classes, H, W) logits


class DiceCELoss(nn.Module):
    """Cross-entropy + soft Dice, a standard robust combo for medical segmentation."""

    def __init__(
        self,
        num_classes: int = 3,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cw = ce_weight
        self.dw = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        probs = torch.softmax(logits, dim=1)
        tgt = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * tgt).sum(dims)
        denom = probs.sum(dims) + tgt.sum(dims)
        dice = (2 * inter + 1e-6) / (denom + 1e-6)
        return self.cw * ce + self.dw * (1.0 - dice.mean())
