"""
stage2_iface.py — the contract stage 3 expects from the stage-2 model.

Stage 2 (RETFound ViT-L + LoRA, seg head, cls head, macula head) is NOT
re-implemented here. Stage 3 only needs two things from it:

    img_feat, seg_logits = model.encode(images)
        images     : [B, 3, Himg, Wimg]
        img_feat   : [B, D]          pooled backbone feature
        seg_logits : [B, 3, Hs, Ws]  {bg, rim, cup} logits

and a way to expose which parameters belong to the segmentation path, so the
stage-3 trainer can freeze/unfreeze them depending on fusion mode:

    model.segmentation_parameters() -> iterator of nn.Parameter
    model.backbone_parameters()     -> iterator of nn.Parameter  (LoRA matrices etc.)

Anything satisfying `Stage2Encoder` below drops in. The MockStage2 is a tiny
stand-in with the right shapes and a real seg head, used by the smoke test and
for wiring validation before the actual RETFound checkpoint is attached.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class Stage2Encoder(Protocol):
    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def segmentation_parameters(self) -> Iterator[nn.Parameter]: ...
    def backbone_parameters(self) -> Iterator[nn.Parameter]: ...


class MockStage2(nn.Module):
    """
    Minimal stand-in. A real call replaces this with the RETFound-backed model.

    - "backbone": a couple conv layers producing a feature map + GAP -> img_feat
    - "seg head": a tiny decoder producing [B,3,Hs,Ws] logits
    Shapes match the contract; weights are random. Just enough to exercise the
    full stage-3 loop (fusion, losses, freeze/unfreeze, metrics).
    """

    def __init__(self, feat_dim: int = 256, seg_size: int = 48):
        super().__init__()
        self.seg_size = seg_size
        self.feat_dim = feat_dim

        # shared stem (stands in for the frozen ViT trunk + LoRA)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
        )
        # backbone projection to img_feat (these emulate LoRA-trained params)
        self.backbone_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, feat_dim)
        )
        # segmentation decoder -> 3-class logits at seg_size x seg_size
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 3, 1),
        )

    def encode(self, images: torch.Tensor):
        f = self.stem(images)                          # [B,64,h,w]
        img_feat = self.backbone_proj(f)               # [B,D]
        seg = self.seg_head(f)                          # [B,3,h,w]
        seg = nn.functional.interpolate(
            seg, size=(self.seg_size, self.seg_size),
            mode="bilinear", align_corners=False,
        )
        return img_feat, seg

    # --- parameter groups for freeze/unfreeze ---
    def segmentation_parameters(self):
        return self.seg_head.parameters()

    def backbone_parameters(self):
        # stem + projection stand in for trunk+LoRA
        for p in self.stem.parameters():
            yield p
        for p in self.backbone_proj.parameters():
            yield p
