# -*- coding: utf-8 -*-
"""
所有任务的 loss —— phase2 与 phase3 共用同一份实现。

FIX(T-2): phase3/losses.py 曾把 phase2 的 SegLoss / HeatmapLoss 整个换掉，
          导致 phase3/trainer.py 的 `from losses import FocalLoss, SegLoss,
          HeatmapLoss` 直接 ImportError (连带 phase3 的 smoke_test/train 崩溃)。
FIX(T-3): phase3/losses.py 的 docstring 写着"stage2 和 stage3 import 同一份实现，
          否则会污染消融比较"，而两份 FocalLoss 实际差了一个 label_smoothing。
          现在真的是同一份了。
FIX(S-5): label_smoothing 由 cfg.cls.label_smoothing 显式传入，不再是默认参数。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 分类
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    """多分类 focal loss。

    alpha            : 各类权重，长度 = num_classes；None 表示不加权
    label_smoothing  : 0 表示关闭。**必须显式传入**，不要依赖默认值——
                       phase2 训练时用的是 0.05，这个数进了 README 的超参表。
    """

    def __init__(self, gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        # FIX(S-10): 之前同时存了 self.alpha(buffer) 和 self._alpha(普通属性)，
        # 后者不随 .to(device) 迁移，每次 forward 都要 host->device 拷贝一次。
        # 现在只保留 buffer。
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, label_smoothing={self.label_smoothing}"

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = logits.shape[1]
        logp = F.log_softmax(logits, dim=1)                       # (B,C)
        pt = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()  # (B,)

        if self.label_smoothing > 0:
            eps = self.label_smoothing
            with torch.no_grad():
                smooth = torch.full_like(logp, eps / C)
                smooth.scatter_(1, target.unsqueeze(1), 1.0 - eps + eps / C)
            ce = -(smooth * logp).sum(dim=1)                       # (B,)
        else:
            ce = -logp.gather(1, target.unsqueeze(1)).squeeze(1)   # (B,)

        focal = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            focal = self.alpha.to(logits.device)[target] * focal
        if self.reduction == "sum":
            return focal.sum()
        if self.reduction == "none":
            return focal
        return focal.mean()


# --------------------------------------------------------------------------- #
# 分割 —— 二分类 multi-label (phase2 主路径: disc/cup 两个独立 sigmoid)
# --------------------------------------------------------------------------- #
def dice_loss(prob: torch.Tensor, target: torch.Tensor,
              eps: float = 1.0) -> torch.Tensor:
    """逐样本逐通道 soft dice loss。prob, target: (B,C,H,W)。"""
    dims = (2, 3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return (1 - dice).mean()


class SegLoss(nn.Module):
    """BCEWithLogits + soft Dice (multi-label: disc / cup 各一个 sigmoid)。"""

    def __init__(self, dice_w: float = 1.0, bce_w: float = 1.0):
        super().__init__()
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.bce_w * self.bce(logits, target)
                + self.dice_w * dice_loss(torch.sigmoid(logits), target))


# --------------------------------------------------------------------------- #
# 分割 —— 三类 softmax (phase3 的 CDR 分支需要 {bg, rim, cup} 形式)
# --------------------------------------------------------------------------- #
class SegCEDiceLoss(nn.Module):
    """类别加权 CE + 前景 soft Dice，用于 stage-3 的分割 anchor。

    不用裸 CrossEntropy 的原因：视盘/视杯相对背景极小，无权重的 CE 会满足于
    "全部预测背景"，在从零初始化的分割头上会让 disc Dice 逐 epoch 下降。
    """

    def __init__(self, ce_class_weights: torch.Tensor | None = None,
                 dice_classes: tuple[int, ...] = (1, 2),
                 dice_weight: float = 1.0, ce_weight: float = 1.0,
                 eps: float = 1e-6):
        super().__init__()
        self.dice_classes = tuple(dice_classes)
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.eps = float(eps)
        if ce_class_weights is not None:
            self.register_buffer("ce_w", ce_class_weights)
        else:
            self.ce_w = None

    def _ce(self, logits, target):
        if self.ce_w is not None:
            w = self.ce_w
        else:
            C = logits.size(1)
            counts = torch.bincount(target.reshape(-1), minlength=C).float()
            freq = counts / counts.sum().clamp_min(1.0)
            inv = 1.0 / (freq + self.eps)
            w = (inv / inv.mean()).to(logits.dtype)
        return F.cross_entropy(logits, target, weight=w)

    def _dice(self, logits, target):
        probs = F.softmax(logits, dim=1)
        terms = []
        for c in self.dice_classes:
            p = probs[:, c]
            t = (target == c).float()
            inter = (p * t).sum(dim=(1, 2))
            denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            terms.append((2 * inter + self.eps) / (denom + self.eps))
        return 1.0 - torch.stack(terms, dim=1).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self._ce(logits, target) + \
               self.dice_weight * self._dice(logits, target)


# --------------------------------------------------------------------------- #
# 黄斑
# --------------------------------------------------------------------------- #
def gaussian_heatmap(coords: torch.Tensor, H: int, W: int,
                     sigma: float = 0.03) -> torch.Tensor:
    """由归一化坐标 (B,2)=(x,y) 生成高斯 heatmap (B,1,H,W)，峰值 1。"""
    B = coords.shape[0]
    device = coords.device
    ys = torch.linspace(0, 1, H, device=device).view(1, H, 1)
    xs = torch.linspace(0, 1, W, device=device).view(1, 1, W)
    cx = coords[:, 0].view(B, 1, 1)
    cy = coords[:, 1].view(B, 1, 1)
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return torch.exp(-d2 / (2 * sigma * sigma)).unsqueeze(1)


class HeatmapLoss(nn.Module):
    """heatmap MSE (对高斯目标) + 坐标 L1。"""

    def __init__(self, sigma: float = 0.03, coord_w: float = 1.0,
                 heat_w: float = 1.0):
        super().__init__()
        self.sigma = sigma
        self.coord_w = coord_w
        self.heat_w = heat_w

    def forward(self, pred_heat: torch.Tensor, pred_coords: torch.Tensor,
                gt_coords: torch.Tensor) -> torch.Tensor:
        B, _, H, W = pred_heat.shape
        gt_heat = gaussian_heatmap(gt_coords, H, W, self.sigma)
        heat_loss = F.mse_loss(torch.sigmoid(pred_heat), gt_heat)
        coord_loss = F.l1_loss(pred_coords, gt_coords)
        return self.heat_w * heat_loss + self.coord_w * coord_loss
