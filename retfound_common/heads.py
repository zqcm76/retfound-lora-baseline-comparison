# -*- coding: utf-8 -*-
"""
三个任务头 + 解码器积木。

约定: ViT 各 level 空间分辨率相同 (gh x gw)，所以 LevelFuse 是把同分辨率、
不同语义层级的特征在通道维拼接后融合，而不是 FPN 那种跨分辨率融合。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 基础积木
# --------------------------------------------------------------------------- #
def _groups(ch: int) -> int:
    """给 GroupNorm 选一个能整除的组数。"""
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(_groups(out_ch), out_ch),
        nn.GELU(),
    )


class UpBlock(nn.Module):
    """双线性上采样 x2 + 两个 conv。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                              align_corners=False)
        self.conv1 = conv_block(in_ch, out_ch)
        self.conv2 = conv_block(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(self.up(x)))


class UpStack(nn.Module):
    """先 1x1 把输入投到 ch，再堆 n_up 个 UpBlock (每个 x2)。"""

    def __init__(self, in_ch: int, ch: int, n_up: int):
        super().__init__()
        self.proj = conv_block(in_ch, ch)
        blocks = []
        for _ in range(max(0, n_up)):
            blocks.append(UpBlock(ch, ch))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        return x


class LevelFuse(nn.Module):
    """把多个同分辨率 ViT level 各自 1x1 投到 out_ch，concat 后再 1x1 融合。"""

    def __init__(self, in_chs: List[int], out_ch: int):
        super().__init__()
        self.projs = nn.ModuleList(
            [nn.Conv2d(c, out_ch, 1) for c in in_chs])
        self.fuse = conv_block(out_ch * len(in_chs), out_ch)

    def forward(self, levels: List[torch.Tensor]) -> torch.Tensor:
        feats = [p(l) for p, l in zip(self.projs, levels)]
        return self.fuse(torch.cat(feats, dim=1))


# --------------------------------------------------------------------------- #
# soft-argmax
# --------------------------------------------------------------------------- #
def soft_argmax_2d(heat: torch.Tensor, beta: float = 10.0
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 (B,1,H,W) 的 heatmap 做 soft-argmax。

    返回:
      coords: (B, 2) 归一化坐标 (x, y) ∈ [0,1]
      prob:   (B,1,H,W) softmax 后的概率图
    """
    B, C, H, W = heat.shape
    flat = heat.reshape(B, C, H * W)
    prob = F.softmax(flat * beta, dim=-1).reshape(B, C, H, W)

    ys = torch.linspace(0, 1, H, device=heat.device, dtype=heat.dtype)
    xs = torch.linspace(0, 1, W, device=heat.device, dtype=heat.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)

    px = (prob[:, 0] * grid_x.unsqueeze(0)).sum(dim=(1, 2))  # (B,)
    py = (prob[:, 0] * grid_y.unsqueeze(0)).sum(dim=(1, 2))  # (B,)
    coords = torch.stack([px, py], dim=1)                    # (B,2) -> (x,y)
    return coords, prob


# --------------------------------------------------------------------------- #
# 分类头
# --------------------------------------------------------------------------- #
class ClsHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.norm(pooled)))


# --------------------------------------------------------------------------- #
# 分割: coarse-to-fine
# --------------------------------------------------------------------------- #
class CoarseSegDecoder(nn.Module):
    """粗分割: 融合多 level → 上采样 → 输出粗 logits + 供 ROI 用的特征。"""

    def __init__(self, in_chs: List[int], ch: int, num_classes: int,
                 n_up: int):
        super().__init__()
        self.fuse = LevelFuse(in_chs, ch)
        self.up = UpStack(ch, ch, n_up)
        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, levels: List[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.fuse(levels)        # (B, ch, gh, gw)
        up = self.up(feat)              # (B, ch, coarse, coarse)
        logits = self.head(up)          # (B, num_classes, coarse, coarse)
        # 返回融合后的低分辨率特征 feat 供 roi_align (在特征图坐标系裁剪)
        return logits, feat


class FineSegDecoder(nn.Module):
    """精修: 输入 roi_align 裁出的特征块 → 上采样 → 精细 logits (ROI 内坐标系)。"""

    def __init__(self, in_ch: int, ch: int, num_classes: int, n_up: int):
        super().__init__()
        self.up = UpStack(in_ch, ch, n_up)
        self.refine = conv_block(ch, ch)
        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, roi_feat: torch.Tensor) -> torch.Tensor:
        x = self.up(roi_feat)
        x = self.refine(x)
        return self.head(x)


class HighResSegDecoder(nn.Module):
    """退路方案: 给分割单独挂的高分辨率解码器 (不做 ROI 裁剪，直接全图上采样)。"""

    def __init__(self, in_chs: List[int], ch: int, num_classes: int,
                 n_up: int):
        super().__init__()
        self.fuse = LevelFuse(in_chs, ch)
        self.up = UpStack(ch, ch, n_up)
        self.refine = conv_block(ch, ch)
        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, levels: List[torch.Tensor]) -> torch.Tensor:
        x = self.fuse(levels)
        x = self.up(x)
        x = self.refine(x)
        return self.head(x)


# --------------------------------------------------------------------------- #
# 黄斑定位头
# --------------------------------------------------------------------------- #
class MaculaHead(nn.Module):
    """heatmap 回归 + soft-argmax 出坐标。"""

    def __init__(self, in_chs: List[int], ch: int, n_up: int,
                 beta: float = 10.0):
        super().__init__()
        self.fuse = LevelFuse(in_chs, ch)
        self.up = UpStack(ch, ch, n_up)
        self.head = nn.Conv2d(ch, 1, 1)
        self.beta = beta

    def forward(self, levels: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self.fuse(levels)
        x = self.up(x)
        heat = self.head(x)                       # (B,1,H,W) 未归一化 logits
        coords, prob = soft_argmax_2d(heat, self.beta)
        return {"heatmap": heat, "coords": coords, "prob": prob}
