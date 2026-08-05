# -*- coding: utf-8 -*-
"""
多任务模型: RETFound 骨干 (冻结 + LoRA) + 分类/分割/黄斑 三头。

coarse-to-fine 流程 (默认):
  backbone 前向一次 → 取多 level 特征
  → 粗分割得到粗 disc/cup logits + 融合特征
  → 由粗 disc 前景框 (训练时可用 GT disc 做 teacher forcing) 在融合特征上 roi_align
  → 精修解码器在 ROI 内坐标系输出精细 disc/cup logits
评测时把精细结果 paste 回全图分辨率再算 Dice。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .backbone import build_backbone
from .heads import (ClsHead, CoarseSegDecoder, FineSegDecoder,
                    HighResSegDecoder, MaculaHead)
from .lora import inject_tuning, freeze_all, count_parameters


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def n_up_for(src: int, dst: int) -> int:
    """需要多少次 x2 上采样把 src 边长升到约 dst。"""
    if dst <= src:
        return 0
    return int(round(math.log2(dst / src)))


@torch.no_grad()
def boxes_from_disc(disc: torch.Tensor, thresh: float = 0.5,
                    margin: float = 0.3) -> torch.Tensor:
    """从 disc 前景 (B,H,W 概率或 0/1) 求归一化框 (B,4) = (x1,y1,x2,y2)∈[0,1]。

    对前景为空的样本回退成整图框。再按 margin 外扩并 clamp 到 [0,1]。
    """
    B, H, W = disc.shape
    boxes = []
    for b in range(B):
        ys, xs = torch.where(disc[b] > thresh)
        if ys.numel() == 0:
            boxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0],
                                      device=disc.device))
            continue
        x1 = xs.min().float() / max(W - 1, 1)
        x2 = xs.max().float() / max(W - 1, 1)
        y1 = ys.min().float() / max(H - 1, 1)
        y2 = ys.max().float() / max(H - 1, 1)
        bw = (x2 - x1).clamp(min=1e-3)
        bh = (y2 - y1).clamp(min=1e-3)
        x1 = (x1 - margin * bw).clamp(0, 1)
        x2 = (x2 + margin * bw).clamp(0, 1)
        y1 = (y1 - margin * bh).clamp(0, 1)
        y2 = (y2 + margin * bh).clamp(0, 1)
        boxes.append(torch.stack([x1, y1, x2, y2]))
    return torch.stack(boxes, dim=0)


def roi_align_norm(feat: torch.Tensor, norm_boxes: torch.Tensor,
                   out_size: int) -> torch.Tensor:
    """在特征图上按归一化框做 roi_align。

    feat:       (B, C, H, W)
    norm_boxes: (B, 4) 归一化 (x1,y1,x2,y2)
    返回:        (B, C, out_size, out_size)
    """
    B, C, H, W = feat.shape
    feat = feat.float()  # roi_align 要 float
    scale = torch.tensor([W - 1, H - 1, W - 1, H - 1],
                         device=feat.device, dtype=feat.dtype)
    px = norm_boxes.to(feat.dtype) * scale                 # 像素坐标
    idx = torch.arange(B, device=feat.device, dtype=feat.dtype).unsqueeze(1)
    rois = torch.cat([idx, px], dim=1)                     # (B,5)
    return roi_align(feat, rois, output_size=(out_size, out_size),
                     spatial_scale=1.0, aligned=True)


@torch.no_grad()
def paste_roi_to_full(fine_logits: torch.Tensor, boxes: torch.Tensor,
                      full_hw) -> torch.Tensor:
    """把每个样本的 ROI 内精细 logits 放回全图位置 (仅评测用)。

    fine_logits: (B, C, fh, fw) ROI 内坐标系
    boxes:       (B, 4) 归一化框
    full_hw:     (H, W)
    返回:         (B, C, H, W) 框外填很负的值 (sigmoid≈0)
    """
    B, C, fh, fw = fine_logits.shape
    H, W = full_hw
    out = fine_logits.new_full((B, C, H, W), -20.0)
    for b in range(B):
        x1 = int(round(float(boxes[b, 0]) * (W - 1)))
        y1 = int(round(float(boxes[b, 1]) * (H - 1)))
        x2 = int(round(float(boxes[b, 2]) * (W - 1)))
        y2 = int(round(float(boxes[b, 3]) * (H - 1)))
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)
        roi = F.interpolate(fine_logits[b:b + 1], size=(y2 - y1, x2 - x1),
                            mode="bilinear", align_corners=False)
        out[b:b + 1, :, y1:y2, x1:x2] = roi
    return out


# --------------------------------------------------------------------------- #
# 多任务模型
# --------------------------------------------------------------------------- #
class MultiTaskModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 1) 骨干
        self.backbone = build_backbone(cfg.backbone)

        # 2) 冻结主干 → 注入 LoRA/Adapter → 按需放开 pos_embed
        freeze_all(self.backbone)
        self.n_injected = inject_tuning(self.backbone.vit, cfg.lora)
        self.backbone.vit.pos_embed.requires_grad_(bool(cfg.backbone.train_pos_embed))

        D = cfg.backbone.embed_dim
        n_levels = len(cfg.backbone.out_indices)
        in_chs = [D] * n_levels
        gh = cfg.backbone.img_size[0] // cfg.backbone.patch_size

        # 3) 分类头
        self.cls_head = ClsHead(D, cfg.cls.num_classes, cfg.cls.dropout)

        # 4) 分割头
        self.seg_mode = cfg.seg.mode
        if self.seg_mode == "coarse2fine":
            self.coarse_seg = CoarseSegDecoder(
                in_chs, cfg.seg.decoder_ch, cfg.seg.num_classes,
                n_up=n_up_for(gh, cfg.seg.coarse_out))
            self.fine_seg = FineSegDecoder(
                cfg.seg.decoder_ch, cfg.seg.decoder_ch, cfg.seg.num_classes,
                n_up=n_up_for(cfg.seg.fine_in, cfg.seg.fine_out))
        elif self.seg_mode == "highres":
            self.highres_seg = HighResSegDecoder(
                in_chs, cfg.seg.decoder_ch, cfg.seg.num_classes,
                n_up=n_up_for(gh, cfg.seg.highres_out))
        else:
            raise ValueError("seg.mode 必须是 'coarse2fine' 或 'highres'")

        # 5) 黄斑头
        self.macula_head = MaculaHead(
            in_chs, cfg.macula.decoder_ch,
            n_up=n_up_for(gh, cfg.macula.heatmap_out),
            beta=cfg.macula.softargmax_beta)

    # ----- 前向 ----- #
    def forward(self, images: torch.Tensor,
                gt_disc: Optional[torch.Tensor] = None,
                use_gt_roi: bool = False) -> Dict[str, object]:
        feats = self.backbone(images)
        levels = feats["levels"]
        out: Dict[str, object] = {}

        # 分类
        out["cls_logits"] = self.cls_head(feats["pooled"])

        # 黄斑
        mac = self.macula_head(levels)
        out["macula_heatmap"] = mac["heatmap"]
        out["macula_coords"] = mac["coords"]
        out["macula_prob"] = mac["prob"]

        # 分割
        if self.seg_mode == "coarse2fine":
            coarse_logits, fuse_feat = self.coarse_seg(levels)
            out["coarse_logits"] = coarse_logits      # (B,2,coarse,coarse)

            # 决定 ROI 框来源
            # FIX(S-4): 训练全程 use_gt_roi=True、评测 use_gt_roi=False，
            # 精修解码器永远只见过完美对齐的 ROI —— 典型 exposure bias，
            # 很可能是 cup Dice 上不去的原因之一。trainer 会按 scheduled
            # sampling 逐 epoch 提高传 use_gt_roi=False 的概率。
            if use_gt_roi and gt_disc is not None:
                # teacher forcing: 用 GT disc (B,1,H,W 或 B,H,W)
                disc = gt_disc
                if disc.dim() == 4:
                    disc = disc[:, 0]
                roi_boxes = boxes_from_disc(
                    disc, thresh=0.5, margin=self.cfg.seg.roi_margin)
            else:
                # 用粗分割预测的 disc (通道0) 概率，detach 不让 ROI 选择反传
                disc_prob = torch.sigmoid(coarse_logits[:, 0]).detach()
                roi_boxes = boxes_from_disc(
                    disc_prob, thresh=0.5, margin=self.cfg.seg.roi_margin)

            roi_feat = roi_align_norm(fuse_feat, roi_boxes,
                                      self.cfg.seg.fine_in)
            fine_logits = self.fine_seg(roi_feat)     # (B,2,fine,fine)
            out["fine_logits"] = fine_logits
            out["roi_boxes"] = roi_boxes
        else:  # highres
            out["seg_logits"] = self.highres_seg(levels)

        return out

    # ----- GradNorm 用的共享参数 (取最后一个 block 的可训练参数) ----- #
    def gradnorm_shared_params(self) -> List[nn.Parameter]:
        last_block = self.backbone.vit.blocks[-1]
        ps = [p for p in last_block.parameters() if p.requires_grad]
        if len(ps) == 0:
            # 万一最后一层没有可训练参数，退而取全部可训练参数的最后几个
            ps = [p for p in self.parameters() if p.requires_grad][-4:]
        return ps

    # ----- 参数统计 ----- #
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_summary(self) -> Dict[str, int]:
        """FIX(P0-2)/FIX(U-3): 之前只报 total/trainable 两个汇总数，导致文档里
        出现 13.5M / 1.35M每任务 / 4.7M 三个互相矛盾的说法。现在强制分解到
        LoRA / pos_embed / 任务头，让"参数高效"这个卖点可复算。"""
        from .config import param_breakdown
        bd = param_breakdown(self)
        bb_tr, bb_tot = count_parameters(self.backbone)
        bd.update(backbone_total=bb_tot, backbone_trainable=bb_tr,
                  lora_injected_layers=self.n_injected)
        return bd
