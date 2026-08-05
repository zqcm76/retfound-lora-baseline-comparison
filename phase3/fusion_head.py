"""
fusion_head.py — clinical-feature fusion classifier for stage 3.

Goal: compare three configurations the plan calls for
    "pure_e2e"      : classify from image features only (the stage-2 baseline,
                      re-exposed here so the comparison uses identical code).
    "cdr_two_stage" : concat HARD CDR (detached) with image features, fine-tune
                      classifier only. Segmentation is frozen / not updated by
                      the classification loss.
    "cdr_soft"      : concat SOFT CDR (differentiable) with image features.
                      Classification loss back-props through CDR into the
                      segmentation branch.

The module is intentionally thin: stage 2 already produces
    - img_feat : [B, D]    pooled backbone feature (RETFound ViT-L CLS or GAP)
    - seg_logits:[B,3,H,W] segmentation logits
We fuse and emit class logits. Whether seg_logits carries gradient is decided
by the CALLER (detach for two-stage), but we also guard it internally so a
mis-wired caller can't silently turn "two_stage" into "soft".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cdr import SoftCDR, compute_cdr_hard


# Normalization stats for the 3 CDR features. vCDR/hCDR/aCDR all live in [0,1]
# but cluster around different means; we standardize with fixed, documented
# constants rather than BatchNorm so that inference on a single image (demo,
# ONNX) behaves identically to training. These are rough population values for
# disc-cup ratios in glaucoma-screening data; they only need to be stable, not
# perfect, because a LayerNorm-free linear layer can absorb the residual scale.
# FIX(T-10): 这三对数原文注释是 "rough population values ... only need to be
# stable, not perfect"。逻辑上没错（后面的线性层能吸收残余尺度），但它们
# **没有来源**，也从没和 REFUGE2 训练集实际分布核对过。
# phase1/metrics.compute_cdr 已经能算出真实的 vCDR/hCDR/aCDR 分布，
# 用 fit_cdr_stats() 从训练集算一次填进去，是十分钟的事。
_CDR_MEAN = torch.tensor([0.55, 0.50, 0.30])
_CDR_STD = torch.tensor([0.18, 0.18, 0.15])


def fit_cdr_stats(cdr_values) -> tuple[torch.Tensor, torch.Tensor]:
    """从训练集实际的 [N,3] CDR 值算标准化常量，替换上面的拍脑袋默认值。"""
    v = torch.as_tensor(cdr_values, dtype=torch.float32)
    return v.mean(0), v.std(0).clamp_min(1e-3)


class CDRFusionClassifier(nn.Module):
    def __init__(
        self,
        img_feat_dim: int,
        num_classes: int = 2,
        mode: str = "cdr_two_stage",
        hidden: int = 256,
        dropout: float = 0.3,
        soft_cdr_tau_col: float = 0.05,
        soft_cdr_tau_occ: float = 0.05,
    ):
        super().__init__()
        # FIX(T-4): pure_e2e_unfrozen = 无 CDR 但解冻分割，消融的第四臂
        if mode not in {"pure_e2e", "pure_e2e_unfrozen", "cdr_two_stage", "cdr_soft"}:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.num_classes = num_classes

        self.soft_cdr = SoftCDR(tau_col=soft_cdr_tau_col, tau_occ=soft_cdr_tau_occ)

        self._uses_cdr = mode in ("cdr_two_stage", "cdr_soft")
        cdr_dim = 3 if self._uses_cdr else 0
        in_dim = img_feat_dim + cdr_dim

        # A small MLP head. For pure_e2e this is just a classifier on img_feat,
        # so the three modes differ ONLY in their input, keeping params close
        # (the head differs by 3*hidden weights — negligible, and reported).
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

        # register normalization buffers (move with .to(device), saved in ckpt)
        self.register_buffer("cdr_mean", _CDR_MEAN.clone())
        self.register_buffer("cdr_std", _CDR_STD.clone())

    def _make_cdr_feat(self, seg_logits: torch.Tensor) -> torch.Tensor:
        """Return [B,3] normalized CDR features appropriate to the mode."""
        if self.mode == "cdr_soft":
            cdr = self.soft_cdr(seg_logits=seg_logits)              # differentiable
        elif self.mode == "cdr_two_stage":
            cdr = compute_cdr_hard(seg_logits)                      # detached inside
        else:
            raise RuntimeError(f"{self.mode} has no CDR features")
        # standardize; clamp raw soft ratios for the *feature* only (not for any
        # reported metric) to keep inputs bounded if soft cup slightly > disc.
        cdr = cdr.clamp(0.0, 1.5)
        return (cdr - self.cdr_mean) / self.cdr_std

    def forward(
        self,
        img_feat: torch.Tensor,
        seg_logits: torch.Tensor | None = None,
        return_cdr: bool = False,
    ):
        """
        img_feat   : [B, D]
        seg_logits : [B, 3, H, W], required unless mode == 'pure_e2e'
        """
        if not self._uses_cdr:
            x = img_feat
            cdr_used = None
        else:
            if seg_logits is None:
                raise ValueError(f"mode {self.mode!r} needs seg_logits")
            # Hard guard: in two-stage we must NOT let grad reach seg, even if
            # the caller forgot to detach. compute_cdr_hard already detaches;
            # this is belt-and-suspenders for the soft-vs-two_stage boundary.
            if self.mode == "cdr_two_stage":
                seg_logits = seg_logits.detach()
            cdr_feat = self._make_cdr_feat(seg_logits)             # [B,3]
            x = torch.cat([img_feat, cdr_feat], dim=1)
            cdr_used = cdr_feat

        logits = self.net(x)
        if return_cdr:
            return logits, cdr_used
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, H, W = 4, 1024, 48, 48           # D=1024 ~ ViT-L width
    img_feat = torch.randn(B, D)
    seg_logits = torch.randn(B, 3, H, W, requires_grad=True)

    for mode in ["pure_e2e", "pure_e2e_unfrozen", "cdr_two_stage", "cdr_soft"]:
        head = CDRFusionClassifier(img_feat_dim=D, num_classes=2, mode=mode)
        seg = seg_logits.clone()
        out = head(img_feat, seg if head._uses_cdr else None)
        loss = out.sum()
        # check grad reaches seg only in cdr_soft
        seg.retain_grad()
        loss.backward(retain_graph=True)
        reaches_seg = (seg.grad is not None) and (seg.grad.abs().sum().item() > 0)
        n_params = sum(p.numel() for p in head.parameters())
        print(f"{mode:14s} out{tuple(out.shape)}  grad->seg={reaches_seg}  head_params={n_params:,}")
