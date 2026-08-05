"""
run_ablation.py — Stage 3 deliverable: the fusion comparison.

Trains the three configurations under IDENTICAL data/seed/optimizer settings
and prints one table:

    mode            AUC    dice_disc  dice_cup  trainable_params  grad->seg
    pure_e2e        ...
    cdr_two_stage   ...
    cdr_soft        ...

This is the "pure end-to-end vs CDR-enhanced" answer the plan asks for, plus
the differentiable-vs-two-stage fork as a third row. Swap MockStage2 for your
real loader to get publishable numbers; the harness is identical.

Run:
    python3 run_ablation.py
"""

from __future__ import annotations

import json

import numpy as np
import torch

import _bootstrap  # noqa: F401

from stage2_iface import MockStage2
from train_stage3 import Stage3Config, Stage3Trainer, _make_synthetic_loader


def run(make_model, train_loader, val_loader, epochs=10, device="cpu",
        img_feat_dim=256, seed=0):
    rows = []
    # size the soft-CDR warmup so it completes partway through training, so the
    # cdr_soft row reflects BOTH phases: anchor-led warmup, then the soft-CDR
    # nudge engaging. Roughly half the total optimizer steps.
    steps_per_epoch = len(train_loader)
    warmup = max(1, (steps_per_epoch * epochs) // 2)
    for mode in ["pure_e2e", "pure_e2e_unfrozen", "cdr_two_stage", "cdr_soft"]:
        torch.manual_seed(seed); np.random.seed(seed)
        model = make_model()
        cfg = Stage3Config(mode=mode, img_feat_dim=img_feat_dim, amp=False,
                           focal_alpha=(0.5, 0.5), seg_loss_weight=1.0,
                           soft_cdr_warmup_steps=warmup)
        tr = Stage3Trainer(model, cfg, device=device)
        tr.fit(train_loader, val_loader, epochs=epochs, log_every=0)
        val = tr.evaluate(val_loader)

        n_head = sum(p.numel() for p in tr.head.parameters() if p.requires_grad)
        n_seg = sum(p.numel() for p in tr.model.segmentation_parameters()
                    if p.requires_grad)
        rows.append({
            "mode": mode,
            "auc": round(val.get("auc", float("nan")), 4),
            "dice_disc": round(val.get("dice_disc", float("nan")), 4),
            "dice_cup": round(val.get("dice_cup", float("nan")), 4),
            "cdr_degenerate_rate": round(val.get("cdr_degenerate_rate", 0.0), 4),
            "trainable_params": n_head + n_seg,
            "grad_to_seg": mode == "cdr_soft",
        })
    return rows


def print_table(rows):
    cols = ["mode", "auc", "dice_disc", "dice_cup", "cdr_degenerate_rate",
            "trainable_params", "grad_to_seg"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    # FIX(T-8): 这个文件整个跑的是 MockStage2 + 合成数据，输出格式却与真实消融表
    # 一模一样。开源清单还把它列为 phase3 核心文件。必须显著标记。
    BANNER = "!" * 78
    print(f"\n{BANNER}\n!!  SYNTHETIC — MockStage2 + 合成数据。这些数字**不可用于任何报告**。"
          f"\n!!  真实消融请用: python train_stage3_real.py --stage2-ckpt ... --manifest ..."
          f"\n{BANNER}\n")
    train_loader = _make_synthetic_loader(n=96, batch=8, seed=1)
    val_loader = _make_synthetic_loader(n=48, batch=8, seed=2)

    rows = run(
        make_model=lambda: MockStage2(feat_dim=256, seg_size=48),
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        device="cpu",
        img_feat_dim=256,
        seed=0,
    )
    print(f"\n{BANNER}\n=== Stage-3 fusion ablation — SYNTHETIC MOCK，仅验证流水线 ===\n{BANNER}\n")
    print_table(rows)
    print("\nJSON:")
    print(json.dumps(rows, indent=2))
