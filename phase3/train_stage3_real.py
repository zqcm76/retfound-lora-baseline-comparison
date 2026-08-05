# -*- coding: utf-8 -*-
"""
train_stage3_real.py — 阶段三真实数据训练入口。

把阶段二的 best.pth checkpoint 加载进来，套上 CDR 融合头，
在真实 REFUGE2/PALM 等数据上做三种融合模式消融。

示例:
    # 全量消融 (三种模式逐个跑, 结果打表)
    python3 train_stage3_real.py \\
        --stage2-ckpt runs/phase2/best.pth \\
        --train-manifest data/train.csv \\
        --val-manifest   data/val.csv \\
        --mode all \\
        --epochs 15

    # 只跑 cdr_soft
    python3 train_stage3_real.py \\
        --stage2-ckpt runs/phase2/best.pth \\
        --train-manifest data/train.csv \\
        --val-manifest   data/val.csv \\
        --mode cdr_soft \\
        --epochs 15

    # 快速验证 (dummy 数据，不需要任何外部文件)
    python3 train_stage3_real.py --dummy --mode all --epochs 5
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401

from retfound_common.config import Config, tiny_config
from retfound_common.dataset import (DummyMultiTaskDataset,
                                     MultiTaskManifestDataset, multitask_collate)
from retfound_common.stats import bootstrap_auc, bootstrap_auc_delta, fmt_ci
from stage2_adapter import (Stage2Adapter, Stage3DataLoaderWrapper,
                            load_stage2_model)
from stage2_iface import MockStage2
from train_stage3 import Stage3Config, Stage3Trainer, _make_synthetic_loader


# --------------------------------------------------------------------------- #
# 帮助函数
# --------------------------------------------------------------------------- #
def _make_real_loader(manifest, img_size, batch_size, train, num_workers, seg_size):
    """从 manifest 构建 stage3 格式的 DataLoader。"""
    ds = MultiTaskManifestDataset(manifest, img_size=img_size, train=train)
    raw = DataLoader(ds, batch_size=batch_size, shuffle=train,
                     num_workers=num_workers, collate_fn=multitask_collate,
                     drop_last=False)
    return Stage3DataLoaderWrapper(raw, seg_size=seg_size)


def _make_dummy_loader(n, img_size, batch_size, seed, seg_size):
    """dummy 数据的 stage3 格式 DataLoader (通过 _make_synthetic_loader)。"""
    return _make_synthetic_loader(n=n, batch=batch_size,
                                  seg_size=seg_size, seed=seed)


def _print_table(rows):
    cols = ["mode", "auc", "auc_ci", "dice_disc", "dice_cup",
            "cdr_degenerate_rate", "trainable_params", "seg_unfrozen", "grad_to_seg"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Stage 3: CDR fusion fine-tuning")
    ap.add_argument("--stage2-ckpt", type=str, default=None,
                    help="阶段二 best.pth 路径 (dummy 模式时不需要)")
    ap.add_argument("--train-manifest", type=str, default=None)
    ap.add_argument("--val-manifest",   type=str, default=None)
    ap.add_argument("--manifest", type=str, default=None,
                    help="含 split 列的单张 CSV")
    ap.add_argument("--split-col", type=str, default="split")
    ap.add_argument("--dummy", action="store_true",
                    help="使用 synthetic dummy 数据 (不需要任何外部文件)")
    ap.add_argument("--mode", type=str, default="all",
                    choices=["pure_e2e", "pure_e2e_unfrozen",
                             "cdr_two_stage", "cdr_soft", "all"],
                    help="融合模式; 'all' 跑**四**臂并打消融表。"
                         "FIX(T-4): pure_e2e_unfrozen 是新增的对照臂 —— "
                         "无 CDR 但同样解冻分割，用来把'解冻'与'CDR'分开。")
    ap.add_argument("--label-smoothing", type=float, default=0.05,
                    help="FIX(T-3): 默认对齐 phase2 训练时的 0.05")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--seg-size", type=int, default=96,
                    help="CDR 用的分割图尺寸 (coarse_logits 会被插值到此尺寸)")
    ap.add_argument("--lr-head", type=float, default=1e-3,
                    help="分类头学习率")
    ap.add_argument("--lr-seg", type=float, default=1e-5,
                    help="cdr_soft 时分割头学习率 (小, 防止过度修改)")
    ap.add_argument("--seg-loss-weight", type=float, default=1.0)
    ap.add_argument("--focal-alpha", type=str, default="0.1,0.9",
                    help="focal loss 类别权重。FIX(T-3): 默认对齐 phase2 的 (0.1,0.9)；"
                         "旧版默认 None，于是 stage3 在 1:9 不平衡上不加权训练。")
    ap.add_argument("--soft-cdr-scale", type=float, default=0.1,
                    help="cls->seg 梯度缩放 (cdr_soft, 默认 0.1)")
    ap.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="./runs/phase3")
    ap.add_argument("--preset", choices=["full", "tiny"], default="full",
                    help="tiny=极小骨干, 用于快速测试")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[stage3] device={device}, mode={args.mode}, epochs={args.epochs}")

    # ---- 阶段二配置 ----
    # FIX(S-2)/FIX(T-1): 真实模式下配置从 checkpoint 恢复（见 load_stage2_model），
    # 这里的 s2cfg 只作为 dummy / fallback。旧版用 phase3 那份**分叉的** Config()
    # 默认值（focal_alpha=None, dropout=0.1, uw_max_weight=20, 无 head_lr），
    # 与 phase2 训练时的配置有 8 处不同。
    s2cfg = tiny_config() if args.preset == "tiny" else Config()
    s2cfg.backbone.img_size = (args.img_size, args.img_size)
    img_feat_dim = s2cfg.backbone.embed_dim

    # ---- 数据 ----
    if args.manifest and not args.train_manifest:
        import pandas as pd
        df_all = pd.read_csv(args.manifest)
        col = args.split_col
        args.train_manifest = df_all[df_all[col] == "train"].reset_index(drop=True)
        if args.val_manifest is None:
            args.val_manifest = df_all[df_all[col] == "val"].reset_index(drop=True)

    # FIX(T-8): 旧版 `python train_stage3_real.py` 不带任何参数时会**静默**走到
    # MockStage2 + 合成数据，输出格式和真实消融表一模一样 —— 而 1_GitHub_README
    # 的 Phase 3 快速开始写的正是这条裸命令。现在必须显式 --dummy。
    if args.train_manifest is None and not args.dummy:
        raise SystemExit(
            "没有提供 --manifest / --train-manifest。\n"
            "若确实要跑合成数据自检，请显式加 --dummy；\n"
            "静默降级会让读者以为拿到的是真实消融结果。")
    if args.stage2_ckpt is None and not args.dummy:
        raise SystemExit("真实模式必须提供 --stage2-ckpt。")

    if args.dummy or args.train_manifest is None:
        print("\n" + "!" * 70)
        print("!!  SYNTHETIC MODE —— MockStage2 + 合成数据，结果不可用于任何报告  !!")
        print("!" * 70 + "\n")
        train_loader = _make_dummy_loader(
            n=96, img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size, seed=1, seg_size=args.seg_size)
        val_loader = _make_dummy_loader(
            n=48, img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size, seed=2, seg_size=args.seg_size)
    else:
        train_loader = _make_real_loader(
            args.train_manifest, (args.img_size, args.img_size),
            args.batch_size, train=True,
            num_workers=args.num_workers, seg_size=args.seg_size)
        val_loader = _make_real_loader(
            args.val_manifest, (args.img_size, args.img_size),
            args.batch_size, train=False,
            num_workers=args.num_workers, seg_size=args.seg_size) \
            if args.val_manifest is not None else None

    # ---- Stage3Config ----
    focal_alpha = None
    if args.focal_alpha:
        focal_alpha = tuple(float(x) for x in args.focal_alpha.split(","))

    # soft-CDR warmup: 让 seg-loss anchor 先稳定约半程，再启用软 CDR 耦合
    steps_per_epoch = len(train_loader)
    warmup_steps = max(1, (steps_per_epoch * args.epochs) // 2)

    def make_s3cfg(mode):
        return Stage3Config(
            mode=mode,
            img_feat_dim=img_feat_dim,
            lr_head=args.lr_head,
            lr_seg=args.lr_seg,
            focal_alpha=focal_alpha,
            label_smoothing=args.label_smoothing,
            seg_loss_weight=args.seg_loss_weight,
            soft_cdr_seg_grad_scale=args.soft_cdr_scale,
            soft_cdr_warmup_steps=warmup_steps,
            amp=(not args.no_amp) and (device == "cuda"),
        )

    # ---- 加载阶段二模型 ----
    def make_stage2_model():
        """每次调用都重新加载，确保各 mode 从同一初始点出发。"""
        torch.manual_seed(args.seed)
        if args.dummy or args.stage2_ckpt is None:
            # dummy 模式: 用 MockStage2
            H = args.img_size
            seg_size_mock = H // (s2cfg.backbone.patch_size * 2)
            seg_size_mock = max(8, seg_size_mock)
            m = MockStage2(feat_dim=img_feat_dim, seg_size=args.seg_size)
            return m
        else:
            # 真实模式: 从 checkpoint 恢复配置 + 严格加载权重
            model_s2, real_cfg = load_stage2_model(args.stage2_ckpt, s2cfg, device)
            return Stage2Adapter(model_s2, seg_out_size=args.seg_size)

    # ---- 运行 ----
    modes = (["pure_e2e", "pure_e2e_unfrozen", "cdr_two_stage", "cdr_soft"]
             if args.mode == "all" else [args.mode])
    rows, evidences = [], {}

    for mode in modes:
        print(f"\n{'='*50}\n[stage3] mode = {mode}\n{'='*50}")
        torch.manual_seed(args.seed); np.random.seed(args.seed)

        model = make_stage2_model()
        cfg   = make_s3cfg(mode)
        trainer = Stage3Trainer(model, cfg, device=device)

        n_head = sum(p.numel() for p in trainer.head.parameters() if p.requires_grad)
        n_seg  = sum(p.numel() for p in trainer.model.segmentation_parameters()
                     if p.requires_grad)
        print(f"[stage3] trainable: head={n_head:,}  seg={n_seg:,}")

        trainer.fit(train_loader, val_loader, epochs=args.epochs, log_every=50)

        val, ev = (trainer.evaluate(val_loader, return_evidence=True)
                   if val_loader else ({}, None))
        print(f"[stage3] final val: {val}")
        ci = (bootstrap_auc(ev["cls_label"], ev["cls_prob"])
              if ev is not None else {})
        if ev is not None:
            evidences[mode] = ev
            ev.to_csv(os.path.join(args.out_dir, f"evidence_{mode}.csv"), index=False)

        rows.append({
            "mode": mode,
            "auc":  round(val.get("auc", float("nan")), 4),
            "auc_ci": fmt_ci(ci) if ci else "n/a",
            "dice_disc": round(val.get("dice_disc", float("nan")), 4),
            "dice_cup":  round(val.get("dice_cup", float("nan")), 4),
            "cdr_degenerate_rate": round(val.get("cdr_degenerate_rate", 0.0), 4),
            "trainable_params": n_head + n_seg,
            # FIX(T-4): 这两列必须并排显示 —— 旧表里 cdr_soft 的可训练参数
            # 是另外两臂的 38 倍，消融根本没控制变量。
            "seg_unfrozen": mode in ("cdr_soft", "pure_e2e_unfrozen"),
            "grad_to_seg": mode == "cdr_soft",
        })

        # 保存各 mode 的 checkpoint
        ckpt_path = os.path.join(args.out_dir, f"head_{mode}.pth")
        torch.save({"head": trainer.head.state_dict(),
                    "mode": mode, "val": val,
                    "img_feat_dim": img_feat_dim},
                   ckpt_path)
        print(f"[stage3] saved → {ckpt_path}")

    # ---- FIX(P1-2)/FIX(T-5): 配对 bootstrap，判断差异是否真的存在 ----
    if len(evidences) > 1:
        base = "pure_e2e" if "pure_e2e" in evidences else list(evidences)[0]
        print("\n" + "=" * 74)
        print(f"配对 bootstrap（相对 {base}）—— 单看点估计无法判断显著性")
        print("=" * 74)
        for m, ev in evidences.items():
            if m == base:
                continue
            d = bootstrap_auc_delta(ev["cls_label"], ev["cls_prob"],
                                    evidences[base]["cls_prob"])
            sig = "显著" if (d["lo"] > 0 or d["hi"] < 0) else "**未达显著**"
            print(f"  {m:20s} ΔAUC = {d['delta']:+.4f} "
                  f"[{d['lo']:+.4f}, {d['hi']:+.4f}]  p={d['p_two_sided']:.3f}  {sig}")
        if "cdr_soft" in evidences and "cdr_two_stage" in evidences:
            d = bootstrap_auc_delta(evidences["cdr_soft"]["cls_label"],
                                    evidences["cdr_soft"]["cls_prob"],
                                    evidences["cdr_two_stage"]["cls_prob"])
            print(f"\n  cdr_soft − cdr_two_stage = {d['delta']:+.4f} "
                  f"[{d['lo']:+.4f}, {d['hi']:+.4f}]  p={d['p_two_sided']:.3f}")
            print("  ⚠️ FIX(T-5): 在**已训练好**的分割器上，soft-CDR 的梯度按 sigmoid")
            print("     尾部衰减约 5507 倍，再乘 scale=0.1 → 总衰减约 5.5 万倍。")
            print("     所以这两臂在机制上几乎等价，差值理论上就该是零。")

    # ---- 打印消融表 ----
    if len(rows) > 1:
        print(f"\n{'='*50}")
        print("Stage-3 融合消融结果")
        print(f"{'='*50}")
        _print_table(rows)

    # 保存 JSON
    result_path = os.path.join(args.out_dir, "ablation_results.json")
    with open(result_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[stage3] JSON 结果 → {result_path}")


if __name__ == "__main__":
    main()
