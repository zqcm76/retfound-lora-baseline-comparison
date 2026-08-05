# -*- coding: utf-8 -*-
"""
阶段二训练入口（LoRA-RETFound 多任务）。

示例:
  # dummy 数据快速跑通
  python train.py --preset tiny --dummy

  # 真实训练（README 里的命令必须是这一条完整版本 —— FIX(P0-5)）
  python train.py --preset full \
      --retfound ../RETFound_MAE/RETFound_mae_natureCFP.pth \
      --manifest data/data_all.csv --split-col split \
      --balancer uncertainty --seg-mode coarse2fine \
      --focal-alpha 0.1,0.9 --label-smoothing 0.05 \
      --head-lr 1e-3 --epochs 80 --out-dir ./runs/phase2_v2
"""
from __future__ import annotations

import argparse
import json
import math
import os

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import DataLoader

from retfound_common.balancing import GradNorm, UncertaintyWeighting
from retfound_common.checkpoint import save_checkpoint
from retfound_common.config import (Config, format_param_breakdown,
                                    param_breakdown, tiny_config)
from retfound_common.dataset import (DummyMultiTaskDataset,
                                     MultiTaskManifestDataset,
                                     make_balanced_sampler, multitask_collate)
from retfound_common.model import MultiTaskModel
from retfound_common.stats import (bootstrap_auc, fmt_ci, youden_threshold)
from retfound_common.trainer import Trainer, build_scheduler


def build_balancer(cfg, device):
    m = cfg.balance.method
    if m == "uncertainty":
        return UncertaintyWeighting(list(cfg.task_names),
                                    max_weight=cfg.balance.uw_max_weight).to(device)
    if m == "gradnorm":
        return GradNorm(list(cfg.task_names), alpha=cfg.balance.gradnorm_alpha).to(device)
    if m == "fixed":
        class _Fixed(torch.nn.Module):
            def __init__(self, names, weights):
                super().__init__()
                self.names = names
                self.weights = dict(zip(names, weights))

            def forward(self, losses):
                total = 0.0
                for n in self.names:
                    if n in losses:
                        total = total + self.weights[n] * losses[n]
                return total, {f"w_{n}": self.weights[n] for n in self.names}
        return _Fixed(list(cfg.task_names), cfg.balance.fixed_weights).to(device)
    raise ValueError(f"未知 balancer: {m}")


def parse_args():
    ap = argparse.ArgumentParser("Phase 2: LoRA-RETFound multi-task training")
    ap.add_argument("--preset", choices=["full", "tiny"], default="full")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--retfound", type=str, default=None)
    ap.add_argument("--train-manifest", type=str, default=None)
    ap.add_argument("--val-manifest", type=str, default=None)
    ap.add_argument("--manifest", type=str, default=None,
                    help="含 split 列的单张 CSV")
    ap.add_argument("--split-col", type=str, default="split")
    ap.add_argument("--balancer", choices=["uncertainty", "gradnorm", "fixed"])
    ap.add_argument("--seg-mode", choices=["coarse2fine", "highres"])
    ap.add_argument("--lora-mode", choices=["lora", "adapter"])
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--focal-alpha", type=str, default=None,
                    help="逗号分隔的每类权重，例 '0.1,0.9'")
    # FIX(S-5): label_smoothing 之前只存在于 FocalLoss 的默认参数里，
    # 文档却把它列为关键超参 —— 不可配置、不可覆盖、不被记录。
    ap.add_argument("--label-smoothing", type=float, default=None)
    ap.add_argument("--no-balanced-sampler", action="store_true")
    ap.add_argument("--scheduler", choices=["warmup_cosine", "none"])
    ap.add_argument("--warmup-epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--head-lr", type=float, default=None)
    ap.add_argument("--roi-pred-prob", type=float, default=None,
                    help="FIX(S-4): ROI scheduled sampling 的上限；0 = 全程 teacher forcing")
    ap.add_argument("--freeze-lora", action="store_true",
                    help="FIX(P1-5): linear probe 对照 —— 不注入 LoRA，只训任务头")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


def apply_overrides(cfg: Config, args) -> Config:
    if args.retfound:            cfg.backbone.pretrained_ckpt = args.retfound
    if args.balancer:            cfg.balance.method = args.balancer
    if args.seg_mode:            cfg.seg.mode = args.seg_mode
    if args.lora_mode:           cfg.lora.mode = args.lora_mode
    if args.lora_r is not None:  cfg.lora.r = args.lora_r; cfg.lora.alpha = 2 * args.lora_r
    if args.epochs is not None:  cfg.train.epochs = args.epochs
    if args.batch_size is not None: cfg.train.batch_size = args.batch_size
    if args.img_size is not None: cfg.backbone.img_size = (args.img_size, args.img_size)
    if args.focal_alpha is not None:
        cfg.cls.focal_alpha = tuple(float(x) for x in args.focal_alpha.split(","))
    if args.label_smoothing is not None:
        cfg.cls.label_smoothing = args.label_smoothing
    if args.no_balanced_sampler: cfg.train.balanced_sampler = False
    if args.scheduler:           cfg.train.scheduler = args.scheduler
    if args.warmup_epochs is not None: cfg.train.warmup_epochs = args.warmup_epochs
    if args.lr is not None:      cfg.train.lr = args.lr
    if args.head_lr is not None: cfg.train.head_lr = args.head_lr
    if args.roi_pred_prob is not None: cfg.seg.roi_pred_prob_max = args.roi_pred_prob
    if args.out_dir:             cfg.train.out_dir = args.out_dir
    if args.seed is not None:    cfg.train.seed = args.seed
    if cfg.balance.method == "gradnorm" and cfg.train.accum_steps != 1:
        print("[warn] GradNorm 要求 accum_steps=1，已自动设为 1。")
        cfg.train.accum_steps = 1
    return cfg


def main():
    args = parse_args()
    cfg = apply_overrides(tiny_config() if args.preset == "tiny" else Config(), args)

    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device} preset={args.preset} balancer={cfg.balance.method} "
          f"seg_mode={cfg.seg.mode} seed={cfg.train.seed}")

    # ---- 数据 ----
    if args.manifest and not args.train_manifest:
        import pandas as pd
        df_all = pd.read_csv(args.manifest)
        col = args.split_col
        args.train_manifest = df_all[df_all[col] == "train"].reset_index(drop=True)
        if args.val_manifest is None:
            args.val_manifest = df_all[df_all[col] == "val"].reset_index(drop=True)

    if args.dummy or args.train_manifest is None:
        if not args.dummy:
            # FIX(T-8) 同类问题：不要静默降级到 dummy 数据
            raise SystemExit(
                "没有提供 --manifest / --train-manifest。若确实要跑合成数据自检，"
                "请显式加 --dummy；静默降级会让人以为跑的是真实实验。")
        H, W = cfg.backbone.img_size
        train_ds = DummyMultiTaskDataset(n=8, img_size=(H, W),
                                         num_classes=cfg.cls.num_classes, seed=1)
        val_ds = DummyMultiTaskDataset(n=4, img_size=(H, W),
                                       num_classes=cfg.cls.num_classes, seed=2)
    else:
        train_ds = MultiTaskManifestDataset(args.train_manifest,
                                            img_size=cfg.backbone.img_size, train=True)
        has_val = args.val_manifest is not None and (
            not hasattr(args.val_manifest, "empty") or not args.val_manifest.empty)
        val_ds = (MultiTaskManifestDataset(args.val_manifest,
                                           img_size=cfg.backbone.img_size, train=False)
                  if has_val else None)

    sampler = None
    if cfg.train.balanced_sampler:
        sampler = make_balanced_sampler(train_ds, cfg.cls.num_classes)
        print("[info] 类别平衡采样:", "已启用" if sampler else "不可用，退回 shuffle")
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=cfg.train.num_workers,
                              collate_fn=multitask_collate, drop_last=False)
    val_loader = (DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                             num_workers=cfg.train.num_workers,
                             collate_fn=multitask_collate) if val_ds else None)

    # ---- 模型 ----
    if args.freeze_lora:
        # FIX(P1-5): linear probe 对照。README 的"为什么用 LoRA 而不是全量微调"
        # 之前只有文字论证、零实验。一次运行就能回答 LoRA 到底买到了什么。
        cfg.lora.targets = ()
        print("[info] --freeze-lora: 不注入 LoRA，仅训练任务头 (linear probe 对照)")
    model = MultiTaskModel(cfg).to(device)
    if cfg.backbone.pretrained_ckpt:
        print(model.backbone.load_pretrained(cfg.backbone.pretrained_ckpt))

    # FIX(P0-2)/FIX(U-3): 参数量必须分解可复算，不能再出现 13.5M / 1.35M每任务 / 4.7M
    bd = param_breakdown(model)
    print("[params]\n" + format_param_breakdown(bd))

    # ---- 优化器（分组学习率 + 分组 min_lr，FIX(S-7)）----
    balancer = build_balancer(cfg, device)
    bb_ids = {id(p) for p in model.backbone.parameters()}
    bb_params = [p for p in model.trainable_parameters() if id(p) in bb_ids]
    head_params = [p for p in model.trainable_parameters() if id(p) not in bb_ids]
    head_params += [p for p in balancer.parameters() if p.requires_grad]
    param_groups = [
        {"params": bb_params, "lr": cfg.train.lr, "min_lr": cfg.train.min_lr},
        {"params": head_params, "lr": cfg.train.head_lr, "min_lr": cfg.train.head_min_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.wd)
    print(f"[info] backbone={len(bb_params)} tensors (lr={cfg.train.lr}, "
          f"min={cfg.train.min_lr}) | heads={len(head_params)} tensors "
          f"(lr={cfg.train.head_lr}, min={cfg.train.head_min_lr})")

    use_amp = cfg.train.amp and device.type == "cuda"
    # FIX(S-10): torch.cuda.amp.GradScaler 在 torch 2.3+ 已弃用
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, cfg.train.accum_steps))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    trainer = Trainer(model, optimizer, balancer, scaler, cfg, device,
                      scheduler=scheduler)

    os.makedirs(cfg.train.out_dir, exist_ok=True)
    with open(os.path.join(cfg.train.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg.to_dict(), "param_breakdown": bd}, f,
                  indent=2, ensure_ascii=False)

    best = -1.0
    for epoch in range(cfg.train.epochs):
        tr = trainer.train_one_epoch(train_loader, epoch)
        print(f"[epoch {epoch}] train: " +
              " ".join(f"{k}={v:.4f}" for k, v in tr.items()))
        if val_loader is None:
            continue
        # 阈值在 val 上标定 —— 这里就是 val，故合法 (FIX(U-7))
        ev, ev_df = trainer.evaluate(val_loader, return_evidence=True,
                                     seg_source="fine")
        thr = 0.5
        if "cls_prob" in ev_df.columns and ev_df["cls_label"].nunique() > 1:
            thr = youden_threshold(ev_df["cls_label"], ev_df["cls_prob"])
            ev, ev_df = trainer.evaluate(val_loader, return_evidence=True,
                                         threshold=thr, seg_source="fine")
        print(f"[epoch {epoch}] val:   " +
              " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in ev.items()))
        score = ev.get("auc", 0.0)
        if isinstance(score, float) and score == score and score > best:
            best = score
            # FIX(S-2): 存完整配置 + seed + 指标 + git commit + 标定好的阈值
            save_checkpoint(os.path.join(cfg.train.out_dir, "best.pth"),
                            model, cfg, epoch=epoch, metrics=ev,
                            balancer=balancer, threshold=thr,
                            extra={"param_breakdown": bd})
            ev_df.to_csv(os.path.join(cfg.train.out_dir, "val_evidence.csv"),
                         index=False)

    save_checkpoint(os.path.join(cfg.train.out_dir, "last.pth"), model, cfg,
                    epoch=cfg.train.epochs - 1, balancer=balancer,
                    extra={"param_breakdown": bd})
    if best > 0:
        print(f"[done] best val AUC = {best:.4f}  (逐样本证据表已落盘，可做 bootstrap CI)")
    print("[done] saved to", cfg.train.out_dir)


if __name__ == "__main__":
    main()
