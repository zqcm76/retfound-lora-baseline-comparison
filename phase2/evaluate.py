# -*- coding: utf-8 -*-
"""
阶段二评测入口 —— 阈值在 val 标定、test 只报告，并落盘逐样本证据表。

FIX(S-3): 旧 trainer.evaluate 只回一个 AUC，逐样本概率被丢弃，bootstrap CI 和
          阈值标定都做不了。
FIX(U-7): 阈值绝不能在 test 上拟合（旧 phase4 就是这么干的）。
FIX(U-9): --seg-source 必须显式指定 fine / coarse —— 两者不可比。
          phase2 训练目标是 fine，phase3/phase4 旧路径用的是 coarse，
          "Coarse-to-Fine 是核心亮点"而 headline Dice 却来自 coarse。

用法:
    python evaluate.py --ckpt runs/phase2_v2/best.pth \
        --manifest data/data_all.csv --calibrate-on val --split test \
        --seg-source fine --out results/phase2_test
"""
from __future__ import annotations

import argparse
import json
import os

import _bootstrap  # noqa: F401
import pandas as pd
import torch
from torch.utils.data import DataLoader

from retfound_common.balancing import UncertaintyWeighting
from retfound_common.checkpoint import build_model_from_checkpoint
from retfound_common.dataset import MultiTaskManifestDataset, multitask_collate
from retfound_common.model import MultiTaskModel
from retfound_common.stats import (bootstrap_auc, bootstrap_mean,
                                   expected_calibration_error, brier_score,
                                   fmt_ci, sens_spec_at,
                                   threshold_at_sensitivity, youden_threshold)
from retfound_common.trainer import Trainer


def make_trainer(model, cfg, device):
    balancer = UncertaintyWeighting(list(cfg.task_names),
                                    max_weight=cfg.balance.uw_max_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return Trainer(model, opt, balancer, scaler, cfg, device)


def loader_for(df, cfg):
    ds = MultiTaskManifestDataset(df, img_size=cfg.backbone.img_size, train=False)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                      num_workers=cfg.train.num_workers,
                      collate_fn=multitask_collate)


def main():
    ap = argparse.ArgumentParser("Phase 2 evaluation")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--calibrate-on", default="val")
    ap.add_argument("--target-sensitivity", type=float, default=0.90)
    ap.add_argument("--seg-source", choices=["fine", "coarse"], default="fine")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.calibrate_on == args.split:
        raise SystemExit(
            f"拒绝执行：--calibrate-on 与 --split 都是 '{args.split}'。"
            f"在 test 上拟合阈值再在 test 上报告 sens/spec 是样本内乐观估计 (FIX(U-7))。")

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    model, cfg, ckpt = build_model_from_checkpoint(args.ckpt, MultiTaskModel,
                                                   map_location=args.device)
    model.to(device).eval()
    trainer = make_trainer(model, cfg, device)
    print(f"[eval] ckpt epoch={ckpt.get('epoch')} seed={ckpt.get('seed')} "
          f"git={str(ckpt.get('git'))[:8]} seg_source={args.seg_source}")

    df = pd.read_csv(args.manifest)
    cal_df = df[df[args.split_col] == args.calibrate_on].reset_index(drop=True)
    test_df = df[df[args.split_col] == args.split].reset_index(drop=True)
    if cal_df.empty or test_df.empty:
        raise SystemExit(f"split 为空: calibrate={len(cal_df)} eval={len(test_df)}")

    # 1) 在 val 上标定阈值
    _, cal_ev = trainer.evaluate(loader_for(cal_df, cfg), return_evidence=True,
                                 seg_source=args.seg_source)
    thr_y = youden_threshold(cal_ev["cls_label"], cal_ev["cls_prob"])
    thr_s = threshold_at_sensitivity(cal_ev["cls_label"], cal_ev["cls_prob"],
                                     args.target_sensitivity)
    print(f"[eval] 阈值在 '{args.calibrate_on}' 上标定: "
          f"youden={thr_y:.4f}  sens>={args.target_sensitivity}: {thr_s:.4f}")

    # 2) 在 test 上只报告
    m, ev = trainer.evaluate(loader_for(test_df, cfg), return_evidence=True,
                             threshold=thr_y, seg_source=args.seg_source)
    ev.to_csv(os.path.join(args.out, f"evidence_{args.split}.csv"), index=False)

    rep = {"split": args.split, "n_images": len(test_df), "ckpt": args.ckpt,
           "seg_source": args.seg_source,
           "img_size": list(cfg.backbone.img_size),
           "threshold_fitted_on": args.calibrate_on}
    ci = bootstrap_auc(ev["cls_label"], ev["cls_prob"], n_boot=args.n_boot)
    rep["auc"] = fmt_ci(ci); rep.update({f"auc_{k}": v for k, v in ci.items()})
    for tag, t in (("youden", thr_y), (f"sens{int(args.target_sensitivity*100)}", thr_s),
                   ("fixed0.5", 0.5)):
        rep[f"op_{tag}"] = {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in sens_spec_at(ev["cls_label"],
                                                     ev["cls_prob"], t).items()}
    rep["calibration"] = {"brier": round(brier_score(ev["cls_label"], ev["cls_prob"]), 4),
                          "ece": round(expected_calibration_error(
                              ev["cls_label"], ev["cls_prob"]), 4)}
    for col in ("dice_disc", "dice_cup", "macula_err_img", "macula_err_disc"):
        if col in ev.columns:
            rep[col] = fmt_ci(bootstrap_mean(ev[col], n_boot=args.n_boot),
                              key="mean", nd=4)
    if "macula_err_disc" in ev.columns:
        # 与 phase4 的 loc_success_rate 同口径，避免 0.341 与 0.013 再被混比
        rep["macula_success_rate@1disc"] = round(
            float((ev["macula_err_disc"].dropna() <= 1.0).mean()), 4)
    rep["_note"] = ("macula_err_img 单位=图像边长；macula_err_disc 单位=视盘直径。"
                    "两者相差约 8 倍，绝不可并排比较 (FIX(P0-6))。")

    path = os.path.join(args.out, f"report_{args.split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\n[eval] 写入 {path}")


if __name__ == "__main__":
    main()
