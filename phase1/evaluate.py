# -*- coding: utf-8 -*-
"""
Phase 1 评测入口 —— 把 test_cls.ipynb / test_seg.ipynb 固化成可复现的脚本。

背景 (FIX(C-3) / FIX(P0-3) / FIX(P1-1)):
  baseline 在 REFUGE2 Test 上的实测结果**一直就在仓库里**，只是埋在两个
  notebook 的 cell 输出里，而且那两个 notebook 含硬编码绝对路径和用户名。
  实测值：
      ResNet34   test AUC 0.8775 (val 0.9310)
      UNet       test Disc/Cup Dice 0.9287 / 0.8448, vCDR MAE 0.0684
  而 README 主表里写的是 `~0.75` / `~0.90` / `~0.78` —— 三个占位数字全部
  低估真实 baseline，且方向一致地利于本方法。ResNet34 那格低估了 12.7 个点。

  正确结论是：基础模型的优势在**分割与 CDR**，不在分类。

用法：
    # 在 val 上标定阈值，然后在 test 上报告（唯一正确的顺序）
    python evaluate.py --config baseline.yaml --task cls \
        --ckpt outputs/baseline_resnet34_best.pt --split test \
        --calibrate-on val --out results/

    python evaluate.py --config baseline.yaml --task seg \
        --ckpt outputs/baseline_unet_best.pt --split test --out results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import metrics as M                                             # noqa: E402
from config import Config                                        # noqa: E402
from datasets import REFUGE2Dataset                              # noqa: E402
from engine import evaluate_classifier, evaluate_segmenter       # noqa: E402
from models import ResNet34Classifier, UNet                      # noqa: E402
from transforms import build_cls_transforms, build_seg_transforms  # noqa: E402
from utils import get_device, param_report, set_seed             # noqa: E402


def _loader(cfg, df, task, size):
    tf = (build_cls_transforms(size, False) if task == "cls"
          else build_seg_transforms(size, False))
    return DataLoader(REFUGE2Dataset(df, tf, task),
                      batch_size=cfg.train.batch_size, shuffle=False,
                      num_workers=cfg.data.num_workers)


def _split(df, name):
    if "split" not in df.columns:
        raise ValueError("索引 CSV 里没有 split 列，无法按 split 评测。")
    sub = df[df["split"] == name].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"split='{name}' 没有任何行。可选: "
                         f"{sorted(df['split'].unique())}")
    return sub


def main() -> None:
    ap = argparse.ArgumentParser("Phase 1 baseline evaluation")
    ap.add_argument("--config", default="baseline.yaml")
    ap.add_argument("--index", default=None, help="覆盖 config 里的索引 CSV")
    ap.add_argument("--task", choices=["cls", "seg"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--calibrate-on", default="val",
                    help="在哪个 split 上标定分类阈值。**绝不能是 test。**")
    ap.add_argument("--target-sensitivity", type=float, default=0.90,
                    help="筛查运行点：固定 sensitivity 反推阈值")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.task == "cls" and args.calibrate_on == args.split:
        raise SystemExit(
            f"拒绝执行：--calibrate-on 与 --split 都是 '{args.split}'。\n"
            f"在 test 上拟合阈值再在 test 上报告 sens/spec 是样本内乐观估计 —— "
            f"这正是 FIX(U-7) 要消除的问题。")

    cfg = Config.from_yaml(args.config)
    if args.index:
        cfg.data.refuge_index = args.index
    set_seed(cfg.train.seed)
    device = get_device()
    os.makedirs(args.out, exist_ok=True)

    df_all = pd.read_csv(cfg.data.refuge_index)
    eval_df = _split(df_all, args.split)
    print(f"[eval] split={args.split}  n={len(eval_df)}  device={device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "cfg" not in ckpt:
        print("[warn] 这个 checkpoint 没有存配置（旧格式）。无法确认它是哪次实验的产物 —— "
              "见 FIX(C-11)。建议用新版 train_*.py 重训或重存。")
    else:
        print(f"[eval] ckpt epoch={ckpt.get('epoch')} seed={ckpt.get('seed')} "
              f"git={str(ckpt.get('git'))[:8]}")

    report: dict = {"split": args.split, "ckpt": args.ckpt, "task": args.task,
                    "n_images": int(len(eval_df))}

    # ------------------------------------------------------------------ cls
    if args.task == "cls":
        model = ResNet34Classifier(num_classes=1, pretrained=False).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        report.update(param_report(model))

        # 1) 在 calibrate split 上标定阈值
        cal_df = _split(df_all, args.calibrate_on)
        _, cal_p, cal_y, _ = evaluate_classifier(
            model, _loader(cfg, cal_df, "cls", cfg.data.cls_image_size),
            device, return_raw=True)
        thr_youden = M.youden_threshold(cal_p, cal_y)
        thr_sens = M.threshold_at_sensitivity(cal_p, cal_y, args.target_sensitivity)
        print(f"[eval] 阈值在 '{args.calibrate_on}' 上标定: "
              f"youden={thr_youden:.4f}  sens>={args.target_sensitivity}: {thr_sens:.4f}")

        # 2) 在 test 上只报告
        _, p, y, paths = evaluate_classifier(
            model, _loader(cfg, eval_df, "cls", cfg.data.cls_image_size),
            device, return_raw=True)

        ci = M.bootstrap_auc(p, y, n_boot=args.n_boot, seed=cfg.train.seed)
        report["auc"] = M.fmt_ci(ci)
        report.update({f"auc_{k}": v for k, v in ci.items()})
        for tag, thr in (("youden", thr_youden), ("sens90", thr_sens),
                         ("fixed0.5", 0.5)):
            m = M.classification_metrics(p, y, threshold=thr)
            report[f"op_{tag}"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in m.items()}
        report["calibration"] = {
            "threshold_fitted_on": args.calibrate_on,
            "brier": round(M.brier_score(p, y), 4),
            "ece": round(M.expected_calibration_error(p, y), 4),
        }
        pd.DataFrame({"image_path": paths, "cls_prob": p, "cls_label": y}) \
          .to_csv(os.path.join(args.out, f"evidence_cls_{args.split}.csv"), index=False)

    # ------------------------------------------------------------------ seg
    else:
        model = UNet(in_channels=3, num_classes=cfg.data.num_seg_classes,
                     base=cfg.model.unet_base_channels).to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
        report.update(param_report(model))

        agg, per, paths = evaluate_segmenter(
            model, _loader(cfg, eval_df, "seg", cfg.data.image_size),
            device, return_raw=True)
        report["aggregate"] = {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in agg.items()}
        for key in ("dice_disc", "dice_cup", "vCDR_mae"):
            ci = M.bootstrap_mean(per[key], n_boot=args.n_boot, seed=cfg.train.seed)
            report[key] = M.fmt_ci(ci, key="mean", nd=4)
        # 评测分辨率必须写进报告：phase1 用 512、phase2 用 384，Dice 不可直接比
        report["eval_resolution"] = cfg.data.image_size
        pd.DataFrame({"image_path": paths, **{k: v for k, v in per.items()}}) \
          .to_csv(os.path.join(args.out, f"evidence_seg_{args.split}.csv"), index=False)

    path = os.path.join(args.out, f"report_{args.task}_{args.split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[eval] 写入 {path}")
    print("[eval] 逐样本证据表已落盘 —— 这是做配对 bootstrap / DeLong 检验的前提。")


if __name__ == "__main__":
    main()
