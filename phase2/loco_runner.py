# -*- coding: utf-8 -*-
r"""
LOCO (Leave-One-Camera-Out) 跨设备泛化实验。

════════════════════════════════════════════════════════════════════════════
FIX(S-1) ★ 本文件最重要的修正：旧版把**训练集**当成跨设备评测集
════════════════════════════════════════════════════════════════════════════
旧代码：
    train = pd.concat([canon_a, zeiss])          # 训练集 = Canon-A + Zeiss
    eval_sets = {"canon_b":…, "zeiss": zeiss, "device_a":…}
    print("Canon-A→Zeiss (cross device): AUC = …")

Zeiss 那 400 张**就在训练集里**，却被标成 "cross device" 并写进了 README 的
LOCO 表（AUC 0.998）。那不是跨设备泛化，是训练集记忆。

三行的真实性质：
    Zeiss      0.998  → ❌ 训练集
    Canon-B    0.978  → ⚠️ 模型选择集（val，用于选 best.pth，乐观偏差）
    Device A   0.893  → ✅ 唯一未参与训练也未参与选择的干净数字

于是"训练分布内 AUC 0.978~0.998"这个叙述完全建立在前两行上，而 0.978→0.812
的"跨设备 gap"把**选择偏差**和**设备偏移**混在了一起。

本版做法：
  1. Canon 800 分层切成三份：Canon-A(训练) / Canon-B(选择) / Canon-C(干净留出)
     —— 这样才有一个既同相机、又没参与训练/选择的基线。
  2. 每个评测集显式带 role 标签，报告里必须打印出来。
  3. 所有路径走 CLI 参数，不再硬编码 E:\ / D:\Anaconda。

用法：
    python loco_runner.py --manifest ../phase3/data/data_all.csv \
        --retfound ../RETFound_MAE/RETFound_mae_natureCFP.pth \
        --out-root ./runs/loco --seeds 42 123 456 --epochs 80
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

from retfound_common.checkpoint import build_model_from_checkpoint
from retfound_common.model import MultiTaskModel
from retfound_common.stats import bootstrap_auc, fmt_ci

# ── 按图像尺寸识别相机 ──────────────────────────────────────────────────────
# 这张表**本来就是对的** —— 错的是 README 把 test 标成了 Canon CR-2。
# REFUGE2: Train 1200 = Zeiss 400 + Canon 800；Val/Test 各 400，来自另外两台相机。
DEVICE_SIZES = {
    (2124, 2056): "zeiss",       # train
    (1634, 1634): "canon",       # train
    (1940, 1940): "device_a",    # val   —— 未见相机
    (1848, 1848): "device_b",    # test  —— 未见相机
}

# 每个评测集的角色，必须随指标一起报告 (FIX(S-1))
ROLE = {
    "canon_a": "TRAIN (训练集，仅作 sanity check，不可当泛化指标)",
    "zeiss":   "TRAIN (训练集，仅作 sanity check，不可当泛化指标)",
    "canon_b": "VAL   (模型选择集，乐观偏差)",
    "canon_c": "HELD-OUT 同相机 (未训练、未选择) ★ 唯一干净的同相机基线",
    "device_a": "HELD-OUT 未见相机 ★ 干净",
    "device_b": "HELD-OUT 未见相机 ★ 干净",
}


def assign_device(path: str) -> str:
    try:
        with Image.open(path) as im:
            return DEVICE_SIZES.get(im.size, "unknown")
    except Exception:
        return "unknown"


def split_canon(canon_df: pd.DataFrame, seed: int = 42):
    """Canon 800 -> Canon-A 400 (训练) / Canon-B 200 (选择) / Canon-C 200 (干净留出)。

    FIX(S-1): 旧版只切 A/B 两份，B 同时当 val 又当"同设备天花板"，
    于是那个 0.978 里混着选择偏差。第三份 C 才是可比的同相机基线。
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=400, random_state=seed)
    a_idx, bc_idx = next(sss.split(canon_df, canon_df["label"]))
    a = canon_df.iloc[a_idx].copy(); a["loco_role"] = "canon_a"
    bc = canon_df.iloc[bc_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    b_idx, c_idx = next(sss2.split(bc, bc["label"]))
    b = bc.iloc[b_idx].copy(); b["loco_role"] = "canon_b"
    c = bc.iloc[c_idx].copy(); c["loco_role"] = "canon_c"
    return a, b, c


def make_train_csv(canon_a, zeiss, canon_b, out_path: Path):
    train = pd.concat([canon_a, zeiss]).copy(); train["split"] = "train"
    val = canon_b.copy(); val["split"] = "val"
    df = pd.concat([train, val]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df


def run_training(csv_path: Path, out_dir: Path, args, seed: int, log_path: Path) -> bool:
    cmd = [sys.executable, "-u", str(Path(__file__).with_name("train.py")),
           "--preset", "full", "--manifest", str(csv_path), "--split-col", "split",
           "--balancer", "uncertainty", "--seg-mode", "coarse2fine",
           "--epochs", str(args.epochs), "--focal-alpha", "0.1,0.9",
           "--label-smoothing", "0.05", "--lr", "2e-4", "--head-lr", "1e-3",
           "--seed", str(seed), "--out-dir", str(out_dir)]
    if args.retfound and os.path.exists(args.retfound):
        cmd += ["--retfound", args.retfound]
    print("\n" + "=" * 64 + f"\nTraining: {out_dir.name}  (seed={seed}, "
          f"epochs={args.epochs})\n" + "=" * 64)
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        print(f"[ERROR] 训练失败 (exit {proc.returncode})，见 {log_path}")
        return False
    return True


def evaluate_per_device(ckpt_path: Path, eval_sets: dict, device_str: str,
                        n_boot: int = 2000):
    """加载 checkpoint 并在每个评测集上单独评。

    FIX(S-2): 用 build_model_from_checkpoint —— 从 ckpt 里恢复配置再建模型，
    并严格加载。旧版是 Config() 默认值 + strict=False，任何形状不匹配的头
    都会静默保持随机初始化。
    """
    import torch
    from torch.utils.data import DataLoader

    from retfound_common.balancing import UncertaintyWeighting
    from retfound_common.dataset import MultiTaskManifestDataset, multitask_collate
    from retfound_common.trainer import Trainer

    model, cfg, ckpt = build_model_from_checkpoint(
        str(ckpt_path), MultiTaskModel, map_location=device_str)
    model.to(device_str).eval()
    thr = float(ckpt.get("cls_threshold", 0.5))   # val 上标定好的阈值，直接沿用
    print(f"  ckpt: epoch={ckpt.get('epoch')} seed={ckpt.get('seed')} "
          f"git={str(ckpt.get('git'))[:8]} threshold={thr:.4f}")

    balancer = UncertaintyWeighting(list(cfg.task_names),
                                    max_weight=cfg.balance.uw_max_weight).to(device_str)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer = Trainer(model, opt, balancer, scaler, cfg, torch.device(device_str))

    results = {}
    for name, df in eval_sets.items():
        if df is None or len(df) == 0:
            continue
        ds = MultiTaskManifestDataset(df, img_size=cfg.backbone.img_size, train=False)
        loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers,
                            collate_fn=multitask_collate)
        m, ev = trainer.evaluate(loader, return_evidence=True, threshold=thr,
                                 seg_source="fine")
        if "cls_prob" in ev.columns:
            ci = bootstrap_auc(ev["cls_label"], ev["cls_prob"], n_boot=n_boot)
            m["auc_ci"] = fmt_ci(ci)
            m.update({f"auc_{k}": v for k, v in ci.items()})
        m["role"] = ROLE.get(name, "?")
        m["n"] = int(len(df))
        results[name] = m
        print(f"  {name:9s} n={len(df):4d}  AUC={m.get('auc_ci', m.get('auc'))}  "
              f"disc={m.get('dice_disc', float('nan')):.4f}  "
              f"cup={m.get('dice_cup', float('nan')):.4f}")
        print(f"            role = {m['role']}")
        results[name]["_evidence"] = ev
    return results


def main():
    ap = argparse.ArgumentParser("LOCO cross-camera experiment")
    # FIX(S-10)/FIX(U-11): 全部走参数，不再硬编码 E:\RETFound / D:\Anaconda
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--retfound", default=None)
    ap.add_argument("--out-root", default="./runs/loco")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)
    df["device"] = df["image_path"].apply(assign_device)
    print("\n全量设备分布:\n" + df["device"].value_counts().to_string())
    if (df["device"] == "unknown").any():
        print(f"[warn] {int((df['device']=='unknown').sum())} 张图尺寸不在 "
              f"DEVICE_SIZES 表里，请核对数据集版本。")

    train_df = df[df["split"] == "train"].copy()
    canon_all = train_df[train_df["device"] == "canon"]
    zeiss = train_df[train_df["device"] == "zeiss"].copy()
    zeiss["loco_role"] = "zeiss"

    all_rows = []
    for seed in args.seeds:
        print("\n" + "#" * 64 + f"\n# Seed {seed}\n" + "#" * 64)
        canon_a, canon_b, canon_c = split_canon(canon_all, seed=seed)
        for nm, d in (("Canon-A(训练)", canon_a), ("Canon-B(选择)", canon_b),
                      ("Canon-C(干净留出)", canon_c), ("Zeiss(训练)", zeiss)):
            print(f"  {nm:18s} n={len(d):4d}  pos={int(d['label'].sum())}")

        run_dir = out_root / f"seed{seed}"; run_dir.mkdir(exist_ok=True)
        csv_path = run_dir / "loco_train.csv"
        make_train_csv(canon_a, zeiss, canon_b, csv_path)

        if not args.skip_train:
            if not run_training(csv_path, run_dir, args, seed,
                                run_dir / "train_log.txt"):
                continue
        ckpt = run_dir / "best.pth"
        if not ckpt.exists():
            print(f"[ERROR] 找不到 checkpoint: {ckpt}"); continue

        print(f"\n--- 逐设备评测 (seed={seed}) ---")
        eval_sets = {
            "canon_a": canon_a,                       # 训练集 (sanity)
            "zeiss": zeiss,                           # 训练集 (sanity)
            "canon_b": canon_b,                       # 选择集
            "canon_c": canon_c,                       # ★ 干净同相机
            "device_a": df[df["split"] == "val"],     # ★ 干净未见相机
            "device_b": df[df["split"] == "test"],    # ★ 干净未见相机
        }
        res = evaluate_per_device(ckpt, eval_sets, args.device, args.n_boot)
        for name, m in res.items():
            ev = m.pop("_evidence", None)
            if ev is not None:
                ev.to_csv(run_dir / f"evidence_{name}.csv", index=False)
        with open(run_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str, ensure_ascii=False)
        all_rows.append({"seed": seed, **{f"{d}_{k}": v for d, m in res.items()
                                          for k, v in m.items()
                                          if isinstance(v, (int, float, str))}})

    # ---- 汇总 ----
    if not all_rows:
        return
    print("\n" + "=" * 78 + "\nLOCO 汇总\n" + "=" * 78)
    print("⚠️  canon_a / zeiss 是**训练集**，它们的 AUC 只是 sanity check，")
    print("    绝不能写进'跨设备泛化'的表格 —— 这正是 FIX(S-1) 修的问题。\n")
    print("可写进报告的对比：")
    print("    canon_c (干净同相机)  vs  device_a / device_b (干净未见相机)")
    print("    两者都没参与训练也没参与选择，差值才是纯粹的设备效应。\n")
    pd.DataFrame(all_rows).to_csv(out_root / "loco_summary.csv", index=False)
    print(f"已保存: {out_root / 'loco_summary.csv'}")
    if len(all_rows) > 1:
        print(f"\n--- 跨 {len(all_rows)} 个 seed 的均值±std (FIX(P1-3)) ---")
        for dev in ["canon_c", "device_a", "device_b"]:
            v = [r.get(f"{dev}_auc", np.nan) for r in all_rows]
            v = [x for x in v if isinstance(x, float) and x == x]
            if v:
                print(f"  {dev:9s}: {np.mean(v):.4f} ± {np.std(v):.4f}")


if __name__ == "__main__":
    main()
