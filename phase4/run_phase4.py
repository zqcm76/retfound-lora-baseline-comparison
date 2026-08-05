"""
run_phase4.py — orchestrate the whole robustness evaluation and write a report.

    python run_phase4.py --mode synthetic --out phase4_report
    python run_phase4.py --mode real --out phase4_report \
        --datasets REFUGE2 RIGA PALM Drishti-GS

Outputs (under --out):
    evidence_table.csv               one row per (model, image): everything
    per_dataset__<model>.csv         per-dataset metric suite, per model
    per_device__<model>.csv          per-device (camera) metric suite, per model
    degradation__<model>.csv         in-domain vs external deltas
    device_degradation__<model>.csv  drop attributed to each camera
    dataset_vs_device_spread.csv     does device explain more variance than dataset?
    comparison_table.csv / .md       baseline vs LoRA-RETFound (cost vs accuracy)
    figs/                            device bar charts + failure overlays
    REPORT.md                        human-readable summary tying it together

Real mode expects you to have wired data_adapters + model_adapters to your
Phase 1-3 artifacts (see the build_real_* hooks below).
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

import compare_baselines as CB
import config as C
import degradation as DG
import evaluate as EV
import failure_analysis as FA
import inference as INF
import stratify_device as SD


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_synthetic(out_dir: str, seed: int):
    import data_adapters as DA
    import model_adapters as MA
    image_root = os.path.join(out_dir, "_synthetic_images")
    datasets = DA.build_synthetic_suite(image_root, seed=seed)
    models = MA.build_synthetic_models(seed=seed)
    train_meta = MA.SYNTHETIC_TRAIN_META
    return models, datasets, train_meta


def build_real(args):
    r"""真实模式 —— 全部走 CLI 参数。

    FIX(U-2) ★ 旧版把三个路径**硬编码在函数体里**，其中
        STAGE2_CKPT = r"E:\RETFound\phase2\runs\loco\seed42\best.pth"
    指向的是 **LOCO** checkpoint，而不是 phase2_v2/best.pth。
    LOCO 模型只用 800 张训练 (Canon-A + Zeiss)、在 Canon-B 上选优。
    于是 headline test AUC 描述的是一个训练数据少 1/3、选优集不同、
    且（见 FIX(U-1)）没有 CDR 融合的模型 —— 和 README 讲的完全不是一回事。
    这也正是旧 TODO 里"Phase 4 checkpoint 与 Phase 2 v2 不一致"的答案。

    FIX(U-11): 旧签名收 dataset_names 却从不使用，README 文档化的 --datasets
    开关静默无效。
    """
    from phase4_adapter import build_real_phase4

    if not args.stage2_ckpt:
        raise SystemExit(
            "真实模式必须提供 --stage2-ckpt。\n"
            "注意区分 runs/phase2_v2/best.pth 与 runs/loco/seed42/best.pth —— "
            "后者只用了 800 张训练，两者不是同一个模型 (FIX(U-2))。")
    if not args.csv:
        raise SystemExit("真实模式必须提供 --csv (data_all.csv)。")

    return build_real_phase4(
        csv_path=args.csv,
        stage2_ckpt=args.stage2_ckpt,
        stage3_head=args.stage3_head,
        split=args.split,
        calib_split=args.calib_split,
        device_str=args.device,
        input_size=args.input_size,
        seg_source=args.seg_source,
        cdr_seg_size=args.cdr_seg_size,
        train_meta_json=args.train_meta_json,
        allow_partial_load=args.allow_partial_load,
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=True)
    except Exception:
        return df.to_string()


def write_report(out_dir: str, evidence: pd.DataFrame, comp_md: str,
                 spread: pd.DataFrame, artifacts: dict, models: list[str],
                 *, mode: str = "real", threshold: float = 0.5,
                 calib_split: str = "val", split: str = "test",
                 seg_source: str = "fine", uses_cdr: bool = False) -> str:
    lines = ["# Phase 4 — Robustness & Honest Evaluation\n"]
    if mode == "synthetic":
        # FIX(U-4)
        lines.append("> # ⚠️ SYNTHETIC DATA — NOT REAL RESULTS\n"
                     "> 本报告由卡通合成眼底图 + 合成检测器产生，仅用于验证流水线。\n"
                     "> 任何数字都不得引用。\n")
    # FIX(U-1)/(U-9)/(U-7): headline 到底描述哪个模型，必须写在最前面
    lines.append(
        f"**评测配置**（决定这些数字的含义）：\n\n"
        f"- 评测 split: `{split}`；分类阈值在 `{calib_split}` 上标定 = "
        f"`{threshold:.4f}`（评测集不参与拟合，FIX(U-7)）\n"
        f"- 分割口径: `{seg_source}`"
        f"{'（coarse-to-fine 精修，与训练目标一致）' if seg_source == 'fine' else '（**仅粗分割**，精修解码器未参与）'}"
        f"，FIX(U-9)\n"
        f"- CDR 融合: {'**启用**（Stage-3 融合头已加载）' if uses_cdr else '**未启用** —— 分类等价于 pure_e2e，不得声称临床特征融合提升了分类性能（FIX(U-1)）'}\n")
    lines.append("Cross-dataset / cross-device generalization of the Phase 2-3 "
                 "models, measured with the in-domain metric suite carried over "
                 "unchanged (including the classification operating point).\n")

    lines.append("## 1. Headline: cost vs accuracy\n")
    lines.append(comp_md)
    lines.append("\n> Accuracy columns are in-domain; `mean_external_auc` and "
                 "`auc_drop_external` summarize the generalization gap. Empty "
                 "cells = not measured (never imputed).\n")

    for i, m in enumerate(models, start=2):
        sub = evidence[evidence["model"] == m]
        per_ds = EV.evaluate_per_dataset(sub, threshold=threshold)
        per_dev = SD.stratify(sub)
        deg = DG.degradation_table(per_ds) if len(per_ds) > 1 else pd.DataFrame()
        # FIX(U-11): 旧版每个模型都用硬编码的 "## 2." / "## 3."，两个模型时重号
        lines.append(f"\n## {i}.1 {m} — per dataset\n")
        lines.append(_fmt(EV.headline(per_ds)))
        lines.append(f"\n## {i}.2 {m} — per acquisition device\n")
        lines.append(_fmt(EV.headline(per_dev)))
        worst = DG.worst_hits(deg, k=5)
        if not worst.empty:
            lines.append(f"\n**Largest relative drops ({m}):**\n")
            lines.append(_fmt(worst.set_index("dataset")))

    lines.append("\n## 90. Does the camera explain more than the dataset?\n")
    if spread is None or spread.empty:
        # FIX(U-5)
        lines.append("_跳过：数据集或设备少于 2 个。单设备时该对比恒为 0/False，"
                     "没有信息量。真正做了跨设备量化的是 `phase2/loco_runner.py`。_\n")
    else:
        lines.append(_fmt(spread.set_index("metric")))
    lines.append("\n> Where `device_explains_more` is True, regrouping the same "
                 "images by camera produces a wider spread than the dataset "
                 "boundary — evidence the drop is a device effect, not a generic "
                 "'different dataset' effect.\n")

    lines.append("\n## 5. Failure analysis\n")
    if artifacts.get("overlays"):
        lines.append("Worst-case overlays (GT solid green/cyan, prediction "
                     "red/orange):\n")
        for task, p in artifacts["overlays"].items():
            lines.append(f"- `{task}`: {os.path.relpath(p, out_dir)}")
    if artifacts.get("tables"):
        lines.append("\nFailure-mode tag frequencies: "
                     + ", ".join(f"`{os.path.relpath(v, out_dir)}`"
                                 for v in artifacts["tables"].values()))

    lines.append("\n## 6. Files\n")
    lines.append("See `evidence_table.csv` (every number traces back to one "
                 "image row), the `per_dataset__*`, `per_device__*`, "
                 "`degradation__*`, `device_degradation__*` CSVs, and `figs/`.\n")

    path = os.path.join(out_dir, "REPORT.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Phase 4 robustness evaluation")
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--out", default="phase4_report")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="real mode: dataset names from config.REGISTRY")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true",
                    help="skip mask cache (disables failure overlays)")
    # ---- 真实模式参数 (FIX(U-2): 不再硬编码路径) ----
    ap.add_argument("--csv", default=None, help="data_all.csv 路径")
    ap.add_argument("--stage2-ckpt", default=None,
                    help="Stage2 checkpoint。**注意区分 phase2_v2 与 loco**")
    ap.add_argument("--stage3-head", default=None,
                    help="Stage3 CDR 融合头。不给则分类无 CDR 融合 (FIX(U-1))")
    ap.add_argument("--split", default="test")
    ap.add_argument("--calib-split", default="val",
                    help="阈值标定集。绝不能等于 --split (FIX(U-7))")
    ap.add_argument("--seg-source", choices=["fine", "coarse"], default="fine",
                    help="FIX(U-9): fine=coarse-to-fine 精修 (与训练一致); "
                         "coarse=旧路径。两者不可比")
    ap.add_argument("--cdr-seg-size", type=int, default=96)
    ap.add_argument("--input-size", type=int, default=None,
                    help="默认沿用 checkpoint 里记录的 img_size")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-meta-json", default=None,
                    help="训练时长/峰值显存等元数据 JSON；缺则留 NaN，绝不臆造")
    ap.add_argument("--allow-partial-load", action="store_true")
    args = ap.parse_args()

    if args.mode == "real" and args.calib_split == args.split:
        raise SystemExit(
            f"拒绝执行：--calib-split 与 --split 都是 '{args.split}'。"
            f"在评测集上拟合阈值再在同一批数据上报告 sens/spec 是样本内乐观估计 "
            f"(FIX(U-7))。")

    os.makedirs(args.out, exist_ok=True)
    figs_dir = os.path.join(args.out, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    cache_dir = None if args.no_cache else os.path.join(args.out, "_mask_cache")

    if args.mode == "synthetic":
        # FIX(U-4): 合成产物必须自带醒目标记 —— 上传的 REPORT.md /
        # comparison_table.md 就是合成模式的输出 (indomain_auc=1.0、
        # 样本量 120/120/80/60 与 build_synthetic_suite 逐个吻合、
        # 成本列等于 DEMO_ONLY_FAKE_TRAIN_META 的占位符)，却与
        # phase4_report_real 混在一起。
        print("\n" + "!" * 78)
        print("!!  SYNTHETIC MODE —— 卡通眼底图 + 合成检测器。")
        print("!!  产出的 REPORT.md / comparison_table.md **不可用于任何报告**。")
        print("!!  真实评测: python run_phase4.py --mode real --csv ... --stage2-ckpt ...")
        print("!" * 78 + "\n")
        models, datasets, train_meta = build_synthetic(args.out, args.seed)
    else:
        models, datasets, train_meta = build_real(args)

    print(f"== Phase 4 [{args.mode}] : {len(models)} models x "
          f"{len(datasets)} datasets ==")

    # 1) evidence table (every model x dataset x image)
    evidence = INF.run_matrix(models, datasets, cache_dir=cache_dir)
    evidence.to_csv(os.path.join(args.out, "evidence_table.csv"), index=False)
    print(f"evidence_table.csv: {len(evidence)} rows")

    # FIX(U-7): 阈值在标定集上拟合一次，之后所有表格沿用同一个值
    calib_split = "val" if args.mode == "synthetic" else args.calib_split
    threshold = EV.fit_threshold(evidence, calib_split=calib_split)
    print(f"[phase4] 分类阈值在 split='{calib_split}' 上标定 = {threshold:.4f}"
          f"（评测集 split='{args.split}' 只报告，不参与拟合）")

    # 评测时把标定集本身从报告里剔除，避免它混进 headline 表
    evidence = evidence[evidence["dataset"] != "REFUGE2_calib"] \
        if "REFUGE2_calib" in set(evidence["dataset"]) else evidence

    # 2) per-model aggregations + degradation + device attribution
    for m in models:
        sub = evidence[evidence["model"] == m]
        per_ds = EV.evaluate_per_dataset(sub, threshold=threshold)
        per_dev = SD.stratify(sub)
        deg = (DG.degradation_table(per_ds) if len(per_ds) > 1
               else pd.DataFrame())      # FIX(U-5): 单数据集时不伪造掉点表
        dev_deg = SD.device_degradation(per_dev, metric="auc")
        safe = m.replace("/", "_").replace(" ", "_")
        per_ds.to_csv(os.path.join(args.out, f"per_dataset__{safe}.csv"))
        per_dev.to_csv(os.path.join(args.out, f"per_device__{safe}.csv"))
        if not deg.empty:
            deg.to_csv(os.path.join(args.out, f"degradation__{safe}.csv"), index=False)
        if not dev_deg.empty:
            dev_deg.to_csv(os.path.join(args.out, f"device_degradation__{safe}.csv"),
                           index=False)
        for metric in ["auc", "disc_dice_mean", "cup_dice_mean"]:
            SD.plot_device_bars(per_dev, metric,
                                os.path.join(figs_dir, f"{safe}__{metric}_by_device.png"))

    # 3) device-vs-dataset variance attribution (use the strongest model)
    strong = list(models)[-1]
    spread = SD.dataset_vs_device_spread(evidence[evidence["model"] == strong])
    spread.to_csv(os.path.join(args.out, "dataset_vs_device_spread.csv"), index=False)

    # 4) baseline comparison table (cost vs accuracy)
    comp = CB.comparison_table(evidence, train_meta)
    comp.to_csv(os.path.join(args.out, "comparison_table.csv"))
    comp_md = CB.to_markdown(comp, "Cost vs accuracy: scratch baseline vs LoRA-RETFound")
    with open(os.path.join(args.out, "comparison_table.md"), "w") as f:
        f.write(comp_md)

    # 5) failure analysis on the strongest model (overlays need the cache)
    artifacts = FA.run_failure_analysis(
        evidence[evidence["model"] == strong], figs_dir)

    # 6) report
    uses_cdr = any(getattr(mm, "uses_cdr", False) for mm in models.values())
    report = write_report(args.out, evidence, comp_md, spread, artifacts,
                          list(models), mode=args.mode, threshold=threshold,
                          calib_split=calib_split, split=args.split,
                          seg_source=args.seg_source, uses_cdr=uses_cdr)

    print("\n--- comparison (cost vs accuracy) ---")
    print(comp.to_string())
    print("\n--- device-vs-dataset spread (strong model) ---")
    print(spread.to_string(index=False))
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
