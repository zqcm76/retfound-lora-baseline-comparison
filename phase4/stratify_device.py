"""
stratify_device.py — the MuCaRD-style view: attribute the cross-dataset drop to
the *acquisition device*, not to a dataset name.

Why this matters for the report: saying "AUC fell on RIGA" conflates everything
that differs between REFUGE and RIGA (population, labeling protocol, prevalence,
camera). Re-binning the exact same per-image rows by device isolates the camera
axis. If the drop tracks device far more tightly than it tracks dataset, that is
direct evidence the model is camera-sensitive — which is the claim MuCaRD is
about, and a much more actionable finding than "it doesn't generalize".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import evaluate as E
from core import HIGHER_IS_BETTER, LOWER_IS_BETTER


def stratify(df: pd.DataFrame, in_domain: str = C.IN_DOMAIN_DATASET) -> pd.DataFrame:
    """Per-device metric table (same suite as per-dataset), with an
    'in_domain_device' flag column."""
    per_device = E.evaluate_per_device(df, in_domain)
    per_device = per_device.copy()
    per_device.insert(
        0, "in_domain_device",
        [dev in C.IN_DOMAIN_DEVICES for dev in per_device.index],
    )
    return per_device


def device_degradation(
    per_device: pd.DataFrame,
    metric: str = "auc",
    reference_devices: set[str] = C.IN_DOMAIN_DEVICES,
) -> pd.DataFrame:
    """Drop of `metric` for each non-reference device against the (sample-size
    weighted) mean of the reference devices."""
    ref = per_device[per_device.index.isin(reference_devices)]
    if ref.empty or metric not in per_device.columns:
        # FIX(U-5): 没有域内参考设备时，锚点无从谈起 —— 返回空表并说明，
        # 而不是让 device_degradation 以自己为锚点、drop 恒为 0。
        print(f"[stratify] device_degradation 跳过: per_device 里没有任何"
              f"域内参考设备 {sorted(reference_devices)}。"
              f"现有设备: {list(per_device.index)}")
        return pd.DataFrame()
    w = ref["n_images"].to_numpy(dtype=float)
    vals = ref[metric].to_numpy(dtype=float)
    m = np.isfinite(vals)
    anchor = float(np.average(vals[m], weights=w[m])) if m.any() else np.nan

    rows = []
    for dev in per_device.index:
        val = float(per_device.loc[dev, metric])
        if metric in HIGHER_IS_BETTER:
            drop = anchor - val
        elif metric in LOWER_IS_BETTER:
            drop = val - anchor
        else:
            drop = anchor - val
        rows.append({
            "device": dev,
            "in_domain_device": dev in reference_devices,
            "n_images": int(per_device.loc[dev, "n_images"]),
            metric: round(val, 4) if np.isfinite(val) else np.nan,
            "ref_anchor": round(anchor, 4) if np.isfinite(anchor) else np.nan,
            "drop": round(drop, 4) if np.isfinite(drop) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("drop", ascending=False).reset_index(drop=True)


def dataset_vs_device_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Compare how much a metric varies BETWEEN datasets vs BETWEEN devices.

    If the device-grouped spread (std across device means) exceeds the
    dataset-grouped spread, the variance is better explained by the camera than
    by the dataset boundary — the quantitative core of the MuCaRD claim.
    """
    thr = E.fit_threshold(df)
    by_ds = E.aggregate(df, by="dataset", cls_threshold=thr)
    by_dev = E.aggregate(df, by="device", cls_threshold=thr)

    # ════════════════════════════════════════════════════════════════════
    # FIX(U-5) ★ 真实模式下 build_real_phase4 只构建一个数据集、且所有样本
    # device_label 相同 —— per_device 表只有 1 行，np.nanstd([单值]) 恒为 0，
    # 于是本节输出一排 0 / False，完全无信息。而 1_GitHub_README 把它列为
    # 亮点（"MuCaRD 框架 + 设备归因分析"）。
    # 真正做了跨设备量化的是 phase2/loco_runner.py，不是这里。
    # 现在：样本不足时明确返回空表 + 原因，而不是伪造一排 0。
    # ════════════════════════════════════════════════════════════════════
    if len(by_ds) < 2 or len(by_dev) < 2:
        print(f"[stratify] 跳过 dataset-vs-device 离散度对比: "
              f"数据集 {len(by_ds)} 个、设备 {len(by_dev)} 个，至少各需 2 个。"
              f"单设备时该对比恒为 0/False，没有信息量 (FIX(U-5))。")
        return pd.DataFrame(columns=["metric", "spread_across_datasets",
                                     "spread_across_devices",
                                     "device_explains_more", "note"])
    rows = []
    for metric in ["auc", "disc_dice_mean", "cup_dice_mean", "vcdr_mae", "loc_norm_mean"]:
        ds_spread = float(np.nanstd(by_ds[metric].to_numpy())) if metric in by_ds else np.nan
        dev_spread = float(np.nanstd(by_dev[metric].to_numpy())) if metric in by_dev else np.nan
        rows.append({
            "metric": metric,
            "spread_across_datasets": round(ds_spread, 4) if np.isfinite(ds_spread) else np.nan,
            "spread_across_devices": round(dev_spread, 4) if np.isfinite(dev_spread) else np.nan,
            "device_explains_more": (np.isfinite(dev_spread) and np.isfinite(ds_spread)
                                     and dev_spread > ds_spread),
        })
    return pd.DataFrame(rows)


def plot_device_bars(per_device: pd.DataFrame, metric: str, out_path: str) -> str:
    """Bar chart of `metric` by device, in-domain devices highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = per_device.dropna(subset=[metric]).sort_values(metric)
    if sub.empty:
        return ""
    colors = ["#2c7fb8" if dev in C.IN_DOMAIN_DEVICES else "#d95f0e" for dev in sub.index]
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(sub)), 3.6))
    ax.bar(range(len(sub)), sub[metric].to_numpy(), color=colors)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by acquisition device "
                 f"(blue = in-domain, orange = shifted)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
