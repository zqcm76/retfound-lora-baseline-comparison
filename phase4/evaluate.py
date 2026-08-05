"""
evaluate.py — aggregate the per-image evidence table into per-group metric
tables (group = dataset by default, or device for the stratified view).

The one subtlety is the classification operating point: AUC is threshold-free,
but sensitivity/specificity are reported at a threshold FIT ON THE IN-DOMAIN
set and then applied unchanged to every external set. That carry-over is what
turns silent calibration drift into a visible sens/spec gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import metrics as M


def fit_threshold(df: pd.DataFrame, *, calib_split: str = "val",
                  in_domain: str = C.IN_DOMAIN_DATASET,
                  target_sensitivity: float | None = None) -> float:
    """在**标定集**上拟合分类阈值。

    ════════════════════════════════════════════════════════════════════════
    FIX(U-7) ★ 旧实现是 `df[df["dataset"] == in_domain]`，而真实模式下唯一的
    dataset 就是 REFUGE2、且它装的是 **test** split。于是 Youden-J 在 test 上
    拟合、sens/spec 又在同一批 test 上报告 —— 样本内乐观估计。
    3_Optimization_Plan.md 记录的 threshold=0.70 / sens=0.675 / spec=0.85
    (J=0.525) 正是在 test 上被最大化出来的。
    所以优化方案里那条 P0「改用 val 拟合阈值」不是优化，是**修 bug**；
    修完之后 sens/spec 大概率会比现在更差，但那才是真实数字。
    ════════════════════════════════════════════════════════════════════════
    """
    sub = df[(df["split"] == calib_split) & (df["has_cls"])]
    if sub.empty:
        # 退回按 dataset 过滤，但必须告警：这可能是在评测集上拟合
        sub = df[(df["dataset"] == in_domain) & (df["has_cls"])]
        splits = sorted(df["split"].dropna().unique().tolist())
        print(f"[evaluate] ⚠️  证据表里没有 split='{calib_split}' 的行 (现有: {splits})。"
              f"退回按 dataset='{in_domain}' 拟合阈值 —— 如果它就是评测集，"
              f"得到的 sens/spec 是样本内乐观估计 (FIX(U-7))。")
    if sub.empty:
        return 0.5
    probs = sub["cls_prob"].to_numpy(); labels = sub["cls_label"].to_numpy()
    if target_sensitivity is not None:
        return M.threshold_at_sensitivity(probs, labels, target_sensitivity)
    return M.youden_threshold(probs, labels)


def fit_indomain_threshold(df: pd.DataFrame,
                           in_domain: str = C.IN_DOMAIN_DATASET) -> float:
    """向后兼容的薄封装。新代码请直接用 fit_threshold(calib_split=...)。"""
    return fit_threshold(df, calib_split="val", in_domain=in_domain)


def aggregate(df: pd.DataFrame, by: str = "dataset", *, cls_threshold: float = 0.5) -> pd.DataFrame:
    """One row per group with the full metric suite + sample counts.

    `by` is the grouping column ("dataset" or "device").
    `cls_threshold` is the carried-over in-domain operating point.
    """
    out: list[dict] = []
    for key, g in df.groupby(by, sort=False):
        probs = g["cls_prob"].to_numpy()
        labels = g["cls_label"].to_numpy()
        sens, spec = M.sens_spec_at_threshold(probs, labels, cls_threshold)

        loc_norm = g["loc_err_norm"].to_numpy()
        finite_loc = np.isfinite(loc_norm)
        succ = (loc_norm[finite_loc] <= C.LOC_SUCCESS_DISC_FRACTION)
        loc_success = float(succ.mean()) if finite_loc.any() else M.NAN

        auc_pt, auc_lo, auc_hi = M.bootstrap_auc_ci(probs, labels)   # FIX(P1-2)
        row = {
            by: key,
            "n_images": int(len(g)),
            # classification
            "auc": auc_pt,
            "auc_lo": auc_lo,
            "auc_hi": auc_hi,
            "n_cls": int((np.isfinite(probs) & np.isin(labels, [0, 1])).sum()),
            "sens_at_thr": sens,
            "spec_at_thr": spec,
            "thr_used": cls_threshold,
            # disc/cup segmentation
            "n_pos": int((np.asarray(labels, dtype=float) == 1).sum()),
            "n_neg": int((np.asarray(labels, dtype=float) == 0).sum()),
            "disc_dice_mean": M.nanmean(g["disc_dice"]),
            "disc_dice_std": M.nanstd(g["disc_dice"]),
            "n_disc": int(np.isfinite(g["disc_dice"]).sum()),
            "cup_dice_mean": M.nanmean(g["cup_dice"]),
            "cup_dice_std": M.nanstd(g["cup_dice"]),
            "n_cup": int(np.isfinite(g["cup_dice"]).sum()),
            "disc_iou_mean": M.nanmean(g["disc_iou"]),
            "cup_iou_mean": M.nanmean(g["cup_iou"]),
            # CDR (clinical fusion target)
            "vcdr_mae": M.nanmean(g["vcdr_abs_err"]),
            "n_vcdr": int(np.isfinite(g["vcdr_abs_err"]).sum()),
            # fovea localization
            "loc_norm_mean": M.nanmean(g["loc_err_norm"]),
            "loc_norm_median": M.nanmedian(g["loc_err_norm"]),
            "loc_px_mean": M.nanmean(g["loc_err_px"]),
            "loc_success_rate": loc_success,
            "n_fovea": int(np.isfinite(g["loc_err_px"]).sum()),
        }
        out.append(row)
    res = pd.DataFrame(out).set_index(by)
    return res


def evaluate_per_dataset(df: pd.DataFrame, in_domain: str = C.IN_DOMAIN_DATASET,
                         *, threshold: float | None = None,
                         calib_split: str = "val") -> pd.DataFrame:
    thr = threshold if threshold is not None else fit_threshold(
        df, calib_split=calib_split, in_domain=in_domain)
    return aggregate(df, by="dataset", cls_threshold=thr)


def evaluate_per_device(df: pd.DataFrame, in_domain: str = C.IN_DOMAIN_DATASET,
                        *, threshold: float | None = None,
                        calib_split: str = "val") -> pd.DataFrame:
    thr = threshold if threshold is not None else fit_threshold(
        df, calib_split=calib_split, in_domain=in_domain)
    return aggregate(df, by="device", cls_threshold=thr)


# Columns worth showing in a compact headline table (kept short on purpose).
HEADLINE_COLS = [
    "n_images", "n_pos", "auc", "auc_lo", "auc_hi", "sens_at_thr", "spec_at_thr",
    "disc_dice_mean", "cup_dice_mean", "vcdr_mae",
    "loc_norm_mean", "loc_success_rate",
]


def headline(table: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in HEADLINE_COLS if c in table.columns]
    return table[cols].round(4)
