"""
metrics.py — the measurement primitives, following the REFUGE convention.

These are the ONLY place metrics are defined. Per-image functions are called by
inference.py while it builds the evidence table; dataset-level functions
(AUC, sensitivity/specificity) are called by evaluate.py on table columns.

Conventions
-----------
* Segmentation: Dice / IoU on boolean masks, with empty-mask handling.
* CDR: vertical cup-to-disc ratio is the clinical default (vCDR); we also expose
  horizontal and area CDR. Vertical disc diameter is the bounding-box height of
  the disc region (standard enough and resolution-robust).
* Localization: Euclidean fovea error in pixels AND normalized by the optic-disc
  diameter of the SAME image, so the number is comparable across datasets shot
  at different resolutions / fields of view.
* Classification: AUC is threshold-free. Sensitivity/specificity are reported at
  a threshold that is *carried over* from the in-domain set (see evaluate.py) —
  re-tuning the threshold on each external set would hide calibration drift,
  which is exactly the failure mode robustness evaluation is meant to expose.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

NAN = float("nan")


# --------------------------- segmentation overlap --------------------------- #
def dice_coef(pred: np.ndarray, gt: np.ndarray,
              empty_is_nan: bool = True) -> float:
    """FIX(U-11): 双空默认改为 NaN。

    旧实现返回 1.0 —— "什么都没预测且 GT 也为空"拿满分，会抬高均值。
    REFUGE2 每张图都有视盘所以不触发，但跨数据集时（PALM 无视杯）会。
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return NAN if empty_is_nan else 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def iou(pred: np.ndarray, gt: np.ndarray, empty_is_nan: bool = True) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return NAN if empty_is_nan else 1.0
    return float(np.logical_and(pred, gt).sum() / union)


# ------------------------------- mask geometry ------------------------------ #
def vertical_diameter(mask: np.ndarray) -> float:
    """Height of the mask bounding box, in pixels (0 if empty)."""
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return 0.0
    return float(rows.max() - rows.min() + 1)


def horizontal_diameter(mask: np.ndarray) -> float:
    cols = np.where(mask.any(axis=0))[0]
    if cols.size == 0:
        return 0.0
    return float(cols.max() - cols.min() + 1)


def equivalent_diameter(mask: np.ndarray) -> float:
    """Diameter of a circle with the same area (alternative normalizer)."""
    area = float(mask.sum())
    if area == 0:
        return 0.0
    return 2.0 * np.sqrt(area / np.pi)


def disc_diameter(mask: np.ndarray, kind: str = "vertical") -> float:
    if kind == "vertical":
        return vertical_diameter(mask)
    if kind == "horizontal":
        return horizontal_diameter(mask)
    if kind == "equivalent":
        return equivalent_diameter(mask)
    raise ValueError(f"unknown disc-diameter kind: {kind}")


# ----------------------------------- CDR ------------------------------------ #
def vcdr(cup: np.ndarray, disc: np.ndarray) -> float:
    d = vertical_diameter(disc)
    return vertical_diameter(cup) / d if d > 0 else NAN


def hcdr(cup: np.ndarray, disc: np.ndarray) -> float:
    d = horizontal_diameter(disc)
    return horizontal_diameter(cup) / d if d > 0 else NAN


def acdr(cup: np.ndarray, disc: np.ndarray) -> float:
    a = float(disc.sum())
    return float(cup.sum()) / a if a > 0 else NAN


# ------------------------------- localization ------------------------------- #
def fovea_localization_error(
    pred_xy: tuple[float, float] | None,
    gt_xy: tuple[float, float] | None,
    disc_diameter_px: float | None = None,
) -> tuple[float, float]:
    """Return (pixel_error, normalized_error). normalized = pixels / disc_diam."""
    if pred_xy is None or gt_xy is None:
        return NAN, NAN
    px = float(np.hypot(pred_xy[0] - gt_xy[0], pred_xy[1] - gt_xy[1]))
    norm = px / disc_diameter_px if (disc_diameter_px and disc_diameter_px > 0) else NAN
    return px, norm


# --------------------------- dataset-level scores --------------------------- #
def classification_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    m = np.isfinite(probs) & np.isin(labels, [0.0, 1.0])
    if m.sum() < 2 or np.unique(labels[m]).size < 2:
        return NAN
    return float(roc_auc_score(labels[m].astype(int), probs[m]))


def youden_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Operating threshold maximizing (sensitivity + specificity - 1) on the
    given set. Fit this on the IN-DOMAIN set, then reuse on external sets."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    m = np.isfinite(probs) & np.isin(labels, [0.0, 1.0])
    if m.sum() < 2 or np.unique(labels[m]).size < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels[m].astype(int), probs[m])
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def sens_spec_at_threshold(
    probs: np.ndarray, labels: np.ndarray, thr: float
) -> tuple[float, float]:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    m = np.isfinite(probs) & np.isin(labels, [0.0, 1.0])
    if m.sum() == 0:
        return NAN, NAN
    p = (probs[m] >= thr).astype(int)
    y = labels[m].astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else NAN
    spec = tn / (tn + fp) if (tn + fp) else NAN
    return sens, spec


def vcdr_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    m = np.isfinite(pred) & np.isfinite(gt)
    return float(np.mean(np.abs(pred[m] - gt[m]))) if m.any() else NAN


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """FIX(P1-2): 任意逐样本指标的均值 bootstrap CI。返回 (mean, lo, hi)。"""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return NAN, NAN, NAN
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def bootstrap_auc_ci(probs: np.ndarray, labels: np.ndarray, n_boot: int = 2000,
                     seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float]:
    """FIX(P1-2): AUC 的 bootstrap CI。

    REFUGE2 test 400 张约 40 阳性时 95% CI 宽约 ±0.08 —— 这意味着
    0.812 与 0.754 在统计上根本区分不开，而优化方案想推到的 0.88
    **就落在当前 CI 内部**。
    """
    probs = np.asarray(probs, dtype=float); labels = np.asarray(labels, dtype=float)
    m = np.isfinite(probs) & np.isin(labels, [0.0, 1.0])
    p, y = probs[m], labels[m].astype(int)
    if y.size < 2 or np.unique(y).size < 2:
        return NAN, NAN, NAN
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, y.size, y.size)
        if np.unique(y[idx]).size < 2:
            continue
        stats.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(roc_auc_score(y, p)), float(lo), float(hi)


def threshold_at_sensitivity(probs, labels, target_sens: float = 0.90) -> float:
    """筛查运行点：固定 sensitivity>=target 反推阈值。"""
    probs = np.asarray(probs, dtype=float); labels = np.asarray(labels, dtype=float)
    m = np.isfinite(probs) & np.isin(labels, [0.0, 1.0])
    if m.sum() < 2 or np.unique(labels[m]).size < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels[m].astype(int), probs[m])
    ok = np.where(tpr >= target_sens)[0]
    return float(thr[-1]) if ok.size == 0 else float(thr[ok[np.argmin(fpr[ok])]])


def nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else NAN


def nanstd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanstd(x)) if np.isfinite(x).any() else NAN


def nanmedian(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else NAN
