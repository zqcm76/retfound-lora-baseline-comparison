# -*- coding: utf-8 -*-
"""
评估指标 —— phase1 的稳定评测核心。

FIX(C-2): 两处硬伤直接卡死了路线图的 P1-2 / P1-4：
          1. classification_metrics 把阈值写死成 0.5，无法传入 val 上标定的阈值；
          2. 只返回聚合值，逐样本概率被丢弃 —— 没法 bootstrap、没法配对检验，
             连 "ResNet34 (0.8775) 是否真的优于 LoRA-RETFound (0.812)" 都判不了。
          现在阈值可传、逐样本概率可回、并直接给出 CI 与校准。
FIX(C-9): aggregate_segmentation 里 `np.mean(dd + dc)` —— dd/dc 是 list，
          `+` 是列表拼接不是逐元素相加。当前数值上恰好等价 (两表等长)，
          但读者极易误读，且某一任务缺失时会静默出错。改成显式写法。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from masks import cup_binary, disc_binary

NAN = float("nan")


# --------------------------------------------------------------------------- #
# 分类
# --------------------------------------------------------------------------- #
def classification_metrics(probs, labels, threshold: float = 0.5) -> dict:
    """阈值可传入。**阈值必须在 val 上标定，绝不能在 test 上拟合。**"""
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    out: dict = {"threshold": float(threshold)}

    m = np.isfinite(probs) & np.isin(labels, [0, 1])
    p, y = probs[m], labels[m]
    out["n"] = int(y.size)
    out["n_pos"] = int((y == 1).sum())
    out["n_neg"] = int((y == 0).sum())

    out["auc"] = float(roc_auc_score(y, p)) if np.unique(y).size > 1 else NAN

    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    d = lambda a, b: a / b if b else NAN
    out.update(tp=tp, fp=fp, tn=tn, fn=fn,
               accuracy=d(tp + tn, y.size),
               sensitivity=d(tp, tp + fn), specificity=d(tn, tn + fp),
               ppv=d(tp, tp + fp), npv=d(tn, tn + fn))
    return out


def youden_threshold(probs, labels) -> float:
    """最大化 sens+spec-1。**只在 val 上调用。**"""
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    m = np.isfinite(probs) & np.isin(labels, [0, 1])
    if m.sum() < 2 or np.unique(labels[m]).size < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels[m], probs[m])
    return float(thr[int(np.argmax(tpr - fpr))])


def threshold_at_sensitivity(probs, labels, target_sens: float = 0.90) -> float:
    """筛查场景：固定 sensitivity>=target 反推阈值。

    青光眼筛查里 sensitivity 0.675（漏诊 32.5%）不可接受，这个运行点更有意义。
    """
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    m = np.isfinite(probs) & np.isin(labels, [0, 1])
    if m.sum() < 2 or np.unique(labels[m]).size < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels[m], probs[m])
    ok = np.where(tpr >= target_sens)[0]
    return float(thr[-1]) if ok.size == 0 else float(thr[ok[np.argmin(fpr[ok])]])


# --------------------------------------------------------------------------- #
# 置信区间 (FIX(P1-2))
# --------------------------------------------------------------------------- #
def bootstrap_auc(probs, labels, n_boot: int = 2000, seed: int = 42,
                  alpha: float = 0.05) -> dict:
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    m = np.isfinite(probs) & np.isin(labels, [0, 1])
    p, y = probs[m], labels[m]
    if y.size < 2 or np.unique(y).size < 2:
        return {"auc": NAN, "lo": NAN, "hi": NAN, "n_pos": 0, "n_neg": 0}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, y.size, y.size)
        if np.unique(y[idx]).size < 2:
            continue
        stats.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"auc": float(roc_auc_score(y, p)), "lo": float(lo), "hi": float(hi),
            "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
            "n_boot": len(stats)}


def bootstrap_mean(values, n_boot: int = 2000, seed: int = 42,
                   alpha: float = 0.05) -> dict:
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": NAN, "lo": NAN, "hi": NAN, "n": 0}
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(v.size)}


def fmt_ci(d: dict, key: str = "auc", nd: int = 3) -> str:
    v, lo, hi = d.get(key, NAN), d.get("lo", NAN), d.get("hi", NAN)
    if not np.isfinite(v):
        return "n/a"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


# --------------------------------------------------------------------------- #
# 校准 (FIX(P1-4))
# --------------------------------------------------------------------------- #
def brier_score(probs, labels) -> float:
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    m = np.isfinite(probs) & np.isin(labels, [0.0, 1.0])
    return float(np.mean((probs[m] - labels[m]) ** 2)) if m.any() else NAN


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    m = np.isfinite(probs) & np.isin(labels, [0.0, 1.0])
    p, y = probs[m], labels[m]
    if p.size == 0:
        return NAN
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p > lo) & (p <= hi) if i else (p >= lo) & (p <= hi)
        if sel.any():
            ece += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(ece)


# --------------------------------------------------------------------------- #
# 分割
# --------------------------------------------------------------------------- #
def dice_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6,
                empty_is_nan: bool = False) -> float:
    """两者皆空时：empty_is_nan=True 返回 NaN，否则返回 1.0。

    FIX(U-11): 默认返回 1.0 会让"什么都没预测且 GT 也为空"拿满分，抬高均值。
    REFUGE2 每张图都有视盘，实际不触发，但跨数据集时要能切换。
    """
    pred = pred.astype(bool); gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return NAN if empty_is_nan else 1.0
    return float((2 * np.logical_and(pred, gt).sum() + eps) / (denom + eps))


def segmentation_metrics(pred_label: np.ndarray, gt_label: np.ndarray) -> dict:
    return {
        "dice_disc": dice_binary(disc_binary(pred_label), disc_binary(gt_label)),
        "dice_cup": dice_binary(cup_binary(pred_label), cup_binary(gt_label)),
    }


# --------------------------------------------------------------------------- #
# CDR
# --------------------------------------------------------------------------- #
def _vh_diameter(binary: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return 0.0, 0.0
    return float(ys.max() - ys.min() + 1), float(xs.max() - xs.min() + 1)


def compute_cdr(label: np.ndarray) -> dict:
    disc = disc_binary(label); cup = cup_binary(label)
    dv, dh = _vh_diameter(disc); cv, ch = _vh_diameter(cup)
    eps = 1e-6
    return {"vCDR": cv / (dv + eps), "hCDR": ch / (dh + eps),
            "areaCDR": float(cup.sum()) / (float(disc.sum()) + eps),
            "disc_vdiam": dv}


def cdr_error(pred_label: np.ndarray, gt_label: np.ndarray) -> dict:
    p, g = compute_cdr(pred_label), compute_cdr(gt_label)
    return {"vCDR_mae": abs(p["vCDR"] - g["vCDR"]),
            "hCDR_mae": abs(p["hCDR"] - g["hCDR"]),
            "areaCDR_mae": abs(p["areaCDR"] - g["areaCDR"])}


# --------------------------------------------------------------------------- #
# 黄斑
# --------------------------------------------------------------------------- #
def fovea_distance(pred_xy, gt_xy, disc_diameter: Optional[float] = None) -> dict:
    """FIX(P0-6): 两个口径分别命名。fovea_px 与 fovea_norm_disc 单位不同，
    绝不能和"按图像边长归一化"的数并排比较。"""
    pred_xy = np.asarray(pred_xy, dtype=float)
    gt_xy = np.asarray(gt_xy, dtype=float)
    d = float(np.sqrt(((pred_xy - gt_xy) ** 2).sum()))
    out = {"fovea_px": d}
    if disc_diameter and disc_diameter > 0:
        out["fovea_norm_disc"] = d / disc_diameter       # 单位 = 视盘直径
    return out


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #
def aggregate_segmentation(pred_labels: Sequence[np.ndarray],
                           gt_labels: Sequence[np.ndarray],
                           return_per_sample: bool = False):
    """FIX(C-9): mean_dice 用显式写法；FIX(C-2): 可回传逐样本值供 bootstrap。"""
    dd, dc, vmae = [], [], []
    for p, g in zip(pred_labels, gt_labels):
        m = segmentation_metrics(p, g)
        dd.append(m["dice_disc"]); dc.append(m["dice_cup"])
        vmae.append(cdr_error(p, g)["vCDR_mae"])
    agg = {
        "dice_disc": float(np.nanmean(dd)) if dd else NAN,
        "dice_cup": float(np.nanmean(dc)) if dc else NAN,
        "mean_dice": (float((np.nanmean(dd) + np.nanmean(dc)) / 2.0) if dd else NAN),
        "vCDR_mae": float(np.nanmean(vmae)) if vmae else NAN,
        "n": len(dd),
    }
    if not return_per_sample:
        return agg
    return agg, {"dice_disc": np.array(dd), "dice_cup": np.array(dc),
                 "vCDR_mae": np.array(vmae)}
