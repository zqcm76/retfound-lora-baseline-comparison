# -*- coding: utf-8 -*-
"""
统计工具：置信区间、配对检验、阈值标定、校准。

FIX(P1-2): 之前所有阶段都只报点估计。REFUGE2 test 400 张约 40 阳性，
           AUC≈0.81 的 95% CI 约 ±0.083 —— 这意味着 0.812 vs 0.754 区分不开，
           cdr_soft(0.934) vs cdr_two_stage(0.931) 的 0.003 更是纯噪声。
FIX(P1-4): 阈值必须在 val 上标定、在 test 上报告 (旧 phase4 在 test 上拟合)。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:  # pragma: no cover
    roc_auc_score = roc_curve = None

NAN = float("nan")


def _clean(y_true, y_score):
    y = np.asarray(y_true, dtype=float).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    m = np.isfinite(s) & np.isin(y, [0.0, 1.0])
    return y[m].astype(int), s[m]


def auc(y_true, y_score) -> float:
    y, s = _clean(y_true, y_score)
    if y.size < 2 or np.unique(y).size < 2:
        return NAN
    return float(roc_auc_score(y, s))


def bootstrap_auc(y_true, y_score, n_boot: int = 2000, seed: int = 42,
                  alpha: float = 0.05) -> dict:
    """AUC 的 bootstrap 置信区间。返回 point / lo / hi / n_pos / n_neg。"""
    y, s = _clean(y_true, y_score)
    if y.size < 2 or np.unique(y).size < 2:
        return {"auc": NAN, "lo": NAN, "hi": NAN, "n_pos": 0, "n_neg": 0,
                "n_boot": 0}
    rng = np.random.default_rng(seed)
    n = y.size
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        stats.append(roc_auc_score(y[idx], s[idx]))
    if not stats:
        return {"auc": float(roc_auc_score(y, s)), "lo": NAN, "hi": NAN,
                "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "n_boot": 0}
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"auc": float(roc_auc_score(y, s)), "lo": float(lo), "hi": float(hi),
            "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            "n_boot": len(stats)}


def bootstrap_auc_delta(y_true, score_a, score_b, n_boot: int = 2000,
                        seed: int = 42, alpha: float = 0.05) -> dict:
    """配对 bootstrap：同一批样本上比较两个模型的 AUC 差。

    这是判断 "cdr_soft 是否真的优于 cdr_two_stage" / "ResNet34 是否真的优于
    LoRA-RETFound" 的正确工具——各自独立的 CI 重叠并不等于差异不显著。
    """
    y = np.asarray(y_true, dtype=float).ravel()
    a = np.asarray(score_a, dtype=float).ravel()
    b = np.asarray(score_b, dtype=float).ravel()
    m = np.isfinite(a) & np.isfinite(b) & np.isin(y, [0.0, 1.0])
    y, a, b = y[m].astype(int), a[m], b[m]
    if y.size < 2 or np.unique(y).size < 2:
        return {"delta": NAN, "lo": NAN, "hi": NAN, "p_two_sided": NAN}
    rng = np.random.default_rng(seed)
    n = y.size
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    d = np.asarray(deltas)
    if d.size == 0:
        return {"delta": NAN, "lo": NAN, "hi": NAN, "p_two_sided": NAN}
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "lo": float(lo), "hi": float(hi), "p_two_sided": float(min(1.0, p)),
            "n_boot": int(d.size)}


def bootstrap_mean(values, n_boot: int = 2000, seed: int = 42,
                   alpha: float = 0.05) -> dict:
    """任意逐样本指标 (Dice / MAE / 定位误差) 的均值 bootstrap CI。"""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": NAN, "lo": NAN, "hi": NAN, "n": 0}
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi),
            "n": int(v.size)}


def fmt_ci(d: dict, key: str = "auc", nd: int = 3) -> str:
    """统一的报告格式：0.812 [0.73, 0.89]"""
    v, lo, hi = d.get(key, NAN), d.get("lo", NAN), d.get("hi", NAN)
    if not np.isfinite(v):
        return "n/a"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


# --------------------------------------------------------------------------- #
# 阈值标定
# --------------------------------------------------------------------------- #
def youden_threshold(y_true, y_score) -> float:
    """最大化 sensitivity + specificity - 1。**只在 val 上调用。**"""
    y, s = _clean(y_true, y_score)
    if y.size < 2 or np.unique(y).size < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[int(np.argmax(tpr - fpr))])


def threshold_at_sensitivity(y_true, y_score, target_sens: float = 0.90) -> float:
    """筛查场景的标准做法：固定 sensitivity>=target，反推阈值。

    青光眼筛查里 sensitivity 0.675 (漏诊 32.5%) 是不可接受的，
    这个函数给出临床上更有意义的运行点。
    """
    y, s = _clean(y_true, y_score)
    if y.size < 2 or np.unique(y).size < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.where(tpr >= target_sens)[0]
    if ok.size == 0:
        return float(thr[-1])
    return float(thr[ok[np.argmin(fpr[ok])]])


def sens_spec_at(y_true, y_score, thr: float) -> dict:
    y, s = _clean(y_true, y_score)
    if y.size == 0:
        return {k: NAN for k in
                ("sens", "spec", "ppv", "npv", "acc", "tp", "fp", "tn", "fn")}
    p = (s >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    d = lambda a, b: a / b if b else NAN
    return {"sens": d(tp, tp + fn), "spec": d(tn, tn + fp),
            "ppv": d(tp, tp + fp), "npv": d(tn, tn + fn),
            "acc": d(tp + tn, y.size),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "threshold": float(thr)}


# --------------------------------------------------------------------------- #
# 校准 —— 临床上比 AUC 更重要，而几乎没有学生项目报这个
# --------------------------------------------------------------------------- #
def brier_score(y_true, y_score) -> float:
    y, s = _clean(y_true, y_score)
    return float(np.mean((s - y) ** 2)) if y.size else NAN


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    y, s = _clean(y_true, y_score)
    if y.size == 0:
        return NAN
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (s > lo) & (s <= hi) if i else (s >= lo) & (s <= hi)
        if not m.any():
            continue
        ece += m.mean() * abs(y[m].mean() - s[m].mean())
    return float(ece)


def reliability_curve(y_true, y_score, n_bins: int = 10):
    """返回 (bin 中心的平均预测概率, 该 bin 的实际阳性率, 该 bin 的样本数)。"""
    y, s = _clean(y_true, y_score)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (s > lo) & (s <= hi) if i else (s >= lo) & (s <= hi)
        if not m.any():
            continue
        xs.append(float(s[m].mean())); ys.append(float(y[m].mean()))
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def classification_report(y_true, y_score, thr: float, *,
                          n_boot: int = 2000, seed: int = 42) -> dict:
    """一次性给出 AUC+CI / 运行点 / 校准，供所有阶段统一调用。"""
    out = {}
    out.update({f"auc_{k}": v for k, v in bootstrap_auc(
        y_true, y_score, n_boot=n_boot, seed=seed).items()})
    out.update(sens_spec_at(y_true, y_score, thr))
    out["brier"] = brier_score(y_true, y_score)
    out["ece"] = expected_calibration_error(y_true, y_score)
    return out
