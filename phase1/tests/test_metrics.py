# -*- coding: utf-8 -*-
"""
指标与掩膜解析的数值单元测试。

FIX(C-7): README_phase1.md 声称代码"经过 numeric unit checks 验证
（相同掩膜 -> Dice 1.0；半径 30/60 的视杯/视盘 -> CDR 0.5；可分离评分 -> AUC 1.0）"，
但这些检查**不在仓库里**（没有 tests/，没有 pytest）。文档里已经把断言写出来了，
落地成文件只要一小时，还能让 GitHub Actions 那条从"低优先级"变成可交付。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics as M            # noqa: E402
import masks as K              # noqa: E402


def _disc_cup(size=200, disc_r=60, cup_r=30):
    yy, xx = np.ogrid[:size, :size]
    c = size // 2
    d2 = (xx - c) ** 2 + (yy - c) ** 2
    lab = np.full((size, size), K.LBL_BG, np.uint8)
    lab[d2 <= disc_r ** 2] = K.LBL_DISC
    lab[d2 <= cup_r ** 2] = K.LBL_CUP
    return lab


# --------------------------------------------------------------- masks
def test_mask_parse_is_true_nearest_of_three():
    """FIX(C-8): 逐灰度值都应归到最近的参考值，中间不能有掉进背景的空隙。"""
    ref = np.array([K.CUP_VALUE, K.DISC_RIM_VALUE, K.BG_VALUE])
    lut = np.array([K.LBL_CUP, K.LBL_DISC, K.LBL_BG])
    for v in range(256):
        got = K.parse_refuge_mask(np.array([[v]], np.uint8))[0, 0]
        assert got == lut[np.abs(v - ref).argmin()], f"灰度 {v} 归类错误"


def test_mask_parse_roundtrip():
    raw = np.array([[0, 128, 255]], np.uint8)
    assert list(K.parse_refuge_mask(raw)[0]) == [K.LBL_CUP, K.LBL_DISC, K.LBL_BG]


def test_disc_includes_cup():
    lab = _disc_cup()
    assert K.disc_binary(lab).sum() > K.cup_binary(lab).sum()
    assert np.all(K.disc_binary(lab)[K.cup_binary(lab) > 0] == 1)


# --------------------------------------------------------------- dice
def test_identical_masks_give_dice_one():
    m = _disc_cup()
    r = M.segmentation_metrics(m, m)
    assert r["dice_disc"] == pytest.approx(1.0, abs=1e-6)
    assert r["dice_cup"] == pytest.approx(1.0, abs=1e-6)


def test_disjoint_masks_give_dice_zero():
    a = np.zeros((50, 50), np.uint8); a[:10, :10] = K.LBL_DISC
    b = np.zeros((50, 50), np.uint8); b[40:, 40:] = K.LBL_DISC
    assert M.segmentation_metrics(a, b)["dice_disc"] == pytest.approx(0.0, abs=1e-5)


def test_empty_pair_behaviour_is_switchable():
    """FIX(U-11): 双空默认给 1.0 会让"什么都没预测"拿满分。"""
    e = np.zeros((10, 10), np.uint8)
    assert M.dice_binary(e, e) == 1.0
    assert np.isnan(M.dice_binary(e, e, empty_is_nan=True))


# --------------------------------------------------------------- CDR
def test_cdr_of_30_60_radii_is_half():
    cdr = M.compute_cdr(_disc_cup(disc_r=60, cup_r=30))
    assert cdr["vCDR"] == pytest.approx(0.5, abs=0.02)
    assert cdr["hCDR"] == pytest.approx(0.5, abs=0.02)
    assert cdr["areaCDR"] == pytest.approx(0.25, abs=0.02)


def test_cdr_is_resolution_invariant():
    """CDR 是比值，跨评测分辨率可比（Dice 不是 —— 见 evaluate.py 的脚注）。"""
    a = M.compute_cdr(_disc_cup(200, 60, 30))["vCDR"]
    b = M.compute_cdr(_disc_cup(400, 120, 60))["vCDR"]
    assert a == pytest.approx(b, abs=0.01)


# --------------------------------------------------------------- 分类
def test_separable_scores_give_auc_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert M.classification_metrics(p, y)["auc"] == pytest.approx(1.0)


def test_threshold_is_actually_applied():
    """FIX(C-2): 旧实现把阈值写死 0.5，这个测试会失败。"""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.65, 0.9])
    assert M.classification_metrics(p, y, threshold=0.5)["sensitivity"] == 1.0
    assert M.classification_metrics(p, y, threshold=0.5)["specificity"] == 0.5
    hi = M.classification_metrics(p, y, threshold=0.7)
    assert hi["specificity"] == 1.0 and hi["sensitivity"] == 0.5


def test_youden_threshold_separates_perfectly():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = M.youden_threshold(p, y)
    m = M.classification_metrics(p, y, threshold=thr)
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0


def test_threshold_at_sensitivity_meets_target():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.1).astype(int)
    p = rng.random(400) + 0.35 * y
    thr = M.threshold_at_sensitivity(p, y, 0.90)
    assert M.classification_metrics(p, y, threshold=thr)["sensitivity"] >= 0.90


# --------------------------------------------------------------- CI
def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.1).astype(int)
    p = rng.random(400) + 0.35 * y
    ci = M.bootstrap_auc(p, y, n_boot=300, seed=0)
    assert ci["lo"] <= ci["auc"] <= ci["hi"]
    assert ci["n_pos"] + ci["n_neg"] == 400


def test_ci_is_wide_at_this_sample_size():
    """40 阳性 / 360 阴性下 95% CI 宽度约 ±0.08 —— 这正是为什么
    0.812 与 0.754 区分不开、0.934 与 0.931 的 0.003 是纯噪声。"""
    rng = np.random.default_rng(1)
    y = np.array([1] * 40 + [0] * 360)
    p = rng.random(400) + 0.35 * y
    ci = M.bootstrap_auc(p, y, n_boot=500, seed=1)
    assert (ci["hi"] - ci["lo"]) > 0.08, "CI 反常地窄，检查 bootstrap 实现"


# --------------------------------------------------------------- 聚合
def test_mean_dice_is_average_of_the_two_means():
    """FIX(C-9): 旧写法 np.mean(dd + dc) 是列表拼接，容易被误读。"""
    a = _disc_cup(); b = _disc_cup(disc_r=58, cup_r=29)
    agg = M.aggregate_segmentation([a, a], [a, b])
    assert agg["mean_dice"] == pytest.approx(
        (agg["dice_disc"] + agg["dice_cup"]) / 2, abs=1e-9)


def test_aggregate_can_return_per_sample():
    a = _disc_cup()
    agg, per = M.aggregate_segmentation([a, a], [a, a], return_per_sample=True)
    assert len(per["dice_disc"]) == 2 and agg["n"] == 2
