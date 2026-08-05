# -*- coding: utf-8 -*-
"""
REFUGE 分割掩膜解析。

!!! 训练前请用 np.unique(mask) 核实你自己那份数据的灰度约定 !!!
常见约定：
    0   = 视杯 (cup)
    128 = 视盘盘沿 (rim)
    255 = 背景

FIX(C-8): 旧实现的 docstring 声称是 "nearest-of-three assignment"，实际是带
          间隙的阈值判断 —— 逐灰度值实测，v=64 会被归到背景而不是视杯
          (真 nearest 应为 cup)。影响极小（只有一个灰度级），但既然文件顶部
          用大写警告要求核实编码，实现就该名副其实。现在是真正的最近邻。
"""
from __future__ import annotations

import numpy as np

# 原始掩膜像素值（按你的数据修改这三个常量即可）
CUP_VALUE = 0
DISC_RIM_VALUE = 128
BG_VALUE = 255

# 内部标签约定（勿改，其余代码依赖它）
LBL_BG, LBL_DISC, LBL_CUP = 0, 1, 2

_RAW = np.array([CUP_VALUE, DISC_RIM_VALUE, BG_VALUE], dtype=np.int16)
_LBL = np.array([LBL_CUP, LBL_DISC, LBL_BG], dtype=np.uint8)


def parse_refuge_mask(mask: np.ndarray) -> np.ndarray:
    """原始灰度掩膜 -> 标签图 {0:bg, 1:rim, 2:cup}。

    真正的 nearest-of-three：对每个像素取到三个参考值中距离最近的那个，
    因此 JPEG / resize 造成的轻微灰度漂移不会把标签打乱，也不会出现
    "落在两个阈值之间就掉进背景"的空隙。
    """
    if mask.ndim == 3:
        mask = mask[..., 0]
    m = mask.astype(np.int16)
    idx = np.abs(m[..., None] - _RAW).argmin(axis=-1)
    return _LBL[idx]


def disc_binary(label: np.ndarray) -> np.ndarray:
    """整个视盘 = 盘沿 + 视杯。"""
    return (label >= LBL_DISC).astype(np.uint8)


def cup_binary(label: np.ndarray) -> np.ndarray:
    return (label == LBL_CUP).astype(np.uint8)


def sanity_check_encoding(mask: np.ndarray) -> dict:
    """打印一张掩膜的灰度直方图与解析结果占比，用于开训前核实编码。"""
    if mask.ndim == 3:
        mask = mask[..., 0]
    vals, counts = np.unique(mask, return_counts=True)
    lab = parse_refuge_mask(mask)
    total = lab.size
    return {
        "raw_values": {int(v): int(c) for v, c in zip(vals, counts)},
        "bg_frac": float((lab == LBL_BG).mean()),
        "rim_frac": float((lab == LBL_DISC).mean()),
        "cup_frac": float((lab == LBL_CUP).mean()),
        "n_pixels": int(total),
    }
