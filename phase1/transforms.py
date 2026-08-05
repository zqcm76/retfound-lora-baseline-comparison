# -*- coding: utf-8 -*-
"""Phase 1 预处理 —— 转发到 retfound_common.transforms，保证与 phase2/3 同一份实现。

FIX(T-1): 之前 phase1/phase2/phase3 各有一份 transforms.py，phase3 那份还少了
两种增广。现在只有 retfound_common 一份。
FIX(U-8): RESIZE_INTERPOLATION 从这里导出，phase4 推理端必须引用同一个常量。
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from retfound_common.transforms import (  # noqa: E402,F401
    IMAGENET_MEAN, IMAGENET_STD, RESIZE_INTERPOLATION,
    build_cls_transforms, build_multitask_transforms, build_seg_transforms,
)

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "RESIZE_INTERPOLATION",
           "build_cls_transforms", "build_seg_transforms",
           "build_multitask_transforms"]
