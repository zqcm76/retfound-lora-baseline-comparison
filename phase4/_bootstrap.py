# -*- coding: utf-8 -*-
"""把仓库根目录与 phase3 目录放进 sys.path。

FIX(U-1-followup): phase4_adapter 需要 phase3 的 CDRFusionClassifier /
binary_to_3class_logits / cdr 模块。旧版只挂了仓库根，于是
`from fusion_head import CDRFusionClassifier` 直接 ModuleNotFoundError。

挂载顺序很重要：phase4 自己的目录是 sys.path[0]（脚本所在目录），
phase3 用 append 挂在最后，因此同名模块永远是 phase4 的优先。
两边唯一同名的文件是 _bootstrap.py 本身，且功能相同。
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
PHASE3_DIR = os.path.join(REPO_ROOT, "phase3")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# append 而非 insert：保证 phase4 的同名模块优先
if os.path.isdir(PHASE3_DIR) and PHASE3_DIR not in sys.path:
    sys.path.append(PHASE3_DIR)


def require_phase3():
    """真正需要 phase3 时调用，给出可操作的错误而不是裸 ModuleNotFoundError。"""
    if not os.path.isdir(PHASE3_DIR):
        raise SystemExit(
            f"找不到 phase3 目录: {PHASE3_DIR}\n"
            f"phase4 的 CDR 融合需要 phase3/fusion_head.py、cdr.py、"
            f"stage2_adapter.py。请确认仓库结构完整，或去掉 --stage3-head "
            f"以不带 CDR 融合的方式评测（报告会明确标注）。")
    missing = [f for f in ("fusion_head.py", "cdr.py", "stage2_adapter.py")
               if not os.path.exists(os.path.join(PHASE3_DIR, f))]
    if missing:
        raise SystemExit(f"phase3 目录缺少文件: {missing}（在 {PHASE3_DIR}）")
