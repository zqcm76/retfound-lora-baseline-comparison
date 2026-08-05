# -*- coding: utf-8 -*-
"""跑真实评测前的 30 秒导入自检 —— 不加载模型、不读数据。

用法:  cd phase4 && python check_imports.py
"""
from __future__ import annotations

import os
import sys

import _bootstrap

print(f"repo root : {_bootstrap.REPO_ROOT}")
print(f"phase3 dir: {_bootstrap.PHASE3_DIR}  "
      f"({'存在' if os.path.isdir(_bootstrap.PHASE3_DIR) else '❌ 缺失'})")
print()

ok = True
CHECKS = [
    ("retfound_common.config",     "Config"),
    ("retfound_common.model",      "MultiTaskModel"),
    ("retfound_common.checkpoint", "build_model_from_checkpoint"),
    ("retfound_common.transforms", "RESIZE_INTERPOLATION"),
    ("cdr",                        "SoftCDR"),
    ("fusion_head",                "CDRFusionClassifier"),
    ("stage2_adapter",             "binary_to_3class_logits"),
    ("config",                     "DEVICE_BY_IMAGE_SIZE"),
    ("phase4_adapter",             "build_real_phase4"),
]
for mod, attr in CHECKS:
    try:
        m = __import__(mod, fromlist=[attr])
        getattr(m, attr)
        src = getattr(m, "__file__", "?")
        print(f"  ✓ {mod:28s}.{attr:28s} <- {src}")
    except Exception as e:
        ok = False
        print(f"  ✗ {mod:28s}.{attr:28s} <- {type(e).__name__}: {e}")

print()
if ok:
    print("全部导入正常，可以跑 run_phase4.py --mode real")
else:
    print("有导入失败，先解决上面标 ✗ 的项")
    sys.exit(1)
