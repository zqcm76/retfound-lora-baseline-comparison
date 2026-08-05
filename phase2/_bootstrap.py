# -*- coding: utf-8 -*-
"""把仓库根目录放进 sys.path，使 `import retfound_common` 可用。

FIX(T-1): phase2 / phase3 不再各持一份模型代码副本，全部 import 同一个包。
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
