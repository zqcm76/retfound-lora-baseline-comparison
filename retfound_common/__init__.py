# -*- coding: utf-8 -*-
"""
retfound_common —— phase2 / phase3 / phase4 共用的模型与训练核心。

FIX(T-1): 开源清单原文写 "phase3 中 backbone.py, lora.py, heads.py, model.py,
          config.py, dataset.py, transforms.py, balancing.py, trainer.py,
          train.py, losses.py 与 phase2 完全相同"。**这句话是错的** ——
          config.py 有 8 处默认值不同、transforms.py 少了两种增广、
          losses.py 是完全不同的文件（导致 phase3/trainer.py 直接 ImportError）。
          按那句话去重会用旧版覆盖新版。现在只有这一份，物理上不可能再分叉。
"""
__version__ = "1.0.0"

from .config import (BackboneCfg, BalanceCfg, ClsCfg, Config, LoRACfg,
                     MaculaCfg, SegCfg, TrainCfg, format_param_breakdown,
                     param_breakdown, tiny_config)

__all__ = [
    "Config", "BackboneCfg", "LoRACfg", "SegCfg", "MaculaCfg", "ClsCfg",
    "BalanceCfg", "TrainCfg", "tiny_config",
    "param_breakdown", "format_param_breakdown",
]
