# -*- coding: utf-8 -*-
"""
LoRA / Adapter 实现 + 注入逻辑。

设计要点:
- 冻结主干、只训低秩矩阵。base Linear 的权重在包装器 __init__ 里就被 requires_grad_(False)。
- LoRA 初始化为恒等 (lora_A kaiming / lora_B 全零)，保证注入瞬间不改变模型输出。
- Adapter 的上投影零初始化，同样保证起步即恒等。
- inject_tuning 只替换 attention 里名为 qkv / proj 的 nn.Linear。
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """用 LoRA 包装一个已有的 nn.Linear。

    y = base(x) + scaling * (dropout(x) @ A^T @ B^T)
    其中 A: (r, in)、B: (out, r)、scaling = alpha / r。
    base 的 weight/bias 被冻结。
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        # 冻结原始权重
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 低秩矩阵 (可训练)
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        # A 用 kaiming，B 留零 → 初始增量为 0 (恒等)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # (.., in) @ (in, r) = (.., r); 再 @ (r, out) = (.., out)
        lora = self.drop(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * lora


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class Adapter(nn.Module):
    """瓶颈式 adapter: down -> GELU -> up，残差相加。up 零初始化 → 起步恒等。"""

    def __init__(self, dim: int, bottleneck: int = 64, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.drop(self.act(self.down(x))))


class AdapterLinear(nn.Module):
    """冻结的 base Linear 之后串一个 Adapter。"""

    def __init__(self, base: nn.Linear, bottleneck: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.adapter = Adapter(base.out_features, bottleneck, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.base(x))


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def freeze_all(module: nn.Module) -> None:
    """把一个模块的所有参数设为不可训练。"""
    for p in module.parameters():
        p.requires_grad_(False)


def inject_tuning(model: nn.Module, cfg) -> int:
    """遍历模型，把名字精确匹配 cfg.targets 的 nn.Linear 替换为 LoRA/Adapter 包装。

    返回被注入的层数。要求先对 backbone 调用 freeze_all。
    """
    n_injected = 0

    def _recurse(parent: nn.Module):
        nonlocal n_injected
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name in cfg.targets:
                if cfg.mode == "lora":
                    wrapped = LoRALinear(child, r=cfg.r, alpha=cfg.alpha,
                                         dropout=cfg.dropout)
                elif cfg.mode == "adapter":
                    wrapped = AdapterLinear(child,
                                            bottleneck=cfg.adapter_bottleneck,
                                            dropout=cfg.dropout)
                else:
                    raise ValueError("lora.mode 必须是 'lora' 或 'adapter'")
                setattr(parent, name, wrapped)
                n_injected += 1
            else:
                _recurse(child)

    _recurse(model)
    return n_injected


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    """返回 (可训练参数量, 总参数量)。"""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total
