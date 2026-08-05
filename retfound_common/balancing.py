# -*- coding: utf-8 -*-
"""
多任务 loss 平衡 (本身作为一组消融):
- UncertaintyWeighting: Kendall et al. 不确定性加权。学一组 log σ²。
- GradNorm: Chen et al. 让各任务在共享层的梯度范数对齐。需双反传。
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# 不确定性加权
# --------------------------------------------------------------------------- #
class UncertaintyWeighting(nn.Module):
    """total = Σ_i [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ]，s_i = log σ_i²。"""

    def __init__(self, task_names: List[str], max_weight: float = 0.0):
        super().__init__()
        self.task_names = list(task_names)
        self.log_sigma2 = nn.Parameter(torch.zeros(len(task_names)))
        # weight = exp(-s) <= max_weight  <=>  s >= -log(max_weight)
        # max_weight<=0 关闭钳制(旧行为)
        self._s_min = (-math.log(max_weight)) if max_weight and max_weight > 0 \
            else None

    def forward(self, losses: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = 0.0
        logs: Dict[str, float] = {}
        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue
            s = self.log_sigma2[i]
            if self._s_min is not None:
                # 软地板: 用 clamp 限制有效 s, 防止权重 exp(-s) 爆高
                s = s.clamp(min=self._s_min)
            L = losses[name]
            total = total + 0.5 * torch.exp(-s) * L + 0.5 * s
            logs[f"w_{name}"] = float(torch.exp(-s).detach())
        return total, logs


# --------------------------------------------------------------------------- #
# GradNorm
# --------------------------------------------------------------------------- #
class GradNorm(nn.Module):
    """GradNorm 动态权重。

    用法 (训练循环里):
      total = balancer.weighted_sum(losses)
      total.backward(retain_graph=True)           # 先回传得到模型梯度
      gn = balancer.grad_norm_loss(losses, shared_params)
      gn.backward()                                # 只更新 w 的梯度
      ... optimizer.step() ...
      balancer.renormalize()                       # 把 w 重新归一化到 sum=n
    注意: 需 accum_steps=1，且这段建议在 fp32 下做 (不要 autocast)。
    """

    def __init__(self, task_names: List[str], alpha: float = 1.5):
        super().__init__()
        self.task_names = list(task_names)
        self.alpha = alpha
        n = len(task_names)
        self.w = nn.Parameter(torch.ones(n))
        self.register_buffer("initial", torch.zeros(n))
        self._init_done = False

    def _idx(self, name: str) -> int:
        return self.task_names.index(name)

    def set_initial(self, losses: Dict[str, torch.Tensor]) -> None:
        """记录第一步各任务 loss 作为基准 L_i(0)。只设一次。"""
        if self._init_done:
            return
        for name in self.task_names:
            if name in losses:
                self.initial[self._idx(name)] = float(losses[name].detach())
        self._init_done = True

    def weighted_sum(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for name in self.task_names:
            if name in losses:
                total = total + self.w[self._idx(name)] * losses[name]
        return total

    def grad_norm_loss(self, losses: Dict[str, torch.Tensor],
                       shared_params: List[torch.Tensor]) -> torch.Tensor:
        """计算 GradNorm 的辅助 loss = Σ_i |G_i - target_i|。

        关键: G_i = ||∇_shared (w_i·L_i)|| = w_i·||∇_shared L_i|| (w_i 为标量，
        可提到范数外)。所以只需用一阶梯度算 g_i=||∇_shared L_i|| (detach)，再把
        G_i 显式写成 w_i·g_i。这样对 w 只需一阶求导，完全避开任何算子的
        double-backward (roi_align / fused-attention 都不支持二阶导)。
        数学上与标准 GradNorm (create_graph=True 让 autograd 处理 ∂G_i/∂w_i) 等价。
        """
        active = [n for n in self.task_names if n in losses]
        gw = []
        for name in active:
            # 一阶梯度 (不建二阶图)，得到该任务对共享参数的梯度范数 (常量)
            gi = torch.autograd.grad(
                losses[name], shared_params,
                retain_graph=True, create_graph=False, allow_unused=True)
            flat = [g.reshape(-1) for g in gi if g is not None]
            if len(flat) == 0:
                gnorm = torch.zeros((), device=self.w.device)
            else:
                gnorm = torch.cat(flat).norm(2)
            wi = self.w[self._idx(name)]
            gw.append(wi * gnorm.detach())      # G_i = w_i · ||∇L_i|| (对 w 可导)
        gw = torch.stack(gw)                     # (A,)
        gw_mean = gw.mean().detach()

        # 相对反向训练速度 r_i
        ratios = []
        for name in active:
            i = self._idx(name)
            init = self.initial[i].clamp(min=1e-8)
            ratios.append((losses[name].detach() / init))
        ratios = torch.stack(ratios)
        r = ratios / ratios.mean().clamp(min=1e-8)

        target = (gw_mean * (r ** self.alpha)).detach()
        return (gw - target).abs().sum()

    @torch.no_grad()
    def renormalize(self) -> None:
        """把 w 钳到非负并重新缩放到 sum = n。"""
        self.w.data.clamp_(min=1e-3)
        n = self.w.numel()
        self.w.data.mul_(n / self.w.data.sum())

    def logs(self) -> Dict[str, float]:
        return {f"w_{n}": float(self.w[self._idx(n)].detach())
                for n in self.task_names}
