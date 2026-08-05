# -*- coding: utf-8 -*-
"""
checkpoint 存取 —— 这是"数字串台"的根因修复。

FIX(S-2): 旧 train.py 存的是 `{"model": ..., "cfg": cfg.task_names}`，
          而 `cfg.task_names` 只是 ("cls","seg","macula") 三个字符串，**不是配置**。
          于是加载方只能用 `Config()` 默认值重建模型 + `strict=False`，
          任何形状不匹配的头都会**静默保持随机初始化**、不报错、指标照常输出。
          这是 0.812/0.754 串台、以及"phase4 checkpoint 与 phase2_v2 不一致"
          最可能的机制。

FIX(T-9)/FIX(U-11): 旧 loader 把 `missing_keys` 里含 "lora" 的键过滤掉再告警
          —— 而 LoRA 权重恰恰是训练出来的东西，等于一个不含 LoRA 的 checkpoint
          可以完全静默地加载成功。`unexpected_keys` 更是从未被检查。

现在：save 存完整 asdict(cfg) + seed + 指标 + git commit；
      load 默认 strict=True，宽松加载必须显式 allow_partial=True 并打印全部键。
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, Optional

import torch

from .config import Config

CKPT_FORMAT_VERSION = 2


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__) or "."
        ).decode().strip()
    except Exception:
        return "unknown"


def save_checkpoint(path: str, model, cfg: Config, *, epoch: int,
                    metrics: Optional[dict] = None,
                    extra: Optional[dict] = None,
                    balancer=None, threshold: Optional[float] = None) -> str:
    """存 checkpoint。cfg 以 dict 形式完整落盘，供 load 端精确重建模型。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": CKPT_FORMAT_VERSION,
        "model": model.state_dict(),
        "cfg": cfg.to_dict(),          # FIX(S-2): 完整配置，不是 task_names
        "seed": cfg.train.seed,
        "epoch": int(epoch),
        "metrics": dict(metrics or {}),
        "git": _git_commit(),
        "torch": torch.__version__,
    }
    if balancer is not None:
        payload["balancer"] = balancer.state_dict()
    if threshold is not None:
        payload["cls_threshold"] = float(threshold)
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    # weights_only=False 是必要的（payload 含 dict/str），但只加载你自己产出的文件
    return torch.load(path, map_location=map_location, weights_only=False)


def config_from_checkpoint(ckpt: dict, fallback: Optional[Config] = None) -> Config:
    """从 checkpoint 恢复 Config。旧格式 (cfg 是 tuple) 时回退并显式警告。"""
    raw = ckpt.get("cfg")
    if isinstance(raw, dict):
        return Config.from_dict(raw)
    print("[checkpoint] ⚠️  这是旧格式 checkpoint（cfg 不是完整配置）。"
          "无法确认它训练时的 img_size / seg_mode / fine_out / lora.r，"
          "只能用调用方提供的配置重建——这正是 FIX(S-2) 要消除的情况。"
          "建议用新版 train.py 重训或重存一次。")
    if fallback is None:
        raise ValueError("旧格式 checkpoint 且未提供 fallback config")
    return fallback


def load_model_state(model, state: dict, *, allow_partial: bool = False,
                     tag: str = "model"):
    """严格加载权重。

    allow_partial=False (默认): 任何 missing / unexpected 键都直接抛异常。
    allow_partial=True        : 打印**完整**键列表后继续，绝不静默。
    """
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = list(missing)
    unexpected = list(unexpected)
    if not missing and not unexpected:
        print(f"[checkpoint] {tag}: 权重完整加载 ({len(state)} 个张量)")
        return missing, unexpected

    msg = (f"[checkpoint] {tag}: 权重未完整匹配\n"
           f"  missing    ({len(missing)}): {missing}\n"
           f"  unexpected ({len(unexpected)}): {unexpected}")
    if not allow_partial:
        raise RuntimeError(
            msg + "\n\n如果这是有意为之（例如只加载骨干），请显式传 "
                  "allow_partial=True。默认拒绝是为了避免任务头静默保持随机初始化。")
    print(msg + "\n  [warn] allow_partial=True，继续执行。")
    return missing, unexpected


def build_model_from_checkpoint(path: str, model_cls, *,
                                map_location: str = "cpu",
                                override_img_size: Optional[int] = None,
                                allow_partial: bool = False,
                                fallback_cfg: Optional[Config] = None):
    """一步到位：读 ckpt → 用 ckpt 里的 cfg 建模型 → 严格加载权重。

    返回 (model, cfg, ckpt)。这是 phase3 / phase4 唯一应该使用的加载入口。
    """
    ckpt = load_checkpoint(path, map_location)
    cfg = config_from_checkpoint(ckpt, fallback_cfg)
    if override_img_size is not None:
        s = int(override_img_size)
        if tuple(cfg.backbone.img_size) != (s, s):
            print(f"[checkpoint] ⚠️  override img_size "
                  f"{tuple(cfg.backbone.img_size)} -> ({s},{s})；"
                  f"pos_embed 会被重新插值，指标可能与训练时不同。")
        cfg.backbone.img_size = (s, s)
    model = model_cls(cfg)
    load_model_state(model, ckpt.get("model", ckpt),
                     allow_partial=allow_partial, tag=os.path.basename(path))
    return model, cfg, ckpt
