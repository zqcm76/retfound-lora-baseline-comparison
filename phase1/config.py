# -*- coding: utf-8 -*-
"""Phase 1 配置。

FIX(C-4): 两个 notebook 都写 `cfg = Config("baseline.yaml")`。Config 是
          dataclass，第一个位置参数是 `data`，于是 cfg.data 被静默设成字符串
          "baseline.yaml"，访问 cfg.data.cls_image_size 才会 AttributeError。
          因为 notebook 后续没真用 cfg，这个错误一直没被发现。
          现在 __post_init__ 会立即拒绝这种用法。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    # REFUGE2 索引 CSV，列: image_path, mask_path, label, fovea_x, fovea_y, [split]
    refuge_index: str = "data/refuge_index.csv"   # FIX: 不再硬编码 E:/ 绝对路径
    image_size: int = 512        # 分割工作分辨率 (16 的倍数)
    cls_image_size: int = 224    # 分类工作分辨率
    num_seg_classes: int = 3     # 0=bg, 1=rim, 2=cup
    val_split: float = 0.2       # 索引无 split 列时才用
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    pretrained: bool = True
    unet_base_channels: int = 32


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_accum_steps: int = 1
    amp: bool = True
    pos_weight: Optional[float] = None   # 青光眼是稀有类，REFUGE2 约 1:9
    seg_ce_weight: float = 1.0
    seg_dice_weight: float = 1.0
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "outputs"
    experiment: str = "baseline"

    def __post_init__(self):
        # FIX(C-4): 让 Config("baseline.yaml") 这种误用立即失败而不是静默通过
        for name, typ in (("data", DataConfig), ("model", ModelConfig),
                          ("train", TrainConfig)):
            v = getattr(self, name)
            if not isinstance(v, typ):
                raise TypeError(
                    f"Config.{name} 应为 {typ.__name__}，收到 {type(v).__name__} "
                    f"({v!r})。\n若要从 YAML 加载，请用 "
                    f"Config.from_yaml(path)，不要用 Config(path)。")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            output_dir=raw.get("output_dir", "outputs"),
            experiment=raw.get("experiment", "baseline"),
        )

    def to_dict(self) -> dict:
        return asdict(self)
