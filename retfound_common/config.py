# -*- coding: utf-8 -*-
"""
共享配置 (原 phase2/config.py 与 phase3/config.py 的合并版)。

FIX(T-1): phase2 与 phase3 曾各持一份分叉的 config.py，默认值有 8 处不同
          (focal_alpha / dropout / uw_max_weight / epochs / head_lr / wd /
          warmup_epochs / min_lr)。此处以 phase2 的值为准，phase3 直接 import。
FIX(S-5): 新增 ClsCfg.label_smoothing —— 它是文档里的关键超参，之前只存在于
          FocalLoss 的默认参数里，不可配置、不可覆盖、不被记录。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple


@dataclass
class BackboneCfg:
    img_size: Tuple[int, int] = (384, 384)
    patch_size: int = 16

    # ViT-L 结构 (与 RETFound 权重对齐，勿改)
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0

    out_indices: Tuple[int, ...] = (5, 11, 17, 23)
    pool: str = "mean"                 # 'token' | 'mean'

    grad_checkpointing: bool = True
    train_pos_embed: bool = True       # 插值后的 pos_embed 是否可训练 (~0.59M)

    pretrained_ckpt: Optional[str] = None
    drop_rate: float = 0.0
    drop_path_rate: float = 0.05


@dataclass
class LoRACfg:
    mode: str = "lora"                 # 'lora' | 'adapter'
    r: int = 8
    alpha: int = 16                    # scaling = alpha / r
    dropout: float = 0.1
    targets: Tuple[str, ...] = ("qkv", "proj")
    adapter_bottleneck: int = 64


@dataclass
class SegCfg:
    mode: str = "coarse2fine"          # 'coarse2fine' | 'highres'
    decoder_ch: int = 256
    num_classes: int = 2               # disc, cup (两个独立 sigmoid)

    coarse_out: int = 96
    fine_in: int = 32
    fine_out: int = 128
    highres_out: int = 384

    roi_margin: float = 0.3
    coarse_loss_w: float = 0.4

    # FIX(S-4): 训练时 ROI 全部用 GT (teacher forcing)、评测时用预测框，
    # 造成曝光偏差。scheduled sampling: 训练中以概率 p 改用预测框，
    # p 在 roi_sampling_warmup 个 epoch 内从 0 线性升到 roi_pred_prob_max。
    # 设 roi_pred_prob_max=0.0 可退回旧行为。
    roi_pred_prob_max: float = 0.5
    roi_sampling_warmup_epochs: int = 20


@dataclass
class MaculaCfg:
    decoder_ch: int = 128
    heatmap_out: int = 96
    softargmax_beta: float = 10.0
    sigma: float = 0.03
    coord_weight: float = 1.0
    heatmap_weight: float = 1.0


@dataclass
class ClsCfg:
    num_classes: int = 2
    dropout: float = 0.2
    focal_gamma: float = 2.0
    focal_alpha: Optional[Tuple[float, ...]] = (0.1, 0.9)
    # FIX(S-5): 之前写死在 FocalLoss 默认参数里，文档却把它列为关键超参
    label_smoothing: float = 0.05


@dataclass
class BalanceCfg:
    method: str = "uncertainty"        # 'uncertainty' | 'gradnorm' | 'fixed'
    fixed_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 2.5e-2
    uw_max_weight: float = 5.0


@dataclass
class TrainCfg:
    epochs: int = 80
    batch_size: int = 2
    accum_steps: int = 8               # 有效 batch = batch_size * accum_steps
    lr: float = 2e-4                   # backbone (LoRA + pos_embed)
    head_lr: float = 1e-3              # 任务头
    wd: float = 0.01
    warmup_epochs: int = 5
    scheduler: str = "warmup_cosine"   # 'warmup_cosine' | 'none'
    min_lr: float = 1e-5
    # FIX(S-7): min_lr 之前只按 backbone 的 base_lr 折算，head 组的实际地板
    # 是 min_lr*head_lr/lr (5 倍于预期)。现按组独立计算，见 build_scheduler。
    head_min_lr: float = 1e-5
    balanced_sampler: bool = True
    amp: bool = True
    amp_dtype: str = "fp16"            # 'fp16' | 'bf16'
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 42
    out_dir: str = "./runs/phase2"
    log_interval: int = 1


@dataclass
class Config:
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    lora: LoRACfg = field(default_factory=LoRACfg)
    seg: SegCfg = field(default_factory=SegCfg)
    macula: MaculaCfg = field(default_factory=MaculaCfg)
    cls: ClsCfg = field(default_factory=ClsCfg)
    balance: BalanceCfg = field(default_factory=BalanceCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    task_names: Tuple[str, ...] = ("cls", "seg", "macula")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """FIX(S-2): checkpoint 里存的是完整 asdict(cfg)，加载时按此重建，
        而不是用 Config() 默认值 + strict=False 静默丢弃不匹配的头。"""
        return cls(
            backbone=BackboneCfg(**{**asdict(BackboneCfg()), **d.get("backbone", {})}),
            lora=LoRACfg(**{**asdict(LoRACfg()), **d.get("lora", {})}),
            seg=SegCfg(**{**asdict(SegCfg()), **d.get("seg", {})}),
            macula=MaculaCfg(**{**asdict(MaculaCfg()), **d.get("macula", {})}),
            cls=ClsCfg(**{**asdict(ClsCfg()), **d.get("cls", {})}),
            balance=BalanceCfg(**{**asdict(BalanceCfg()), **d.get("balance", {})}),
            train=TrainCfg(**{**asdict(TrainCfg()), **d.get("train", {})}),
            task_names=tuple(d.get("task_names", ("cls", "seg", "macula"))),
        )


def tiny_config() -> Config:
    """CPU 冒烟测试用，结构同构但极小。"""
    cfg = Config()
    cfg.backbone = BackboneCfg(
        img_size=(128, 128), patch_size=16, embed_dim=192, depth=4, num_heads=3,
        out_indices=(1, 3), pool="mean", grad_checkpointing=False,
        train_pos_embed=True, pretrained_ckpt=None, drop_path_rate=0.0)
    cfg.lora = LoRACfg(mode="lora", r=4, alpha=8, dropout=0.0,
                       targets=("qkv", "proj"), adapter_bottleneck=16)
    cfg.seg = SegCfg(mode="coarse2fine", decoder_ch=32, num_classes=2,
                     coarse_out=32, fine_in=8, fine_out=32, highres_out=64,
                     roi_margin=0.3, coarse_loss_w=0.4,
                     roi_pred_prob_max=0.5, roi_sampling_warmup_epochs=1)
    cfg.macula = MaculaCfg(decoder_ch=32, heatmap_out=32, softargmax_beta=10.0,
                           sigma=0.05, coord_weight=1.0, heatmap_weight=1.0)
    cfg.cls = ClsCfg(num_classes=2, dropout=0.0, focal_gamma=2.0,
                     focal_alpha=None, label_smoothing=0.0)
    cfg.balance = BalanceCfg(method="uncertainty")
    cfg.train = TrainCfg(epochs=1, batch_size=2, accum_steps=2, lr=1e-3,
                         head_lr=1e-3, warmup_epochs=0, amp=False, grad_clip=1.0,
                         num_workers=0, seed=0, out_dir="./runs/tiny")
    return cfg


# --------------------------------------------------------------------------- #
# 参数量分解 (FIX(P0-2)/FIX(U-3): 之前文档里 13.5M / 1.35M/任务 / 4.7M 三个数
# 互相矛盾，且 4.7M 溯源到 phase4 的合成占位符。这里给出唯一可复算的口径。)
# --------------------------------------------------------------------------- #
def param_breakdown(model) -> dict:
    """返回 LoRA / pos_embed / 任务头 / 合计 / 占比 的精确分解。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and ".lora_" in n)
    adapter = sum(p.numel() for n, p in model.named_parameters()
                  if p.requires_grad and ".adapter." in n)
    pos = sum(p.numel() for n, p in model.named_parameters()
              if p.requires_grad and n.endswith("pos_embed"))
    heads = trainable - lora - adapter - pos
    return {
        "total": total, "trainable": trainable,
        "lora": lora, "adapter": adapter, "pos_embed": pos, "heads": heads,
        "trainable_ratio": trainable / total if total else float("nan"),
    }


def format_param_breakdown(bd: dict) -> str:
    return (
        f"total={bd['total']/1e6:.2f}M  trainable={bd['trainable']/1e6:.2f}M "
        f"({bd['trainable_ratio']:.2%})\n"
        f"  ├─ lora      = {bd['lora']/1e6:.3f}M\n"
        f"  ├─ adapter   = {bd['adapter']/1e6:.3f}M\n"
        f"  ├─ pos_embed = {bd['pos_embed']/1e6:.3f}M\n"
        f"  └─ heads     = {bd['heads']/1e6:.3f}M"
    )
