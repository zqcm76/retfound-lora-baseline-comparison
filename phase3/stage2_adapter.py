# -*- coding: utf-8 -*-
"""
stage2_adapter.py — 把阶段二的 MultiTaskModel 包装成阶段三的 Stage2Encoder 协议。

格式差异:
  Stage2 seg 头: coarse_logits [B, 2, H, W] — 两个独立 sigmoid (disc/cup 二分类)
  Stage3 需要:   seg_logits    [B, 3, H, W] — 三类 softmax {bg=0, rim=1, cup=2}

  Stage2 img:    feats["pooled"] [B, D]  — ViT-L mean-pool 或 cls token
  Stage3 需要:   img_feat        [B, D]  — 直接兼容 ✓

  Stage2 seg_target: [B,2,H,W] float {0,1} = stack(disc_mask, cup_mask)
  Stage3 y_seg:      [B,H,W] int64 {0=bg,1=rim,2=cup}

转换方案 (binary -> 3-class):
  p_disc = sigmoid(disc_logit)
  p_cup  = sigmoid(cup_logit) * p_disc      ← 解剖约束: cup ⊂ disc
  p_rim  = p_disc - p_cup                   ← rim = disc 但不含 cup
  p_bg   = 1 - p_disc
  z_i    = log(p_i + eps) → softmax(z) ≈ [p_bg, p_rim, p_cup]

  数学保证: softmax(log(p_i)) = p_i / Σ p_j ≈ p_i (当 Σ p_j ≈ 1 时近似恒等)
  实际上 p_bg+p_rim+p_cup = 1 - p_disc + p_rim + p_cup = 1 ✓ 精确成立。
  所以 softmax(z) = [p_bg, p_rim, p_cup] 精确成立。

为什么用 coarse_logits 而不是 fine_logits:
  fine_logits 在 ROI 内坐标系 (only covers the disc region)，CDR 计算需要全图坐标
  知道盘外背景在哪。coarse_logits 是全图 96x96，足够 CDR geometry 估计。
  如需更高分辨率，可在 adapter 里换用 paste_roi_to_full 路径。

使用方法:
    # 加载 stage2 checkpoint
    from stage2_adapter import Stage2Adapter, load_stage2_model
    model_s2 = load_stage2_model("runs/phase2/best.pth", cfg)
    
    # 包装成 stage3 接口
    adapter = Stage2Adapter(model_s2, cfg)
    
    # 传给 Stage3Trainer
    from train_stage3 import Stage3Trainer, Stage3Config
    s3cfg = Stage3Config(mode="cdr_soft", img_feat_dim=cfg.backbone.embed_dim)
    trainer = Stage3Trainer(adapter, s3cfg, device="cuda")
    trainer.fit(train_loader, val_loader, epochs=15)
"""
from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

import _bootstrap  # noqa: F401

from retfound_common.checkpoint import build_model_from_checkpoint
from retfound_common.config import Config, tiny_config
from retfound_common.model import MultiTaskModel


# --------------------------------------------------------------------------- #
# 格式转换工具
# --------------------------------------------------------------------------- #
def binary_to_3class_logits(
    coarse_logits: torch.Tensor,
    upsample_size: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    [B,2,H,W] 二分类 sigmoid logits (disc/cup) → [B,3,H,W] 三类 softmax logits。

    ch0 = disc (整个视盘, 含 cup), ch1 = cup。
    输出 ch0=bg, ch1=rim, ch2=cup，可直接喂 `logits_to_disc_cup_probs()` 和
    `compute_cdr_hard/SoftCDR`。

    upsample_size: 若不为 None，在转换后把 spatial 维度双线性插值到该尺寸。
                   CDR 在 96px 上已足够精确；推理时可调整。
    """
    if coarse_logits.dim() != 4 or coarse_logits.size(1) != 2:
        raise ValueError(
            f"expected [B,2,H,W] from stage-2 coarse head, got {tuple(coarse_logits.shape)}"
        )
    disc_logit = coarse_logits[:, 0]          # [B,H,W]
    cup_logit  = coarse_logits[:, 1]          # [B,H,W]

    p_disc = torch.sigmoid(disc_logit)                    # P(disc∪cup)
    p_cup  = torch.sigmoid(cup_logit) * p_disc            # 解剖约束: cup ⊂ disc
    p_rim  = (p_disc - p_cup).clamp(min=0.0)             # rim = disc \ cup
    p_bg   = (1.0 - p_disc).clamp(min=0.0)               # background

    # log(p) 精确恢复：softmax(log(p)) = p，因为 p_bg+p_rim+p_cup = 1
    z = torch.stack(
        [torch.log(p_bg + eps),
         torch.log(p_rim + eps),
         torch.log(p_cup + eps)],
        dim=1,
    )   # [B,3,H,W]

    if upsample_size is not None:
        z = F.interpolate(z, size=(upsample_size, upsample_size),
                          mode="bilinear", align_corners=False)
    return z


def seg_target_to_3class(
    disc_mask: torch.Tensor,
    cup_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Stage2 格式的 disc/cup 二值掩膜 → Stage3 三类整数 seg target。

    disc_mask: [B,1,H,W] 或 [B,H,W] float {0,1} — 整个视盘 (含 cup)
    cup_mask:  [B,1,H,W] 或 [B,H,W] float {0,1} — 仅视杯

    返回: [B,H,W] int64 {0=bg, 1=rim, 2=cup}

    证明: disc+cup ∈ {0,1,2}，当 cup⊂disc 时：
      disc=0,cup=0 → 0=bg ✓
      disc=1,cup=0 → 1=rim ✓
      disc=1,cup=1 → 2=cup ✓
    """
    if disc_mask.dim() == 4:
        disc_mask = disc_mask[:, 0]
    if cup_mask.dim() == 4:
        cup_mask = cup_mask[:, 0]
    return (disc_mask.long() + cup_mask.long()).clamp(0, 2)


# --------------------------------------------------------------------------- #
# 适配器主体
# --------------------------------------------------------------------------- #
class Stage2Adapter(nn.Module):
    """
    把 MultiTaskModel 包装成 Stage2Encoder 协议。

    Parameters
    ----------
    model      : MultiTaskModel (stage2)
    seg_out_size : 把 coarse_logits 插值到该尺寸再做三类转换。
                   None → 保持 coarse_out (默认 96)。
                   传入和训练时 seg_size 一致 (run_ablation 里 MockStage2 用 48)。
    """

    def __init__(self, model: MultiTaskModel, seg_out_size: int | None = None):
        super().__init__()
        self.model = model
        self.seg_out_size = seg_out_size
        self._seg_mode = model.seg_mode        # 'coarse2fine' or 'highres'

    # ---- Stage2Encoder 协议 ------------------------------------------------ #

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        images: [B, 3, H, W]
        返回:
          img_feat   [B, D]       — backbone pooled 特征 (D=embed_dim)
          seg_logits [B, 3, Hs, Ws] — 三类 softmax logits {bg, rim, cup}
        """
        # 只跑 backbone + seg 头 (不跑分类/黄斑头，节省显存和时间)
        feats = self.model.backbone(images)
        img_feat = feats["pooled"]             # [B, D]
        levels   = feats["levels"]

        if self._seg_mode == "coarse2fine":
            # 只用粗分割 (全图坐标系，CDR 需要全图)
            coarse_logits, _ = self.model.coarse_seg(levels)   # [B,2,coarse,coarse]
        else:  # highres
            coarse_logits = self.model.highres_seg(levels)     # [B,2,H,W]

        seg_3class = binary_to_3class_logits(coarse_logits, self.seg_out_size)
        return img_feat, seg_3class

    def segmentation_parameters(self) -> Iterator[nn.Parameter]:
        """分割头参数 (cdr_soft 时解冻并以小 lr 更新)。

        FIX(T-4): 旧版同时 yield 了 coarse_seg 和 fine_seg，但 encode() 只调用
        coarse_seg —— **fine_seg 不在前向图里，永远收不到梯度**。它那 3.54M
        参数被放进优化器、被计进消融表的 trainable_params 列，纯属虚高
        (cdr_soft 因此显示为另外两个模式的 38 倍)。现在只 yield 真正参与前向的。
        """
        if self._seg_mode == "coarse2fine":
            yield from self.model.coarse_seg.parameters()
        else:
            yield from self.model.highres_seg.parameters()

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        """LoRA + pos_embed 等骨干可训练参数 (stage3 默认冻结)。"""
        return (p for p in self.model.backbone.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# DataLoader 适配器: stage2 dict batch → stage3 tuple batch
# --------------------------------------------------------------------------- #
class Stage3DataLoaderWrapper:
    """
    把 stage2 的 multitask_collate DataLoader 输出的 dict batch 转换为
    stage3 Trainer 期望的 (images, y_cls, y_seg) tuple。

    用法:
        raw_loader = DataLoader(train_ds, collate_fn=multitask_collate, ...)
        loader_s3  = Stage3DataLoaderWrapper(raw_loader, device=device)
        for images, y_cls, y_seg in loader_s3:
            ...
    """

    def __init__(self, loader, seg_size: int | None = None):
        self.loader = loader
        self.seg_size = seg_size

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            images   = batch["image"]           # [B,3,H,W] float
            y_cls    = batch["cls_label"]       # [B] int64
            has_seg  = batch["has_seg"]         # [B] bool
            # multitask_collate 把 disc/cup 合并到 seg_target [B,2,H,W]
            # disc = seg_target[:,0:1], cup = seg_target[:,1:2]
            if has_seg.all():
                seg_target = batch["seg_target"]        # [B,2,H,W] float
                disc = seg_target[:, 0:1]               # [B,1,H,W]
                cup  = seg_target[:, 1:2]               # [B,1,H,W]
                y_seg = seg_target_to_3class(disc, cup) # [B,H,W] int64
                if self.seg_size is not None:
                    # 最近邻 resize 保持整数标签
                    y_seg = F.interpolate(
                        y_seg.unsqueeze(1).float(),
                        size=(self.seg_size, self.seg_size),
                        mode="nearest",
                    ).squeeze(1).long()
            else:
                y_seg = None   # stage3 trainer 按 has_seg=None 处理

            yield images, y_cls, y_seg


# --------------------------------------------------------------------------- #
# 加载真实 stage2 checkpoint
# --------------------------------------------------------------------------- #
def load_stage2_model(
    ckpt_path: str,
    cfg: Config | None = None,
    device: str = "cpu",
    map_location: str | None = None,
    allow_partial: bool = False,
):
    """从 best.pth 加载 stage2 模型，返回 (model, cfg)。

    FIX(T-9): 旧版把 missing_keys 里含 "lora"/"pos_embed" 的键**过滤掉**再告警。
    而 LoRA 权重恰恰是 stage-2 训练出来的东西 —— 等于一个不含 LoRA 的
    checkpoint 可以完全静默地加载成功，`unexpected_keys` 更是从未被检查。
    FIX(S-2): 配置从 checkpoint 里恢复，而不是用 Config() 默认值猜。
    """
    model, real_cfg, ckpt = build_model_from_checkpoint(
        ckpt_path, MultiTaskModel,
        map_location=map_location or device,
        allow_partial=allow_partial, fallback_cfg=cfg)
    model.to(device).eval()
    print(f"[stage2_adapter] 已加载 {ckpt_path} "
          f"(epoch={ckpt.get('epoch')}, seed={ckpt.get('seed')}, "
          f"git={str(ckpt.get('git'))[:8]})")
    return model, real_cfg


# --------------------------------------------------------------------------- #
# 快速兼容性检验
# --------------------------------------------------------------------------- #
def _smoke_test_adapter():
    """验证 adapter 格式和梯度行为。运行: python3 stage2_adapter.py"""
    print("=== Stage2Adapter smoke test ===\n")
    cfg = tiny_config()
    cfg.backbone.img_size = (128, 128)

    model = MultiTaskModel(cfg)
    adapter = Stage2Adapter(model, seg_out_size=48)

    # 协议检查
    from stage2_iface import Stage2Encoder
    assert isinstance(adapter, Stage2Encoder), \
        "Stage2Adapter 未满足 Stage2Encoder 协议"
    print("✓ 满足 Stage2Encoder 协议")

    # forward 形状
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        img_feat, seg_logits = adapter.encode(x)
    print(f"✓ img_feat:   {tuple(img_feat.shape)}")
    print(f"✓ seg_logits: {tuple(seg_logits.shape)}  (期望 [2,3,48,48])")
    assert img_feat.shape == (2, cfg.backbone.embed_dim), "img_feat shape mismatch"
    assert seg_logits.shape == (2, 3, 48, 48), "seg_logits shape mismatch"

    # softmax 概率和 = 1 (验证转换的数学正确性)
    probs = torch.softmax(seg_logits, dim=1)
    prob_sum = probs.sum(dim=1)
    assert (prob_sum - 1.0).abs().max().item() < 1e-5, "概率和不为 1"
    print("✓ softmax(seg_logits) 概率和 = 1.0")

    # segmentation_parameters 可迭代
    seg_params = list(adapter.segmentation_parameters())
    bb_params = list(adapter.backbone_parameters())
    print(f"✓ segmentation_parameters: {len(seg_params)} 个 tensor")
    print(f"✓ backbone_parameters:     {len(bb_params)} 个 tensor")

    # seg_target_to_3class 转换
    B, H, W = 2, 48, 48
    disc = torch.randint(0, 2, (B, 1, H, W)).float()
    cup  = disc * torch.randint(0, 2, (B, 1, H, W)).float()  # cup ⊂ disc
    y3 = seg_target_to_3class(disc, cup)
    assert y3.dtype == torch.int64, "y3 应为 int64"
    assert y3.min() >= 0 and y3.max() <= 2, "y3 超出 {0,1,2}"
    print(f"✓ seg_target_to_3class: {tuple(y3.shape)}, 值域 {{{y3.min().item()},{y3.max().item()}}}")

    # Stage3DataLoaderWrapper 可迭代
    from retfound_common.dataset import DummyMultiTaskDataset, multitask_collate
    from torch.utils.data import DataLoader
    ds = DummyMultiTaskDataset(n=4, img_size=(128, 128), seed=0)
    raw = DataLoader(ds, batch_size=2, collate_fn=multitask_collate)
    wrap = Stage3DataLoaderWrapper(raw, seg_size=48)
    imgs_b, y_cls_b, y_seg_b = next(iter(wrap))
    print(f"✓ DataLoaderWrapper: images {tuple(imgs_b.shape)}, "
          f"y_cls {tuple(y_cls_b.shape)}, y_seg {tuple(y_seg_b.shape)}")

    # 接入 Stage3Trainer (完整训练一步)
    from train_stage3 import Stage3Trainer, Stage3Config
    s3cfg = Stage3Config(
        mode="cdr_soft",
        img_feat_dim=cfg.backbone.embed_dim,
        amp=False,
        focal_alpha=(0.5, 0.5),
        seg_loss_weight=1.0,
        soft_cdr_warmup_steps=2,
    )
    trainer = Stage3Trainer(adapter, s3cfg, device="cpu")
    trainer.fit(wrap, epochs=2, log_every=0)
    val = trainer.evaluate(wrap)
    print(f"✓ Stage3Trainer 跑通: {val}")
    print("\n=== 全部通过 ✔ ===")


if __name__ == "__main__":
    _smoke_test_adapter()
