# -*- coding: utf-8 -*-
"""
RETFound (ViT-L/16, MAE) 骨干封装。

关键工程点:
1. 位置编码插值: RETFound 在 224 上预训练 (14x14 grid)，我们要吃 384 甚至更高，
   必须把 pos_embed 的 patch 部分双三次插值到新 grid，cls token 那一行单独保留。
2. 多尺度特征: 从 out_indices 指定的 block 抓 token 序列，去掉 cls 后 reshape 成
   (B, D, gh, gw) 给分割/黄斑解码器。
   注意: ViT 各层空间分辨率相同 (都是 gh x gw)，差别在语义层级，不是分辨率金字塔。
3. 显存: grad_checkpointing 打开时对每个 block 用 checkpoint(use_reentrant=False)。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

import timm


# --------------------------------------------------------------------------- #
# 位置编码插值
# --------------------------------------------------------------------------- #
def interpolate_pos_embed(pos_embed: torch.Tensor,
                          new_grid: Tuple[int, int],
                          num_extra_tokens: int = 1) -> torch.Tensor:
    """把 pos_embed 的 patch 部分双三次插值到 new_grid。

    pos_embed: (1, num_extra + old_gh*old_gw, D)
    返回:      (1, num_extra + new_gh*new_gw, D)
    前 num_extra_tokens 个 (cls 等) 原样保留。
    """
    n_total = pos_embed.shape[1]
    dim = pos_embed.shape[2]
    n_patch = n_total - num_extra_tokens
    old_side = int(round(n_patch ** 0.5))
    assert old_side * old_side == n_patch, \
        "旧 pos_embed 不是正方形 grid，无法推断边长"

    new_gh, new_gw = new_grid
    if old_side == new_gh and old_side == new_gw:
        return pos_embed  # 无需插值

    extra = pos_embed[:, :num_extra_tokens]                 # (1, extra, D)
    patch = pos_embed[:, num_extra_tokens:]                 # (1, n_patch, D)
    patch = patch.reshape(1, old_side, old_side, dim).permute(0, 3, 1, 2)
    patch = F.interpolate(patch, size=(new_gh, new_gw),
                          mode="bicubic", align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, new_gh * new_gw, dim)
    return torch.cat([extra, patch], dim=1)


# --------------------------------------------------------------------------- #
# 骨干
# --------------------------------------------------------------------------- #
class RETFoundBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.img_size = cfg.img_size
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.embed_dim
        self.out_indices = tuple(cfg.out_indices)
        self.pool = cfg.pool

        gh = cfg.img_size[0] // cfg.patch_size
        gw = cfg.img_size[1] // cfg.patch_size
        self._grid = (gh, gw)

        # 用 timm 造一个无分类头的 ViT。class_token=True 以对齐 RETFound。
        # global_pool='token' 只影响 timm 自带 forward_head，我们用自定义 forward，
        # 所以这里取值不影响中间特征抽取。
        self.vit = timm.create_model(
            self._timm_name(cfg),
            pretrained=False,
            num_classes=0,
            class_token=True,
            global_pool="token",
            img_size=cfg.img_size,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
        )

        # timm 已按 img_size 造好 pos_embed，但为稳妥仍按需插值到我们的 grid。
        self._ensure_pos_embed()

        # 位置编码是否可训练
        self.vit.pos_embed.requires_grad_(bool(cfg.train_pos_embed))

        self.grad_checkpointing = bool(cfg.grad_checkpointing)

    # ----- 构建辅助 ----- #
    def _timm_name(self, cfg) -> str:
        """按 embed_dim/depth 选 timm 结构名 (full ViT-L 或 tiny smoke)。"""
        if cfg.embed_dim == 1024 and cfg.depth == 24:
            return "vit_large_patch16_224"
        # FIX(S-10): 旧代码在这里返回 "__custom__"，timm.create_model 会直接崩，
        # 只是因为 build_backbone() 提前分流才没暴露。给出明确错误而不是死代码。
        raise ValueError(
            f"RETFoundBackbone 只支持 ViT-L/16 (embed_dim=1024, depth=24)，"
            f"收到 embed_dim={cfg.embed_dim}, depth={cfg.depth}。"
            f"其它尺寸请用 build_backbone()，它会分流到 _CustomViTBackbone。")

    def _ensure_pos_embed(self):
        pe = self.vit.pos_embed
        n_extra = getattr(self.vit, "num_prefix_tokens", 1)
        gh, gw = self._grid
        if pe.shape[1] != n_extra + gh * gw:
            new_pe = interpolate_pos_embed(pe.data, (gh, gw), n_extra)
            self.vit.pos_embed = nn.Parameter(new_pe)

    # ----- 属性 ----- #
    @property
    def grid_size(self) -> Tuple[int, int]:
        return self._grid

    @property
    def num_prefix_tokens(self) -> int:
        return getattr(self.vit, "num_prefix_tokens", 1)

    def set_fused_attn(self, enabled: bool) -> None:
        """开/关 timm attention 的 fused SDPA。

        GradNorm 需要二阶导 (对梯度范数再求导)，而 fused/flash SDPA 没有
        double-backward 实现 (CPU/部分 CUDA 都如此)，故 GradNorm 时关掉，
        退回可二阶求导的手写 softmax(q@k^T)@v 路径。
        """
        for m in self.vit.modules():
            if hasattr(m, "fused_attn"):
                m.fused_attn = enabled

    # ----- 前向 ----- #
    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        v = self.vit
        B = x.shape[0]

        x = v.patch_embed(x)                                # (B, N, D)
        # 拼 cls token
        cls = v.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)                      # (B, 1+N, D)
        # 加位置编码 (已插值到当前 grid)
        x = x + v.pos_embed
        x = v.pos_drop(x)
        # 一些 timm 版本有 patch_drop / norm_pre，没有则是 Identity
        x = getattr(v, "patch_drop", nn.Identity())(x)
        x = getattr(v, "norm_pre", nn.Identity())(x)

        levels: List[torch.Tensor] = []
        gh, gw = self._grid
        n_pre = self.num_prefix_tokens
        for i, blk in enumerate(v.blocks):
            if self.grad_checkpointing and self.training:
                x = cp.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if i in self.out_indices:
                tok = x[:, n_pre:]                          # 去掉前缀 token
                feat = tok.transpose(1, 2).reshape(B, self.embed_dim, gh, gw)
                levels.append(feat)

        x = v.norm(x)
        cls_out = x[:, 0]
        tokens = x[:, n_pre:]                                # (B, N, D)
        if self.pool == "mean":
            pooled = tokens.mean(dim=1)
        else:  # 'token'
            pooled = cls_out

        return {
            "pooled": pooled,           # (B, D) 给分类头
            "cls": cls_out,             # (B, D)
            "tokens": tokens,           # (B, N, D)
            "levels": levels,           # list of (B, D, gh, gw)
            "grid": self._grid,
        }

    # ----- 加载 RETFound 预训练权重 ----- #
    def load_pretrained(self, ckpt_path: str) -> str:
        """加载 RETFound .pth。处理 'model'/'state_dict' 包装、'module.' 前缀、
        丢弃 decoder/mask_token/head 键、把 pos_embed 插值到当前 grid、
        只保留 shape 匹配的键。返回 load 信息字符串。
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        # 去前缀 + 丢无关键
        cleaned = {}
        for k, val in state.items():
            nk = k[7:] if k.startswith("module.") else k
            if nk.startswith(("decoder", "mask_token", "head")):
                continue
            cleaned[nk] = val

        tgt = self.vit.state_dict()

        # pos_embed 插值到当前 grid
        if "pos_embed" in cleaned and "pos_embed" in tgt:
            if cleaned["pos_embed"].shape != tgt["pos_embed"].shape:
                gh, gw = self._grid
                cleaned["pos_embed"] = interpolate_pos_embed(
                    cleaned["pos_embed"], (gh, gw), self.num_prefix_tokens)

        # 只留 shape 匹配的键
        keep = {k: v for k, v in cleaned.items()
                if k in tgt and v.shape == tgt[k].shape}
        missing = [k for k in tgt if k not in keep]
        msg = self.vit.load_state_dict(keep, strict=False)
        info = (f"[RETFound] loaded {len(keep)}/{len(tgt)} tensors; "
                f"missing {len(missing)}; "
                f"unexpected {len(getattr(msg, 'unexpected_keys', []))}")
        return info


def build_backbone(cfg) -> RETFoundBackbone:
    """工厂: full ViT-L 用命名模型; 其它 (tiny smoke) 用 VisionTransformer 直接构造。"""
    if cfg.embed_dim == 1024 and cfg.depth == 24:
        return RETFoundBackbone(cfg)
    # ---- 自定义小 ViT (smoke test) ---- #
    return _CustomViTBackbone(cfg)


class _CustomViTBackbone(RETFoundBackbone):
    """smoke test 用: 跳过命名模型，直接用 timm.models.vision_transformer。"""

    def __init__(self, cfg):
        nn.Module.__init__(self)
        from timm.models.vision_transformer import VisionTransformer
        self.cfg = cfg
        self.img_size = cfg.img_size
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.embed_dim
        self.out_indices = tuple(cfg.out_indices)
        self.pool = cfg.pool
        gh = cfg.img_size[0] // cfg.patch_size
        gw = cfg.img_size[1] // cfg.patch_size
        self._grid = (gh, gw)

        self.vit = VisionTransformer(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=3,
            num_classes=0,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            class_token=True,
            global_pool="token",
            drop_path_rate=cfg.drop_path_rate,
        )
        self._ensure_pos_embed()
        self.vit.pos_embed.requires_grad_(bool(cfg.train_pos_embed))
        self.grad_checkpointing = bool(cfg.grad_checkpointing)
