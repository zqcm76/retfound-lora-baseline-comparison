# -*- coding: utf-8 -*-
r"""
phase4_adapter.py — 把真实数据 + Stage2/3 模型接入 Phase4 引擎。

════════════════════════════════════════════════════════════════════════════
本文件的两处修正决定了 headline 数字到底在描述哪个模型
════════════════════════════════════════════════════════════════════════════

FIX(U-1) ★ Phase 4 从未使用 Phase 3 的 CDR 融合头
    旧签名:  def build_real_phase4(..., stage3_head=None, ...)  # 暂时不用，CDR已经在Stage2里
    `stage3_head` 在整个函数体里**再也没有出现过**，而 run_phase4.py 老老实实
    传了 head_cdr_soft.pth —— 参数被静默丢弃。
    那句注释是错的：Stage 2 的分类头是 ClsHead(feats["pooled"])，只吃 backbone
    池化特征，**代码里没有任何 CDR**。CDR 融合完全是 Phase 3 的
    CDRFusionClassifier 才有的东西。
    后果：README 亮点写着"临床特征融合提升分类性能"，而 test 上评的是等价于
    pure_e2e 的裸 Stage-2 模型 —— 整条 CDR 贡献从未在 test 上被验证过。
    本版真正加载并使用 stage3_head；不提供时显式声明"无 CDR 融合"。

FIX(U-9) ★ 分割用的是粗解码器，精修解码器从未参与
    旧代码:  if "coarse_logits" in out: seg = out["coarse_logits"]
             elif "fine_logits" in out: ...
    coarse2fine 模式下 coarse_logits 永远存在 → **elif 分支永远不执行**。
    于是 headline 的 Disc/Cup Dice 根本没用到那个 3.54M 参数的精修解码器，
    而"Coarse-to-Fine 分割"是 README 的核心设计亮点。
    本版默认走 fine（paste_roi_to_full，与 phase2 训练目标一致），
    并保留 --seg-source coarse 做消融。

FIX(U-8) ★ 评测缩放与训练不一致
    训练 (retfound_common/transforms): A.Resize -> cv2.INTER_LINEAR，无抗锯齿
    旧推理:                            skimage.resize(anti_aliasing=True)
    1848->384 是 4.8 倍降采样，实测逐像素差 mean=0.0131、纹理被抹平 16.1%。
    模型在有锯齿的图上训练、评测却喂抗锯齿平滑的图 —— 这是评测代码自己造出来
    的分布偏移，和真实设备偏移叠在一起被算进了"域偏移"。本版与训练完全一致。

FIX(U-6) ★ 设备标签按图像尺寸判定，不再硬编码 "Canon_CR2"
"""
from __future__ import annotations

import os
from typing import Iterator, Optional

import _bootstrap  # noqa: F401
import numpy as np

import config as C
from core import EvalDataset, Model, Prediction, Sample


# ═══════════════════════════════════════════════════════════════════════════
# 1. 数据适配器
# ═══════════════════════════════════════════════════════════════════════════
class RefugeCSVAdapter:
    """从 data_all.csv 读取指定 split，输出 Sample 流。

    掩膜约定 (REFUGE2 灰度 BMP): 255=背景, 128=盘沿, 0=视杯
        disc_gt = gray < disc_gray_max (200)   整个视盘 (盘沿 + 视杯)
        cup_gt  = gray < cup_gray_max  (64)    仅视杯
    与 retfound_common/dataset._read_combined_mask 同一套阈值。
    """

    def __init__(self, csv_path: str, split: str = "test",
                 device_label: Optional[str] = None,
                 disc_gray_max: int = 200, cup_gray_max: int = 64,
                 dataset_name: str = "REFUGE2"):
        import pandas as pd
        self.name = dataset_name
        self.split = split
        self.device_label = device_label       # None = 按图像尺寸自动判定
        self.disc_gray_max = disc_gray_max
        self.cup_gray_max = cup_gray_max

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        self._rows = df.to_dict("records")
        self._n_bad_image = 0
        self._n_bad_mask = 0
        print(f"[RefugeCSVAdapter] split={split}, {len(self._rows)} 行")

    def __len__(self) -> int:
        return len(self._rows)

    def _device_for(self, image: Optional[np.ndarray]) -> str:
        """FIX(U-6): 按图像尺寸判定相机，与 phase2/loco_runner 同一份事实。

        旧版硬写 device_label="Canon_CR2"  # REFUGE2 test set = Canon CR-2
        —— Canon CR-2 是**训练集**相机。REFUGE2 的 test 来自第四台相机。
        """
        if self.device_label:
            return self.device_label
        if image is None:
            return "unknown"
        h, w = image.shape[:2]
        return C.DEVICE_BY_IMAGE_SIZE.get((h, w),
                                          C.DEVICE_BY_IMAGE_SIZE.get((w, h), "unknown"))

    def __iter__(self) -> Iterator[Sample]:
        from skimage import io as skio

        for r in self._rows:
            img_path = r.get("image_path", "")
            mask_path = r.get("mask_path", "")
            row_id = str(r.get("ID", os.path.basename(str(img_path))))

            try:
                image = skio.imread(img_path)
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                image = image[..., :3]
            except Exception as e:
                print(f"  [warn] 读图失败 {img_path}: {e}")
                image = None
                self._n_bad_image += 1

            disc_gt = cup_gt = None
            has_disc = has_cup = False
            if mask_path and os.path.exists(str(mask_path)):
                try:
                    m = skio.imread(mask_path)
                    if m.ndim == 3:
                        m = m[..., 0]
                    disc_gt = (m < self.disc_gray_max)
                    cup_gt = (m < self.cup_gray_max)
                    has_disc = bool(disc_gt.any())
                    has_cup = bool(cup_gt.any())
                except Exception as e:
                    print(f"  [warn] 读掩膜失败 {mask_path}: {e}")
                    self._n_bad_mask += 1
            elif mask_path:
                self._n_bad_mask += 1

            try:
                cls_label = int(r.get("label")); has_cls = True
            except (TypeError, ValueError):
                cls_label, has_cls = None, False

            fovea_gt, has_fovea = None, False
            try:
                fx, fy = float(r.get("fovea_x")), float(r.get("fovea_y"))
                if np.isfinite(fx) and np.isfinite(fy):
                    fovea_gt, has_fovea = (fx, fy), True
            except (TypeError, ValueError):
                pass

            yield Sample(
                id=row_id, device=self._device_for(image), image=image,
                cls_label=cls_label, disc_gt=disc_gt, cup_gt=cup_gt,
                fovea_gt=fovea_gt, has_disc=has_disc, has_cup=has_cup,
                has_fovea=has_fovea, has_cls=has_cls,
                split=self.split,                      # FIX(U-11): 不再写死 "test"
                image_path=str(img_path),
            )

        # FIX(U-11): 读失败之前只 print 不计数，400 张里丢几张不会被发现
        if self._n_bad_image or self._n_bad_mask:
            raise RuntimeError(
                f"[RefugeCSVAdapter] {self._n_bad_image} 张图 / {self._n_bad_mask} "
                f"张掩膜读取失败。静默跳过会让指标建立在一个未知子集上 —— "
                f"请先修数据路径。")


# ═══════════════════════════════════════════════════════════════════════════
# 2. 模型适配器
# ═══════════════════════════════════════════════════════════════════════════
class Phase2ModelAdapter:
    """把 Stage-2 MultiTaskModel（可选 + Stage-3 CDR 融合头）包成 Phase4 Model。

    参数
    ----
    stage3_head : CDRFusionClassifier | None
        FIX(U-1): 提供时，分类概率由 **CDR 融合头**给出（这才是 README 宣称的
        "临床特征融合"）。为 None 时用 Stage-2 自己的 ClsHead —— 那等价于
        pure_e2e，报告里必须写明"无 CDR 融合"。
    seg_source : 'fine' | 'coarse'
        FIX(U-9): 'fine' 走 paste_roi_to_full（与 phase2 训练目标一致），
        'coarse' 是旧路径。两者不可比，必须显式选。
    """

    def __init__(self, name: str, torch_model, *, input_size: int = 384,
                 device_str: str = "cuda", stage3_head=None,
                 seg_source: str = "fine", cdr_seg_size: int = 96):
        self.name = name
        self.model = torch_model
        self.input_size = int(input_size)
        self.device_str = device_str
        self.stage3_head = stage3_head
        self.seg_source = seg_source
        self.cdr_seg_size = int(cdr_seg_size)
        self.uses_cdr = stage3_head is not None
        self.model.to(device_str).eval()
        if self.stage3_head is not None:
            self.stage3_head.to(device_str).eval()
        print(f"[Phase2ModelAdapter] name={name} seg_source={seg_source} "
              f"CDR融合={'启用' if self.uses_cdr else '未启用(等价 pure_e2e)'}")

    # ------------------------------------------------------------------ #
    def _forward(self, chw):
        import torch
        import torch.nn.functional as F

        from retfound_common.model import paste_roi_to_full

        with torch.no_grad():
            out = self.model(chw, use_gt_roi=False)

            S = self.input_size
            # ---- 分割：FIX(U-9) 默认用精修结果，而不是永远只用 coarse ----
            if self.seg_source == "fine" and "fine_logits" in out:
                seg_logits = paste_roi_to_full(out["fine_logits"],
                                               out["roi_boxes"], (S, S))
            elif "coarse_logits" in out:
                seg_logits = F.interpolate(out["coarse_logits"], size=(S, S),
                                           mode="bilinear", align_corners=False)
            elif "seg_logits" in out:
                seg_logits = F.interpolate(out["seg_logits"], size=(S, S),
                                           mode="bilinear", align_corners=False)
            else:
                seg_logits = None

            # ---- 分类：FIX(U-1) 有 Stage-3 头就走 CDR 融合 ----
            if self.stage3_head is not None:
                from stage2_adapter import binary_to_3class_logits
                feats = self.model.backbone(chw)
                coarse, _ = self.model.coarse_seg(feats["levels"])
                seg3 = binary_to_3class_logits(coarse, self.cdr_seg_size)
                cls_logits = self.stage3_head(feats["pooled"], seg3)[0]
            else:
                cls_logits = out["cls_logits"][0]
            cls_logits = cls_logits.cpu().numpy()

            disc_prob = cup_prob = None
            if seg_logits is not None:
                s = seg_logits[0].cpu().numpy()
                disc_prob = 1.0 / (1.0 + np.exp(-s[0]))
                cup_prob = 1.0 / (1.0 + np.exp(-s[1]))

            fovea_xy = None
            if "macula_coords" in out:
                c = out["macula_coords"][0].cpu().numpy()
                fovea_xy = (float(c[0]) * S, float(c[1]) * S)

        return cls_logits, disc_prob, cup_prob, fovea_xy

    # ------------------------------------------------------------------ #
    def predict(self, image: np.ndarray) -> Prediction:
        import cv2
        import torch

        from retfound_common.transforms import (IMAGENET_MEAN, IMAGENET_STD,
                                                RESIZE_INTERPOLATION)

        H, W = image.shape[:2]
        S = self.input_size

        # FIX(U-8): 与训练侧完全一致的缩放 —— cv2.INTER_LINEAR、无抗锯齿。
        # 旧版用 skimage.resize(anti_aliasing=True)，1848->384 的 4.8 倍降采样下
        # 纹理被多抹平 16.1%，等于评测自带一层分布偏移。
        img_s = cv2.resize(np.ascontiguousarray(image), (S, S),
                           interpolation=RESIZE_INTERPOLATION).astype(np.float32) / 255.0
        img_s = (img_s - np.array(IMAGENET_MEAN, np.float32)) / \
                np.array(IMAGENET_STD, np.float32)
        chw = torch.from_numpy(img_s.transpose(2, 0, 1))[None].float().to(self.device_str)

        cls_logits, disc_prob, cup_prob, fovea_xy = self._forward(chw)

        cl = np.asarray(cls_logits, dtype=float).ravel()
        e = np.exp(cl - cl.max())
        cls_prob = float(e[1] / e.sum()) if cl.size > 1 else float(1 / (1 + np.exp(-cl[0])))

        def up(p):
            if p is None:
                return None
            return cv2.resize(p.astype(np.float32), (W, H),
                              interpolation=cv2.INTER_LINEAR)

        fov_o = None
        if fovea_xy is not None:
            fov_o = (fovea_xy[0] * W / S, fovea_xy[1] * H / S)

        return Prediction(cls_prob=cls_prob, disc_prob=up(disc_prob),
                          cup_prob=up(cup_prob), fovea_xy=fov_o)


# ═══════════════════════════════════════════════════════════════════════════
# 3. 组装
# ═══════════════════════════════════════════════════════════════════════════
def build_real_phase4(csv_path: str, stage2_ckpt: str,
                      stage3_head: Optional[str] = None,
                      split: str = "test", calib_split: str = "val",
                      device_str: str = "cuda", input_size: Optional[int] = None,
                      seg_source: str = "fine", cdr_seg_size: int = 96,
                      train_meta_json: Optional[str] = None,
                      allow_partial_load: bool = False):
    """返回 (models, datasets, train_meta)。

    FIX(U-1): stage3_head 现在**真的会被加载并使用**。
    FIX(S-2): stage2 配置从 checkpoint 恢复，不再用 Config() 默认值 + strict=False。
    FIX(U-7): 同时构建 calib_split，让阈值能在 val 上标定。
    """
    import torch

    from retfound_common.checkpoint import build_model_from_checkpoint
    from retfound_common.model import MultiTaskModel

    # ---- Stage 2 ----
    model_s2, s2cfg, ckpt = build_model_from_checkpoint(
        stage2_ckpt, MultiTaskModel, map_location=device_str,
        override_img_size=input_size, allow_partial=allow_partial_load)
    S = int(s2cfg.backbone.img_size[0])
    print(f"[Phase4] Stage2 已加载: {stage2_ckpt}")
    print(f"         epoch={ckpt.get('epoch')} seed={ckpt.get('seed')} "
          f"git={str(ckpt.get('git'))[:8]} img_size={S}")
    if "loco" in str(stage2_ckpt).replace("\\", "/").lower():
        # FIX(U-2): run_phase4 旧版硬编码的正是 runs/loco/seed42/best.pth
        print("[Phase4] ⚠️  这个 checkpoint 路径里含 'loco' —— LOCO 模型只用了 800 张"
              "训练 (Canon-A + Zeiss)，且在 Canon-B 上选优，"
              "与 phase2_v2 (全部 1200 张) 不是同一个模型。"
              "报告里必须写明 headline 数字来自哪个 checkpoint。")

    # ---- Stage 3 CDR 融合头 (FIX(U-1)) ----
    head = None
    if stage3_head:
        from fusion_head import CDRFusionClassifier
        hck = torch.load(stage3_head, map_location=device_str, weights_only=False)
        mode = hck.get("mode", "cdr_soft")
        head = CDRFusionClassifier(
            img_feat_dim=int(hck.get("img_feat_dim", s2cfg.backbone.embed_dim)),
            num_classes=s2cfg.cls.num_classes, mode=mode)
        missing, unexpected = head.load_state_dict(hck["head"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Stage3 head 权重未完整加载: "
                               f"missing={list(missing)} unexpected={list(unexpected)}")
        print(f"[Phase4] Stage3 CDR 融合头已加载: {stage3_head} (mode={mode})")
    else:
        print("[Phase4] ⚠️  未提供 --stage3-head。分类走 Stage-2 自带的 ClsHead，"
              "**没有任何 CDR 融合** —— 等价于 pure_e2e。"
              "报告里不得声称'临床特征融合提升分类性能' (FIX(U-1))。")

    name = "LoRA-RETFound" + (" + CDR fusion" if head is not None else " (no CDR)")
    models = {name: Phase2ModelAdapter(name, model_s2, input_size=S,
                                       device_str=device_str, stage3_head=head,
                                       seg_source=seg_source,
                                       cdr_seg_size=cdr_seg_size)}

    # ---- 数据：评测集 + 标定集 (FIX(U-7)) ----
    datasets = {"REFUGE2": RefugeCSVAdapter(csv_path, split=split)}
    if calib_split and calib_split != split:
        datasets["REFUGE2_calib"] = RefugeCSVAdapter(
            csv_path, split=calib_split, dataset_name="REFUGE2_calib")
        print(f"[Phase4] 标定集 split='{calib_split}' 已加入，"
              f"阈值将在其上拟合、在 '{split}' 上报告。")

    # ---- 训练元数据 (FIX(U-3)) ----
    train_meta = {}
    if train_meta_json and os.path.exists(train_meta_json):
        import json
        with open(train_meta_json, encoding="utf-8") as f:
            train_meta[name] = json.load(f)
    else:
        # 真实参数量现场统计，不再依赖任何手写占位符
        from retfound_common.config import param_breakdown
        bd = param_breakdown(model_s2)
        train_meta[name] = {
            "params_total_m": bd["total"] / 1e6,
            "params_trainable_m": bd["trainable"] / 1e6,
            "params_lora_m": bd["lora"] / 1e6,
            "params_heads_m": bd["heads"] / 1e6,
            "trainable_ratio": bd["trainable"] / bd["total"],
        }
        print(f"[Phase4] 参数量实测: total={bd['total']/1e6:.2f}M "
              f"trainable={bd['trainable']/1e6:.2f}M "
              f"({bd['trainable']/bd['total']:.2%})，其中 LoRA {bd['lora']/1e6:.2f}M。"
              f"训练时长/显存等需 --train-meta-json 提供，缺则留 NaN。")
    return models, datasets, train_meta
