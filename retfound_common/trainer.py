# -*- coding: utf-8 -*-
"""
训练器：指标、loss 汇总 (任务屏蔽 + 分割目标路由)、两条训练路径、评测。

本次修复：
  FIX(S-3): evaluate() 之前只返回聚合后的 AUC，逐样本概率被丢弃 —— 直接卡死
            bootstrap CI (P1-2) 与阈值标定 (P1-4)。现在可返回逐样本证据表。
  FIX(S-4): 训练 ROI 全用 GT、评测用预测框的 exposure bias，改为 scheduled sampling。
  FIX(S-5): label_smoothing 从 cfg.cls 显式传入 FocalLoss，不再吃默认值。
  FIX(S-6): Dice 之前是"batch 均值的均值"(dice_n 计的是 batch 数)，标签不齐时有偏。
            改为累加逐样本 Dice。
  FIX(S-7): min_lr 之前只按 backbone 的 base_lr 折算，head 组实际地板是 5 倍。
            改为每个 param group 独立的 lr_lambda。
  FIX(P0-6): 黄斑两个口径 (图像边长归一化 / 视盘直径归一化) 显式区分命名，
            杜绝 0.013 与 0.341 被当成同一个指标比较。
  FIX(U-9): evaluate 增加 seg_source 参数，强制显式选择 fine / coarse 口径。
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import FocalLoss, SegLoss, HeatmapLoss
from .model import roi_align_norm, paste_roi_to_full


# --------------------------------------------------------------------------- #
# 学习率调度
# --------------------------------------------------------------------------- #
def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    """线性 warmup + 余弦退火，按**优化器 step** 粒度。

    FIX(S-7): LambdaLR 把同一个倍率乘到所有 param group 的初始 lr 上。旧实现只
    算了一个 min_ratio = min_lr / cfg.train.lr（backbone 的），于是 head 组
    (初始 1e-3) 的实际地板变成 1e-3*(1e-5/2e-4)=5e-5，而不是文档写的 1e-5。
    现在为每个 group 单独构造 lambda，group 可自带 'min_lr'。
    """
    if getattr(cfg.train, "scheduler", "none") != "warmup_cosine":
        return None
    steps_per_epoch = max(1, steps_per_epoch)
    warmup_steps = int(cfg.train.warmup_epochs * steps_per_epoch)
    total_steps = max(1, cfg.train.epochs * steps_per_epoch)

    ratios = []
    for g in optimizer.param_groups:
        base = float(g["lr"])
        floor = float(g.get("min_lr", cfg.train.min_lr))
        ratios.append((floor / base) if base > 0 else 0.0)

    def make(min_ratio: float):
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            prog = min(1.0, max(0.0, prog))
            cos = 0.5 * (1.0 + math.cos(math.pi * prog))
            return min_ratio + (1.0 - min_ratio) * cos
        return lr_lambda

    return torch.optim.lr_scheduler.LambdaLR(optimizer, [make(r) for r in ratios])


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """二分类 AUC (Mann-Whitney U，带并列校正)。

    已验证与 sklearn.roc_auc_score 数值一致 (200 次含大量并列的随机试验，最大
    差异 1.1e-16)，所以 phase1 (sklearn) 与本模块的 AUC 可直接对比 —— 这是
    P1-1 跨阶段基线对比成立的前提。
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((labels == 1).sum()); n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def dice_per_sample(prob: torch.Tensor, target: torch.Tensor,
                    thresh: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """硬 Dice，返回 (B, C) 的逐样本逐通道值（FIX(S-6): 不再提前求均值）。"""
    pred = (prob > thresh).float()
    dims = (2, 3)
    inter = (pred * target).sum(dims)
    denom = pred.sum(dims) + target.sum(dims)
    return (2 * inter + eps) / (denom + eps)


def macula_error_img(pred_xy: torch.Tensor, gt_xy: torch.Tensor,
                     eps: float = 1e-12) -> torch.Tensor:
    """口径 A：归一化坐标欧氏距离，单位 = **图像边长**。

    FIX(P0-6): LOCO 报的 0.009~0.013 是这个口径，phase4 报的 0.341 是下面的口径 B
    (单位 = 视盘直径)。两者相差约 1/disc_frac ≈ 8 倍。把它们并排比较会得出
    "黄斑定位退化 26 倍"的假结论。函数名强制区分。
    """
    return torch.sqrt(((pred_xy - gt_xy) ** 2).sum(dim=1) + eps)


def macula_error_disc(pred_xy: torch.Tensor, gt_xy: torch.Tensor,
                      disc_mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """口径 B：再除以视盘等效直径，单位 = **视盘直径** (REFUGE / phase4 口径)。"""
    d = macula_error_img(pred_xy, gt_xy, eps)
    m = disc_mask
    if m.dim() == 4:
        m = m[:, 0]
    m = (m > 0.5).float()
    H, W = m.shape[-2], m.shape[-1]
    side = float(H + W) / 2.0
    area = m.flatten(1).sum(dim=1)
    diam = (2.0 * torch.sqrt(area / math.pi + eps)) / side
    diam = torch.where(area > 0, diam.clamp(min=eps), torch.ones_like(diam))
    return d / diam


# --------------------------------------------------------------------------- #
# 训练器
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, model, optimizer, balancer, scaler, cfg, device,
                 scheduler=None):
        self.model = model
        self.opt = optimizer
        self.balancer = balancer
        self.scaler = scaler
        self.cfg = cfg
        self.device = device
        self.scheduler = scheduler
        self.epoch = 0                      # FIX(S-4): scheduled sampling 需要

        alpha = None
        if cfg.cls.focal_alpha is not None:
            alpha = torch.tensor(cfg.cls.focal_alpha, dtype=torch.float32)
        # FIX(S-5): label_smoothing 显式传入，可配置、可记录、可复现
        self.focal = FocalLoss(
            gamma=cfg.cls.focal_gamma, alpha=alpha,
            label_smoothing=getattr(cfg.cls, "label_smoothing", 0.0))
        self.seg_loss = SegLoss(dice_w=1.0, bce_w=1.0)
        self.heat_loss = HeatmapLoss(sigma=cfg.macula.sigma,
                                     coord_w=cfg.macula.coord_weight,
                                     heat_w=cfg.macula.heatmap_weight)
        self.is_gradnorm = (cfg.balance.method == "gradnorm")

    # ----- FIX(S-4): ROI scheduled sampling ----- #
    def _use_gt_roi(self) -> bool:
        """训练时按概率决定用 GT ROI 还是模型自己的粗预测框。

        p_pred 在 roi_sampling_warmup_epochs 内从 0 线性升到 roi_pred_prob_max。
        roi_pred_prob_max=0 即退回旧的全程 teacher forcing 行为。
        """
        pmax = float(getattr(self.cfg.seg, "roi_pred_prob_max", 0.0))
        if pmax <= 0:
            return True
        warm = max(1, int(getattr(self.cfg.seg, "roi_sampling_warmup_epochs", 1)))
        p_pred = pmax * min(1.0, self.epoch / warm)
        return bool(torch.rand(()).item() >= p_pred)

    # ----- loss 汇总 ----- #
    def compute_losses(self, out: Dict[str, object],
                       batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        dev = self.device
        losses: Dict[str, torch.Tensor] = {}

        has_cls = batch["has_cls"].to(dev)
        if has_cls.any():
            losses["cls"] = self.focal(out["cls_logits"][has_cls],
                                       batch["cls_label"].to(dev)[has_cls])

        has_seg = batch["has_seg"].to(dev)
        if has_seg.any():
            seg_target = batch["seg_target"].to(dev)
            if self.model.seg_mode == "coarse2fine":
                coarse_logits = out["coarse_logits"]
                ch = coarse_logits.shape[-1]
                coarse_tgt = F.interpolate(seg_target, size=(ch, ch),
                                           mode="bilinear", align_corners=False)
                coarse_tgt = (coarse_tgt > 0.5).float()
                l_coarse = self.seg_loss(coarse_logits[has_seg], coarse_tgt[has_seg])

                fine_logits = out["fine_logits"]
                fh = fine_logits.shape[-1]
                fine_tgt = (roi_align_norm(seg_target, out["roi_boxes"], fh) > 0.5).float()
                l_fine = self.seg_loss(fine_logits[has_seg], fine_tgt[has_seg])
                losses["seg"] = self.cfg.seg.coarse_loss_w * l_coarse + l_fine
            else:
                seg_logits = out["seg_logits"]
                sh, sw = seg_logits.shape[-2:]
                tgt = (F.interpolate(seg_target, size=(sh, sw), mode="bilinear",
                                     align_corners=False) > 0.5).float()
                losses["seg"] = self.seg_loss(seg_logits[has_seg], tgt[has_seg])

        has_macula = batch["has_macula"].to(dev)
        if has_macula.any():
            losses["macula"] = self.heat_loss(
                out["macula_heatmap"][has_macula],
                out["macula_coords"][has_macula],
                batch["macula"].to(dev)[has_macula])
        return losses

    # ----- 一个 epoch ----- #
    def train_one_epoch(self, loader, epoch: int = 0) -> Dict[str, float]:
        self.epoch = int(epoch)
        self.model.train()
        if self.is_gradnorm:
            return self._train_gradnorm(loader, epoch)
        return self._train_standard(loader, epoch)

    def _train_standard(self, loader, epoch: int) -> Dict[str, float]:
        cfg = self.cfg
        accum = max(1, cfg.train.accum_steps)
        use_amp = cfg.train.amp and self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if cfg.train.amp_dtype == "bf16" else torch.float16

        running: Dict[str, float] = {}
        n_pred_roi = 0
        self.opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            images = batch["image"].to(self.device)
            gt_disc = batch["disc_mask"].to(self.device)
            use_gt = self._use_gt_roi()
            n_pred_roi += int(not use_gt)

            with torch.autocast(device_type=self.device.type, dtype=amp_dtype,
                                enabled=use_amp):
                out = self.model(images, gt_disc=gt_disc, use_gt_roi=use_gt)
                losses = self.compute_losses(out, batch)
                if len(losses) == 0:
                    continue
                total, logs = self.balancer(losses)
                total = total / accum

            self.scaler.scale(total).backward()

            if (step + 1) % accum == 0:
                if cfg.train.grad_clip > 0:
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.model.trainable_parameters(),
                                             cfg.train.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            running["total"] = running.get("total", 0.0) + float(total.detach()) * accum

        n = max(1, len(loader))
        stats = {k: v / n for k, v in running.items()}
        stats["roi_pred_frac"] = n_pred_roi / n
        return stats

    def _train_gradnorm(self, loader, epoch: int) -> Dict[str, float]:
        cfg = self.cfg
        running: Dict[str, float] = {}
        shared = self.model.gradnorm_shared_params()

        for step, batch in enumerate(loader):
            images = batch["image"].to(self.device)
            gt_disc = batch["disc_mask"].to(self.device)
            self.opt.zero_grad(set_to_none=True)

            out = self.model(images, gt_disc=gt_disc, use_gt_roi=self._use_gt_roi())
            losses = self.compute_losses(out, batch)
            if len(losses) == 0:
                continue

            self.balancer.set_initial(losses)
            total = self.balancer.weighted_sum(losses)
            total.backward(retain_graph=True)

            gn = self.balancer.grad_norm_loss(losses, shared)
            w = self.balancer.w
            gw = torch.autograd.grad(gn, w, retain_graph=False)[0]
            if w.grad is None:
                w.grad = gw.detach().clone()
            else:
                w.grad.copy_(gw.detach())

            if cfg.train.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.trainable_parameters(),
                                         cfg.train.grad_clip)
            self.opt.step()
            self.balancer.renormalize()
            if self.scheduler is not None:
                self.scheduler.step()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            running["total"] = running.get("total", 0.0) + float(total.detach())

        n = max(1, len(loader))
        return {k: v / n for k, v in running.items()}

    # ----- 评测 ----- #
    @torch.no_grad()
    def evaluate(self, loader, *, return_evidence: bool = False,
                 threshold: float = 0.5, seg_source: str = "fine"):
        """评测。

        FIX(S-3): return_evidence=True 时额外返回逐样本 DataFrame ——
                  bootstrap CI / 阈值标定 / 失败分析的前提。
        FIX(U-9): seg_source 必须显式选择：
                  'fine'   = coarse-to-fine 精修 paste 回全图（与训练目标一致）
                  'coarse' = 只用粗分割（phase3/phase4 旧路径的口径）
                  两者不可比，报告里必须写明用的哪个。
        """
        assert seg_source in {"fine", "coarse"}, seg_source
        self.model.eval()
        rows: List[dict] = []
        dice_sum = torch.zeros(self.cfg.seg.num_classes)
        dice_n = 0                                    # FIX(S-6): 计样本数
        cls_scores, cls_labels = [], []
        mac_img, mac_disc = [], []

        for batch in loader:
            images = batch["image"].to(self.device)
            out = self.model(images, gt_disc=None, use_gt_roi=False)
            B = images.shape[0]
            paths = batch.get("image_path", [""] * B)
            devs = batch.get("device", [""] * B)
            rec = [{"image_path": paths[i] if i < len(paths) else "",
                    "device": devs[i] if i < len(devs) else ""} for i in range(B)]

            has_cls = batch["has_cls"]
            if has_cls.any():
                prob = torch.softmax(out["cls_logits"], dim=1)[:, 1].cpu()
                cls_scores.append(prob[has_cls].numpy())
                cls_labels.append(batch["cls_label"][has_cls].cpu().numpy())
                for i in range(B):
                    if bool(has_cls[i]):
                        rec[i]["cls_prob"] = float(prob[i])
                        rec[i]["cls_label"] = int(batch["cls_label"][i])

            has_seg = batch["has_seg"]
            if has_seg.any():
                seg_target = batch["seg_target"]
                H, W = seg_target.shape[-2:]
                if self.model.seg_mode == "coarse2fine":
                    if seg_source == "fine":
                        full_logits = paste_roi_to_full(
                            out["fine_logits"], out["roi_boxes"], (H, W))
                    else:
                        full_logits = F.interpolate(
                            out["coarse_logits"], size=(H, W),
                            mode="bilinear", align_corners=False)
                else:
                    full_logits = F.interpolate(out["seg_logits"], size=(H, W),
                                                mode="bilinear", align_corners=False)
                prob = torch.sigmoid(full_logits).cpu()
                d = dice_per_sample(prob[has_seg], seg_target[has_seg])
                dice_sum += d.sum(dim=0)
                dice_n += d.shape[0]
                j = 0
                for i in range(B):
                    if bool(has_seg[i]):
                        rec[i]["dice_disc"] = float(d[j, 0])
                        rec[i]["dice_cup"] = (float(d[j, 1]) if d.shape[1] > 1
                                              else float("nan"))
                        j += 1

            has_macula = batch["has_macula"]
            if has_macula.any():
                coords = out["macula_coords"].cpu()
                gt = batch["macula"]
                e_img = macula_error_img(coords[has_macula], gt[has_macula]).numpy()
                mac_img.append(e_img)
                both = has_macula & batch["has_seg"]
                e_disc = None
                if both.any():
                    e_disc = macula_error_disc(coords[both], gt[both],
                                               batch["disc_mask"][both]).numpy()
                    mac_disc.append(e_disc)
                j = k = 0
                for i in range(B):
                    if bool(has_macula[i]):
                        rec[i]["macula_err_img"] = float(e_img[j]); j += 1
                    if e_disc is not None and bool(both[i]):
                        rec[i]["macula_err_disc"] = float(e_disc[k]); k += 1

            rows.extend(rec)

        metrics: Dict[str, object] = {"seg_source": seg_source}
        if cls_scores:
            s = np.concatenate(cls_scores); l = np.concatenate(cls_labels)
            metrics["auc"] = auc_score(s, l)
            metrics["n_pos"] = int((l == 1).sum())
            metrics["n_neg"] = int((l == 0).sum())
            from .stats import sens_spec_at
            ss = sens_spec_at(l, s, threshold)
            metrics.update(sens=ss["sens"], spec=ss["spec"], ppv=ss["ppv"],
                           npv=ss["npv"], threshold=float(threshold))
        if dice_n > 0:
            dm = dice_sum / dice_n                    # FIX(S-6): 真正的逐样本均值
            metrics["dice_disc"] = float(dm[0])
            metrics["dice_cup"] = float(dm[1]) if dm.numel() > 1 else float("nan")
            metrics["dice_mean"] = float(dm.mean())
            metrics["n_seg"] = int(dice_n)
        if mac_img:
            metrics["macula_err_img"] = float(np.concatenate(mac_img).mean())
        if mac_disc:
            arr = np.concatenate(mac_disc)
            metrics["macula_err_disc"] = float(arr.mean())
            # 与 phase4 的 loc_success_rate 同口径：落在 1 个视盘直径内
            metrics["macula_success_rate"] = float((arr <= 1.0).mean())

        if not return_evidence:
            return metrics
        import pandas as pd
        return metrics, pd.DataFrame(rows)
