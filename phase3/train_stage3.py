"""
train_stage3.py — Stage 3: clinical (CDR) fusion fine-tuning.

What this stage does (per the plan):
  - load the stage-2 model (RETFound+LoRA backbone, seg head, cls features)
  - compute CDR from the segmentation output
  - fuse CDR with image features and re-fine-tune the classification head
  - compare three configurations:
        pure_e2e       : image features only
        cdr_two_stage  : + hard CDR, segmentation FROZEN (no grad from cls loss)
        cdr_soft       : + soft CDR, segmentation UNFROZEN (cls loss flows in)

Freeze policy (the part that actually differentiates the modes):
  - pure_e2e      : backbone + seg frozen; train only the classifier head.
  - cdr_two_stage : backbone + seg frozen; train only the classifier head.
                    (CDR is a fixed, detached input feature.)
  - cdr_soft      : seg UNFROZEN (small LR) so the classification loss can
                    refine cup/disc through the soft CDR. Backbone stays frozen
                    to respect the 3060/12GB budget and the LoRA spirit; if you
                    want the trunk to move too, add it to the seg param group.

  In every mode we ALSO keep a segmentation loss term when seg ground-truth is
  available, so cdr_soft doesn't drift the segmenter purely to satisfy the
  classifier (which would wreck the stage-4 Dice numbers). The seg loss is the
  anchor; the soft-CDR coupling is a nudge on top.

Usage (real run):
    model = load_real_stage2(ckpt_path)          # you provide this
    trainer = Stage3Trainer(model, mode="cdr_soft", device="cuda")
    trainer.fit(train_loader, val_loader, epochs=15)

This file's __main__ runs a CPU smoke test on synthetic data for all 3 modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import _bootstrap  # noqa: F401

from cdr import compute_cdr_hard
from fusion_head import CDRFusionClassifier
from retfound_common.losses import FocalLoss, SegCEDiceLoss   # FIX(T-2/T-3)
from stage2_iface import Stage2Encoder


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def dice_from_logits(seg_logits: torch.Tensor, seg_target: torch.Tensor,
                     n_classes: int = 3, eps: float = 1e-6) -> dict[str, float]:
    """Per-class Dice for {rim(1), cup(2)} plus the full disc (1∪2).
    seg_target: [B,Hs,Ws] int64 in {0,1,2}. Returns disc/cup Dice."""
    pred = seg_logits.argmax(1)                              # [B,H,W]
    out = {}
    # full disc = classes {1,2}
    for name, cls_set in [("disc", {1, 2}), ("cup", {2})]:
        p = torch.zeros_like(pred, dtype=torch.float32)
        t = torch.zeros_like(pred, dtype=torch.float32)
        for c in cls_set:
            p += (pred == c).float()
            t += (seg_target == c).float()
        p = p.clamp(0, 1); t = t.clamp(0, 1)
        inter = (p * t).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        dice = ((2 * inter + eps) / (denom + eps)).mean().item()
        out[f"dice_{name}"] = dice
    return out


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class Stage3Config:
    """FIX(T-4): mode 新增 'pure_e2e_unfrozen' —— 消融的第四臂。

    旧的三个模式同时变了**两件事**：有无 CDR 特征、分割冻不冻。
        pure_e2e       head 265k, seg 冻结
        cdr_two_stage  head 266k, seg 冻结
        cdr_soft       head 266k, seg 解冻 9.9M   <- 可训练参数是另外两个的 38 倍
    所以 cdr_soft - cdr_two_stage 的差**不能归因于"可微"**。
    第四臂 pure_e2e_unfrozen（无 CDR、但同样解冻分割、同样有 seg anchor）
    把"解冻"这个因素单独隔离出来。

    FIX(T-3): label_smoothing 默认与 phase2 训练时一致 (0.05)。旧 phase3 用的是
    另一份没有 label smoothing 的 FocalLoss，于是"0.934 > phase2 的 0.931"
    根本不是同一条流水线上的比较。
    """
    mode: str = "cdr_two_stage"
    num_classes: int = 2
    img_feat_dim: int = 256
    lr_head: float = 1e-3
    lr_seg: float = 1e-5            # only used when seg is unfrozen (cdr_soft)
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    focal_alpha: tuple[float, ...] | None = (0.1, 0.9)   # 对齐 phase2
    label_smoothing: float = 0.05                        # FIX(T-3): 对齐 phase2
    seg_loss_weight: float = 1.0    # anchor weight on segmentation CE
    cls_loss_weight: float = 1.0
    seg_loss_type: str = "ce_dice"  # "ce_dice" (class-balanced, default) or "ce" (legacy)
    grad_accum: int = 1             # for 3060: accumulate to fake a bigger batch
    amp: bool = True                # mixed precision
    max_grad_norm: float = 5.0
    # --- cdr_soft only: controls how hard the classification loss is allowed to
    # push back through the soft-CDR into the segmentation branch. The soft-CDR
    # coupling is meant to be a NUDGE on top of the seg-loss anchor; left
    # unscaled it can out-muscle the anchor while segmentation is still diffuse
    # and collapse the disc (degenerate-rate climbs, Dice falls). We scale that
    # one gradient path and warm it up so the anchor establishes a sane segmenter
    # first. scale applies ONLY to the cls->seg path; the seg-loss anchor and the
    # forward CDR values are untouched. Set scale=1.0, warmup=0 to recover the
    # original (unthrottled) behavior.
    soft_cdr_seg_grad_scale: float = 0.1   # upper bound on the cls->seg nudge
    soft_cdr_warmup_steps: int = 100       # linear 0 -> scale over this many opt steps


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
class Stage3Trainer:
    def __init__(self, model: Stage2Encoder, cfg: Stage3Config | None = None,
                 device: str = "cpu"):
        if not isinstance(model, Stage2Encoder):
            raise TypeError("model must satisfy the Stage2Encoder protocol "
                            "(encode / segmentation_parameters / backbone_parameters)")
        self.cfg = cfg or Stage3Config()
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.head = CDRFusionClassifier(
            img_feat_dim=self.cfg.img_feat_dim,
            num_classes=self.cfg.num_classes,
            mode=self.cfg.mode,
        ).to(self.device)
        self._uses_cdr = self.cfg.mode in ("cdr_two_stage", "cdr_soft")
        self._seg_unfrozen = self.cfg.mode in ("cdr_soft", "pure_e2e_unfrozen")

        # ---- freeze policy ----
        self._apply_freeze_policy()

        # ---- losses ----
        alpha = (torch.tensor(self.cfg.focal_alpha, device=self.device)
                 if self.cfg.focal_alpha else None)
        self.cls_loss = FocalLoss(gamma=self.cfg.focal_gamma, alpha=alpha,
                                  label_smoothing=self.cfg.label_smoothing)
        if self.cfg.seg_loss_type == "ce_dice":
            # class-balanced anchor: robust to the bg-dominant class imbalance
            # that otherwise lets the segmenter collapse to all-background.
            self.seg_loss = SegCEDiceLoss()
        elif self.cfg.seg_loss_type == "ce":
            self.seg_loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"unknown seg_loss_type {self.cfg.seg_loss_type!r}")

        # ---- optimizer (param groups) ----
        param_groups = [{"params": self.head.parameters(), "lr": self.cfg.lr_head}]
        if self._seg_unfrozen:
            seg_params = [p for p in self.model.segmentation_parameters() if p.requires_grad]
            if seg_params:
                param_groups.append({"params": seg_params, "lr": self.cfg.lr_seg})
        self.opt = torch.optim.AdamW(param_groups, weight_decay=self.cfg.weight_decay)

        use_amp = self.cfg.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self._amp_enabled = use_amp
        self._opt_step = 0   # counts completed optimizer steps, for soft-CDR warmup

    def _apply_freeze_policy(self):
        # default: freeze everything in the stage-2 model
        for p in self.model.parameters():
            p.requires_grad_(False)
        # FIX(T-4): 'pure_e2e_unfrozen' 也解冻分割（只受 seg anchor 驱动，
        # 没有 CDR 通路），用来把"解冻"与"CDR"两个因素分开。
        if self._seg_unfrozen:
            for p in self.model.segmentation_parameters():
                p.requires_grad_(True)
        # backbone 在所有模式下都冻结 (LoRA 已在 stage 2 训好)

    # ----- one optimization step over a micro-batch loop (grad accumulation) -
    def _run_batch(self, images, y_cls, y_seg, train: bool):
        images = images.to(self.device)
        y_cls = y_cls.to(self.device)
        has_seg = y_seg is not None
        if has_seg:
            y_seg = y_seg.to(self.device)

        amp_ctx = torch.amp.autocast("cuda", enabled=self._amp_enabled)
        with amp_ctx:
            img_feat, seg_logits = self.model.encode(images)

            # Decide what the classification head sees as segmentation.
            #   pure_e2e / cdr_two_stage : detach -> no cls gradient to seg, and
            #       a cheaper graph (no activations retained through the decoder).
            #   cdr_soft : we WANT cls gradient to reach seg, but only as a
            #       controlled nudge. Straight-through scaling keeps the forward
            #       value EXACTLY equal to seg_logits (so the CDR numbers are
            #       unchanged) while multiplying the backward cls->seg gradient
            #       by `scale`. With warmup, `scale` ramps 0 -> cfg cap so the
            #       seg-loss anchor shapes a sane segmenter before the soft-CDR
            #       is allowed to pull on it.
            cls_seg_scale = 0.0
            if self.cfg.mode in ("pure_e2e", "pure_e2e_unfrozen", "cdr_two_stage"):
                seg_for_head = seg_logits.detach()
            elif self.cfg.mode == "cdr_soft":
                if self.cfg.soft_cdr_warmup_steps > 0:
                    warm = min(1.0, self._opt_step / float(self.cfg.soft_cdr_warmup_steps))
                else:
                    warm = 1.0
                cls_seg_scale = self.cfg.soft_cdr_seg_grad_scale * warm
                # straight-through: value == seg_logits, grad == scale * d/dseg
                seg_for_head = (seg_logits.detach()
                                + cls_seg_scale * (seg_logits - seg_logits.detach()))
            else:
                seg_for_head = seg_logits

            cls_logits = self.head(
                img_feat, seg_for_head if self._uses_cdr else None)

            loss = self.cfg.cls_loss_weight * self.cls_loss(cls_logits, y_cls)
            # segmentation anchor: keep the segmenter honest. In cdr_soft this
            # term shares the seg branch with the (now scaled) soft-CDR gradient,
            # and uses the UN-scaled seg_logits so the anchor is at full strength.
            # FIX(T-4b): 旧版在所有模式下都算 seg anchor 并加进 loss，
            # 但冻结模式下它没有梯度路径 —— 白算一次，还让三个模式打印出来的
            # train_loss 不可比。现在只在分割解冻时计入，并分项记录。
            seg_term = 0.0
            if has_seg and self._seg_unfrozen and train:
                sl = self.seg_loss(seg_logits, y_seg)
                loss = loss + self.cfg.seg_loss_weight * sl
                seg_term = float(sl.detach().cpu())

        metrics = {"loss": float(loss.detach().cpu()),
                   "cls_seg_scale": cls_seg_scale, "seg_anchor": seg_term}
        return loss, cls_logits.detach(), seg_logits.detach(), metrics

    def fit(self, train_loader, val_loader=None, epochs: int = 10, log_every: int = 50):
        for ep in range(1, epochs + 1):
            self.head.train(); self.model.train()
            self.opt.zero_grad(set_to_none=True)
            running = []
            for it, batch in enumerate(train_loader):
                images, y_cls, y_seg = batch
                loss, _, _, m = self._run_batch(images, y_cls, y_seg, train=True)
                loss = loss / self.cfg.grad_accum
                self.scaler.scale(loss).backward()

                if (it + 1) % self.cfg.grad_accum == 0:
                    # clip across all trainable params
                    self.scaler.unscale_(self.opt)
                    trainable = [p for g in self.opt.param_groups for p in g["params"]]
                    torch.nn.utils.clip_grad_norm_(trainable, self.cfg.max_grad_norm)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    self._opt_step += 1

                running.append(m["loss"])
                if log_every and (it + 1) % log_every == 0:
                    print(f"  ep{ep} it{it+1} loss={np.mean(running[-log_every:]):.4f}")

            msg = f"[{self.cfg.mode}] epoch {ep}: train_loss={np.mean(running):.4f}"
            if val_loader is not None:
                val = self.evaluate(val_loader)
                msg += "  " + "  ".join(f"{k}={v:.4f}" for k, v in val.items())
            print(msg)

    @torch.no_grad()
    def evaluate(self, loader, return_evidence: bool = False):
        """FIX(T-11): return_evidence=True 时回传逐样本概率，
        这是 bootstrap CI 与"cdr_soft 是否真的优于 cdr_two_stage"配对检验的前提。
        单看 0.934 vs 0.931 完全无法判断 —— n=400/约40阳性时 CI 宽约 ±0.08。"""
        return self._evaluate(loader, return_evidence)

    @torch.no_grad()
    def _evaluate(self, loader, return_evidence: bool = False):
        self.head.eval(); self.model.eval()
        all_prob, all_y = [], []
        dice_acc, n_seg = {"dice_disc": 0.0, "dice_cup": 0.0}, 0
        degen_count, n_total = 0, 0
        for batch in loader:
            images, y_cls, y_seg = batch
            images = images.to(self.device); y_cls = y_cls.to(self.device)
            img_feat, seg_logits = self.model.encode(images)
            cls_logits = self.head(img_feat, seg_logits if self._uses_cdr else None)
            prob = F.softmax(cls_logits, dim=1)[:, 1]   # P(glaucoma)
            all_prob.append(prob.cpu().numpy())
            all_y.append(y_cls.cpu().numpy())

            # track CDR degeneracy rate (only meaningful when CDR is used)
            if self._uses_cdr:
                hard = compute_cdr_hard(seg_logits)
                degen_count += int(hard.degenerate.sum().item())
                n_total += seg_logits.size(0)

            if y_seg is not None:
                d = dice_from_logits(seg_logits, y_seg.to(self.device))
                for k in dice_acc:
                    dice_acc[k] += d[k] * images.size(0)
                n_seg += images.size(0)

        y = np.concatenate(all_y); p = np.concatenate(all_prob)
        out = {}
        # AUC needs both classes present
        if len(np.unique(y)) == 2:
            out["auc"] = float(roc_auc_score(y, p))
        else:
            out["auc"] = float("nan")
        if n_seg > 0:
            out["dice_disc"] = dice_acc["dice_disc"] / n_seg
            out["dice_cup"] = dice_acc["dice_cup"] / n_seg
        if n_total > 0:
            out["cdr_degenerate_rate"] = degen_count / n_total
        if not return_evidence:
            return out
        import pandas as pd
        return out, pd.DataFrame({"cls_prob": p, "cls_label": y})


# --------------------------------------------------------------------------- #
# smoke test on synthetic data, all three modes
# --------------------------------------------------------------------------- #
def _make_synthetic_loader(n=64, batch=8, seg_size=48, seed=0):
    """Synthetic fundus-like batch: random images, a disc/cup phantom whose
    cup size correlates with the label (bigger cup -> glaucoma), so AUC should
    rise above 0.5 and CDR should carry real signal — letting us see the modes
    actually train rather than just run."""
    rng = np.random.default_rng(seed)
    Himg = Wimg = 96
    images = torch.randn(n, 3, Himg, Wimg)
    y_cls = torch.from_numpy(rng.integers(0, 2, size=n)).long()

    # build seg targets: disc radius ~ fixed, cup radius grows with label + noise
    yy, xx = torch.meshgrid(torch.arange(seg_size), torch.arange(seg_size), indexing="ij")
    cy = cx = seg_size / 2
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    segs = torch.zeros(n, seg_size, seg_size, dtype=torch.long)
    for i in range(n):
        disc_r = 18 + rng.normal(0, 1)
        cup_r = (7 + 6 * int(y_cls[i])) + rng.normal(0, 1)   # label-dependent
        cup_r = min(cup_r, disc_r - 1)
        disc = r < disc_r
        cup = r < cup_r
        seg = torch.zeros(seg_size, seg_size, dtype=torch.long)
        seg[disc] = 1            # rim+...
        seg[cup] = 2             # cup overrides
        segs[i] = seg
        # bake a faint cup-sized bright blob into the image so img_feat also
        # has *some* signal (otherwise pure_e2e can't learn and the comparison
        # is unfair to it)
        cup_np = cup.numpy()
        cup_up = torch.from_numpy(
            np.kron(cup_np, np.ones((Himg // seg_size, Wimg // seg_size)))
        ).float()
        images[i, 0] += 1.5 * cup_up

    ds = torch.utils.data.TensorDataset(images, y_cls, segs)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)


if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    # READ THIS before interpreting the numbers below.
    #
    # This smoke test uses MockStage2, whose segmentation head is RANDOMLY
    # initialized and never pre-trained. What you SHOULD see, and why:
    #
    #   * AUC = 1.0 everywhere — the synthetic data bakes the label into both
    #     the cup size and the image, so even the mock backbone separates the
    #     classes trivially. Real REFUGE2 data will not. The smoke test answers
    #     "does it train / evaluate / back-prop correctly", not "is it good".
    #
    #   * pure_e2e and cdr_two_stage FREEZE segmentation, so their disc/cup Dice
    #     sit at the random-init value and do not move across epochs. That is
    #     correct: in these modes the classification loss never touches seg.
    #
    #   * cdr_soft UNFREEZES seg. With the class-balanced CE+Dice anchor, disc
    #     Dice jumps within the first epoch (the anchor teaches the random head
    #     foreground) and cup Dice then climbs steadily. Once the optimizer
    #     passes soft_cdr_warmup_steps, the soft-CDR gradient engages and you
    #     should see cup-Dice improvement *accelerate* — that is the
    #     classification signal refining the cup through the differentiable CDR.
    #     The CDR-degenerate rate should stay ~0 throughout.
    #
    #   * Earlier drafts used a bare CrossEntropy anchor; on a from-scratch seg
    #     head that anchor MINIMIZED loss by predicting all-background, so disc
    #     Dice *fell* epoch over epoch and the degenerate rate climbed. That was
    #     a real class-imbalance failure, not a SoftCDR bug; SegCEDiceLoss fixes
    #     it. A real (already-trained) stage-2 head won't start from that mode,
    #     but the anchor is robust either way now.
    #
    # On your REAL stage-2 checkpoint: two-stage's Dice == stage-2 Dice (seg
    # frozen); cdr_soft's Dice may move as seg is fine-tuned. That delta is what
    # stage 4 measures, and the cls->seg coupling is throttled by
    # soft_cdr_seg_grad_scale / soft_cdr_warmup_steps so it stays a nudge.
    # ---------------------------------------------------------------------- #
    from stage2_iface import MockStage2

    torch.manual_seed(0); np.random.seed(0)
    train_loader = _make_synthetic_loader(n=96, batch=8, seed=1)
    val_loader = _make_synthetic_loader(n=48, batch=8, seed=2)

    for mode in ["pure_e2e", "cdr_two_stage", "cdr_soft"]:
        print(f"\n===== mode = {mode} =====")
        # reset RNG per mode so all three start from the SAME random MockStage2
        # weights — otherwise the frozen-seg modes show different Dice baselines
        # purely from different inits, which is confusing in a smoke test.
        torch.manual_seed(0); np.random.seed(0)
        model = MockStage2(feat_dim=256, seg_size=48)
        cfg = Stage3Config(mode=mode, img_feat_dim=256, amp=False,
                           focal_alpha=(0.5, 0.5), seg_loss_weight=1.0,
                           soft_cdr_warmup_steps=48)  # ~4 epochs at 12 steps/epoch
        tr = Stage3Trainer(model, cfg, device="cpu")
        # count trainable params by group for the honest-comparison table
        n_head = sum(p.numel() for p in tr.head.parameters() if p.requires_grad)
        n_seg = sum(p.numel() for p in tr.model.segmentation_parameters() if p.requires_grad)
        print(f"trainable: head={n_head:,}  seg(unfrozen)={n_seg:,}")
        tr.fit(train_loader, val_loader, epochs=8, log_every=0)
