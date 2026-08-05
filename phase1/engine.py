"""Training engine.

A single generic `Trainer` handles the boilerplate that is identical for every task and
every phase: epoch loop, mixed precision, gradient accumulation, best-checkpoint tracking,
logging. Task-specific behaviour is injected via `prepare_batch` and `evaluate_fn`, so the
exact same Trainer is reused for the classifier, the segmenter, and (later) the RETFound
multi-task model.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from metrics import aggregate_segmentation, classification_metrics
from utils import AverageMeter, git_commit, save_checkpoint


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        cfg,
        prepare_batch,
        evaluate_fn,
        monitor: str = "auc",
        mode: str = "max",
        scheduler=None,
        logger=None,
    ) -> None:
        self.model = model.to(device)
        self.opt = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.cfg = cfg
        self.prepare_batch = prepare_batch
        self.evaluate_fn = evaluate_fn
        self.monitor = monitor
        self.mode = mode
        self.scheduler = scheduler
        self.logger = logger
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=cfg.train.amp and device.type == "cuda"
        )
        self.best = float("-inf") if mode == "max" else float("inf")

    def _log(self, msg: str) -> None:
        (self.logger.info if self.logger else print)(msg)

    def _is_better(self, v: float) -> bool:
        return v > self.best if self.mode == "max" else v < self.best

    def train_one_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        meter = AverageMeter()
        accum = max(self.cfg.train.grad_accum_steps, 1)
        self.opt.zero_grad(set_to_none=True)
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"train e{epoch}", leave=False)
        for step, batch in pbar:
            inputs, targets = self.prepare_batch(batch, self.device)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets) / accum
            self.scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
            meter.update(loss.item() * accum, inputs.size(0))
            pbar.set_postfix(loss=f"{meter.avg:.4f}")
        return meter.avg

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        return self.evaluate_fn(self.model, loader, self.device)

    def fit(self, train_loader, val_loader, out_dir: str, run_name: str) -> float:
        for epoch in range(1, self.cfg.train.epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            metrics = self.evaluate(val_loader)
            score = metrics.get(self.monitor, float("nan"))
            self._log(
                f"epoch {epoch:03d} | train_loss {train_loss:.4f} | "
                + " | ".join(f"{k} {v:.4f}" for k, v in metrics.items())
            )
            if score == score and self._is_better(score):  # score == score -> not NaN
                self.best = score
                # FIX(C-11): checkpoint 之前只存 model/epoch/metrics/score，
                # 拿到 .pt 无法确定它是哪次实验、什么配置、什么种子的产物 ——
                # 这正是 0.812/0.754 这类数字串台的温床。
                save_checkpoint(
                    {
                        "model": self.model.state_dict(),
                        "epoch": epoch,
                        "metrics": metrics,
                        "score": score,
                        "cfg": self.cfg.to_dict(),
                        "seed": self.cfg.train.seed,
                        "monitor": self.monitor,
                        "git": git_commit(),
                    },
                    f"{out_dir}/{run_name}_best.pt",
                )
        return self.best


# --------------------------------------------------------------------------- evaluators
@torch.no_grad()
def evaluate_classifier(model, loader, device, threshold: float = 0.5,
                        return_raw: bool = False):
    """FIX(C-2): 增加 threshold 参数与 return_raw。

    旧版把逐样本概率直接丢掉，导致 bootstrap CI (P1-2) 和阈值标定 (P1-4)
    都无从下手 —— 连"ResNet34 是否真的优于 LoRA-RETFound"都判不了。
    """
    probs, labels, paths = [], [], []
    for batch in loader:
        logits = model(batch["image"].to(device)).squeeze(1)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend(batch["label"].numpy().tolist())
        paths.extend(list(batch.get("image_path", [""] * len(batch["label"]))))
    m = classification_metrics(probs, labels, threshold=threshold)
    if not return_raw:
        return m
    return m, np.asarray(probs, dtype=float), np.asarray(labels, dtype=float), paths


@torch.no_grad()
def evaluate_segmenter(model, loader, device, return_raw: bool = False):
    """FIX(C-2): return_raw 时回传逐样本 Dice/CDR 误差，供 bootstrap CI。"""
    preds, gts, paths = [], [], []
    for batch in loader:
        logits = model(batch["image"].to(device))
        pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
        gt = batch["mask"].numpy().astype(np.uint8)
        paths.extend(list(batch.get("image_path", [""] * pred.shape[0])))
        for i in range(pred.shape[0]):
            preds.append(pred[i]); gts.append(gt[i])
    if not return_raw:
        return aggregate_segmentation(preds, gts)
    agg, per = aggregate_segmentation(preds, gts, return_per_sample=True)
    return agg, per, paths
