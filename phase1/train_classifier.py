"""Train the ResNet34 glaucoma-classification baseline on REFUGE2.

    python train_classifier.py --config baseline.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)          # FIX(C-1): 四个脚本统一同一套 sys.path 约定

from config import Config  # noqa: E402
from datasets import REFUGE2Dataset  # noqa: E402
from engine import Trainer, evaluate_classifier  # noqa: E402
from models import BCEClsLoss, ResNet34Classifier  # noqa: E402
from transforms import build_cls_transforms  # noqa: E402
from utils import count_parameters, get_device, set_seed, setup_logger  # noqa: E402


def prepare_batch(batch, device):
    return batch["image"].to(device), batch["label"].to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline.yaml")
    ap.add_argument("--index", default=None, help="覆盖 config 里的索引 CSV")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.index:
        cfg.data.refuge_index = args.index
    if args.epochs:
        cfg.train.epochs = args.epochs
    set_seed(cfg.train.seed)
    device = get_device()
    logger = setup_logger("cls", f"{cfg.output_dir}/cls.log")
    logger.info(f"device={device} | epochs={cfg.train.epochs}")

    df = pd.read_csv(cfg.data.refuge_index)
    if "split" in df.columns:
        tr_df = df[df.split == "train"]
        va_df = df[df.split == "val"]
    else:
        tr_df, va_df = train_test_split(
            df, test_size=cfg.data.val_split, stratify=df["label"],
            random_state=cfg.train.seed,
        )
    logger.info(f"train={len(tr_df)} val={len(va_df)}")

    train_loader = DataLoader(
        REFUGE2Dataset(tr_df, build_cls_transforms(cfg.data.cls_image_size, True), "cls"),
        batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
    )
    val_loader = DataLoader(
        REFUGE2Dataset(va_df, build_cls_transforms(cfg.data.cls_image_size, False), "cls"),
        batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
    )

    model = ResNet34Classifier(num_classes=1, pretrained=cfg.model.pretrained)
    logger.info(f"ResNet34 trainable params: {count_parameters(model):,}")
    loss_fn = BCEClsLoss(cfg.train.pos_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)

    trainer = Trainer(
        model, opt, loss_fn, device, cfg,
        prepare_batch=prepare_batch, evaluate_fn=evaluate_classifier,
        monitor="auc", mode="max", scheduler=sched, logger=logger,
    )
    best = trainer.fit(train_loader, val_loader, cfg.output_dir, f"{cfg.experiment}_resnet34")
    logger.info(f"best val AUC: {best:.4f}")


if __name__ == "__main__":
    main()
