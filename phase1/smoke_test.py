# -*- coding: utf-8 -*-
"""
Phase 1 端到端冒烟测试（CPU，合成数据）。

FIX(C-1): 旧版**必然 ImportError**，从来跑不通：
    ROOT = dirname(dirname(__file__))          # -> RETFound/ 而不是 phase1/
    sys.path.insert(0, ROOT)
    sys.path.insert(0, ROOT + "/scripts")      # 这个目录不存在
    from make_synthetic_data import ...        # 不在这两个路径下 -> ImportError
    from src.config import Config              # 没有 src/ 包        -> ImportError
  根因是 README_phase1.md 描述的是 src/ + scripts/ + configs/ 三层布局，
  实际文件全部平铺在 phase1/。而 1_GitHub_README 让读者第一步就跑这个脚本。
  现在按平铺布局修好，并且真的能输出 ALL SMOKE TESTS PASSED。
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch
from torch.utils.data import DataLoader

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from config import Config                                          # noqa: E402
from datasets import REFUGE2Dataset                                # noqa: E402
from engine import Trainer, evaluate_classifier, evaluate_segmenter  # noqa: E402
from make_synthetic_data import make_synthetic                     # noqa: E402
from metrics import (bootstrap_auc, classification_metrics,        # noqa: E402
                     compute_cdr, youden_threshold)
from models import BCEClsLoss, DiceCELoss, ResNet34Classifier, UNet  # noqa: E402
from transforms import build_cls_transforms, build_seg_transforms  # noqa: E402
from utils import count_parameters, param_report, set_seed         # noqa: E402


def main() -> None:
    set_seed(0)
    device = torch.device("cpu")
    tmp = tempfile.mkdtemp()
    df = make_synthetic(tmp, n_train=16, n_val=8, size=192)
    print(f"[ok] synthetic data: {len(df)} samples")

    cfg = Config()
    cfg.train.epochs = 1
    cfg.train.batch_size = 4
    cfg.train.amp = False
    cfg.output_dir = os.path.join(tmp, "out")

    tr_df = df[df.split == "train"]
    va_df = df[df.split == "val"]
    sz = 128

    # ----- classification -----
    tl = DataLoader(REFUGE2Dataset(tr_df, build_cls_transforms(sz, True), "cls"),
                    batch_size=4, shuffle=True, num_workers=0)
    vl = DataLoader(REFUGE2Dataset(va_df, build_cls_transforms(sz, False), "cls"),
                    batch_size=4, num_workers=0)
    b = next(iter(tl))
    assert tuple(b["image"].shape[1:]) == (3, sz, sz), b["image"].shape
    assert "image_path" in b, "FIX(C-2): 证据表需要 image_path 透传"
    clf = ResNet34Classifier(num_classes=1, pretrained=False)
    print(f"[ok] ResNet34 trainable params: {count_parameters(clf):,} "
          f"| {param_report(clf)}")
    Trainer(clf, torch.optim.AdamW(clf.parameters(), lr=1e-3),
            BCEClsLoss().to(device), device, cfg,
            lambda bt, d: (bt["image"].to(d), bt["label"].to(d)),
            evaluate_classifier, monitor="auc", mode="max"
            ).fit(tl, vl, cfg.output_dir, "smoke_cls")
    print("[ok] classifier: 1-epoch fit + eval + checkpoint")

    # FIX(C-11): checkpoint 必须自带配置/种子/commit
    ck = torch.load(os.path.join(cfg.output_dir, "smoke_cls_best.pt"),
                    map_location="cpu", weights_only=False)
    for k in ("cfg", "seed", "git", "metrics"):
        assert k in ck, f"checkpoint 缺少 {k}（FIX(C-11) 未生效）"
    print("[ok] checkpoint 含 cfg/seed/git/metrics，可回溯")

    # FIX(C-2): 阈值可传 + 逐样本概率可回 + bootstrap CI
    m, probs, labels, paths = evaluate_classifier(clf, vl, device, return_raw=True)
    assert probs.shape == labels.shape and len(paths) == len(probs)
    thr = youden_threshold(probs, labels)
    m_thr = classification_metrics(probs, labels, threshold=thr)
    ci = bootstrap_auc(probs, labels, n_boot=200, seed=0)
    assert m_thr["threshold"] == thr, "阈值未生效"
    print(f"[ok] threshold 可传 (youden={thr:.3f}), 逐样本概率可回, "
          f"AUC CI=[{ci['lo']:.3f}, {ci['hi']:.3f}] n_pos={ci['n_pos']}")

    # ----- segmentation -----
    tl2 = DataLoader(REFUGE2Dataset(tr_df, build_seg_transforms(sz, True), "seg"),
                     batch_size=4, shuffle=True, num_workers=0)
    vl2 = DataLoader(REFUGE2Dataset(va_df, build_seg_transforms(sz, False), "seg"),
                     batch_size=4, num_workers=0)
    b2 = next(iter(tl2))
    assert tuple(b2["mask"].shape[1:]) == (sz, sz) and b2["mask"].dtype == torch.long
    seg = UNet(in_channels=3, num_classes=3, base=16)
    print(f"[ok] UNet trainable params: {count_parameters(seg):,}")
    out = seg(b2["image"])
    assert tuple(out.shape) == (4, 3, sz, sz), out.shape
    sl = DiceCELoss(num_classes=3).to(device)
    assert torch.isfinite(sl(out, b2["mask"])), "seg loss not finite"
    Trainer(seg, torch.optim.AdamW(seg.parameters(), lr=1e-3), sl, device, cfg,
            lambda bt, d: (bt["image"].to(d), bt["mask"].to(d)),
            evaluate_segmenter, monitor="mean_dice", mode="max"
            ).fit(tl2, vl2, cfg.output_dir, "smoke_seg")
    agg, per, spaths = evaluate_segmenter(seg, vl2, device, return_raw=True)
    assert len(per["dice_disc"]) == len(spaths)
    print(f"[ok] segmenter: 逐样本 Dice 可回 (n={len(spaths)})")

    cdr = compute_cdr(b2["mask"][0].numpy())
    print("[ok] CDR on a ground-truth mask:", {k: round(v, 3) for k, v in cdr.items()})

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
