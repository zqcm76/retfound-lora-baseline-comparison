"""Datasets.

`REFUGE2Dataset` 是本项目**唯一**使用的数据集（训练 / 验证 / 测试）。

FIX(C-6): 旧 docstring 宣称 FundusSegDataset 用于 "Drishti-GS / RIM-ONE / RIGA /
PALM 的零样本跨设备评测，这就是 headline 结果之所以是 zero-shot 的原因"。
**这些数据集从未被使用过**，FundusSegDataset 是死代码。对外宣传一个不存在的
headline result 是开源诚信问题。现在它被明确标记为未使用的预留接口。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from masks import parse_refuge_mask


def _imread_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return m


class REFUGE2Dataset(Dataset):
    """Multi-task REFUGE2 dataset.

    Required dataframe columns by task:
        task='cls' : image_path, label
        task='seg' : image_path, mask_path
    'fovea_x' / 'fovea_y' are read when present (plumbed through for Phase 2; the Phase 1
    baseline scripts do not use them).
    """

    def __init__(self, df: pd.DataFrame, transforms, task: str = "seg") -> None:
        assert task in {"cls", "seg"}, task
        self.df = df.reset_index(drop=True)
        self.t = transforms
        self.task = task

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        img = _imread_rgb(row["image_path"])
        if self.task == "cls":
            out = self.t(image=img)
            return {"image": out["image"],
                    "label": torch.tensor(float(row["label"])),
                    "image_path": str(row["image_path"])}
        # segmentation
        label = parse_refuge_mask(_read_mask(row["mask_path"]))
        out = self.t(image=img, mask=label)
        return {"image": out["image"], "mask": out["mask"].long(),
                "image_path": str(row["image_path"])}


class FundusSegDataset(Dataset):
    """⚠️ NOT USED —— 为将来接入外部数据集预留的接口，本项目未使用。

    FIX(C-6): 本项目只用 REFUGE2。若将来要接 RIGA / PALM / Drishti-GS，
    传入数据集特定的 `mask_parser(np.ndarray) -> label map`（各公开集编码都不同）。
    在此之前，任何文档都不应把它描述成已完成的零样本评测。
    """

    def __init__(self, pairs, transforms, mask_parser=parse_refuge_mask) -> None:
        self.pairs = list(pairs)  # list of (image_path, mask_path)
        self.t = transforms
        self.parse = mask_parser

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        image_path, mask_path = self.pairs[i]
        img = _imread_rgb(image_path)
        label = self.parse(_read_mask(mask_path))
        out = self.t(image=img, mask=label)
        return {"image": out["image"], "mask": out["mask"].long()}


def build_refuge_index(root, csv_out: str | None = None) -> pd.DataFrame:
    """BEST-EFFORT indexer — REFUGE2 ships in several folder layouts, so treat this as a
    starting point and adapt the path logic to YOUR download.

    Assumed layout (edit me):
        <root>/images/<name>.jpg
        <root>/masks/<name>.(bmp|png)
        <root>/glaucoma_labels.csv   columns: [image, label]
        <root>/fovea.csv             columns: [image, Fovea_X, Fovea_Y]

    Returns a dataframe with columns image_path, mask_path, label, fovea_x, fovea_y and
    optionally writes it to `csv_out`.
    """
    root = Path(root)
    img_dir, msk_dir = root / "images", root / "masks"
    labels, fovea = {}, {}

    lab_csv, fov_csv = root / "glaucoma_labels.csv", root / "fovea.csv"
    if lab_csv.exists():
        ldf = pd.read_csv(lab_csv)
        labels = dict(zip(ldf.iloc[:, 0], ldf.iloc[:, 1]))
    if fov_csv.exists():
        fdf = pd.read_csv(fov_csv)
        fovea = {r.iloc[0]: (r.iloc[1], r.iloc[2]) for _, r in fdf.iterrows()}

    rows = []
    n_missing_label = 0
    for img_path in sorted(img_dir.glob("*")):
        if img_path.is_dir():
            continue
        mask_path = next(iter(msk_dir.glob(img_path.stem + ".*")), None)
        fx, fy = fovea.get(img_path.name, (np.nan, np.nan))
        # FIX(C-5): 旧代码是 labels.get(name, 0) —— 文件名对不上时（.jpg/.png、
        # 大小写、路径前缀差异）**所有图像静默变成 label=0**。全阴性时 AUC 会是
        # NaN 还算兜住了，但部分不匹配会静默污染标签且毫无迹象。
        if img_path.name in labels:
            lab = int(labels[img_path.name])
        else:
            lab = -1
            n_missing_label += 1
        rows.append(
            {
                "image_path": str(img_path),
                "mask_path": str(mask_path) if mask_path else "",
                "label": lab,
                "fovea_x": fx,
                "fovea_y": fy,
            }
        )
    df = pd.DataFrame(rows)

    if n_missing_label:
        raise ValueError(
            f"{n_missing_label}/{len(df)} 张图像在 {lab_csv} 里找不到标签。"
            f"通常是文件名后缀/大小写不一致。请修正后重试 —— "
            f"静默填 0 会造成无法察觉的标签污染。")
    present = sorted(set(df["label"]) - {-1})
    if len(present) < 2:
        raise ValueError(
            f"索引里只有一个类别 {present}，AUC 无法计算。请核对 {lab_csv}。")
    print(f"[build_refuge_index] {len(df)} 张，类别分布: "
          f"{df['label'].value_counts().to_dict()}")
    if csv_out:
        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_out, index=False)
    return df
