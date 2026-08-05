# -*- coding: utf-8 -*-
"""
数据集与 collate。

样本 schema (dict):
  image:      (3,H,W) float
  cls_label:  long 标量;          has_cls:    bool
  disc_mask:  (1,H,W) {0,1};       cup_mask: (1,H,W) {0,1}; has_seg: bool
  macula:     (2,) 归一化 (x,y);    has_macula: bool

每个样本带 has_* 标志，因为不同数据集标签不全 (REFUGE2 三任务齐全，
分割集只有 disc/cup，PALM 侧重分类等)。loss 按标志屏蔽。

DummyMultiTaskDataset 仅供 CPU 冒烟测试; 真实数据集骨架在文件末尾，
TODO 指向阶段一的预处理/增广 pipeline。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
# 注: 真实数据集才需要 transforms(albumentations)；放到类里惰性导入，
# 这样只用 DummyMultiTaskDataset / 跑 smoke_test 时无需安装 albumentations+cv2。


# --------------------------------------------------------------------------- #
# 类别不平衡: 按 cls_label 频率构造 WeightedRandomSampler
# --------------------------------------------------------------------------- #
def _extract_cls_labels(ds, num_classes: int):
    """从数据集*不解码图像*地抽出每样本 (cls_label, has_cls)。

    支持 DummyMultiTaskDataset(._cls) 与 MultiTaskManifestDataset(.rows)。
    其它数据集回退: 返回 None 表示无法构造平衡采样。
    """
    # Dummy: 预生成的 _cls + all_tasks/取模规则
    if hasattr(ds, "_cls"):
        labels, has = [], []
        for i in range(len(ds)):
            labels.append(int(ds._cls[i]))
            has.append(bool(getattr(ds, "all_tasks", True) or True))  # dummy 恒有 cls
        return np.asarray(labels), np.asarray(has, dtype=bool)
    # Manifest: 直接读 rows 里的 cls 列 (复用 _get 的别名解析)
    if hasattr(ds, "rows") and hasattr(ds, "_get"):
        labels, has = [], []
        for row in ds.rows:
            s = ds._get(row, "cls")
            if s != "":
                labels.append(int(float(s)))
                has.append(True)
            else:
                labels.append(-1)
                has.append(False)
        return np.asarray(labels), np.asarray(has, dtype=bool)
    return None


def make_balanced_sampler(ds, num_classes: int) -> Optional[WeightedRandomSampler]:
    """按逆频率给每个样本赋采样权重 (只对有 cls 标注的样本生效)。

    无 cls 标注的样本: 赋所有有标注样本的平均权重 (既不抑制也不放大, 保证仍被采到)。
    若数据集不支持标签抽取, 返回 None (调用方退回普通 shuffle)。
    """
    got = _extract_cls_labels(ds, num_classes)
    if got is None:
        return None
    labels, has = got
    n = len(labels)
    if n == 0 or not has.any():
        return None

    # 各类频率 (仅统计有标注样本)
    weights = np.ones(n, dtype=np.float64)
    labeled = labels[has]
    counts = np.bincount(labeled, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)             # 防 0
    inv = 1.0 / counts                              # 逆频率
    inv = inv / inv.sum() * num_classes             # 归一, 量级稳定
    per_class_w = inv

    for i in range(n):
        if has[i]:
            weights[i] = per_class_w[labels[i]]
    # 无标注样本 → 给有标注样本的平均权重
    mean_labeled_w = float(weights[has].mean())
    weights[~has] = mean_labeled_w

    w = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=n, replacement=True)



# --------------------------------------------------------------------------- #
# Dummy 数据集 (smoke test)
# --------------------------------------------------------------------------- #
class DummyMultiTaskDataset(Dataset):
    """随机图 + 程序生成的圆形 disc / 内嵌 cup 掩膜 + 随机黄斑点。"""

    def __init__(self, n: int = 8, img_size=(128, 128), num_classes: int = 2,
                 all_tasks: bool = True, seed: int = 0):
        self.n = n
        self.H, self.W = img_size
        self.num_classes = num_classes
        self.all_tasks = all_tasks
        self.rng = np.random.RandomState(seed)
        # 预生成标签让 __getitem__ 确定性
        self._cls = self.rng.randint(0, num_classes, size=n)
        self._cx = self.rng.uniform(0.3, 0.7, size=n)
        self._cy = self.rng.uniform(0.3, 0.7, size=n)
        self._disc_r = self.rng.uniform(0.12, 0.20, size=n)
        self._mac_x = self.rng.uniform(0.1, 0.9, size=n)
        self._mac_y = self.rng.uniform(0.1, 0.9, size=n)

    def __len__(self) -> int:
        return self.n

    def _disk(self, cx, cy, r) -> torch.Tensor:
        ys = torch.linspace(0, 1, self.H).view(self.H, 1)
        xs = torch.linspace(0, 1, self.W).view(1, self.W)
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        return (d2 <= r * r).float().unsqueeze(0)        # (1,H,W)

    def __getitem__(self, i: int) -> Dict[str, object]:
        image = torch.from_numpy(
            np.random.RandomState(i).randn(3, self.H, self.W).astype("float32"))
        disc = self._disk(self._cx[i], self._cy[i], self._disc_r[i])
        cup = self._disk(self._cx[i], self._cy[i], self._disc_r[i] * 0.55)
        macula = torch.tensor([self._mac_x[i], self._mac_y[i]],
                              dtype=torch.float32)
        return {
            "image": image,
            "cls_label": torch.tensor(int(self._cls[i]), dtype=torch.long),
            "has_cls": True,
            "disc_mask": disc,
            "cup_mask": cup,
            "has_seg": True if self.all_tasks else (i % 2 == 0),
            "macula": macula,
            "has_macula": True if self.all_tasks else (i % 3 == 0),
        }


# --------------------------------------------------------------------------- #
# collate
# --------------------------------------------------------------------------- #
def multitask_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """把样本列表拼成 batch，并构造 seg_target=(B,2,H,W)=cat(disc,cup)。"""
    images = torch.stack([b["image"] for b in batch], dim=0)
    cls_label = torch.stack([b["cls_label"] for b in batch], dim=0)
    disc = torch.stack([b["disc_mask"] for b in batch], dim=0)   # (B,1,H,W)
    cup = torch.stack([b["cup_mask"] for b in batch], dim=0)
    seg_target = torch.cat([disc, cup], dim=1)                   # (B,2,H,W)
    macula = torch.stack([b["macula"] for b in batch], dim=0)    # (B,2)

    has_cls = torch.tensor([bool(b["has_cls"]) for b in batch])
    has_seg = torch.tensor([bool(b["has_seg"]) for b in batch])
    has_macula = torch.tensor([bool(b["has_macula"]) for b in batch])

    return {
        "image": images,
        "cls_label": cls_label,
        "has_cls": has_cls,
        "disc_mask": disc,
        "seg_target": seg_target,
        "has_seg": has_seg,
        "macula": macula,
        "has_macula": has_macula,
        # FIX(S-3): 逐样本回溯所需的元信息 (evidence table / bootstrap CI)
        "image_path": [b.get("image_path", "") for b in batch],
        "device": [b.get("device", "") for b in batch],
    }


# --------------------------------------------------------------------------- #
# 真实数据集骨架 (阶段一 pipeline 接入点)
# --------------------------------------------------------------------------- #
class MultiTaskManifestDataset(Dataset):
    """从 manifest 读取真实数据 (青光眼分类 + 视盘/视杯分割 + 黄斑定位)。

    manifest 可以是:
      - CSV 路径 (str / pathlib.Path)，或
      - 已经 pandas.read_csv 后的 DataFrame (本类不强依赖 pandas，按鸭子类型判断)，或
      - dict 记录列表。
    这样 train.py(传 CSV 路径) 和 check.py(传 DataFrame) 都能直接用。

    规范字段 (大小写敏感；某列留空 = 该任务对此样本无监督，loss 会按 has_* 屏蔽):
        image_path   必填，眼底图路径
        cls_label    分类标签 (整数)；空 → has_cls=False
        disc_path    视盘掩膜路径；空 → has_seg=False
        cup_path     视杯掩膜路径；空 → 视杯全 0 (但只要有 disc 仍 has_seg=True)
        macula_xy    黄斑归一化坐标 "x,y" (∈[0,1])；或改用 macula_x / macula_y 两列
        dataset      数据来源/设备 (阶段四按维度拆指标，可空)
    若你的列名不同，改下面的 _COLS 别名即可(已内置一批常见写法)。

    读图/resize/归一化/增广统一交给阶段一的 transforms.build_multitask_transforms，
    图像 + disc/cup 掩膜 + 黄斑关键点随同一组几何变换联动 (albumentations)。
    输出 schema 与 DummyMultiTaskDataset 完全一致，可直接喂 multitask_collate。
    """

    # 列名别名 (按需扩展)
    _COLS = {
        "image": ("image_path", "image", "img_path", "path", "filepath"),
        "cls":   ("cls_label", "label", "class", "glaucoma", "y"),
        "disc":  ("disc_path", "disc", "disc_mask", "od_path", "od"),
        "cup":   ("cup_path", "cup", "cup_mask", "oc_path", "oc"),
        # REFUGE2 风格: 单张掩膜里用灰度等级同时编码视盘+视杯 (见 _read_combined_mask)
        "seg":   ("mask_path", "mask", "seg_path", "seg", "gt_path", "gt"),
        "mac":   ("macula_xy", "macula", "fovea_xy", "fovea"),
        "mac_x": ("macula_x", "fovea_x", "mac_x", "fx"),
        "mac_y": ("macula_y", "fovea_y", "mac_y", "fy"),
    }

    def __init__(self, manifest, img_size=None, size=None, transform=None,
                 train: bool = True, num_classes: int = 2,
                 disc_gray_max: int = 200, cup_gray_max: int = 64,
                 mask_dark_is_fg: bool = True, device_col: str = "device"):
        # ---- REFUGE 合并掩膜的灰度阈值 (背景=255, 视盘=128, 视杯=0 约定) ----
        #   视盘 = 视杯∪盘沿 = 灰度 < disc_gray_max (默认 200，含 0 和 128)
        #   视杯 = 灰度 < cup_gray_max (默认 64，仅含 0)
        #   若你的掩膜灰度反过来 (背景=0)，把这两个阈值/逻辑改一下即可。
        self.disc_gray_max = disc_gray_max
        self.cup_gray_max = cup_gray_max
        # FIX(S-8): 分开的 disc/cup 掩膜文件的前景极性 (REFUGE 是"暗=前景")
        self.mask_dark_is_fg = bool(mask_dark_is_fg)
        # 供 phase4 按相机分层用；manifest 里没有该列就留空字符串
        self.device_col = device_col
        # ---- 解析输入分辨率: size 优先；其次 img_size；默认 384 (方图) ----
        if size is not None:
            side = int(size)
        elif img_size is not None:
            side = int(img_size[0]) if isinstance(img_size, (tuple, list)) \
                else int(img_size)
        else:
            side = 384
        self.size = side
        self.img_size = (side, side)
        self.train = train
        self.num_classes = num_classes

        # ---- 读取 manifest 行 (兼容 DataFrame / CSV 路径 / 记录列表) ----
        if hasattr(manifest, "to_dict"):            # pandas DataFrame
            self.rows = manifest.to_dict(orient="records")
        elif isinstance(manifest, (list, tuple)):   # 已是记录列表
            self.rows = list(manifest)
        else:                                        # CSV 路径
            import csv
            with open(str(manifest), newline="", encoding="utf-8") as f:
                self.rows = list(csv.DictReader(f))

        # ---- 增广/归一化 pipeline (阶段一 transforms，含黄斑关键点联动) ----
        # 惰性导入: 只有真正用真实数据集时才需要 albumentations + cv2。
        if transform is not None:
            self.transform = transform
        else:
            from .transforms import build_multitask_transforms
            self.transform = build_multitask_transforms(side, train)

    def __len__(self) -> int:
        return len(self.rows)

    # ----- 字段读取 (按别名查找，自动跳过空串 / pandas 的 NaN) ----- #
    def _get(self, row, key) -> str:
        for name in self._COLS[key]:
            if name in row and row[name] is not None:
                s = str(row[name]).strip()
                if s != "" and s.lower() != "nan":
                    return s
        return ""

    # ----- 读图/读掩膜: 返回原始 numpy，resize/归一化交给 transform ----- #
    def _read_image(self, path: str) -> np.ndarray:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))      # (H,W,3) uint8

    def _read_mask(self, path: str, hw, *, invert: bool = None) -> np.ndarray:
        """读单张二值掩膜。

        FIX(S-8): 旧实现固定用 `> 127` 当前景，而同文件的 _read_combined_mask
        用的是 `< 200` / `< 64` (暗=前景，REFUGE 约定)。两者极性相反：如果有人
        按 disc_path/cup_path 提供分开的掩膜且沿用 REFUGE 极性，前景背景会被
        整个反过来，而且不会报错。现在极性由 self.mask_dark_is_fg 统一控制。
        """
        from PIL import Image
        nearest = getattr(Image, "Resampling", Image).NEAREST
        m = Image.open(path).convert("L")
        if m.size != (hw[1], hw[0]):        # PIL.size = (W,H)
            m = m.resize((hw[1], hw[0]), nearest)
        g = np.asarray(m)
        dark_is_fg = self.mask_dark_is_fg if invert is None else invert
        fg = (g < 128) if dark_is_fg else (g > 127)
        return fg.astype(np.uint8)

    def _read_combined_mask(self, path: str, hw):
        """REFUGE2 单张掩膜 → (disc, cup) 两个 {0,1} 图。

        约定 背景=255 / 视盘(盘沿)=128 / 视杯=0:
          disc = 灰度 < disc_gray_max  (视杯 + 盘沿，即整个视盘)
          cup  = 灰度 < cup_gray_max   (仅视杯)
        用最近邻把掩膜对齐到图像尺寸，避免插值出中间灰度。
        """
        from PIL import Image
        nearest = getattr(Image, "Resampling", Image).NEAREST
        m = Image.open(path).convert("L")
        if m.size != (hw[1], hw[0]):
            m = m.resize((hw[1], hw[0]), nearest)
        g = np.asarray(m)
        disc = (g < self.disc_gray_max).astype(np.uint8)
        cup = (g < self.cup_gray_max).astype(np.uint8)
        return disc, cup

    def __getitem__(self, i: int) -> Dict[str, object]:
        row = self.rows[i]

        # 图像 (原始尺寸 H0,W0；transform 内的 Resize 会统一到 self.size)
        image = self._read_image(self._get(row, "image"))
        H0, W0 = image.shape[:2]

        # 分类标签
        cls_s = self._get(row, "cls")
        has_cls = cls_s != ""
        cls_label = torch.tensor(int(float(cls_s)) if has_cls else 0,
                                 dtype=torch.long)

        # 分割掩膜 (无标注则全 0)。优先用分开的 disc/cup 列；
        # 否则用 REFUGE2 风格的单张合并掩膜 (mask_path) 拆成视盘/视杯。
        disc_s = self._get(row, "disc")
        seg_s = self._get(row, "seg")
        has_seg = (disc_s != "") or (seg_s != "")
        if disc_s:
            disc = self._read_mask(disc_s, (H0, W0))
            cup_s = self._get(row, "cup")
            cup = self._read_mask(cup_s, (H0, W0)) if cup_s \
                else np.zeros((H0, W0), np.uint8)
        elif seg_s:
            disc, cup = self._read_combined_mask(seg_s, (H0, W0))
        else:
            disc = np.zeros((H0, W0), np.uint8)
            cup = np.zeros((H0, W0), np.uint8)

        # 黄斑坐标: 自动识别绝对像素坐标 (如 REFUGE2 的 fovea_x/fovea_y，值 >1)
        # 还是归一化坐标 (∈[0,1])，统一换成原图像素的关键点 (x,y)。
        mac_s = self._get(row, "mac")
        if mac_s:
            mx, my = [float(v) for v in mac_s.replace(";", ",").split(",")[:2]]
            has_macula = True
        else:
            mx_s, my_s = self._get(row, "mac_x"), self._get(row, "mac_y")
            if mx_s and my_s:
                mx, my = float(mx_s), float(my_s)
                has_macula = True
            else:
                mx, my, has_macula = 0.5, 0.5, False     # 占位，结果会被丢弃
        if mx > 1.0 or my > 1.0:                          # 绝对像素坐标，直接用
            kpt = (mx, my)
        else:                                             # 归一化坐标 → 像素
            kpt = (mx * W0, my * H0)

        # 联合变换: image + 两个 mask + 黄斑关键点 (同一组几何变换)
        out = self.transform(image=image, masks=[disc, cup], keypoints=[kpt])
        image_t = out["image"]                            # (3,size,size) float
        disc_t = out["masks"][0].unsqueeze(0).float()     # (1,size,size) {0,1}
        cup_t = out["masks"][1].unsqueeze(0).float()

        kps = out.get("keypoints", [])
        if has_macula and len(kps) > 0:
            kx, ky = float(kps[0][0]), float(kps[0][1])
            macula = torch.tensor([kx / self.size, ky / self.size],
                                  dtype=torch.float32).clamp_(0.0, 1.0)
        else:
            macula = torch.zeros(2, dtype=torch.float32)

        return {
            "image": image_t,
            "cls_label": cls_label,
            "has_cls": has_cls,
            "disc_mask": disc_t,
            "cup_mask": cup_t,
            "has_seg": has_seg,
            "macula": macula,
            "has_macula": has_macula,
            # 供逐样本证据表回溯 (FIX(S-3)/FIX(P1-2) 需要)
            "image_path": self._get(row, "image"),
            "device": str(row.get(self.device_col, "") or ""),
        }
