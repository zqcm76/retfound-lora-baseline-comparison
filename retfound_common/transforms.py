# -*- coding: utf-8 -*-
"""
预处理与数据增广 —— phase1/2/3 共用同一份实现。

FIX(T-1): phase2 与 phase3 曾各持一份 transforms.py，phase3 那份缺了
          GaussianBlur 与 CoarseDropout、色彩抖动也更弱。于是 Stage-3 消融
          用的增广和 Stage-2 训练时不同，"0.934 > phase2 的 0.931"不是同一
          条流水线上的比较。现在只有这一份。
FIX(S-9): CoarseDropout 的 mask_fill_value 默认为 None，即图像被挖洞、掩膜
          不动。384 尺度下洞最大 ~31px，而视盘直径约 46~58px —— 一个洞能盖住
          视盘一半以上，模型却仍被要求把那里预测成 disc/cup。现在显式设
          mask_fill_value=0 让掩膜同步挖洞（语义一致），并把强度降下来；
          想做消融可以用 coarse_dropout=False 关掉。
FIX(U-8): 这里记录训练侧的缩放实现，phase4 推理必须与之一致 ——
          A.Resize 底层是 cv2.INTER_LINEAR 且**不做抗锯齿**。
          phase4 曾用 skimage.resize(anti_aliasing=True)，1848->384 的 4.8 倍
          降采样下两者纹理差异达 16%，等于评测代码自己引入了一层分布偏移。
"""
from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# phase4 推理端必须引用这个常量，保证与训练同一种插值
RESIZE_INTERPOLATION = cv2.INTER_LINEAR


def _geometric_aug() -> A.Affine:
    # 轻微仿射；CONSTANT 边界使填充像素为 0 (= 掩膜里的背景)
    return A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(-0.05, 0.05),
        rotate=(-15, 15),
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5,
    )


def build_cls_transforms(size: int, train: bool) -> A.Compose:
    if train:
        aug = [
            A.Resize(height=size, width=size, interpolation=RESIZE_INTERPOLATION),
            A.HorizontalFlip(p=0.5),
            _geometric_aug(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                                 val_shift_limit=10, p=0.3),
        ]
    else:
        aug = [A.Resize(height=size, width=size, interpolation=RESIZE_INTERPOLATION)]
    aug += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(aug)


def build_seg_transforms(size: int, train: bool) -> A.Compose:
    if train:
        aug = [
            A.Resize(height=size, width=size, interpolation=RESIZE_INTERPOLATION),
            A.HorizontalFlip(p=0.5),
            _geometric_aug(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ]
    else:
        aug = [A.Resize(height=size, width=size, interpolation=RESIZE_INTERPOLATION)]
    aug += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(aug)


def build_multitask_transforms(size: int, train: bool, *,
                               coarse_dropout: bool = True,
                               blur: bool = True) -> A.Compose:
    """联合变换：image + disc/cup 掩膜 + 黄斑关键点同步做几何变换。

    coarse_dropout / blur 暴露成开关，方便做 FIX(S-9) 的消融
    （怀疑 CoarseDropout 压住了 cup Dice 时可以直接关掉重训一版对比）。
    """
    if train:
        aug = [
            A.Resize(height=size, width=size, interpolation=RESIZE_INTERPOLATION),
            A.HorizontalFlip(p=0.5),
            _geometric_aug(),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20,
                                 val_shift_limit=15, p=0.4),
        ]
        if blur:
            aug.append(A.GaussianBlur(blur_limit=(3, 7), p=0.2))
        if coarse_dropout:
            # FIX(S-9): 强度从 8% 降到 4%、p 从 0.3 降到 0.15，
            # 并让掩膜同步挖洞 (mask_fill_value=0 = 背景)
            aug.append(A.CoarseDropout(
                max_holes=4,
                max_height=int(size * 0.04), max_width=int(size * 0.04),
                fill_value=0, mask_fill_value=0, p=0.15))
    else:
        aug = [A.Resize(height=size, width=size, interpolation=RESIZE_INTERPOLATION)]
    aug += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(
        aug,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
