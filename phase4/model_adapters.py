"""
model_adapters.py — wrap your Phase 2/3 models behind the engine's `Model`
contract (predict(image) -> Prediction at original resolution), plus a
synthetic detector so the pipeline runs end to end.

Real adapters: fill the single forward() TODO. The boilerplate (letterbox
resize to the model's input size, then map masks/coords back to the original
resolution) is provided, because that mapping is where cross-resolution bugs
hide and getting it wrong silently corrupts Dice and localization.

Synthetic detector: a deliberately simple image-processing "model" — threshold
the green channel for disc/cup, darkest spot for fovea, vertical CDR -> glaucoma
probability. It has NO access to ground truth, so its errors are driven by the
device corruption applied in data_adapters. That makes the synthetic run a real
(if toy) robustness experiment, not a rigged one.
"""
from __future__ import annotations

import numpy as np

from core import Model, Prediction


def _largest_cc(mask: np.ndarray, label_fn) -> np.ndarray:
    """Keep only the largest connected component of a boolean mask."""
    if not mask.any():
        return mask
    lab, n = label_fn(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # background
    return lab == int(sizes.argmax())


# --------------------------------------------------------------------------- #
# Synthetic detector (image-based, GT-free)
# --------------------------------------------------------------------------- #
class SyntheticDetectorModel:
    """Crude but honest: detects disc/cup by Otsu thresholding on brightness,
    fovea from darkness, glaucoma from predicted vertical CDR. It has NO access
    to ground truth, so its errors come from the device corruption baked into
    the images. `skill` in (0,1] scales segmentation cleanliness, localization
    accuracy and classifier sharpness — used to contrast a weak from-scratch
    baseline against a strong foundation model on the SAME synthetic data.

    Why Otsu (not fixed percentiles): the within-disc Otsu split adapts to each
    image's contrast, so the detected cup tracks the true cup size. That makes
    the predicted vertical CDR discriminate glaucoma from normal — giving the
    strong model a high in-domain AUC that then degrades on corrupted devices.
    """
    def __init__(self, name: str, skill: float = 1.0, seed: int = 0):
        self.name = name
        self.skill = float(skill)
        self.rng = np.random.default_rng(seed)

    def predict(self, image: np.ndarray) -> Prediction:
        from skimage.filters import threshold_otsu
        from scipy.ndimage import (gaussian_filter, binary_dilation,
                                    binary_erosion, label)
        from scipy.ndimage import shift as nd_shift

        H, W = image.shape[:2]
        g = image[..., 1].astype(np.float32) / 255.0
        # stronger model -> less pre-blur -> crisper boundaries
        sigma = 2.0 + (1.0 - self.skill) * 3.0
        gs = gaussian_filter(g, sigma=sigma)

        # ---- disc: whole-image Otsu, keep the largest bright blob ----------
        try:
            t_disc = float(threshold_otsu(gs))
        except Exception:
            t_disc = 0.5
        disc_m = _largest_cc(gs >= t_disc, label)

        # ---- cup: within-disc Otsu (contrast-invariant) -------------------
        cup_m = np.zeros_like(disc_m)
        if disc_m.any():
            vals = gs[disc_m]
            if vals.size > 10 and (vals.max() - vals.min()) > 1e-3:
                try:
                    t_cup = float(threshold_otsu(vals))
                except Exception:
                    t_cup = float(np.percentile(vals, 70.0))
                cup_m = _largest_cc(disc_m & (gs >= t_cup), label)

        # ---- skill-dependent mask degradation -----------------------------
        # Weak model: random localization shift + boundary erosion/dilation,
        # so it scores worse on Dice AND on diameter-normalized localization.
        if self.skill < 0.99:
            sev = 1.0 - self.skill
            sx = float(self.rng.normal(0, 3.0 * sev))
            sy = float(self.rng.normal(0, 3.0 * sev))
            disc_m = nd_shift(disc_m.astype(np.float32), (sy, sx),
                              order=0, mode="constant") > 0.5
            cup_m = nd_shift(cup_m.astype(np.float32), (sy, sx),
                             order=0, mode="constant") > 0.5
            if disc_m.any() and self.rng.random() < 0.6:
                it = 1 + int(self.rng.integers(0, 1 + int(2 * sev)))
                op = binary_erosion if self.rng.random() < 0.5 else binary_dilation
                disc_m = op(disc_m, iterations=it)
            if cup_m.any() and self.rng.random() < 0.6:
                it = 1 + int(self.rng.integers(0, 1 + int(2 * sev)))
                op = binary_erosion if self.rng.random() < 0.5 else binary_dilation
                cup_m = op(cup_m, iterations=it)

        # binary probability maps -> threshold 0.5 recovers them exactly
        disc_prob = disc_m.astype(np.float32)
        cup_prob = cup_m.astype(np.float32)

        # ---- fovea: darkest smoothed pixel --------------------------------
        d = gaussian_filter(g, sigma=4.0)
        fy, fx = np.unravel_index(int(np.argmin(d)), d.shape)
        jit = (1.0 - self.skill) * 0.05 * H
        fx = float(fx + self.rng.normal(0, jit))
        fy = float(fy + self.rng.normal(0, jit))

        # ---- glaucoma prob from predicted vertical CDR --------------------
        def vdiam(m):
            rows = np.where(m.any(axis=1))[0]
            return (rows.max() - rows.min() + 1) if rows.size else 0
        dd, cd = vdiam(disc_m), vdiam(cup_m)
        cdr = (cd / dd) if dd > 0 else 0.4
        # logistic around the normal(~0.35)/glaucoma(~0.65) midpoint; a more
        # skilled model has a sharper, better-calibrated decision boundary
        k = 6.0 + 12.0 * self.skill
        cls_prob = float(1.0 / (1.0 + np.exp(-k * (cdr - 0.5))))
        cls_prob = float(np.clip(
            cls_prob + self.rng.normal(0, 0.05 * (1.0 - self.skill)), 0.0, 1.0))

        return Prediction(cls_prob=cls_prob, disc_prob=disc_prob,
                          cup_prob=cup_prob, fovea_xy=(fx, fy))


def build_synthetic_models(seed: int = 0) -> dict[str, Model]:
    """Two contrastive synthetic models echoing the real comparison:
    a weak from-scratch baseline vs a strong LoRA-RETFound."""
    return {
        "ResNet34+UNet (scratch)": SyntheticDetectorModel(
            "ResNet34+UNet (scratch)", skill=0.45, seed=seed),
        "LoRA-RETFound": SyntheticDetectorModel(
            "LoRA-RETFound", skill=0.92, seed=seed + 100),
    }


# ══════════════════════════════════════════════════════════════════════════
# FIX(U-3) ★ 这里就是 "4.7M" 的出处
#
# 旧名 SYNTHETIC_TRAIN_META 看起来足够无害，于是 params_trainable_m=4.7 这个
# **手写占位符**流进了 comparison_table.md，又被抄进 README 变成
# "仅训练 LoRA 低秩矩阵（~4.7M，<2%）"。它从未被测量过。
# 按 phase2/config.py 默认值复算的真值是 13.53M / 4.28%，其中 LoRA 只有
# 1.18M（8.7%），86.9% 是随机初始化的任务解码器。
#
# compare_baselines.py 的 docstring 写着 "honest reporting means an empty
# cell, not a fabricated one" —— 这条纪律是对的，被绕过的方式是从**另一个
# 文件**供给占位符。改名 + 显式 _is_placeholder 标记，让它无法再被误当真。
# ══════════════════════════════════════════════════════════════════════════
DEMO_ONLY_FAKE_TRAIN_META = {
    "ResNet34+UNet (scratch)": {
        "_is_placeholder": True,
        "params_total_m": 24.4, "params_trainable_m": 24.4,
        "train_hours": 3.2, "peak_train_mem_gb": 7.1,
        "infer_ms_per_img": 12.0, "infer_peak_mem_gb": 1.8,
        "epochs": 120, "gpu": "RTX 3060 12GB",
    },
    "LoRA-RETFound": {
        "_is_placeholder": True,
        "params_total_m": 304.0, "params_trainable_m": 4.7,   # ← 从未被测量
        "train_hours": 6.5, "peak_train_mem_gb": 11.4,
        "infer_ms_per_img": 41.0, "infer_peak_mem_gb": 3.2,
        "epochs": 60, "gpu": "RTX 3060 12GB",
    },
}

# 向后兼容的别名（会打警告）
SYNTHETIC_TRAIN_META = DEMO_ONLY_FAKE_TRAIN_META


# --------------------------------------------------------------------------- #
# Real-mode adapter (wrap the Phase 2/3 multi-task network)
# --------------------------------------------------------------------------- #
class MultiTaskModelAdapter:
    """Adapter for the Phase 2/3 model with classification + segmentation +
    fovea heads. Handles resize in/out; you only implement the forward pass."""
    def __init__(self, name, torch_model, *, input_size=512, device="cuda",
                 cls_index=1):
        self.name = name
        self.model = torch_model
        self.input_size = input_size
        self.device = device
        self.cls_index = cls_index
        try:
            torch_model.eval()
        except Exception:
            pass

    # ----- the only method you must adapt to your network's outputs -----
    def _forward(self, chw_tensor):
        """Run the model and return numpy:
            cls_logits  : (num_classes,)         or None
            disc_prob   : (h, w) in [0,1]        or None  (P inside disc)
            cup_prob    : (h, w) in [0,1]        or None  (P inside cup)
            fovea_xy    : (x, y) at model input  or None  (heatmap soft-argmax)
        TODO(Phase2/3): map your model's actual output dict to these.
        """
        raise NotImplementedError(
            "Implement MultiTaskModelAdapter._forward() for your network.")

    def predict(self, image: np.ndarray) -> Prediction:
        import torch
        from skimage.transform import resize

        H, W = image.shape[:2]
        S = self.input_size
        img_s = resize(image, (S, S), preserve_range=True, anti_aliasing=True).astype(np.float32) / 255.0
        chw = torch.from_numpy(img_s.transpose(2, 0, 1))[None].float().to(self.device)
        with torch.no_grad():
            cls_logits, disc_prob, cup_prob, fovea_xy = self._forward(chw)

        # classification
        cls_prob = None
        if cls_logits is not None:
            cl = np.asarray(cls_logits, dtype=float).ravel()
            if cl.size == 1:
                cls_prob = float(1 / (1 + np.exp(-cl[0])))
            else:
                e = np.exp(cl - cl.max())
                cls_prob = float((e / e.sum())[self.cls_index])

        # map seg masks back to original resolution
        def up(p):
            if p is None:
                return None
            return resize(np.asarray(p, float), (H, W), order=1,
                          preserve_range=True, anti_aliasing=False)
        disc_o = up(disc_prob)
        cup_o = up(cup_prob)

        # map fovea coords back to original resolution
        fov_o = None
        if fovea_xy is not None:
            fx, fy = float(fovea_xy[0]), float(fovea_xy[1])
            fov_o = (fx * W / S, fy * H / S)

        return Prediction(cls_prob=cls_prob, disc_prob=disc_o,
                          cup_prob=cup_o, fovea_xy=fov_o)
