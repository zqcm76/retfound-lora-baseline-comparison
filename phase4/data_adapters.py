"""
data_adapters.py — turn whatever Phase 1 produced into the engine's `Sample`
stream, plus a synthetic dataset so the whole pipeline runs without real data.

Two real-mode adapters are sketched with explicit TODOs at the only spots that
depend on YOUR Phase 1 layout (mask channel order, fovea coordinate file, the
per-image device for multi-camera RIGA). Everything else is generic.

The synthetic dataset is the smoke-test workhorse: it renders a cartoon fundus
(bright disc, brighter cup, dark fovea, vessel-like noise), then applies a
DEVICE-SPECIFIC corruption (blur / contrast / brightness / geometric shift).
Ground truth comes from the clean geometry, so the device corruption makes the
downstream detector drift — i.e. the degradation in the report *emerges* from a
camera effect rather than being hard-coded. That is the behaviour Phase 4 is
meant to surface, demonstrated on data we fully control.
"""
from __future__ import annotations

import os
from typing import Iterator

import numpy as np

import config as C
from core import EvalDataset, Sample

# Device corruption profiles for the synthetic generator.
# (blur_sigma, contrast_gain, brightness_add, dx, dy) — dx/dy in pixels.
_DEVICE_PROFILES = {
    "Canon_CR2":      (0.6, 1.00, 0.00, 0.0, 0.0),   # in-domain reference
    "Zeiss_Visucam":  (0.8, 0.98, 0.02, 1.0, 0.0),   # in-domain-ish
    "TopCon_NW6":     (1.2, 0.82, 0.06, 2.0, 1.0),   # lower contrast
    "BinRushed":      (2.0, 0.90, -0.04, 4.0, 3.0),  # blur + geometric shift
    "Magrabia":       (1.4, 1.10, 0.10, 2.0, -2.0),  # over-bright
    "Drishti_fundus": (1.6, 0.88, 0.03, 3.0, 2.0),
    "Nidek_AFC210":   (1.8, 0.84, -0.02, 3.0, -3.0),
}


# --------------------------------------------------------------------------- #
# Synthetic dataset
# --------------------------------------------------------------------------- #
def _draw_disk(mask: np.ndarray, cx: float, cy: float, r: float) -> None:
    from skimage.draw import disk
    rr, cc = disk((cy, cx), r, shape=mask.shape)
    mask[rr, cc] = True


def _apply_device(img: np.ndarray, profile) -> np.ndarray:
    from scipy.ndimage import gaussian_filter, shift
    blur, gain, bright, dx, dy = profile
    out = img.astype(np.float32) / 255.0
    out = gaussian_filter(out, sigma=(blur, blur, 0))
    out = np.clip((out - 0.5) * gain + 0.5 + bright, 0, 1)
    out = shift(out, shift=(dy, dx, 0), order=1, mode="nearest")
    return (out * 255).astype(np.uint8)


class SyntheticGlaucomaDataset:
    """Renders `n` fake fundus images on a single device. Glaucoma cases get a
    larger cup (higher CDR), matching the clinical signal the model keys on."""
    def __init__(self, name: str, device: str, n: int, *, seed: int = 0,
                 prevalence: float = 0.4, has_cup: bool = True,
                 has_fovea: bool = True, size: int = 384,
                 image_dir: str | None = None):
        self.name = name
        self.device = device
        self.n = n
        self.seed = seed
        self.prevalence = prevalence
        self.has_cup = has_cup
        self.has_fovea = has_fovea
        self.size = size
        self.image_dir = image_dir
        if image_dir:
            os.makedirs(image_dir, exist_ok=True)

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[Sample]:
        from skimage import io as skio
        rng = np.random.default_rng(self.seed)
        prof = _DEVICE_PROFILES.get(self.device, _DEVICE_PROFILES["Canon_CR2"])
        H = W = self.size

        for i in range(self.n):
            label = int(rng.random() < self.prevalence)

            # disc geometry, jittered
            cx = W * 0.5 + rng.normal(0, W * 0.04)
            cy = H * 0.5 + rng.normal(0, H * 0.04)
            disc_r = H * (0.16 + rng.normal(0, 0.01))
            # cup ratio: normals ~0.35, glaucoma ~0.65 (vertical)
            cdr = (0.65 if label else 0.35) + rng.normal(0, 0.06)
            cdr = float(np.clip(cdr, 0.15, 0.9))
            cup_r = disc_r * cdr

            disc_gt = np.zeros((H, W), bool)
            cup_gt = np.zeros((H, W), bool)
            _draw_disk(disc_gt, cx, cy, disc_r)
            if self.has_cup:
                _draw_disk(cup_gt, cx, cy, cup_r)

            # fovea ~2.5 disc-diameters temporal (to the side), with jitter
            side = -1 if rng.random() < 0.5 else 1
            fx = cx + side * disc_r * 5.0 + rng.normal(0, W * 0.01)
            fy = cy + rng.normal(0, H * 0.02)
            fx = float(np.clip(fx, 5, W - 5)); fy = float(np.clip(fy, 5, H - 5))

            # render a cartoon fundus
            img = np.zeros((H, W, 3), np.float32)
            yy, xx = np.mgrid[0:H, 0:W]
            retina = np.exp(-(((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (2 * (H * 0.45) ** 2)))
            img[..., 0] = 0.55 * retina            # reddish background
            img[..., 1] = 0.28 * retina
            img[..., 2] = 0.12 * retina
            img[disc_gt] += np.array([0.30, 0.32, 0.18])     # bright disc
            if self.has_cup:
                img[cup_gt] += np.array([0.18, 0.20, 0.12])  # brighter cup
            # dark fovea blob
            fov = np.exp(-(((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * (H * 0.03) ** 2)))
            img -= (0.25 * fov)[..., None]
            # vessel-like streaks (noise) so the detector isn't trivial
            img += rng.normal(0, 0.02, img.shape)
            img = np.clip(img, 0, 1)
            image = (img * 255).astype(np.uint8)

            # device corruption -> emergent degradation
            image = _apply_device(image, prof)

            image_path = None
            if self.image_dir:
                image_path = os.path.join(self.image_dir, f"{self.name}_{i:04d}.png")
                if not os.path.exists(image_path):
                    skio.imsave(image_path, image, check_contrast=False)

            yield Sample(
                id=f"{i:04d}",
                device=self.device,
                image=image,
                cls_label=label,
                disc_gt=disc_gt,
                cup_gt=cup_gt if self.has_cup else None,
                fovea_gt=(fx, fy) if self.has_fovea else None,
                has_disc=True,
                has_cup=self.has_cup,
                has_fovea=self.has_fovea,
                has_cls=True,
                image_path=image_path,
            )


def build_synthetic_suite(image_root: str, seed: int = 0) -> dict[str, EvalDataset]:
    """A small multi-dataset / multi-device suite mirroring the real plan:
      REFUGE2 (in-domain Canon), RIGA (3 cameras, no fovea), PALM (no cup),
      Drishti-GS (third camera). Sizes kept tiny for a fast smoke test.
    """
    img = lambda d: os.path.join(image_root, d)
    suite: dict[str, EvalDataset] = {}
    suite["REFUGE2"] = SyntheticGlaucomaDataset(
        "REFUGE2", "Canon_CR2", 120, seed=seed, prevalence=0.5,
        image_dir=img("REFUGE2"))
    # RIGA spans three cameras -> concatenate as one logical dataset with
    # per-image device set by the sub-stream.
    suite["RIGA"] = _ConcatDataset("RIGA", [
        SyntheticGlaucomaDataset("RIGA", "BinRushed", 40, seed=seed + 1,
                                 has_fovea=False, image_dir=img("RIGA")),
        SyntheticGlaucomaDataset("RIGA", "Magrabia", 40, seed=seed + 2,
                                 has_fovea=False, image_dir=img("RIGA")),
        SyntheticGlaucomaDataset("RIGA", "TopCon_NW6", 40, seed=seed + 3,
                                 has_fovea=False, image_dir=img("RIGA")),
    ])
    suite["PALM"] = SyntheticGlaucomaDataset(
        "PALM", "Zeiss_Visucam", 80, seed=seed + 4, prevalence=0.5,
        has_cup=False, image_dir=img("PALM"))
    suite["Drishti-GS"] = SyntheticGlaucomaDataset(
        "Drishti-GS", "Drishti_fundus", 60, seed=seed + 5,
        has_fovea=False, image_dir=img("Drishti-GS"))
    return suite


class _ConcatDataset:
    """Chain several SyntheticGlaucomaDatasets under one dataset name, keeping
    each stream's own device (so RIGA carries per-image camera labels)."""
    def __init__(self, name: str, parts: list[SyntheticGlaucomaDataset]):
        self.name = name
        self.parts = parts
        for p in parts:               # ensure ids are globally unique
            p.name = name

    def __len__(self) -> int:
        return sum(len(p) for p in self.parts)

    def __iter__(self) -> Iterator[Sample]:
        for p in self.parts:
            for s in p:
                s.id = f"{p.device}_{s.id}"
                yield s


# --------------------------------------------------------------------------- #
# Real-mode adapters (skeletons — wire to your Phase 1 datasets)
# --------------------------------------------------------------------------- #
class RefugeStyleAdapter:
    """Generic adapter for a REFUGE-style folder: images + disc/cup masks +
    a fovea CSV + a glaucoma-label CSV. Adjust the three TODOs to your layout."""
    def __init__(self, spec, *, disc_gray_max=200, cup_gray_max=64):
        """FIX(U-11): 旧默认值 mask_disc_value=255 配合 `disc_gt = m <= 255`
        **恒为全 True**（整张图都是视盘）；`cup_gt = m == 128` 取的是盘沿不是视杯。
        docstring 里的 "0=disc,128=cup,255=bg" 也写反了 —— 正确是
        0=cup, 128=rim, 255=bg。现在与 retfound_common/dataset 同一套阈值。
        """
        self.name = spec.name
        self.spec = spec
        self.disc_gray_max = disc_gray_max
        self.cup_gray_max = cup_gray_max
        self._items = self._index()

    def _index(self) -> list[dict]:
        # TODO(Phase1): list your image files and pair them with mask / label
        # rows. Return a list of dicts: {id, image_path, mask_path, label,
        # fovea_xy, device}.
        raise NotImplementedError(
            "Wire RefugeStyleAdapter._index() to your Phase 1 file layout.")

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Sample]:
        from skimage import io as skio
        for it in self._items:
            image = skio.imread(it["image_path"])
            disc_gt = cup_gt = None
            if it.get("mask_path"):
                m = skio.imread(it["mask_path"])
                m = m[..., 0] if m.ndim == 3 else m
                # REFUGE 约定: 0=cup, 128=rim, 255=bg (暗 = 前景)
                disc_gt = m < self.disc_gray_max     # 盘沿 + 视杯 = 整个视盘
                cup_gt = m < self.cup_gray_max       # 仅视杯
            yield Sample(
                id=str(it["id"]), device=it.get("device", self.spec.device_default),
                image=image, cls_label=it.get("label"),
                disc_gt=disc_gt, cup_gt=cup_gt,
                fovea_gt=it.get("fovea_xy"),
                has_disc=self.spec.has_disc and disc_gt is not None,
                has_cup=self.spec.has_cup and cup_gt is not None,
                has_fovea=self.spec.has_fovea and it.get("fovea_xy") is not None,
                has_cls=self.spec.has_cls and it.get("label") is not None,
                split=self.spec.split, image_path=it["image_path"],
            )


class TorchDatasetAdapter:
    """Wrap an existing Phase 1 torch Dataset that returns dicts. Map its keys
    to Sample fields here. Yields numpy so the engine stays torch-free."""
    def __init__(self, torch_dataset, spec, key_map: dict | None = None):
        self.name = spec.name
        self.ds = torch_dataset
        self.spec = spec
        self.key_map = key_map or {}

    def __len__(self) -> int:
        return len(self.ds)

    def __iter__(self) -> Iterator[Sample]:
        km = self.key_map
        for idx in range(len(self.ds)):
            item = self.ds[idx]
            def get(k, default=None):
                return item.get(km.get(k, k), default)
            img = get("image")
            if img is not None and hasattr(img, "numpy"):  # torch tensor CHW
                img = (img.numpy().transpose(1, 2, 0) * 255).astype("uint8")
            yield Sample(
                id=str(get("id", idx)),
                device=get("device", self.spec.device_default),
                image=img, cls_label=get("cls_label"),
                disc_gt=get("disc_gt"), cup_gt=get("cup_gt"),
                fovea_gt=get("fovea_gt"),
                has_disc=self.spec.has_disc, has_cup=self.spec.has_cup,
                has_fovea=self.spec.has_fovea, has_cls=self.spec.has_cls,
                split=self.spec.split, image_path=get("image_path"),
            )
