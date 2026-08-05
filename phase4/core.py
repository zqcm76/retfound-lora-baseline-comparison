"""
core.py — shared types for the Phase 4 robustness-evaluation engine.

Design note
-----------
The whole phase is built around ONE artifact: a per-image "evidence table"
(a pandas DataFrame, one row per evaluated image). Every Phase 4 deliverable
is a *view* of that table:

  * evaluate.py        -> per-dataset aggregation (Dice / AUC / CDR / loc-err)
  * degradation.py     -> in-domain vs external deltas
  * stratify_device.py -> the same aggregation grouped by camera/device
  * failure_analysis.py-> the worst rows, rendered as overlays
  * compare_baselines.py-> in-domain row of two models + their training cost

Keeping every number traceable to a single image row is what makes the
"honest evaluation" story auditable: nothing is computed twice with different
conventions, and any cell in any table can be drilled back to its images.

This module is intentionally torch-free so the analysis core can be unit
tested without GPU / model weights. Only model_adapters.py (real mode) imports
torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

# --------------------------------------------------------------------------- #
# Per-image evidence-table schema (column names live here so every module agrees)
# --------------------------------------------------------------------------- #
COLS = [
    "id", "model", "dataset", "device", "split",
    # classification
    "cls_prob", "cls_label", "has_cls",
    # segmentation
    "disc_dice", "cup_dice", "disc_iou", "cup_iou", "has_disc", "has_cup",
    # cup-to-disc ratio (clinical fusion target from Phase 3)
    "vcdr_pred", "vcdr_gt", "vcdr_abs_err",
    "hcdr_abs_err", "acdr_abs_err",
    # fovea localization
    "loc_err_px", "loc_err_norm", "disc_diam_px", "has_fovea",
    # diagnostics used by failure analysis
    "img_contrast", "failure_tags", "cache_path", "image_path",
]

# Metrics where a HIGHER value is better (used by degradation direction logic).
HIGHER_IS_BETTER = {
    "auc", "sens_at_thr", "spec_at_thr",
    "disc_dice_mean", "cup_dice_mean", "disc_iou_mean", "cup_iou_mean",
    "loc_success_rate",
}
# Metrics where a LOWER value is better.
LOWER_IS_BETTER = {
    "vcdr_mae", "loc_norm_mean", "loc_norm_median", "loc_px_mean",
}


# --------------------------------------------------------------------------- #
# Data contract — what an evaluation dataset must yield, per image
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One evaluation image plus whatever ground truth the dataset provides.

    Availability flags matter: cross-dataset eval is only honest if a metric is
    scored exactly where its label exists. E.g. RIGA has disc+cup but no fovea;
    PALM has disc+fovea but no cup. Setting has_* correctly is the mechanism
    that keeps missing labels out of the means (instead of silently scoring
    them as 0 and inventing a degradation that is really a missing annotation).
    """
    id: str
    device: str                              # camera/device label for this image
    image: np.ndarray | None                 # HxWx3 uint8 (None = no overlay/contrast)
    cls_label: int | None                    # 1 = glaucoma, 0 = normal
    disc_gt: np.ndarray | None               # HxW bool, optic disc region (cup INCLUDED)
    cup_gt: np.ndarray | None                # HxW bool, optic cup region (subset of disc)
    fovea_gt: tuple[float, float] | None      # (x=col, y=row) in image pixels
    has_disc: bool = False
    has_cup: bool = False
    has_fovea: bool = False
    has_cls: bool = False
    split: str = "test"
    image_path: str | None = None
    extra: dict = field(default_factory=dict)


@runtime_checkable
class EvalDataset(Protocol):
    """Iterable of Samples. Real adapters wrap a Phase 1 torch Dataset and yield
    numpy Samples; the engine never touches torch."""
    name: str
    def __iter__(self) -> Iterator[Sample]: ...
    def __len__(self) -> int: ...


# --------------------------------------------------------------------------- #
# Model contract — what a model must return, per image
# --------------------------------------------------------------------------- #
@dataclass
class Prediction:
    """Model output for one image, already mapped back to the ORIGINAL image
    resolution by the adapter.

    Convention (must match Phase 2/3 outputs):
      * disc_prob = P(pixel is inside the optic disc, i.e. disc OR cup)   [0,1]
      * cup_prob  = P(pixel is inside the optic cup)                      [0,1]
        => cup is nested inside disc; both are thresholded at 0.5 here.
      * fovea_xy  = (x=col, y=row) predicted fovea centre, image pixels
      * cls_prob  = P(glaucoma)
    Any field may be None if the model does not produce that head.
    """
    cls_prob: float | None
    disc_prob: np.ndarray | None
    cup_prob: np.ndarray | None
    fovea_xy: tuple[float, float] | None


@runtime_checkable
class Model(Protocol):
    name: str
    def predict(self, image: np.ndarray) -> Prediction: ...


# --------------------------------------------------------------------------- #
# Geometry helpers shared by metrics + overlays
# --------------------------------------------------------------------------- #
def prob_to_mask(prob: np.ndarray | None, thr: float = 0.5) -> np.ndarray | None:
    if prob is None:
        return None
    return prob >= thr


def resize_mask_like(mask: np.ndarray, ref_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize a boolean mask to ref_shape (H, W).

    Used as a safety net when a model returns predictions at a different
    resolution than the GT (it should not, but cross-dataset glue is messy).
    """
    if mask.shape == ref_shape:
        return mask.astype(bool)
    from skimage.transform import resize
    out = resize(mask.astype(float), ref_shape, order=0, preserve_range=True,
                 anti_aliasing=False)
    return out >= 0.5
