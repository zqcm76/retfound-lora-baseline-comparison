"""
inference.py — run one model over one dataset and emit the per-image evidence
table (a pandas DataFrame, one row per image), computing every per-image metric
on the way and (optionally) caching predicted masks for failure overlays.

This is where the data contract (Sample) meets the model contract (Prediction).
The engine stays torch-free: it only calls model.predict(image) and treats the
result as numpy.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config as C
import metrics as M
from core import COLS, EvalDataset, Model, Prediction, prob_to_mask, resize_mask_like


def _img_contrast(image: np.ndarray | None) -> float:
    if image is None:
        return M.NAN
    g = image[..., 1] if image.ndim == 3 else image  # green channel ~ vessel/disc contrast
    return float(np.std(g.astype(np.float32) / 255.0))


def _failure_tags(row: dict, cup_area_pred: float, cup_area_gt: float) -> str:
    tags: list[str] = []
    if row["has_disc"] and np.isfinite(row["disc_dice"]) and row["disc_dice"] < C.FAIL_DISC_DICE:
        tags.append("disc_miss")
    if row["has_cup"] and np.isfinite(row["cup_dice"]):
        if row["cup_dice"] < C.FAIL_CUP_DICE:
            tags.append("cup_miss")
        if cup_area_gt > 0 and cup_area_pred > C.FAIL_CUP_OVERSEG_RATIO * cup_area_gt:
            tags.append("cup_overseg")
    if row["has_fovea"] and np.isfinite(row["loc_err_norm"]) and row["loc_err_norm"] > C.FAIL_LOC_NORM:
        tags.append("loc_gross")
    if row["has_cls"] and np.isfinite(row["cls_prob"]) and row["cls_label"] in (0, 1):
        confident = abs(row["cls_prob"] - 0.5) > C.FAIL_CLS_CONF
        wrong = int(row["cls_prob"] >= 0.5) != int(row["cls_label"])
        if confident and wrong:
            tags.append("cls_confident_wrong")
    return ",".join(tags)


def run_inference(
    model: Model,
    dataset: EvalDataset,
    *,
    cache_dir: str | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Evaluate `model` on `dataset`. Returns a DataFrame with COLS columns.

    If cache_dir is given, predicted masks + fovea for each image are written to
    `cache_dir/<dataset>/<id>.npz` so failure_analysis can render overlays
    without keeping every mask in memory (and without re-running the model).
    """
    rows: list[dict] = []
    n = len(dataset)
    ds_cache = None
    if cache_dir:
        ds_cache = os.path.join(cache_dir, dataset.name)
        os.makedirs(ds_cache, exist_ok=True)

    for i, s in enumerate(dataset):
        if progress and (i % 50 == 0 or i == n - 1):
            print(f"  [{model.name} | {dataset.name}] {i + 1}/{n}", flush=True)

        pred: Prediction = model.predict(s.image) if s.image is not None else \
            model.predict(np.zeros((8, 8, 3), np.uint8))

        ref_shape = None
        if s.disc_gt is not None:
            ref_shape = s.disc_gt.shape
        elif s.cup_gt is not None:
            ref_shape = s.cup_gt.shape
        elif s.image is not None:
            ref_shape = s.image.shape[:2]

        disc_pred = prob_to_mask(pred.disc_prob, C.MASK_THRESHOLD)
        cup_pred = prob_to_mask(pred.cup_prob, C.MASK_THRESHOLD)
        if ref_shape is not None:
            if disc_pred is not None:
                disc_pred = resize_mask_like(disc_pred, ref_shape)
            if cup_pred is not None:
                cup_pred = resize_mask_like(cup_pred, ref_shape)

        row = {c: M.NAN for c in COLS}
        row.update(
            id=s.id, model=model.name, dataset=dataset.name,
            device=s.device, split=s.split,
            has_cls=bool(s.has_cls), has_disc=bool(s.has_disc),
            has_cup=bool(s.has_cup), has_fovea=bool(s.has_fovea),
            image_path=s.image_path,
        )

        # classification
        if s.has_cls:
            row["cls_prob"] = M.NAN if pred.cls_prob is None else float(pred.cls_prob)
            row["cls_label"] = M.NAN if s.cls_label is None else int(s.cls_label)

        # segmentation + CDR (need a predicted disc mask)
        cup_area_pred = cup_area_gt = 0.0
        if s.has_disc and s.disc_gt is not None and disc_pred is not None:
            row["disc_dice"] = M.dice_coef(disc_pred, s.disc_gt)
            row["disc_iou"] = M.iou(disc_pred, s.disc_gt)
            row["disc_diam_px"] = M.disc_diameter(s.disc_gt, C.LOC_NORM_DIAMETER)

            if s.has_cup and s.cup_gt is not None and cup_pred is not None:
                row["cup_dice"] = M.dice_coef(cup_pred, s.cup_gt)
                row["cup_iou"] = M.iou(cup_pred, s.cup_gt)
                cup_area_pred = float(cup_pred.sum())
                cup_area_gt = float(s.cup_gt.sum())
                row["vcdr_pred"] = M.vcdr(cup_pred, disc_pred)
                row["vcdr_gt"] = M.vcdr(s.cup_gt, s.disc_gt)
                if np.isfinite(row["vcdr_pred"]) and np.isfinite(row["vcdr_gt"]):
                    row["vcdr_abs_err"] = abs(row["vcdr_pred"] - row["vcdr_gt"])
                    row["hcdr_abs_err"] = abs(
                        M.hcdr(cup_pred, disc_pred) - M.hcdr(s.cup_gt, s.disc_gt))
                    row["acdr_abs_err"] = abs(
                        M.acdr(cup_pred, disc_pred) - M.acdr(s.cup_gt, s.disc_gt))

        # fovea localization (normalize by GT disc diameter if we have one)
        if s.has_fovea and s.fovea_gt is not None:
            diam = row["disc_diam_px"] if np.isfinite(row["disc_diam_px"]) else None
            px, norm = M.fovea_localization_error(pred.fovea_xy, s.fovea_gt, diam)
            row["loc_err_px"] = px
            row["loc_err_norm"] = norm

        row["img_contrast"] = _img_contrast(s.image)
        row["failure_tags"] = _failure_tags(row, cup_area_pred, cup_area_gt)

        # cache masks for overlays
        if ds_cache is not None and s.image is not None:
            cache_path = os.path.join(ds_cache, f"{s.id}.npz")
            np.savez_compressed(
                cache_path,
                disc_pred=(disc_pred if disc_pred is not None else np.zeros((1, 1), bool)),
                cup_pred=(cup_pred if cup_pred is not None else np.zeros((1, 1), bool)),
                disc_gt=(s.disc_gt if s.disc_gt is not None else np.zeros((1, 1), bool)),
                cup_gt=(s.cup_gt if s.cup_gt is not None else np.zeros((1, 1), bool)),
                fovea_pred=np.array(pred.fovea_xy if pred.fovea_xy is not None else (np.nan, np.nan)),
                fovea_gt=np.array(s.fovea_gt if s.fovea_gt is not None else (np.nan, np.nan)),
            )
            row["cache_path"] = cache_path

        rows.append(row)

    df = pd.DataFrame(rows, columns=COLS)
    return df


def run_matrix(
    models: dict[str, Model],
    datasets: dict[str, EvalDataset],
    *,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Evaluate every model on every dataset; return the concatenated evidence
    table. cache_dir is namespaced per model so overlays don't collide."""
    frames = []
    for mname, model in models.items():
        for dname, ds in datasets.items():
            mc = os.path.join(cache_dir, mname) if cache_dir else None
            frames.append(run_inference(model, ds, cache_dir=mc))
    return pd.concat(frames, ignore_index=True)
