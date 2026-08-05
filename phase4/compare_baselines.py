"""
compare_baselines.py — the headline "is the big foundation model actually worth
it?" table: from-scratch ResNet34+UNet vs LoRA-RETFound, putting COST next to
ACCURACY so the trade-off is explicit.

Cost numbers (params, train time, peak train memory, inference latency/memory)
are not all measurable from a checkpoint after the fact — train time and peak
*training* memory must come from logs the training phase wrote. So this module
reads a small JSON per model (written in Phase 2) and merges it with the
accuracy numbers computed here. Anything missing shows as NaN rather than being
guessed — honest reporting means an empty cell, not a fabricated one.

Expected train-meta JSON (one per model), e.g. outputs/<model>/train_meta.json:
{
  "params_total_m": 304.0,
  "params_trainable_m": 4.7,
  "train_hours": 6.5,
  "peak_train_mem_gb": 11.4,
  "infer_ms_per_img": 41.0,
  "infer_peak_mem_gb": 3.2,
  "epochs": 80,
  "gpu": "RTX 3060 12GB"
}
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import config as C
import evaluate as E


def load_train_meta(path: str | None) -> dict:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def count_parameters(model) -> tuple[float, float]:
    """(total_M, trainable_M) for a torch model. Returns (nan, nan) if model is
    not a torch.nn.Module (e.g. synthetic mode)."""
    try:
        total = sum(p.numel() for p in model.parameters())
        train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total / 1e6, train / 1e6
    except Exception:
        return float("nan"), float("nan")


def _mean_external_auc(per_dataset: pd.DataFrame, in_domain: str) -> float:
    ext = per_dataset.drop(index=in_domain, errors="ignore")["auc"]
    return float(np.nanmean(ext.to_numpy())) if len(ext) else float("nan")


def comparison_table(
    df: pd.DataFrame,
    train_meta: dict[str, dict],
    in_domain: str = C.IN_DOMAIN_DATASET,
) -> pd.DataFrame:
    """One row per model. df is the full evidence table (model column present);
    train_meta maps model_name -> meta dict (see module docstring)."""
    rows = []
    for model_name, g in df.groupby("model", sort=False):
        per_ds = E.evaluate_per_dataset(g, in_domain)
        meta = train_meta.get(model_name, {})
        # FIX(U-3): 占位符必须在表里显形，不能和实测值长得一模一样
        is_fake = bool(meta.get("_is_placeholder", False))
        idn = per_ds.loc[in_domain] if in_domain in per_ds.index else None

        def idv(col):
            return float(idn[col]) if idn is not None and np.isfinite(idn[col]) else np.nan

        ext_auc = _mean_external_auc(per_ds, in_domain)
        rows.append({
            "model": (f"{model_name} [PLACEHOLDER COSTS]" if is_fake else model_name),
            "params_total_M": meta.get("params_total_m", np.nan),
            "params_trainable_M": meta.get("params_trainable_m", np.nan),
            "train_hours": meta.get("train_hours", np.nan),
            "peak_train_mem_GB": meta.get("peak_train_mem_gb", np.nan),
            "infer_ms_per_img": meta.get("infer_ms_per_img", np.nan),
            "infer_peak_mem_GB": meta.get("infer_peak_mem_gb", np.nan),
            "indomain_auc": idv("auc"),
            "indomain_disc_dice": idv("disc_dice_mean"),
            "indomain_cup_dice": idv("cup_dice_mean"),
            "indomain_vcdr_mae": idv("vcdr_mae"),
            "mean_external_auc": round(ext_auc, 4) if np.isfinite(ext_auc) else np.nan,
            "auc_drop_external": (round(idv("auc") - ext_auc, 4)
                                  if np.isfinite(idv("auc")) and np.isfinite(ext_auc)
                                  else np.nan),
        })
    table = pd.DataFrame(rows).set_index("model")
    return table.round(4)


def to_markdown(table: pd.DataFrame, title: str = "Baseline comparison") -> str:
    """Render the comparison table as a markdown block for the tech report."""
    try:
        body = table.reset_index().to_markdown(index=False)
    except Exception:
        # to_markdown needs `tabulate`; fall back to a fixed-width string.
        body = table.reset_index().to_string(index=False)
    return f"### {title}\n\n{body}\n"
