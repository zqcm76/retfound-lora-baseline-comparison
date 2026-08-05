"""
degradation.py — quantify how many points each metric loses off the in-domain
anchor, per external dataset. Direction-aware: for higher-is-better metrics a
negative delta is a drop; for error metrics (vCDR MAE, localization) a positive
delta is a drop. We report both, plus a relative %, so the report can say
"disc Dice fell 0.07 (-8.1%) on RIGA" rather than a vague "it got worse".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from core import HIGHER_IS_BETTER, LOWER_IS_BETTER

# Metrics we summarize degradation for (must exist in the aggregated table).
DEGRADE_METRICS = [
    "auc", "sens_at_thr", "spec_at_thr",
    "disc_dice_mean", "cup_dice_mean",
    "vcdr_mae", "loc_norm_mean",
]


def _delta(metric: str, external: float, anchor: float) -> tuple[float, float]:
    """Return (signed_drop, relative_pct). signed_drop > 0 always means WORSE."""
    if not (np.isfinite(external) and np.isfinite(anchor)):
        return np.nan, np.nan
    if metric in HIGHER_IS_BETTER:
        drop = anchor - external           # positive => lost points
    elif metric in LOWER_IS_BETTER:
        drop = external - anchor           # positive => error grew
    else:
        drop = anchor - external
    rel = (drop / abs(anchor) * 100.0) if anchor != 0 else np.nan
    return float(drop), float(rel)


def degradation_table(
    per_dataset: pd.DataFrame, in_domain: str = C.IN_DOMAIN_DATASET
) -> pd.DataFrame:
    """Long-form table: one row per (dataset, metric) with value / drop / rel%."""
    if in_domain not in per_dataset.index:
        raise ValueError(f"in-domain dataset {in_domain!r} missing from results")
    anchor = per_dataset.loc[in_domain]
    rows = []
    for ds in per_dataset.index:
        if ds == in_domain:
            continue
        for metric in DEGRADE_METRICS:
            if metric not in per_dataset.columns:
                continue
            ext_val = per_dataset.loc[ds, metric]
            drop, rel = _delta(metric, ext_val, anchor[metric])
            rows.append({
                "dataset": ds, "metric": metric,
                "in_domain": round(float(anchor[metric]), 4) if np.isfinite(anchor[metric]) else np.nan,
                "external": round(float(ext_val), 4) if np.isfinite(ext_val) else np.nan,
                "drop": round(drop, 4) if np.isfinite(drop) else np.nan,
                "rel_pct": round(rel, 2) if np.isfinite(rel) else np.nan,
            })
    return pd.DataFrame(rows)


def degradation_wide(long_df: pd.DataFrame, value: str = "drop") -> pd.DataFrame:
    """Pivot to dataset x metric for a quick scan of where points were lost."""
    if long_df.empty:
        return long_df
    return long_df.pivot(index="dataset", columns="metric", values=value)


def worst_hits(long_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """The k largest relative drops across all (dataset, metric) pairs — the
    headline 'where it broke most' list for the report."""
    if long_df.empty:
        return long_df
    return (long_df.dropna(subset=["rel_pct"])
            .sort_values("rel_pct", ascending=False)
            .head(k)
            .reset_index(drop=True))
