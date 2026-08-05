"""
failure_analysis.py — pick the worst cases per task, render them as GT-vs-pred
overlays, and tabulate failure-mode tag frequencies.

Overlays are reconstructed from the npz mask cache written by inference.py
(image loaded from its path). This keeps memory flat: nothing is held for all
images, and the model is never re-run. If no cache is available, only the tag
tables are produced.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config as C


# ----------------------------- tag frequencies ----------------------------- #
def tag_counts(df: pd.DataFrame, by: str = "dataset") -> pd.DataFrame:
    """Count each failure tag per group. Returns group x tag (counts + rate)."""
    known = ["disc_miss", "cup_miss", "cup_overseg", "loc_gross", "cls_confident_wrong"]
    rows = []
    for key, g in df.groupby(by, sort=False):
        rec = {by: key, "n_images": int(len(g))}
        joined = g["failure_tags"].fillna("").astype(str)
        for tag in known:
            c = int(joined.str.contains(rf"\b{tag}\b").sum())
            rec[tag] = c
            rec[f"{tag}_rate"] = round(c / len(g), 4) if len(g) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------- worst cases -------------------------------- #
_TASK_SORT = {
    # task -> (column, ascending?, predicate on the row to be eligible)
    "disc_dice": ("disc_dice", True, lambda r: r["has_disc"]),
    "cup_dice": ("cup_dice", True, lambda r: r["has_cup"]),
    "loc_norm": ("loc_err_norm", False, lambda r: r["has_fovea"]),
    "vcdr": ("vcdr_abs_err", False, lambda r: r["has_cup"]),
}


def worst_cases(df: pd.DataFrame, task: str, k: int = C.N_FAILURE_CASES) -> pd.DataFrame:
    col, ascending, pred = _TASK_SORT[task]
    sub = df[df.apply(pred, axis=1)].dropna(subset=[col])
    if sub.empty:
        return sub
    return sub.sort_values(col, ascending=ascending).head(k)


def confident_wrong_cases(df: pd.DataFrame, k: int = C.N_FAILURE_CASES) -> pd.DataFrame:
    sub = df[df["has_cls"] & np.isfinite(df["cls_prob"]) & df["cls_label"].isin([0, 1])].copy()
    if sub.empty:
        return sub
    sub["conf"] = (sub["cls_prob"] - 0.5).abs()
    sub["wrong"] = (sub["cls_prob"] >= 0.5).astype(int) != sub["cls_label"].astype(int)
    sub = sub[sub["wrong"]].sort_values("conf", ascending=False)
    return sub.head(k)


# --------------------------------- overlays --------------------------------- #
def _contour_overlay(ax, image: np.ndarray, cache_path: str) -> None:
    from skimage import measure
    ax.imshow(image)
    data = np.load(cache_path, allow_pickle=True)

    def draw(mask_gt, mask_pred, c_gt, c_pred):
        for m, color in ((mask_gt, c_gt), (mask_pred, c_pred)):
            if m is None or m.size <= 1 or not m.any():
                continue
            for contour in measure.find_contours(m.astype(float), 0.5):
                ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.6)

    draw(data["disc_gt"], data["disc_pred"], "#00d000", "#ff3030")     # green GT / red pred
    draw(data["cup_gt"], data["cup_pred"], "#00ffff", "#ffaa00")        # cyan GT / orange pred
    fg, fp = data["fovea_gt"], data["fovea_pred"]
    if np.all(np.isfinite(fg)):
        ax.plot(fg[0], fg[1], marker="+", color="#00d000", markersize=12, markeredgewidth=2)
    if np.all(np.isfinite(fp)):
        ax.plot(fp[0], fp[1], marker="x", color="#ff3030", markersize=10, markeredgewidth=2)
    ax.set_xticks([]); ax.set_yticks([])


def render_worst(df: pd.DataFrame, task: str, out_dir: str,
                 k: int = C.N_FAILURE_CASES, ncols: int = 4) -> str | None:
    """Render a grid of the k worst cases for `task` to a PNG. Needs cache_path
    + image_path on the rows. Returns the PNG path (or None if nothing to show)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import io as skio

    cases = (confident_wrong_cases(df, k) if task == "cls"
             else worst_cases(df, task, k))
    cases = cases[cases["cache_path"].notna() & cases["image_path"].notna()]
    if cases.empty:
        return None

    n = len(cases)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    metric_col = {"disc_dice": "disc_dice", "cup_dice": "cup_dice",
                  "loc_norm": "loc_err_norm", "vcdr": "vcdr_abs_err",
                  "cls": "cls_prob"}[task]
    for ax, (_, r) in zip(axes, cases.iterrows()):
        try:
            img = skio.imread(r["image_path"])
        except Exception:
            ax.axis("off"); continue
        _contour_overlay(ax, img, r["cache_path"])
        ax.set_title(f"{r['dataset']}/{r['id']}\n{r['device']} · "
                     f"{metric_col}={r[metric_col]:.3f}", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Worst cases — {task} "
                 f"(GT solid green/cyan, pred red/orange)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"failures_{task}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_failure_analysis(df: pd.DataFrame, out_dir: str) -> dict:
    """Produce tag tables (CSV) + worst-case overlays for every task. Returns a
    dict of artifact paths for the orchestrator/report."""
    os.makedirs(out_dir, exist_ok=True)
    artifacts: dict = {"overlays": {}, "tables": {}}

    by_ds = tag_counts(df, "dataset")
    by_dev = tag_counts(df, "device")
    p1 = os.path.join(out_dir, "failure_tags_by_dataset.csv")
    p2 = os.path.join(out_dir, "failure_tags_by_device.csv")
    by_ds.to_csv(p1, index=False)
    by_dev.to_csv(p2, index=False)
    artifacts["tables"]["by_dataset"] = p1
    artifacts["tables"]["by_device"] = p2

    for task in ["disc_dice", "cup_dice", "loc_norm", "vcdr", "cls"]:
        png = render_worst(df, task, out_dir)
        if png:
            artifacts["overlays"][task] = png
    return artifacts
