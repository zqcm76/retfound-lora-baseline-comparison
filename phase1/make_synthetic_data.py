"""Generate a small SYNTHETIC fundus-like dataset.

This is NOT a substitute for REFUGE2 — it exists so you (and the smoke test) can run the
entire pipeline end to end on day one, before any real data is downloaded, and confirm the
plumbing works. Each sample has a bright "optic disc" with a concentric "cup", a mask in
the REFUGE {0,128,255} convention, a fovea point, and a glaucoma label correlated with the
cup-to-disc ratio (so a sanity-check epoch can actually learn something).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def make_synthetic(root, n_train: int = 24, n_val: int = 8, size: int = 256, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    rows = []

    def gen(split: str, n: int) -> None:
        for k in range(n):
            img = rng.integers(20, 80, (size, size, 3), dtype=np.uint8)
            img[..., 0] = np.clip(img[..., 0].astype(int) + 60, 0, 255)  # reddish fundus
            cx, cy = rng.integers(int(size * 0.3), int(size * 0.7), size=2)
            disc_r = int(rng.integers(int(size * 0.10), int(size * 0.16)))
            cup_r = int(disc_r * rng.uniform(0.4, 0.8))
            yy, xx = np.ogrid[:size, :size]
            dist = (xx - cx) ** 2 + (yy - cy) ** 2
            disc_area = dist <= disc_r ** 2
            img[disc_area] = np.clip(
                img[disc_area].astype(int) + np.array([120, 120, 40]), 0, 255
            ).astype(np.uint8)

            mask = np.full((size, size), 255, np.uint8)  # background
            mask[disc_area] = 128                         # disc rim
            mask[dist <= cup_r ** 2] = 0                  # cup

            fx = int(np.clip(cx + rng.choice([-1, 1]) * disc_r * 2.5, 0, size - 1))
            fy = int(np.clip(cy + rng.integers(-disc_r, disc_r), 0, size - 1))
            label = int(cup_r / disc_r > 0.6)             # high CDR -> "glaucoma"

            ip = root / "images" / f"{split}_{k:03d}.png"
            mp = root / "masks" / f"{split}_{k:03d}.png"
            cv2.imwrite(str(ip), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mp), mask)
            rows.append(
                {
                    "image_path": str(ip),
                    "mask_path": str(mp),
                    "label": label,
                    "fovea_x": fx,
                    "fovea_y": fy,
                    "split": split,
                }
            )

    gen("train", n_train)
    gen("val", n_val)
    df = pd.DataFrame(rows)
    df.to_csv(root / "refuge_index.csv", index=False)
    return df


if __name__ == "__main__":
    df = make_synthetic("data/synthetic")
    print(f"wrote {len(df)} synthetic samples to data/synthetic/ and refuge_index.csv")
