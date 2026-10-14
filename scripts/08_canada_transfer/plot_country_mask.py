"""
Plot the coarse-grid country mask (see scripts/08_canada_transfer/build_country_mask.py).

Plotted in grid space (row/col), north-up (row 0 is the SW corner, so
origin="lower"). Left: country of each cell center. Right: Canadian area
fraction with border-straddling cells marked.

Usage:
    python scripts/08_canada_transfer/plot_country_mask.py
    python scripts/08_canada_transfer/plot_country_mask.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
MASK_PATH = REPO_ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13_countries.nc"
OUT_PATH = REPO_ROOT / "outputs" / "country_mask.png"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mask", type=Path, default=MASK_PATH)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    with xr.open_dataset(args.mask) as d:
        d = d.load()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cmap = ListedColormap(["#a6cee3", "#b2df8a", "#e31a1c", "#fdbf6f", "#999999"])
    im = axes[0].imshow(d.country, origin="lower", cmap=cmap, vmin=-0.5, vmax=4.5)
    cb = fig.colorbar(im, ax=axes[0], ticks=range(5), shrink=0.8)
    cb.ax.set_yticklabels(["water", "US", "CA", "MX", "other land"])
    axes[0].set_title("Country (cell center)")

    im = axes[1].imshow(d.frac_ca, origin="lower", cmap="Reds", vmin=0, vmax=1)
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="Canadian area fraction")
    by, bx = np.where(d.straddles_border.values)
    axes[1].scatter(bx, by, s=6, c="k", label="straddles border")
    axes[1].legend(loc="lower right")
    axes[1].set_title("Canadian area fraction + border cells")

    for ax in axes:
        ax.set_xlabel("col")
        ax.set_ylabel("row")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
