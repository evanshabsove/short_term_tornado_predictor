"""
Build the ~40km (39km actual) dense-grid used for tornado forecasting, by
block-aligned stride coarsening of the native HRRR Lambert Conformal
Conic grid (see src/tornado_predictor/grid.py for the projection math).

This does NOT download any HRRR data -- the grid is derived purely from
documented HRRR native-grid projection parameters. Run
scripts/verify_hrrr_grid_params.py separately (requires network access)
to confirm those hardcoded parameters against a live HRRR grib file.

Usage:
    python scripts/build_grid.py
    python scripts/build_grid.py --stride 13 --out data/processed/hrrr_coarse_grid_stride13.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyproj import Geod

from tornado_predictor.grid import DEFAULT_STRIDE, HRRR_EARTH_RADIUS_M, build_coarse_grid

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def print_sanity_summary(grid) -> None:
    print(f"Grid shape: {grid.n_rows} rows x {grid.n_cols} cols  (cell size = {grid.cell_size_m / 1000:.1f} km)")
    print(f"Lat range:  {grid.lat_center.min():.3f} to {grid.lat_center.max():.3f}")
    print(f"Lon range:  {grid.lon_center.min():.3f} to {grid.lon_center.max():.3f}")

    # A perfect sphere matching HRRR's own earth model -- a self-consistency
    # check of this grid's own geometry, not a comparison against WGS84.
    geod = Geod(a=HRRR_EARTH_RADIUS_M, f=0.0)
    checkpoints = {
        "domain center": (grid.n_rows // 2, grid.n_cols // 2),
        "near domain edge": (5, 5),
    }
    for label, (r, c) in checkpoints.items():
        lon1, lat1 = grid.lon_center[r, c], grid.lat_center[r, c]
        lon2, lat2 = grid.lon_center[r, c + 1], grid.lat_center[r, c + 1]
        _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
        print(f"  {label}: adjacent-cell spacing (east) = {dist_m / 1000:.2f} km")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    grid = build_coarse_grid(stride=args.stride)
    out_path = args.out or PROCESSED_DIR / f"hrrr_coarse_grid_stride{args.stride}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grid.to_dataset().to_netcdf(out_path)

    print_sanity_summary(grid)
    print(f"\nSaved grid to {out_path}")


if __name__ == "__main__":
    main()
