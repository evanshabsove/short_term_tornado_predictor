"""
Build a country mask for the ~39km coarse grid from Natural Earth boundaries,
and report how much of Canada the HRRR domain covers.

Per cell (row, col):
  country         int8: 0 = ocean/water/other, 1 = US, 2 = CA, 3 = MX, 4 = other land
                  (Cuba, Bahamas, ...). Assigned by the cell CENTER (point-in-polygon).
  frac_{us,ca,mx,other_land,water}
                  fraction of the cell's exact LCC square covered by each class
                  (computed in LCC meters, same projection as the grid, so areas
                  are true cell-relative fractions).
  straddles_border  True if the cell holds >= 2 of US/CA/MX/other-land, each
                  covering more than --min-frac (default 1%) of the cell.
  straddles_coast   True if the cell holds both land and water, each > --min-frac.

Natural Earth admin-0 country polygons INCLUDE the Great Lakes (the border runs
through the lake middles), so the Natural Earth 10m lakes layer is subtracted
from every country before assignment; lake cells therefore count as water
(class 0, "ocean_or_water"). Province areas in the coverage report are NOT
lake-subtracted (they are land + inland water, as Natural Earth draws them).

Data (not committed), all from naturalearthdata.com in data/raw/natural_earth_admin0/:
ne_10m_admin_0_countries.shp, ne_10m_admin_1_states_provinces.shp, ne_10m_lakes.shp.

Usage:
    python scripts/08_canada_transfer/build_country_mask.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import shapefile  # pyshp
import xarray as xr
from pyproj import Transformer
from shapely import STRtree, points
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union

from tornado_predictor.grid import HrrrCoarseGrid

REPO_ROOT = Path(__file__).resolve().parents[2]
GRID_PATH = REPO_ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13.nc"
OUT_PATH = REPO_ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13_countries.nc"
NE_DIR = REPO_ROOT / "data" / "raw" / "natural_earth_admin0"

CODES = {"OCEAN": 0, "US": 1, "CA": 2, "MX": 3, "OTHER": 4}
ADM0_TO_CLASS = {"USA": "US", "CAN": "CA", "MEX": "MX"}
# Clip window (lon/lat) applied before projecting: keeps antimeridian/polar
# polygons (Russia, Alaska's Aleutians) out of the LCC transform.
CLIP = box(-175, 5, -30, 85)


def read_shapes(path: Path, attr: str):
    r = shapefile.Reader(str(path), encoding="utf-8")
    for sr in r.iterShapeRecords():
        yield sr.record.as_dict(), shape(sr.shape.__geo_interface__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=Path, default=GRID_PATH)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--min-frac", type=float, default=0.01)
    args = p.parse_args()

    grid = HrrrCoarseGrid.from_netcdf(args.grid)
    to_lcc = Transformer.from_crs(grid.geodetic_crs_proj4, grid.crs_proj4, always_xy=True).transform
    project = lambda g: transform(to_lcc, g)  # noqa: E731

    # --- countries -> LCC class geometries -----------------------------
    groups: dict[str, list] = {"US": [], "CA": [], "MX": [], "OTHER": []}
    for rec, geom in read_shapes(NE_DIR / "ne_10m_admin_0_countries.shp", "ADM0_A3"):
        g = geom.intersection(CLIP)
        if g.is_empty:
            continue
        groups[ADM0_TO_CLASS.get(rec["ADM0_A3"], "OTHER")].append(g)
    lake_parts = [g for _, geom in read_shapes(NE_DIR / "ne_10m_lakes.shp", "name")
                  if not (g := geom.intersection(CLIP)).is_empty]
    lakes = project(unary_union(lake_parts).buffer(0))
    class_geom = {k: project(unary_union(v).buffer(0)).difference(lakes) for k, v in groups.items() if v}

    # --- cell-center assignment ----------------------------------------
    cx, cy = (np.asarray(a) for a in to_lcc(grid.lon_center, grid.lat_center))
    centers = points(cx.ravel(), cy.ravel())
    country = np.zeros(centers.shape, dtype=np.int8)
    for name, geom in class_geom.items():
        tree = STRtree([geom])
        hit = tree.query(centers, predicate="within")
        country[hit[0]] = CODES[name]
    country = country.reshape(grid.n_rows, grid.n_cols)

    # --- per-cell area fractions in exact LCC squares -------------------
    names = ["US", "CA", "MX", "OTHER"]
    frac = {n: np.zeros((grid.n_rows, grid.n_cols), dtype=np.float32) for n in names}
    cell_area = grid.cell_size_m**2
    for r in range(grid.n_rows):
        for c in range(grid.n_cols):
            sq = box(*grid.cell_bounds_xy(r, c))
            for n, g in class_geom.items():
                if g.intersects(sq):
                    frac[n][r, c] = g.intersection(sq).area / cell_area
    land = sum(frac[n] for n in names)
    water = np.clip(1.0 - land, 0.0, 1.0)

    m = args.min_frac
    n_land_classes = sum((frac[n] > m).astype(int) for n in names)
    straddles_border = n_land_classes >= 2
    straddles_coast = (land > m) & (water > m)

    ds = xr.Dataset(
        {
            "country": (("row", "col"), country, {
                "long_name": "country of cell center",
                "flag_values": list(CODES.values()), "flag_meanings": "ocean_or_water us ca mx other_land",
            }),
            "frac_us": (("row", "col"), frac["US"]),
            "frac_ca": (("row", "col"), frac["CA"]),
            "frac_mx": (("row", "col"), frac["MX"]),
            "frac_other_land": (("row", "col"), frac["OTHER"]),
            "frac_water": (("row", "col"), water.astype(np.float32)),
            "straddles_border": (("row", "col"), straddles_border),
            "straddles_coast": (("row", "col"), straddles_coast),
        },
        coords={"row": np.arange(grid.n_rows), "col": np.arange(grid.n_cols)},
        attrs={
            "title": "Country mask for the HRRR coarse grid",
            "source": "Natural Earth 1:10m admin-0 countries minus 1:10m lakes",
            "min_frac_for_straddle_flags": m,
            "note": "Natural Earth 10m lakes (incl. the Great Lakes) are subtracted from country polygons, so lake cells are water.",
        },
    )
    ds["country"].encoding["dtype"] = "int8"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.out)

    # --- report ---------------------------------------------------------
    total = country.size
    print(f"Grid {grid.n_rows}x{grid.n_cols} = {total} cells")
    for name, code in CODES.items():
        k = int((country == code).sum())
        print(f"  {name:6s} {k:6d}  ({100 * k / total:.1f}%)")
    print(f"  straddles_border: {int(straddles_border.sum())}   straddles_coast: {int(straddles_coast.sum())}")
    ca_any = int((frac["CA"] > m).sum())
    print(f"  cells with >{m:.0%} Canadian area: {ca_any}  (center-assigned CA: {int((country == 2).sum())})")
    print(f"  total Canadian area in grid: {frac['CA'].sum() * cell_area / 1e6:,.0f} km2")

    top = grid.lat_center[-1]
    print(f"\nTop row center lat: {top.min():.2f}-{top.max():.2f}N (top edge of grid extent a bit higher)")
    ca_rows = np.where((country == 2).any(axis=1))[0]
    ca_lat = np.where(country == 2, grid.lat_center, np.nan)
    print(f"CA cells span rows {ca_rows.min()}-{ca_rows.max()}, lat {np.nanmin(ca_lat):.2f}-{np.nanmax(ca_lat):.2f}N")

    # --- Canadian provinces/territories: share of area inside the grid ---
    x_min, y_min, _, _ = grid.cell_bounds_xy(0, 0)
    _, _, x_max, y_max = grid.cell_bounds_xy(grid.n_rows - 1, grid.n_cols - 1)
    domain = box(x_min, y_min, x_max, y_max)
    print("\nCanadian provinces/territories (area inside coarse grid extent):")
    rows = []
    for rec, geom in read_shapes(NE_DIR / "ne_10m_admin_1_states_provinces.shp", "adm0_a3"):
        if rec["adm0_a3"] != "CAN":
            continue
        g = project(geom.intersection(CLIP).buffer(0))
        inside = g.intersection(domain)
        lat_max = geom.bounds[3]
        rows.append((rec["name"], g.area, inside.area, lat_max))
    for name, a, ia, lat_max in sorted(rows, key=lambda t: -t[2] / t[1]):
        print(f"  {name:28s} {100 * ia / a:5.1f}% inside  ({a / 1e6:>11,.0f} km2, northmost {lat_max:.1f}N)")
    tot = sum(r[1] for r in rows)
    ins = sum(r[2] for r in rows)
    print(f"  {'ALL CANADA':28s} {100 * ins / tot:5.1f}% inside  ({tot / 1e6:>11,.0f} km2)")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
