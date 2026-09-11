"""
Defines the ~40km (39km actual) dense-grid used for tornado forecasting,
by block-aligned stride coarsening of HRRR's native Lambert Conformal
Conic (LCC) grid.

This reuses HRRR's own native projection rather than building an
independent one, and every coarse cell is an exact contiguous block of
native HRRR pixels (13x13 -> 39km, the closest integer stride to
"~40km"). Building the grid is pure projection math and requires no
HRRR data download.

The HRRR_* constants below are the standard, documented HRRR CONUS grid
parameters. Run scripts/verify_hrrr_grid_params.py (requires network
access) to confirm them against a live HRRR grib2 file before trusting
this module for anything beyond exploratory use.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import Polygon

# --- HRRR native grid (NCEP HRRR CONUS Lambert Conformal Conic) ---
HRRR_LAT_1_DEG = 38.5  # Latin1 == Latin2 -> tangent cone
HRRR_LAT_0_DEG = 38.5  # LaD, latitude of origin
HRRR_LON_0_DEG = -97.5  # LoV, central meridian
HRRR_LA1_DEG = 21.138123  # lat of native pixel (row=0, col=0) center
HRRR_LO1_DEG = 237.280472  # lon of native pixel (row=0, col=0) center, 0-360 convention
HRRR_DX_M = 3000.0
HRRR_DY_M = 3000.0
HRRR_NX = 1799
HRRR_NY = 1059
HRRR_EARTH_RADIUS_M = 6371229.0  # spherical earth, standard for NCEP HRRR/RAP/NAM native grids

DEFAULT_STRIDE = 13  # 13 * 3km = 39km, closest integer stride to "~40km"

NATIVE_LCC_PROJ4 = (
    f"+proj=lcc +lat_1={HRRR_LAT_1_DEG} +lat_2={HRRR_LAT_1_DEG} "
    f"+lat_0={HRRR_LAT_0_DEG} +lon_0={HRRR_LON_0_DEG} "
    f"+x_0=0 +y_0=0 +R={HRRR_EARTH_RADIUS_M} +units=m +no_defs +type=crs"
)
# Lon/lat on the SAME spherical datum as the LCC grid above, not WGS84 --
# HRRR's own lat/lon arrays are computed against this spherical earth, so
# mixing in WGS84 here would introduce a small but confusing mismatch.
SPHERICAL_LONLAT_PROJ4 = f"+proj=longlat +R={HRRR_EARTH_RADIUS_M} +no_defs +type=crs"


@lru_cache(maxsize=None)
def _transformer(from_proj4: str, to_proj4: str) -> Transformer:
    return Transformer.from_crs(CRS.from_proj4(from_proj4), CRS.from_proj4(to_proj4), always_xy=True)


def lonlat_to_xy(lon, lat, crs_proj4: str = NATIVE_LCC_PROJ4, geodetic_proj4: str = SPHERICAL_LONLAT_PROJ4):
    """Forward-project lon/lat (deg) to native LCC x/y (m)."""
    return _transformer(geodetic_proj4, crs_proj4).transform(lon, lat)


def xy_to_lonlat(x, y, crs_proj4: str = NATIVE_LCC_PROJ4, geodetic_proj4: str = SPHERICAL_LONLAT_PROJ4):
    """Inverse-project native LCC x/y (m) to lon/lat (deg)."""
    return _transformer(crs_proj4, geodetic_proj4).transform(x, y)


@dataclass(frozen=True, eq=False)
class HrrrCoarseGrid:
    """A ~40km grid derived from block-aligned stride coarsening of the
    native HRRR grid. `crs_proj4`/`geodetic_crs_proj4` are carried as
    instance fields (not just module constants) so a grid rebuilt via
    `from_dataset`/`from_netcdf` is fully self-contained and correct even
    if this module's constants are later revised."""

    stride: int
    n_rows: int
    n_cols: int
    cell_size_m: float
    origin_x_m: float  # LCC x of native pixel (row=0, col=0) center
    origin_y_m: float  # LCC y of native pixel (row=0, col=0) center
    native_dx_m: float
    native_dy_m: float
    lat_center: np.ndarray  # shape (n_rows, n_cols)
    lon_center: np.ndarray  # shape (n_rows, n_cols)
    crs_proj4: str = NATIVE_LCC_PROJ4
    geodetic_crs_proj4: str = SPHERICAL_LONLAT_PROJ4

    @property
    def x_min_m(self) -> float:
        return self.origin_x_m - self.native_dx_m / 2.0

    @property
    def y_min_m(self) -> float:
        return self.origin_y_m - self.native_dy_m / 2.0

    def cell_bounds_xy(self, row, col):
        """(x_min, y_min, x_max, y_max) of cell (row, col) in native LCC meters."""
        x_min = self.x_min_m + col * self.cell_size_m
        y_min = self.y_min_m + row * self.cell_size_m
        return x_min, y_min, x_min + self.cell_size_m, y_min + self.cell_size_m

    def cell_polygon(self, row: int, col: int) -> Polygon:
        """One cell's lat/lon boundary, as a straight-edge polygon
        (inverse-projected from the LCC square) -- a negligible
        approximation of the true, very slightly curved cell boundary at
        this ~39km scale."""
        x_min, y_min, x_max, y_max = self.cell_bounds_xy(row, col)
        xs = np.array([x_min, x_max, x_max, x_min])
        ys = np.array([y_min, y_min, y_max, y_max])
        lons, lats = xy_to_lonlat(xs, ys, self.crs_proj4, self.geodetic_crs_proj4)
        return Polygon(zip(lons, lats))

    def cell_polygons(self, rows, cols) -> list[Polygon]:
        """Vectorized batch variant of cell_polygon for many cells at once
        (e.g. every cell a tornado track could plausibly touch)."""
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        x_min = self.x_min_m + cols * self.cell_size_m
        y_min = self.y_min_m + rows * self.cell_size_m
        x_max, y_max = x_min + self.cell_size_m, y_min + self.cell_size_m
        corner_x = np.stack([x_min, x_max, x_max, x_min], axis=-1)
        corner_y = np.stack([y_min, y_min, y_max, y_max], axis=-1)
        lon, lat = xy_to_lonlat(corner_x, corner_y, self.crs_proj4, self.geodetic_crs_proj4)
        return [Polygon(zip(lon[i], lat[i])) for i in range(len(rows))]

    def assign_cells(self, lon, lat):
        """Vectorized lon/lat -> (row, col). Out-of-bounds or non-finite
        input gets sentinel -1 (never raises)."""
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x, y = lonlat_to_xy(lon, lat, self.crs_proj4, self.geodetic_crs_proj4)
        col = np.floor((x - self.x_min_m) / self.cell_size_m)
        row = np.floor((y - self.y_min_m) / self.cell_size_m)
        valid = (
            np.isfinite(row)
            & np.isfinite(col)
            & (row >= 0)
            & (row < self.n_rows)
            & (col >= 0)
            & (col < self.n_cols)
        )
        row_idx = np.where(valid, row, -1).astype(int)
        col_idx = np.where(valid, col, -1).astype(int)
        return row_idx, col_idx

    def to_dataset(self) -> xr.Dataset:
        return xr.Dataset(
            data_vars={
                "lat": (("row", "col"), self.lat_center, {"long_name": "coarse grid cell center latitude", "units": "degrees_north"}),
                "lon": (("row", "col"), self.lon_center, {"long_name": "coarse grid cell center longitude", "units": "degrees_east"}),
            },
            coords={
                "row": np.arange(self.n_rows),
                "col": np.arange(self.n_cols),
            },
            attrs={
                "title": "HRRR-derived coarse dense-grid for tornado forecasting",
                "description": (
                    "Block-aligned stride coarsening of the native HRRR Lambert "
                    "Conformal Conic grid; each cell is an exact stride x stride "
                    "block of native HRRR pixels."
                ),
                "projection": "lambert_conformal_conic",
                "lat_1_deg": HRRR_LAT_1_DEG,
                "lat_0_deg": HRRR_LAT_0_DEG,
                "lon_0_deg": HRRR_LON_0_DEG,
                "earth_radius_m": HRRR_EARTH_RADIUS_M,
                "crs_proj4": self.crs_proj4,
                "geodetic_crs_proj4": self.geodetic_crs_proj4,
                "native_nx": HRRR_NX,
                "native_ny": HRRR_NY,
                "native_dx_m": self.native_dx_m,
                "native_dy_m": self.native_dy_m,
                "native_la1_deg": HRRR_LA1_DEG,
                "native_lo1_deg": HRRR_LO1_DEG,
                "stride": self.stride,
                "cell_size_m": self.cell_size_m,
                "origin_x_m": self.origin_x_m,
                "origin_y_m": self.origin_y_m,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> "HrrrCoarseGrid":
        a = ds.attrs
        return cls(
            stride=int(a["stride"]),
            n_rows=ds.sizes["row"],
            n_cols=ds.sizes["col"],
            cell_size_m=float(a["cell_size_m"]),
            origin_x_m=float(a["origin_x_m"]),
            origin_y_m=float(a["origin_y_m"]),
            native_dx_m=float(a["native_dx_m"]),
            native_dy_m=float(a["native_dy_m"]),
            lat_center=ds["lat"].values,
            lon_center=ds["lon"].values,
            crs_proj4=str(a["crs_proj4"]),
            geodetic_crs_proj4=str(a["geodetic_crs_proj4"]),
        )

    @classmethod
    def from_netcdf(cls, path: Path) -> "HrrrCoarseGrid":
        with xr.open_dataset(path) as ds:
            return cls.from_dataset(ds)


def build_coarse_grid(stride: int = DEFAULT_STRIDE) -> HrrrCoarseGrid:
    assert HRRR_DX_M == HRRR_DY_M, "formulas below assume square native pixels"

    # Truncate any partial block at the domain's north/east edge -- every
    # coarse cell must be an exact stride x stride block of native pixels.
    n_rows = HRRR_NY // stride
    n_cols = HRRR_NX // stride
    cell_size_m = stride * HRRR_DX_M

    # HRRR_LO1_DEG is in GRIB's 0-360 convention; subtract 360 before
    # forward-projecting.
    lo1_signed = HRRR_LO1_DEG - 360.0
    x0, y0 = lonlat_to_xy(lo1_signed, HRRR_LA1_DEG)  # native pixel (0,0) center, LCC meters

    # A coarse cell's center is the native block's first-pixel position
    # plus (stride-1)/2 native pixels (the block's midpoint), not stride/2.
    offset_m = (stride - 1) / 2.0 * HRRR_DX_M
    x_centers_1d = x0 + np.arange(n_cols) * cell_size_m + offset_m
    y_centers_1d = y0 + np.arange(n_rows) * cell_size_m + offset_m
    grid_x, grid_y = np.meshgrid(x_centers_1d, y_centers_1d)
    lon_center, lat_center = xy_to_lonlat(grid_x, grid_y)

    return HrrrCoarseGrid(
        stride=stride,
        n_rows=n_rows,
        n_cols=n_cols,
        cell_size_m=cell_size_m,
        origin_x_m=x0,
        origin_y_m=y0,
        native_dx_m=HRRR_DX_M,
        native_dy_m=HRRR_DY_M,
        lat_center=lat_center,
        lon_center=lon_center,
    )
