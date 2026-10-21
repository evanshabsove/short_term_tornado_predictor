"""
SPC tornado reports as GeoJSON for the forecast API's GET /reports (ticket 8).

Source: data/processed/spc_tornado_reports_2014_2025.csv (the table the
labeling pipeline uses). Each report becomes up to two features sharing an
`event_id`: a `touchdown` Point and, when the report has a real track, a
`track` LineString. The track is SPC's straight line from touchdown to lift-off
(not a surveyed path), and its single timestamp is the touchdown time -- SPC
has no per-point timing, so the track does not say where the tornado was at any
other moment.

Window semantics: half-open [start, end) on the touchdown timestamp, UTC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import mapping

DEFAULT_REPORTS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "spc_tornado_reports_2014_2025.csv"
MAX_SPAN_DAYS = 31
SCHEMA_VERSION = 1


def _naive_utc(ts) -> pd.Timestamp:
    """Project convention: naive timestamps, treated as UTC (tz-aware input is converted)."""
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def _iso_z(ts) -> str:
    return f"{_naive_utc(ts):%Y-%m-%dT%H:%M:%S}Z"


class ReportsCatalog:
    """The reports table, loaded once. Timestamps are naive UTC."""

    def __init__(self, df: pd.DataFrame):
        required = {"event_id", "timestamp_utc", "start_lat", "start_lon", "length_mi", "width_yd", "path_wkt", "ef_rating", "state"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"reports table is missing columns {sorted(missing)}")
        df = df.copy()
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
        self.df = df.sort_values("timestamp_utc").reset_index(drop=True)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_REPORTS_PATH) -> "ReportsCatalog":
        return cls(pd.read_csv(path))

    @property
    def first(self) -> pd.Timestamp:
        return self.df["timestamp_utc"].iloc[0]

    @property
    def last(self) -> pd.Timestamp:
        return self.df["timestamp_utc"].iloc[-1]

    def between(self, start, end) -> pd.DataFrame:
        """Reports with start <= touchdown time < end."""
        t = self.df["timestamp_utc"]
        return self.df[(t >= _naive_utc(start)) & (t < _naive_utc(end))]

    def geojson(self, start, end) -> dict:
        rows = self.between(start, end)
        features = []
        for r in rows.itertuples(index=False):
            props = {
                "event_id": r.event_id,
                "timestamp_utc": _iso_z(r.timestamp_utc),
                "ef_rating": None if pd.isna(r.ef_rating) else int(r.ef_rating),
                "state": r.state,
                "length_mi": float(r.length_mi),
                "width_yd": int(r.width_yd),
            }
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(r.start_lon), float(r.start_lat)]},
                "properties": {"kind": "touchdown", **props},
            })
            line = wkt.loads(r.path_wkt)
            coords = np.asarray(line.coords)
            if len(coords) >= 2 and not np.allclose(coords[0], coords[-1]):  # a zero-length track is just the touchdown point
                features.append({"type": "Feature", "geometry": mapping(line), "properties": {"kind": "track", **props}})
        return {
            "type": "FeatureCollection",
            "schema_version": SCHEMA_VERSION,
            "start": _iso_z(start),
            "end": _iso_z(end),
            "count": int(len(rows)),
            "features": features,
        }
