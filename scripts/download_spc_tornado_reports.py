"""
Download and parse the SPC (Storm Prediction Center) tornado report
database into a clean dataframe with full track geometry (start point,
end point, and a straight-line path) rather than just the touchdown point.

Source: SPC Severe Weather Database, "actual tornadoes" file (single row
per tornado track — no duplicate state-line segments).
See https://www.spc.noaa.gov/wcm/#data

The track path is approximated as a straight line between the reported
start and end points — the same approximation used by Sobash et al.
(2020) — since SPC does not provide intermediate waypoints. Downstream
grid-cell labeling should buffer `path_wkt` by (roughly) `width_yd` to
find every grid cell the tornado crossed, not just the cell it touched
down in.

Usage:
    python scripts/download_spc_tornado_reports.py
    python scripts/download_spc_tornado_reports.py --start-year 2012 --end-year 2022
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests
from shapely.geometry import LineString

SPC_URL = "https://www.spc.noaa.gov/wcm/data/1950-2025_actual_tornadoes.csv"

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "spc"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# SPC time zone codes -> hours to ADD to local time to get UTC.
# The database standardizes report times to tz=3 (CST) except when the
# original report was already in GMT (tz=9); tz=0 is "unknown" and dropped.
TZ_TO_UTC_OFFSET = {
    1: 5,  # EST
    2: 4,  # EDT
    3: 6,  # CST
    4: 5,  # CDT
    5: 7,  # MST
    6: 6,  # MDT
    7: 8,  # PST
    8: 7,  # PDT
    9: 0,  # GMT
}


def download_raw_csv(dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"Raw file already exists, skipping download: {dest}")
        return dest
    print(f"Downloading {SPC_URL}\n  -> {dest}")
    resp = requests.get(SPC_URL, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def parse_tornado_reports(raw_csv: Path, start_year: int, end_year: int) -> pd.DataFrame:
    df = pd.read_csv(raw_csv, dtype={"om": str}, low_memory=False)
    df = df[(df["yr"] >= start_year) & (df["yr"] <= end_year)].copy()

    # Drop reports with missing/placeholder start coordinates (0.0, 0.0).
    n_before = len(df)
    df = df[(df["slat"] != 0.0) | (df["slon"] != 0.0)]
    if (dropped := n_before - len(df)):
        print(f"Dropped {dropped} reports with missing start coordinates")

    # Build a UTC timestamp from date + time + tz, and drop any rows with an
    # unrecognized tz code *before* deriving anything else from df, so every
    # later column is computed from the same final set of rows/index.
    local_dt = pd.to_datetime(df["date"] + " " + df["time"], format="%Y-%m-%d %H:%M:%S")
    offset_hours = df["tz"].map(TZ_TO_UTC_OFFSET)
    n_unknown_tz = int(offset_hours.isna().sum())
    if n_unknown_tz:
        print(f"Dropped {n_unknown_tz} reports with an unrecognized tz code")
    df = df.assign(_local_dt=local_dt, _offset_hours=offset_hours).dropna(subset=["_offset_hours"])
    timestamp_utc = df["_local_dt"] + pd.to_timedelta(df["_offset_hours"], unit="h")

    # A missing end point (0.0, 0.0) means SPC didn't report one (common in
    # older records, rare in 2012+); fall back to a degenerate point track.
    missing_end = (df["elat"] == 0.0) & (df["elon"] == 0.0)
    if (n_missing_end := int(missing_end.sum())):
        print(f"{n_missing_end} reports have no end point; treating track as a point at touchdown")
    end_lat = df["elat"].where(~missing_end, df["slat"])
    end_lon = df["elon"].where(~missing_end, df["slon"])

    # mag == -9 means unrated ("EFU"); treat as missing, not a rating of -9.
    ef_rating = df["mag"].where(df["mag"] != -9)

    # Straight-line path from touchdown to lift-off (WKT, lon/lat order per
    # GIS convention). Note this is a straight-line approximation, not the
    # true (possibly curved) track.
    path_wkt = [
        LineString([(slon, slat), (elon, elat)]).wkt
        for slat, slon, elat, elon in zip(df["slat"], df["slon"], end_lat, end_lon)
    ]

    clean = pd.DataFrame(
        {
            "event_id": df["yr"].astype(str) + "_" + df["om"].astype(str),
            "timestamp_utc": timestamp_utc,
            "start_lat": df["slat"],
            "start_lon": df["slon"],
            "end_lat": end_lat,
            "end_lon": end_lon,
            "length_mi": df["len"],
            "width_yd": df["wid"],
            "path_wkt": path_wkt,
            "ef_rating": ef_rating,
            "state": df["st"],
        }
    )
    return clean.sort_values("timestamp_utc").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--force-download", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    raw_csv = RAW_DIR / "1950-2025_actual_tornadoes.csv"
    download_raw_csv(raw_csv, force=args.force_download)

    clean = parse_tornado_reports(raw_csv, args.start_year, args.end_year)

    out_path = PROCESSED_DIR / f"spc_tornado_reports_{args.start_year}_{args.end_year}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)

    print(f"\nParsed {len(clean)} tornado reports ({args.start_year}-{args.end_year})")
    print(clean.head())
    print(f"\nSaved clean dataframe to {out_path}")


if __name__ == "__main__":
    main()
