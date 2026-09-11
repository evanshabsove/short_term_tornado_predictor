# Short-Term Tornado Predictor

## Goal

A short-range (0–8 hour lead time) tornado forecasting model driven by
HRRR (High-Resolution Rapid Refresh) numerical weather prediction fields,
trained against SPC (Storm Prediction Center) storm reports as ground truth.

This is a **separate but complementary** project to an existing CNN-based
tornado detection model built on radar data. That model detects tornadoes
in near-real-time from radar imagery; this project forecasts tornado
probability ahead of time from NWP model output.

## Methodology

This project follows a **dense-grid classification** approach, in the
style of Sobash et al. (2020), rather than a storm-centered approach
(e.g. TorNet):

- The prediction domain is discretized into a grid of **~40 km** cells.
- Time is discretized into two **4-hour bins** covering the 0–8h forecast
  window (bin 0 = 0–4h, bin 1 = 4–8h lead time), computed independently
  per hourly HRRR run — see `src/tornado_predictor/time_bins.py`.
- Every **(grid cell, time bin)** pair is one training sample — not just
  cells/times near an observed storm. This gives a dense, exhaustive
  labeling of the domain rather than a sparse, storm-relative one.
- **Features**: HRRR forecast fields (e.g. CAPE, shear, helicity, and
  other severe-weather-relevant fields) valid over each time bin, pulled
  via [Herbie](https://github.com/blaylockbk/Herbie).
- **Labels**: a grid cell/time bin is labeled positive if an SPC tornado
  report falls within that cell and time bin (spatial/temporal
  neighborhood rules TBD as the labeling pipeline is built out).
- The result is a dense binary (or probabilistic) classification problem
  over the full grid and forecast period, suitable for gridded
  verification against observed tornado occurrence.

## Project Structure

```
short_term_tornado_predictor/
├── src/tornado_predictor/   # package code (data pipeline, features, model)
├── data/
│   ├── raw/                 # raw HRRR pulls, raw SPC report files
│   ├── interim/             # regridded / aligned intermediate data
│   └── processed/           # final gridded sample tables ready for modeling
├── scripts/                   # CLI entry points (data download, grid building, verification)
├── notebooks/                # exploratory analysis
├── tests/
├── requirements.txt
├── pyproject.toml             # editable install for src/tornado_predictor
└── CLAUDE.md                 # methodology/context notes for AI-assisted sessions
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Status

Data collection and the spatial grid are underway:

- SPC tornado reports (2012–2022) downloaded and parsed with full track
  geometry: `scripts/download_spc_tornado_reports.py`
- HRRR retrieval via Herbie confirmed working: `scripts/smoke_test_herbie.py`
- The ~40km (39km actual) dense grid is defined by block-aligned stride
  coarsening of HRRR's native grid: `src/tornado_predictor/grid.py`,
  built via `scripts/build_grid.py`
- The 0–8h time-bin scheme (two 4-hour bins, defined independently per
  hourly HRRR run) is defined: `src/tornado_predictor/time_bins.py`

Not yet started: aggregating HRRR fields onto the grid, labeling grid
cells from tornado tracks using the grid and time bins, and modeling —
ML libraries will be added to `requirements.txt` once the feature/label
pipeline is in place.
