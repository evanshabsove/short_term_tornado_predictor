import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tornado_predictor import live_features
from tornado_predictor.grid import build_coarse_grid
from tornado_predictor.inference import load_checkpoint, predict_proba
from tornado_predictor.live_features import (
    LIVE_FIELD_SPECS,
    PUBLISH_LATENCY_MINUTES,
    extract_live_inputs,
    latest_complete_run,
)
from tornado_predictor.training import DenseGridDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = REPO_ROOT / "data" / "interim" / "scaled_build_v2"
REFERENCE_CHECKPOINT = REPO_ROOT / "models" / "ema_study" / "seed10" / "tornado_unet_heldout_2020_ema0.9999.pt"

GRID = build_coarse_grid()
NAMES = ["cape_mean", "cape_max", "t2m_mean"]


# --- latest_complete_run (no network: is_complete is injected) ---


def _recorder(complete: set):
    probed = []

    def is_complete(ts):
        probed.append(ts)
        return ts in complete

    return is_complete, probed


def test_search_starts_at_latency_floor_and_steps_back_hourly():
    # 02:40 - 50 min = 01:50 -> floor 01:00. 01:00 is not published yet, 00:00 is.
    is_complete, probed = _recorder({pd.Timestamp("2026-10-19 00:00")})
    assert latest_complete_run(now="2026-10-19 02:40", is_complete=is_complete) == pd.Timestamp("2026-10-19 00:00")
    assert probed == [pd.Timestamp("2026-10-19 01:00"), pd.Timestamp("2026-10-19 00:00")]


def test_newest_run_chosen_when_available():
    is_complete, _ = _recorder({pd.Timestamp("2026-10-19 01:00"), pd.Timestamp("2026-10-19 00:00")})
    assert latest_complete_run(now="2026-10-19 02:40", is_complete=is_complete) == pd.Timestamp("2026-10-19 01:00")


def test_latency_boundary_is_exactly_50_minutes():
    assert PUBLISH_LATENCY_MINUTES == 50
    is_complete, probed = _recorder(set())
    latest_complete_run(now="2026-10-19 02:50", max_lookback_hours=0, is_complete=is_complete)
    assert probed == [pd.Timestamp("2026-10-19 02:00")]
    is_complete, probed = _recorder(set())
    latest_complete_run(now="2026-10-19 02:49:59", max_lookback_hours=0, is_complete=is_complete)
    assert probed == [pd.Timestamp("2026-10-19 01:00")]


def test_returns_none_when_nothing_complete_and_probes_lookback_plus_one():
    is_complete, probed = _recorder(set())
    assert latest_complete_run(now="2026-10-19 02:40", max_lookback_hours=3, is_complete=is_complete) is None
    assert len(probed) == 4


def test_timezone_aware_now_is_converted_to_naive_utc():
    is_complete, probed = _recorder(set())
    latest_complete_run(now=pd.Timestamp("2026-10-18 22:40", tz="US/Central"), max_lookback_hours=0, is_complete=is_complete)
    # 22:40 CDT = 03:40 UTC; minus 50 min = 02:50; floor -> 02:00
    assert probed == [pd.Timestamp("2026-10-19 02:00")]


# --- extract_live_inputs (extract_features faked) ---


def _fake_dataset(value_for, include_nan=False, drop=None):
    data = {}
    for i, name in enumerate(NAMES):
        if name == drop:
            continue
        arr = np.full((GRID.n_rows, GRID.n_cols), value_for(i), dtype="float64")
        if include_nan and i == 0:
            arr[3, 4] = np.nan
        data[name] = (("row", "col"), arr)
    return xr.Dataset(data)


def test_stacks_in_feature_order_with_one_plane_per_bin(monkeypatch, tmp_path):
    calls = []

    def fake_extract(init_time, fxx, grid, include_uh=True):
        calls.append((str(init_time), fxx, include_uh))
        return _fake_dataset(lambda i: 100 * fxx + i)

    monkeypatch.setattr(live_features, "extract_features", fake_extract)
    live = extract_live_inputs("2026-10-19 00:00", GRID, NAMES, cache_root=tmp_path)

    assert live.X.shape == (2, 3, GRID.n_rows, GRID.n_cols) and live.X.dtype == np.float32
    assert live.feature_names == NAMES
    # bin 0 uses fxx=0, bin 1 uses fxx=4; channels follow the requested order
    assert live.X[0, :, 0, 0].tolist() == [0, 1, 2]
    assert live.X[1, :, 0, 0].tolist() == [400, 401, 402]
    assert [c[1] for c in calls] == [0, 4]
    assert all(c[2] is False for c in calls)  # UH never pulled on the live path
    assert [b["fxx"] for b in live.bins] == [0, 4]
    assert live.bins[0]["valid_start"] == pd.Timestamp("2026-10-19 00:00")
    assert live.bins[0]["valid_end"] == live.bins[1]["valid_start"] == pd.Timestamp("2026-10-19 04:00")
    assert live.bins[1]["valid_end"] == pd.Timestamp("2026-10-19 08:00")


def test_reordered_feature_names_reorder_channels(monkeypatch, tmp_path):
    monkeypatch.setattr(live_features, "extract_features", lambda *a, **k: _fake_dataset(lambda i: i))
    live = extract_live_inputs("2026-10-19 00:00", GRID, ["t2m_mean", "cape_mean", "cape_max"], cache_root=tmp_path)
    assert live.X[0, :, 0, 0].tolist() == [2, 0, 1]


def test_nan_in_base_features_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(live_features, "extract_features", lambda *a, **k: _fake_dataset(lambda i: i, include_nan=True))
    with pytest.raises(ValueError, match="NaN"):
        extract_live_inputs("2026-10-19 00:00", GRID, NAMES, cache_root=tmp_path)


def test_missing_feature_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(live_features, "extract_features", lambda *a, **k: _fake_dataset(lambda i: i, drop="t2m_mean"))
    with pytest.raises(ValueError, match="t2m_mean"):
        extract_live_inputs("2026-10-19 00:00", GRID, NAMES, cache_root=tmp_path)


def test_cache_cleanup_removes_only_that_days_directory_and_runs_on_failure(monkeypatch, tmp_path):
    day_dir = tmp_path / "hrrr" / "20261019"
    other_dir = tmp_path / "hrrr" / "20261018"
    for d in (day_dir, other_dir):
        d.mkdir(parents=True)
        (d / "x.grib2").write_text("x")

    monkeypatch.setattr(live_features, "extract_features", lambda *a, **k: _fake_dataset(lambda i: i))
    extract_live_inputs("2026-10-19 00:00", GRID, NAMES, cache_root=tmp_path, cleanup=False)
    assert day_dir.exists()

    extract_live_inputs("2026-10-19 00:00", GRID, NAMES, cache_root=tmp_path)
    assert not day_dir.exists() and other_dir.exists()

    day_dir.mkdir()
    monkeypatch.setattr(live_features, "extract_features", lambda *a, **k: _fake_dataset(lambda i: i, include_nan=True))
    with pytest.raises(ValueError):
        extract_live_inputs("2026-10-19 00:00", GRID, NAMES, cache_root=tmp_path)
    assert not day_dir.exists()  # cleaned up even though extraction failed


def test_live_field_specs_are_the_eight_base_pulls():
    names = [n for n, _ in LIVE_FIELD_SPECS]
    assert names == ["cape", "cin", "srh_0_1km", "srh_0_3km", "shear_u_0_6km", "shear_v_0_6km", "t2m", "d2m"]


# --- parity with the training path (network + local staged data) ---

network = pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1" or not STAGING_DIR.exists() or not REFERENCE_CHECKPOINT.exists(),
    reason="set RUN_NETWORK_TESTS=1 and have the staged runs + a reference checkpoint locally",
)


def _staged_run_on_or_after(date: str) -> pd.Timestamp:
    manifest = json.loads((STAGING_DIR / "manifest.json").read_text())
    done = sorted(pd.Timestamp(k) for k, v in manifest.items() if v["status"] == "done")
    return next(t for t in done if t >= pd.Timestamp(date))


@network
@pytest.mark.parametrize("date", ["2023-01-01", "2019-06-01"])
def test_live_path_reproduces_staged_training_features_and_predictions(date, tmp_path):
    model, checkpoint = load_checkpoint(str(REFERENCE_CHECKPOINT))
    names = checkpoint["feature_names"]
    init_time = _staged_run_on_or_after(date)
    staged_path = STAGING_DIR / f"{init_time:%Y-%m-%dT%H}.nc"

    # Reduce the staged file to the 14 base variables so DenseGridDataset sees
    # exactly what training saw (staged files from different eras mix schemas).
    with xr.open_dataset(staged_path) as staged:
        staged[[v for v in staged.data_vars if not str(v).startswith("uh_")]].load().to_netcdf(tmp_path / "base.nc")
    dataset = DenseGridDataset(str(tmp_path / "base.nc"))
    assert dataset.feature_names == names  # training channel order == checkpoint order

    live = extract_live_inputs(init_time, GRID, names)

    np.testing.assert_allclose(live.X, dataset.X, rtol=1e-6, atol=1e-6)
    for bin_index in range(2):
        np.testing.assert_array_equal(predict_proba(model, live.X[bin_index]), predict_proba(model, dataset.X[bin_index]))
