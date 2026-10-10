"""
Evaluation-only diagnostics for the year-holdout CV models (old = v2-run
set, models/cv_year_holdout/; v4 = expanded quiet set,
models/cv_year_holdout_v4/), scored on the full v4 held-out years.

Three questions (see CLAUDE.md, "Post-v4 diagnostics"):
  1. Sample-mix artifacts: does skill hold when restricted to init hour
     20Z only (where quiet and active samples are both common) and per
     forecast bin (bin 0 vs bin 1)? Tests the time-of-day-shortcut and
     short-lead-skew concerns.
  2. Map-level discrimination: does max predicted probability separate
     active maps from quiet maps (AUC-ROC, AUC-PR, and quiet-map false-
     alarm rate at fixed recall of active maps)? This is what the extra
     quiet training data was meant to improve; cell-pooled AUC-PR is a
     blunt instrument for it. Also: spatial skill *within* active maps.
  3. Neighborhood-dilated verification: a cell counts as positive if a
     tornado touched it or any cell within k cells (k=1,2; ~39/78 km).
     Model scores are unchanged -- only the verification label is relaxed.

Stage A scores each fold once and caches float16 per-cell probabilities
(+labels, init_time, bin_index) in data/interim/diag_scores/, so Stage B
(metrics) can be re-run/extended without re-scoring. Resumable.

Usage: python scripts/diagnose_sample_mix.py
Output: models/cv_year_holdout_v4/sample_mix_diagnostics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from sklearn.metrics import average_precision_score, roc_auc_score

from tornado_predictor.build_scaled import combine_staged_runs, load_manifest
from tornado_predictor.evaluate import compute_metrics, get_proba_labels
from tornado_predictor.inference import assert_feature_order_matches, load_checkpoint
from tornado_predictor.split import available_years, split_run_bins_year_holdout
from tornado_predictor.time_bins import N_BINS
from tornado_predictor.training import DenseGridDataset

REPO = Path(__file__).resolve().parents[1]
STAGING = REPO / "data" / "interim" / "scaled_build_v2"
CKPTS = {"old": REPO / "models" / "cv_year_holdout", "v4": REPO / "models" / "cv_year_holdout_v4"}
SCRATCH = REPO / "data" / "interim" / "cv_scratch"
CACHE = REPO / "data" / "interim" / "diag_scores"
OUT = REPO / "models" / "cv_year_holdout_v4" / "sample_mix_diagnostics.json"
UH = ["uh_0_2km_mean", "uh_0_2km_max", "uh_0_3km_mean", "uh_0_3km_max",
      "uh_2_5km_mean", "uh_2_5km_max", "uh_layers_available"]
KEEP = ["pos_rate", "auc_pr", "auc_roc", "f1_at_best_f1", "precision_at_best_f1", "recall_at_best_f1"]
RECALLS = (0.5, 0.8)


def stage_a(years, run_bins):
    CACHE.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for year in years:
        if all((CACHE / f"{t}_{year}.npy").exists() for t in CKPTS) and (CACHE / f"meta_{year}.npz").exists():
            continue
        print(f"[score] {year}", flush=True)
        _, test_runs = split_run_bins_year_holdout(run_bins, year, buffer_hours=24.0)
        combined = combine_staged_runs(STAGING, test_runs)
        init = pd.to_datetime(combined["init_time"].values)
        bins = combined["bin_index"].values
        path = SCRATCH / f"diag_test_{year}.nc"
        try:
            combined.drop_vars(UH, errors="ignore").to_netcdf(path)
            ds = DenseGridDataset(str(path))
            n = len(ds)
            for tag, d in CKPTS.items():
                model, ckpt = load_checkpoint(str(d / f"tornado_unet_heldout_{year}.pt"))
                assert_feature_order_matches(ds.feature_names, ckpt)
                s, y = get_proba_labels(model, ds)
                np.save(CACHE / f"{tag}_{year}.npy", s.reshape(n, 81, 138).astype("float16"))
            np.savez(CACHE / f"meta_{year}.npz", label=ds.y.astype("uint8"),
                     init_hour=init.hour.values, bin_index=bins,
                     init_time=init.values.astype("datetime64[s]").astype("int64"))
        finally:
            path.unlink(missing_ok=True)


def cell_metrics(y, s):
    m = compute_metrics(y.ravel(), s.ravel(), include_curves=False)
    out = {k: float(m[k]) for k in KEEP}
    out["n_samples"] = int(y.shape[0])
    out["n_pos_cells"] = int(y.sum())
    out["lift"] = out["auc_pr"] / out["pos_rate"] if out["pos_rate"] > 0 else float("nan")
    return out


def map_level(s, y, lab_quiet_run):
    act = y.sum((1, 2)) > 0
    smax = s.max((1, 2)).astype("float64")
    out = {"n_maps": int(len(act)), "n_active": int(act.sum()), "active_frac": float(act.mean())}
    if act.sum() == 0 or (~act).sum() == 0:
        return out
    out["map_auc_roc"] = float(roc_auc_score(act, smax))
    out["map_auc_pr"] = float(average_precision_score(act, smax))
    out["map_lift"] = out["map_auc_pr"] / out["active_frac"]
    for r in RECALLS:
        thr = np.quantile(smax[act], 1 - r)
        out[f"far_all_quiet@recall{r}"] = float((smax[~act] >= thr).mean())
        qr = ~act & lab_quiet_run
        qa = ~act & ~lab_quiet_run
        out[f"far_quietdate_runs@recall{r}"] = float((smax[qr] >= thr).mean()) if qr.sum() else float("nan")
        out[f"far_quiet_bins_of_active_runs@recall{r}"] = float((smax[qa] >= thr).mean()) if qa.sum() else float("nan")
    # spatial skill within active maps only
    ya, sa = y[act], s[act]
    out["within_active_auc_pr"] = float(average_precision_score(ya.ravel(), sa.ravel().astype("float64")))
    out["within_active_pos_rate"] = float(ya.mean())
    out["within_active_lift"] = out["within_active_auc_pr"] / out["within_active_pos_rate"]
    return out


def stage_b(years):
    results = {}
    for year in years:
        meta = np.load(CACHE / f"meta_{year}.npz")
        y = meta["label"].astype("float32")
        hr, bi, it = meta["init_hour"], meta["bin_index"], meta["init_time"]
        act = y.sum((1, 2)) > 0
        # a run is a "quiet-date run" if neither of its bins is active
        run_active = pd.Series(act).groupby(it).transform("any").values
        quiet_run = ~run_active
        subsets = {
            "all": np.ones(len(y), bool), "init20z": hr == 20, "init_not20z": hr != 20,
            "bin0": bi == 0, "bin1": bi == 1, "init20z_bin0": (hr == 20) & (bi == 0),
            "init20z_bin1": (hr == 20) & (bi == 1),
        }
        row = {"year": year, "mix": {
            "frac_init20z": float((hr == 20).mean()),
            "active_frac_20z": float(act[hr == 20].mean()) if (hr == 20).any() else float("nan"),
            "active_frac_not20z": float(act[hr != 20].mean()) if (hr != 20).any() else float("nan"),
            "active_frac_bin0": float(act[bi == 0].mean()), "active_frac_bin1": float(act[bi == 1].mean()),
            "frac_quiet_date_runs": float(quiet_run.mean()),
        }}
        for tag in CKPTS:
            s = np.load(CACHE / f"{tag}_{year}.npy").astype("float32")
            r = {"subsets": {}, "dilation": {}}
            for name, m in subsets.items():
                if m.sum() == 0 or y[m].sum() == 0:
                    r["subsets"][name] = {"n_samples": int(m.sum()), "n_pos_cells": 0}
                    continue
                r["subsets"][name] = cell_metrics(y[m], s[m])
            r["map_level"] = map_level(s, y, quiet_run)
            for k in (0, 1, 2):
                yd = y if k == 0 else maximum_filter(y, size=(1, 2 * k + 1, 2 * k + 1), mode="constant")
                r["dilation"][f"k{k}"] = cell_metrics(yd, s)
            row[tag] = r
        results[year] = row
        print(f"[metrics] {year}: v4 all={row['v4']['subsets']['all']['auc_pr']:.4f} "
              f"20Z={row['v4']['subsets']['init20z'].get('auc_pr', float('nan')):.4f} "
              f"map_auc={row['v4']['map_level'].get('map_auc_roc', float('nan')):.3f} "
              f"k1={row['v4']['dilation']['k1']['auc_pr']:.4f}", flush=True)
    OUT.write_text(json.dumps({"folds": [results[y] for y in sorted(results)]}, indent=1))


def main():
    manifest = load_manifest(STAGING / "manifest.json")
    run_bins = [(pd.Timestamp(k), b) for k, v in sorted(manifest.items()) if v["status"] == "done" for b in range(N_BINS)]
    years = available_years(run_bins)
    stage_a(years, run_bins)
    stage_b(years)


if __name__ == "__main__":
    main()
