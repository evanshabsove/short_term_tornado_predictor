"""
Like-for-like comparison of the original year-holdout CV models (trained
on the v2 run set, models/cv_year_holdout/) against the v4 CV models
(trained on v2 + the expanded quiet dates, models/cv_year_holdout_v4/).

Why: the two CVs' reported AUC-PR values are measured on DIFFERENT test
sets -- v4's held-out years contain many extra quiet maps, which lowers
the per-cell positive rate and mechanically lowers AUC-PR/F1 even for an
identical model. This script scores BOTH sets of fold checkpoints on the
SAME test samples, two ways, per held-out year:

  * full_v4   -- the full v4 held-out year (includes the new quiet runs)
  * old_subset -- only the v4 held-out-year samples whose run was part of
                  the original v2 run set (data/interim/scaled_build_v3
                  manifest = the 3,182 runs the original CV used)

Inference only -- no retraining. UH columns are dropped (14 features).
Output: models/cv_year_holdout_v4/like_for_like.json

Usage: python scripts/compare_cv_old_vs_v4.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tornado_predictor.build_scaled import combine_staged_runs, load_manifest
from tornado_predictor.evaluate import compute_metrics, get_proba_labels
from tornado_predictor.inference import assert_feature_order_matches, load_checkpoint
from tornado_predictor.split import available_years, split_run_bins_year_holdout
from tornado_predictor.time_bins import N_BINS
from tornado_predictor.training import DenseGridDataset

REPO = Path(__file__).resolve().parents[1]
STAGING = REPO / "data" / "interim" / "scaled_build_v2"
V4_MANIFEST = STAGING / "manifest.json"
OLD_MANIFEST = REPO / "data" / "interim" / "scaled_build_v3" / "manifest.json"
OLD_CKPTS = REPO / "models" / "cv_year_holdout"
NEW_CKPTS = REPO / "models" / "cv_year_holdout_v4"
SCRATCH = REPO / "data" / "interim" / "cv_scratch"
OUT = NEW_CKPTS / "like_for_like.json"
UH = ["uh_0_2km_mean", "uh_0_2km_max", "uh_0_3km_mean", "uh_0_3km_max",
      "uh_2_5km_mean", "uh_2_5km_max", "uh_layers_available"]
KEEP = ["n_cells", "pos_rate", "auc_roc", "auc_pr", "best_f1_threshold",
        "precision_at_best_f1", "recall_at_best_f1", "f1_at_best_f1",
        "cohens_d", "mean_proba_positive", "mean_proba_negative"]


def done_runs(path):
    return {pd.Timestamp(k) for k, v in load_manifest(path).items() if v["status"] == "done"}


def score(model, ckpt, ds):
    assert_feature_order_matches(ds.feature_names, ckpt)
    y_score, y_true = get_proba_labels(model, ds)
    n = len(ds)
    return y_score.reshape(n, -1), y_true.reshape(n, -1)


def metrics(y_score, y_true, mask):
    m = compute_metrics(y_true[mask].ravel(), y_score[mask].ravel(), include_curves=False)
    out = {k: float(m[k]) for k in KEEP}
    out["n_samples"] = int(mask.sum())
    out["n_active"] = int((y_true[mask].sum(1) > 0).sum())
    return out


def main():
    v4_runs = done_runs(V4_MANIFEST)
    old_runs = done_runs(OLD_MANIFEST)
    run_bins = [(t, b) for t in sorted(v4_runs) for b in range(N_BINS)]
    results = json.loads(OUT.read_text())["folds"] if OUT.exists() else []
    done = {r["held_out_year"] for r in results}
    SCRATCH.mkdir(parents=True, exist_ok=True)

    for year in available_years(run_bins):
        if year in done:
            continue
        print(f"=== {year} ===", flush=True)
        _, test_runs = split_run_bins_year_holdout(run_bins, year, buffer_hours=24.0)
        path = SCRATCH / f"l4l_test_{year}.nc"
        try:
            combine_staged_runs(STAGING, test_runs).drop_vars(UH, errors="ignore").to_netcdf(path)
            ds = DenseGridDataset(str(path))
            in_old = np.array([t in old_runs for t, _ in test_runs])
            full = np.ones(len(ds), dtype=bool)
            row = {"held_out_year": year}
            for tag, ckdir in (("old_model", OLD_CKPTS), ("v4_model", NEW_CKPTS)):
                model, ckpt = load_checkpoint(str(ckdir / f"tornado_unet_heldout_{year}.pt"))
                s, y = score(model, ckpt, ds)
                row[tag] = {"full_v4": metrics(s, y, full), "old_subset": metrics(s, y, in_old)}
                print(f"  {tag}: full AUC-PR={row[tag]['full_v4']['auc_pr']:.4f} "
                      f"old-subset AUC-PR={row[tag]['old_subset']['auc_pr']:.4f}", flush=True)
            results.append(row)
            results.sort(key=lambda r: r["held_out_year"])
            OUT.write_text(json.dumps({"folds": results}, indent=2))
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
