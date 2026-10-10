"""
Evaluate the seed study (scripts/run_seed_study.sh): for each fold, score
seed 0 (the existing v4 CV model), seeds 1 and 2 (models/seed_study/), and
the old-recipe model on the SAME full v4 held-out year, plus three
ensembles (mean predicted probability):
  ens_seeds  = mean of the v4 seeds available (0, 1, 2)
  ens_old_v4 = mean of old model and v4 seed 0
  ens_all    = mean of old + every v4 seed
Per config: cell-pooled metrics (AUC-PR, best-F1, precision/recall), map-level
AUC-ROC + quiet-map false-alarm rate, and AUC-PR under 1-cell-dilated labels.

Reuses cached labels/meta/old/seed-0 scores from data/interim/diag_scores/
(written by scripts/diagnose_sample_mix.py); caches new seed scores there too.
Seeds whose checkpoint does not exist yet are skipped, so this can be tested
on partial results.

Usage: python scripts/seed_study_eval.py [--years 2016 2020 2024] [--out PATH]
Output: models/seed_study/seed_study_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_sample_mix as dsm  # noqa: E402  (reuses its metric helpers + paths)

from tornado_predictor.build_scaled import combine_staged_runs, load_manifest  # noqa: E402
from tornado_predictor.evaluate import get_proba_labels  # noqa: E402
from tornado_predictor.inference import assert_feature_order_matches, load_checkpoint  # noqa: E402
from tornado_predictor.split import split_run_bins_year_holdout  # noqa: E402
from tornado_predictor.time_bins import N_BINS  # noqa: E402
from tornado_predictor.training import DenseGridDataset  # noqa: E402

REPO = dsm.REPO
SEED_DIR = REPO / "models" / "seed_study"
SEEDS = (1, 2)


def score_new_seeds(year, run_bins):
    todo = [s for s in SEEDS
            if (SEED_DIR / f"seed{s}" / f"tornado_unet_heldout_{year}.pt").exists()
            and not (dsm.CACHE / f"seed{s}_{year}.npy").exists()]
    if not todo:
        return
    print(f"[score] {year}: seeds {todo}", flush=True)
    _, test_runs = split_run_bins_year_holdout(run_bins, year, buffer_hours=24.0)
    combined = combine_staged_runs(dsm.STAGING, test_runs)
    meta = np.load(dsm.CACHE / f"meta_{year}.npz")
    init = pd.to_datetime(combined["init_time"].values).values.astype("datetime64[s]").astype("int64")
    assert np.array_equal(init, meta["init_time"]), "test-sample order differs from cached diag scores"
    path = dsm.SCRATCH / f"seedstudy_test_{year}.nc"
    try:
        combined.drop_vars(dsm.UH, errors="ignore").to_netcdf(path)
        ds = DenseGridDataset(str(path))
        for s in todo:
            model, ckpt = load_checkpoint(str(SEED_DIR / f"seed{s}" / f"tornado_unet_heldout_{year}.pt"))
            assert_feature_order_matches(ds.feature_names, ckpt)
            sc, _ = get_proba_labels(model, ds)
            np.save(dsm.CACHE / f"seed{s}_{year}.npy", sc.reshape(len(ds), 81, 138).astype("float16"))
    finally:
        path.unlink(missing_ok=True)


def evaluate_config(s, y, quiet_run, y_dil1):
    out = dsm.cell_metrics(y, s)
    ml = dsm.map_level(s, y, quiet_run)
    out.update({k: ml[k] for k in ("map_auc_roc", "map_auc_pr", "far_all_quiet@recall0.5", "far_all_quiet@recall0.8",
                                   "far_quietdate_runs@recall0.5", "far_quiet_bins_of_active_runs@recall0.5") if k in ml})
    out["auc_pr_k1"] = dsm.cell_metrics(y_dil1, s)["auc_pr"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2016, 2020, 2024])
    ap.add_argument("--out", type=Path, default=SEED_DIR / "seed_study_results.json")
    args = ap.parse_args()
    manifest = load_manifest(dsm.STAGING / "manifest.json")
    run_bins = [(pd.Timestamp(k), b) for k, v in sorted(manifest.items()) if v["status"] == "done" for b in range(N_BINS)]

    results = []
    for year in args.years:
        score_new_seeds(year, run_bins)
        meta = np.load(dsm.CACHE / f"meta_{year}.npz")
        y = meta["label"].astype("float32")
        act = y.sum((1, 2)) > 0
        quiet_run = ~pd.Series(act).groupby(meta["init_time"]).transform("any").values
        y_dil1 = maximum_filter(y, size=(1, 3, 3), mode="constant")
        scores = {"old": np.load(dsm.CACHE / f"old_{year}.npy").astype("float32"),
                  "seed0": np.load(dsm.CACHE / f"v4_{year}.npy").astype("float32")}
        for s in SEEDS:
            p = dsm.CACHE / f"seed{s}_{year}.npy"
            if p.exists():
                scores[f"seed{s}"] = np.load(p).astype("float32")
        v4_keys = [k for k in scores if k.startswith("seed")]
        scores["ens_seeds"] = np.mean([scores[k] for k in v4_keys], axis=0)
        scores["ens_old_v4"] = np.mean([scores["old"], scores["seed0"]], axis=0)
        scores["ens_all"] = np.mean([scores["old"]] + [scores[k] for k in v4_keys], axis=0)
        row = {"year": year, "v4_seeds_available": v4_keys,
               "n_samples": int(len(y)), "pos_rate": float(y.mean())}
        for name, s in scores.items():
            row[name] = evaluate_config(s, y, quiet_run, y_dil1)
            print(f"[eval] {year} {name:11s} AUC-PR={row[name]['auc_pr']:.4f} F1={row[name]['f1_at_best_f1']:.3f} "
                  f"mapAUC={row[name].get('map_auc_roc', float('nan')):.3f}", flush=True)
        results.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"folds": results}, indent=1))


if __name__ == "__main__":
    main()
