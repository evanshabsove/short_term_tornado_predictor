"""
Leave-one-year-out cross-validation for the project's best-known
training config (TornadoUNet, base_channels=16, alpha=0.25, gamma=2.0,
quiet_keep_fraction=0.2, 50 epochs, seed=0 -- the exact recipe behind
models/unet/tornado_unet.pt).

Why: every AUC-PR reported so far comes from one fixed train/val(/test)
split. Comparing the 2-way val (0.0398) against the 3-way test (0.0836)
for the SAME frozen checkpoint showed a >2x swing -- with only ~2,344
usable archive dates, which specific storms land in which split carries
real variance that has never been quantified. This script holds out one
full calendar year at a time, retrains the identical config on
everything else, and evaluates on the held-out year -- giving a
mean +/- spread across many independent folds instead of trusting any
single split.

Reuses the already-staged v2 runs (data/interim/scaled_build_v2/,
14 features, matching tornado_unet.pt) -- no new HRRR pulls needed.
Each fold's train/test netCDF is built via build_scaled.combine_staged_runs,
written to a scratch file just long enough to load via DenseGridDataset,
then deleted -- keeps disk usage bounded to ~one fold at a time instead
of accumulating a full extra copy of the dataset per fold.

Resumable: re-running skips any held-out year already present in
--results (same spirit as the manifest-based build scripts). Supports
--years to run a subset (e.g. a single year for a feasibility/timing
check before committing to the full multi-hour run across every year).

Usage:
    python scripts/cross_validate_year.py --years 2022   # feasibility check, one fold
    python scripts/cross_validate_year.py                # full run, all available years
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from tornado_predictor.build_scaled import combine_staged_runs, load_manifest
from tornado_predictor.evaluate import evaluate_model
from tornado_predictor.split import available_years, split_run_bins_year_holdout
from tornado_predictor.time_bins import N_BINS
from tornado_predictor.training import DenseGridDataset, train_model
from tornado_predictor.unet import TornadoUNet

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING_DIR = REPO_ROOT / "data" / "interim" / "scaled_build_v2"
DEFAULT_MANIFEST_PATH = DEFAULT_STAGING_DIR / "manifest.json"
DEFAULT_SCRATCH_DIR = REPO_ROOT / "data" / "interim" / "cv_scratch"
DEFAULT_OUT_DIR = REPO_ROOT / "models" / "cv_year_holdout"
DEFAULT_RESULTS_PATH = DEFAULT_OUT_DIR / "results.json"

BASE_CHANNELS = 16
ALPHA = 0.25
GAMMA = 2.0
LR = 1e-3
BATCH_SIZE = 2
QUIET_KEEP_FRACTION = 0.2
EPOCHS = 50
SEED = 0
BUFFER_HOURS = 24.0


def completed_run_bins(manifest: dict) -> list[tuple]:
    completed_runs = [pd.Timestamp(k) for k, v in manifest.items() if v["status"] == "done"]
    return [(init_time, bin_index) for init_time in completed_runs for bin_index in range(N_BINS)]


def summarize(results: list[dict]) -> dict:
    import numpy as np

    auc_prs = [r["auc_pr"] for r in results]
    return {
        "n_folds": len(results),
        "auc_pr_mean": float(np.mean(auc_prs)),
        "auc_pr_std": float(np.std(auc_prs)),
        "auc_pr_min": float(np.min(auc_prs)),
        "auc_pr_max": float(np.max(auc_prs)),
    }


def run_one_fold(held_out_year: int, run_bins: list[tuple], staging_dir: Path, scratch_dir: Path, out_dir: Path, epochs: int, drop_columns: list[str] | None = None, seed: int = SEED, ema_decays: tuple[float, ...] = ()) -> dict:
    train_runs, test_runs = split_run_bins_year_holdout(run_bins, held_out_year, buffer_hours=BUFFER_HOURS)

    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_train = scratch_dir / f"train_heldout_{held_out_year}.nc"
    scratch_test = scratch_dir / f"test_heldout_{held_out_year}.nc"

    try:
        combine_staged_runs(staging_dir, train_runs).drop_vars(drop_columns or [], errors="ignore").to_netcdf(scratch_train)
        combine_staged_runs(staging_dir, test_runs).drop_vars(drop_columns or [], errors="ignore").to_netcdf(scratch_test)

        train_ds = DenseGridDataset(str(scratch_train))
        test_ds = DenseGridDataset(str(scratch_test))
        # Channel order follows the FIRST staged run of each combined set, and staged files from different sources order their
        # variables differently (e.g. UH columns sit after shear in the *_v3 augmented copies but before it in natively-pulled
        # files), so train and test can disagree. Align the test set to the training order so evaluation feeds the model the
        # channels it was trained on. (A no-op whenever the orders already match, e.g. every 14-feature study.)
        if test_ds.feature_names != train_ds.feature_names:
            assert set(test_ds.feature_names) == set(train_ds.feature_names), "train/test feature sets differ"
            idx = [test_ds.feature_names.index(n) for n in train_ds.feature_names]
            test_ds.X = test_ds.X[:, idx]
            test_ds.feature_names = list(train_ds.feature_names)
            print("  (aligned test-set channel order to the training set's)", flush=True)
        print(f"  train: {len(train_ds)} ({int(train_ds.is_active.sum())} active) | "
              f"test (held-out {held_out_year}): {len(test_ds)} ({int(test_ds.is_active.sum())} active)")

        t0 = time.time()
        torch.manual_seed(seed)
        model = TornadoUNet(in_channels=len(train_ds.feature_names), base_channels=BASE_CHANNELS)
        train_result = train_model(
            train_ds, model, val_dataset=test_ds, epochs=epochs, batch_size=BATCH_SIZE,
            lr=LR, alpha=ALPHA, gamma=GAMMA, quiet_keep_fraction=QUIET_KEEP_FRACTION, seed=seed, ema_decays=tuple(ema_decays),
        )
        metrics = evaluate_model(model, test_ds, include_curves=False)
        ema_metrics = {}
        for decay, sd in train_result.get("ema_state_dicts", {}).items():
            ema_model = TornadoUNet(in_channels=len(train_ds.feature_names), base_channels=BASE_CHANNELS)
            ema_model.load_state_dict(sd)
            ema_model.eval()
            ema_metrics[str(decay)] = evaluate_model(ema_model, test_ds, include_curves=False)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_class": "TornadoUNet", "model_state_dict": sd, "ema_decay": decay,
                "training_hyperparameters": train_result["hyperparameters"],
                "model_hyperparameters": {"in_channels": len(train_ds.feature_names), "base_channels": BASE_CHANNELS},
                "feature_names": train_ds.feature_names,
            }, out_dir / f"tornado_unet_heldout_{held_out_year}_ema{decay}.pt")
        elapsed = time.time() - t0

        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"tornado_unet_heldout_{held_out_year}.pt"
        torch.save({
            "model_class": "TornadoUNet",
            "model_state_dict": model.state_dict(),
            "loss_history": train_result["loss_history"],
            "val_loss_history": train_result["val_loss_history"],
            "training_hyperparameters": train_result["hyperparameters"],
            "model_hyperparameters": {"in_channels": len(train_ds.feature_names), "base_channels": BASE_CHANNELS},
            "feature_names": train_ds.feature_names,
        }, ckpt_path)

        return {
            "held_out_year": held_out_year,
            "seed": seed,
            "n_train": len(train_ds),
            "n_train_active": int(train_ds.is_active.sum()),
            "n_test": len(test_ds),
            "n_test_active": int(test_ds.is_active.sum()),
            "n_test_positive_cells": int(test_ds.y.sum()),
            "elapsed_sec": elapsed,
            "final_train_loss": train_result["loss_history"][-1],
            "final_val_loss": train_result["val_loss_history"][-1],
            "checkpoint": str(ckpt_path),
            **({"ema_metrics": ema_metrics} if ema_metrics else {}),
            **metrics,
        }
    finally:
        scratch_train.unlink(missing_ok=True)
        scratch_test.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--ema-decays", type=float, nargs="*", default=[], help="also track an exponential moving average of the weights at each decay; saves tornado_unet_heldout_<year>_ema<decay>.pt and reports ema_metrics (the plain model is saved/evaluated as always)")
    parser.add_argument("--seed", type=int, default=SEED, help="seeds both model init (torch.manual_seed) and train_model's quiet-map sampling/shuffling; use a separate --out-dir/--results per seed")
    parser.add_argument("--drop-columns", nargs="+", default=None, help="variables to drop from each fold's combined data (e.g. the UH columns, which are NaN-filled for runs staged without them)")
    parser.add_argument("--years", type=int, nargs="+", default=None, help="restrict to these held-out years (default: every year present in the manifest)")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    run_bins = completed_run_bins(manifest)
    years = args.years if args.years else available_years(run_bins)
    print(f"{len(run_bins)} run_bins across {len(available_years(run_bins))} years available; running folds for: {years}")

    results = json.loads(args.results.read_text())["folds"] if args.results.exists() else []
    done_years = {r["held_out_year"] for r in results}

    for year in years:
        if year in done_years:
            print(f"\n=== held-out year {year}: already done, skipping ===")
            continue
        print(f"\n=== held-out year {year} ===")
        fold_result = run_one_fold(year, run_bins, args.staging_dir, args.scratch_dir, args.out_dir, args.epochs, args.drop_columns, args.seed, tuple(args.ema_decays))
        print(f"  AUC-PR={fold_result['auc_pr']:.4f} AUC-ROC={fold_result['auc_roc']:.4f} "
              f"cohens_d={fold_result['cohens_d']:.2f}  ({fold_result['elapsed_sec']:.0f}s)")
        results.append(fold_result)
        results.sort(key=lambda r: r["held_out_year"])
        args.out_dir.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps({"folds": results, "summary": summarize(results)}, indent=2, default=str))

    summary = summarize(results)
    print(f"\n=== summary across {summary['n_folds']} folds ===")
    print(f"AUC-PR mean={summary['auc_pr_mean']:.4f} std={summary['auc_pr_std']:.4f} "
          f"min={summary['auc_pr_min']:.4f} max={summary['auc_pr_max']:.4f}")
    print(f"Saved to {args.results}")


if __name__ == "__main__":
    main()
