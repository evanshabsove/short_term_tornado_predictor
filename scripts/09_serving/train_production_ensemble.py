"""
Trains the production model bundle: a 5-member ensemble of TornadoUNet
(base_channels=16), every member trained on ALL completed runs (2014-2025,
no held-out year, no validation set), with the EMA(0.9999) weights saved
per member. Writes models/production/member_seed<N>.pt plus manifest.json,
which ensemble.load_bundle verifies and the forecast API serves.

Recipe = the project's established one, unchanged (CLAUDE.md, "Seed-variance
study and ensembles" / EMA study): alpha=0.25, gamma=2.0, lr=1e-3, batch 2,
quiet_keep_fraction=0.2, 50 epochs, 14 base features (UH columns dropped,
as the CV scripts do), unmasked loss. Seeds 20-24 (10-15 are used by the
EMA study).

There is NO held-out score for these exact weights -- every year is in the
training set. The manifest's expected_skill block points at the 12-fold
year-holdout CV of the same recipe; do not quote an in-sample number as
skill (check_production_bundle.py labels its in-sample metrics as such).

Resumable: a member whose checkpoint already exists is skipped, so an
interrupted run continues where it stopped. All members share one dataset
load. The manifest is (re)written whenever every member exists.

Usage:
    python scripts/09_serving/train_production_ensemble.py                       # full run (~4-5 h CPU)
    python scripts/09_serving/train_production_ensemble.py --n-members 2 --epochs 1 --max-runs 40 \
        --out-dir <scratch>/prod_smoke --scratch-dir <scratch>/prod_scratch      # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tornado_predictor.build_scaled import _drop_columns_and_verify, combine_staged_runs, load_manifest
from tornado_predictor.ensemble import compute_bundle_id, file_sha256
from tornado_predictor.inference import load_checkpoint
from tornado_predictor.time_bins import N_BINS
from tornado_predictor.training import DenseGridDataset, train_model
from tornado_predictor.unet import TornadoUNet

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGING_DIR = REPO_ROOT / "data" / "interim" / "scaled_build_v2"
DEFAULT_SOURCE_MANIFEST = DEFAULT_STAGING_DIR / "manifest.json"
DEFAULT_OUT_DIR = REPO_ROOT / "models" / "production"
DEFAULT_SCRATCH_DIR = REPO_ROOT / "data" / "interim" / "production_scratch"
# Any existing 14-feature CV checkpoint: its feature_names are the order every
# trained-and-evaluated model in this project uses.
REFERENCE_CHECKPOINT = REPO_ROOT / "models" / "ema_study" / "seed10" / "tornado_unet_heldout_2020_ema0.9999.pt"

BASE_CHANNELS = 16
ALPHA = 0.25
GAMMA = 2.0
LR = 1e-3
BATCH_SIZE = 2
QUIET_KEEP_FRACTION = 0.2
EPOCHS = 50
EMA_DECAY = 0.9999
FIRST_SEED = 20
N_MEMBERS = 5
OLD_SPC_CEILING = pd.Timestamp("2025-09-09")  # the build's original label ceiling; see manifest note


def member_path(out_dir: Path, seed: int) -> Path:
    return out_dir / f"member_seed{seed}.pt"


def sidecar_path(out_dir: Path, seed: int) -> Path:
    return out_dir / f"member_seed{seed}.json"


def done_runs(source_manifest: dict) -> list[pd.Timestamp]:
    return sorted(pd.Timestamp(k) for k, v in source_manifest.items() if v["status"] == "done")


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def build_dataset(args, runs: list[pd.Timestamp], reference_names: list[str]) -> tuple[DenseGridDataset, dict]:
    """Combines every staged run into one DenseGridDataset (via a scratch
    netCDF, since DenseGridDataset reads a path -- same approach as the CV
    script), dropping the UH columns and verifying nothing else is NaN."""
    run_bins = [(t, b) for t in runs for b in range(N_BINS)]
    print(f"combining {len(runs)} runs ({len(run_bins)} samples)...", flush=True)
    t0 = time.time()
    combined = combine_staged_runs(args.staging_dir, run_bins)
    uh_columns = [v for v in combined.data_vars if str(v).startswith("uh_")]
    combined = _drop_columns_and_verify(combined, uh_columns)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.scratch_dir / "all_runs.nc"
    combined.to_netcdf(scratch)
    del combined
    dataset = DenseGridDataset(str(scratch))
    scratch.unlink(missing_ok=True)

    if dataset.feature_names != reference_names:
        raise SystemExit(
            f"feature order {dataset.feature_names} differs from the reference checkpoint's {reference_names} "
            "-- refusing to train: members would not match the order every earlier model uses"
        )

    inits = pd.to_datetime([t for t, _ in run_bins[::N_BINS]])
    summary = {
        "source_manifest": str(args.source_manifest),
        "n_runs": len(runs),
        "n_samples": len(dataset),
        "n_active_samples": int(dataset.is_active.sum()),
        "active_fraction": float(dataset.is_active.mean()),
        "n_positive_cells": int(dataset.y.sum()),
        "years": sorted({int(y) for y in inits.year}),
        "first_init": str(inits.min()),
        "last_init": str(inits.max()),
        "n_runs_after_2025_09_09": int((inits > OLD_SPC_CEILING).sum()),
        "note_post_ceiling_runs": (
            "Runs after 2025-09-09 are active-date runs only (quiet dates were selected up to the old "
            "label ceiling); 36 of ~4,900 runs, negligible but recorded."
        ),
        "dropped_columns": uh_columns,
        "load_seconds": time.time() - t0,
    }
    print(f"  {summary['n_samples']} samples, {summary['n_active_samples']} active "
          f"({summary['active_fraction']:.1%}), loaded in {summary['load_seconds']:.0f}s", flush=True)
    return dataset, summary


def train_member(dataset: DenseGridDataset, seed: int, args, data_summary: dict) -> None:
    print(f"\n=== member seed {seed}: training {args.epochs} epochs ===", flush=True)
    t0 = time.time()
    torch.manual_seed(seed)
    model = TornadoUNet(in_channels=len(dataset.feature_names), base_channels=BASE_CHANNELS)
    result = train_model(
        dataset, model, epochs=args.epochs, batch_size=BATCH_SIZE, lr=LR, alpha=ALPHA, gamma=GAMMA,
        quiet_keep_fraction=QUIET_KEEP_FRACTION, seed=seed, ema_decays=(EMA_DECAY,),
    )
    elapsed = time.time() - t0
    checkpoint = {
        "model_class": "TornadoUNet",
        "model_state_dict": result["ema_state_dicts"][EMA_DECAY],
        "ema_decay": EMA_DECAY,
        "loss_history": result["loss_history"],
        "training_hyperparameters": result["hyperparameters"],
        "model_hyperparameters": {"in_channels": len(dataset.feature_names), "base_channels": BASE_CHANNELS},
        "feature_names": dataset.feature_names,
        "training_data": {k: data_summary[k] for k in ("n_runs", "n_samples", "years", "first_init", "last_init")},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = member_path(args.out_dir, seed)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)  # atomic: a half-written checkpoint never looks complete
    sidecar_path(args.out_dir, seed).write_text(json.dumps({
        "seed": seed,
        "epochs": args.epochs,
        "train_seconds": elapsed,
        "first_epoch_loss": result["loss_history"][0],
        "final_train_loss": result["loss_history"][-1],
    }, indent=2))
    print(f"  seed {seed} done in {elapsed / 60:.1f} min; loss {result['loss_history'][0]:.6f} -> "
          f"{result['loss_history'][-1]:.6f}", flush=True)


def write_manifest(args, seeds: list[int], data_summary: dict, reference_names: list[str]) -> Path:
    recipe = {
        "model_class": "TornadoUNet", "base_channels": BASE_CHANNELS, "alpha": ALPHA, "gamma": GAMMA, "lr": LR,
        "batch_size": BATCH_SIZE, "quiet_keep_fraction": QUIET_KEEP_FRACTION, "epochs": args.epochs,
        "ema_decay": EMA_DECAY, "cell_mask": "none", "combine": "plain mean of member probabilities",
    }
    members = []
    for seed in seeds:
        info = json.loads(sidecar_path(args.out_dir, seed).read_text())
        members.append({
            "path": member_path(args.out_dir, seed).name,
            "sha256": file_sha256(member_path(args.out_dir, seed)),
            "seed": seed,
            "ema_decay": EMA_DECAY,
            "final_train_loss": info["final_train_loss"],
            "train_seconds": info["train_seconds"],
        })
    manifest = {
        "bundle_id": compute_bundle_id([m["sha256"] for m in members], recipe),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "recipe": recipe,
        "feature_names": reference_names,
        "members": members,
        "training_data": data_summary,
        "expected_skill": {
            "held_out_score_for_these_weights": None,
            "why": "every year 2014-2025 is in the training set, so there is no held-out evaluation of these exact weights",
            "reference": (
                "12-fold leave-one-year-out CV of the same recipe (CLAUDE.md): single seed AUC-PR ~0.044-0.049 "
                "(+/-~19-25% seed noise); 2-model ensemble 0.0532; deployment-like expectation ~0.02-0.03 "
                "(CV mix is richer in positives than an all-hourly-runs stream)."
            ),
            "caveats": [
                "Ensembling reduces model variance only; the accuracy ceiling is unchanged.",
                "Raw probabilities are uncalibrated (focal loss alpha=0.25, non-natural sample mix).",
                "Member spread is a disagreement signal of untested meaning, not a confidence interval.",
            ],
        },
    }
    path = args.out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH_DIR)
    parser.add_argument("--reference-checkpoint", type=Path, default=REFERENCE_CHECKPOINT)
    parser.add_argument("--n-members", type=int, default=N_MEMBERS)
    parser.add_argument("--first-seed", type=int, default=FIRST_SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-runs", type=int, default=None, help="smoke tests only: use a random subset of this many runs")
    args = parser.parse_args()

    _, reference = load_checkpoint(str(args.reference_checkpoint))
    reference_names = list(reference["feature_names"])
    seeds = [args.first_seed + i for i in range(args.n_members)]
    todo = [s for s in seeds if not member_path(args.out_dir, s).exists()]
    print(f"members {seeds}; already trained: {[s for s in seeds if s not in todo]}; to train: {todo}", flush=True)

    runs = done_runs(load_manifest(args.source_manifest))
    if args.max_runs:
        rng = np.random.default_rng(0)
        runs = sorted(rng.choice(np.array(runs, dtype="datetime64[ns]"), size=args.max_runs, replace=False))
        runs = [pd.Timestamp(t) for t in runs]

    summary_path = args.out_dir / "training_data_summary.json"
    if todo:
        dataset, data_summary = build_dataset(args, runs, reference_names)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(data_summary, indent=2))
        for seed in todo:
            train_member(dataset, seed, args, data_summary)
    else:
        data_summary = json.loads(summary_path.read_text())

    manifest_path = write_manifest(args, seeds, data_summary, reference_names)
    print(f"\nmanifest written: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
