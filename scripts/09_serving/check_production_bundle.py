"""
Mechanics checks for the production bundle (models/production/manifest.json).

The bundle has NO held-out score: every year is in its training set (see the
manifest's expected_skill block). So this script checks that training behaved
and that the ensemble works -- it does not measure skill:

  1. Loss: each member's final training loss vs. the year-holdout CV models'
     (same recipe; ~0.00015 expected).
  2. Agreement: pairwise correlation between members' probability maps, and the
     ensemble's spread, on a sample of runs.
  3. Level: mean predicted probability over US-trusted cells vs. the CV EMA
     models scored on the same default sample (0.0064-0.0084).
  4. In-sample AUC-PR for each member and for the ensemble, printed under a
     "TRAINING-SET FIT, NOT SKILL" header. It exists to catch a broken model
     (near the no-skill rate), not to be quoted.

Usage:
    python scripts/09_serving/check_production_bundle.py
    python scripts/09_serving/check_production_bundle.py --manifest <path> --n-runs 150
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tornado_predictor.build_scaled import _drop_columns_and_verify, combine_staged_runs, load_manifest
from tornado_predictor.cell_mask import load_us_trusted_mask
from tornado_predictor.ensemble import DEFAULT_BUNDLE_MANIFEST_PATH, load_bundle
from tornado_predictor.evaluate import compute_metrics
from tornado_predictor.time_bins import N_BINS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGING_DIR = REPO_ROOT / "data" / "interim" / "scaled_build_v2"
CV_PLAIN_CHECKPOINTS = "models/ema_study/seed*/tornado_unet_heldout_*.pt"  # plain (non-EMA) files carry loss_history
# Mean predicted p over US-trusted cells for the 9 CV EMA(0.9999) models (seeds 10-12, folds 2016/2020/2024), scored on
# THIS script's default sample (200 evenly spaced runs, 2026-10-19): 0.0064-0.0084. Like-for-like on purpose --
# CLAUDE.md's Stage A figure (~0.0128, pooled over held-out test years) is a different sample and not comparable.
CV_US_MEAN_PROBA_RANGE = (0.0064, 0.0084)


def cv_final_losses() -> list[float]:
    losses = []
    for path in sorted(REPO_ROOT.glob(CV_PLAIN_CHECKPOINTS)):
        if "ema" in path.name:
            continue
        history = torch.load(path, weights_only=False).get("loss_history")
        if history:
            losses.append(float(history[-1]))
    return losses


def load_sample(staging_dir: Path, source_manifest: Path, n_runs: int, feature_names: list[str]):
    runs = sorted(pd.Timestamp(k) for k, v in load_manifest(source_manifest).items() if v["status"] == "done")
    runs = [runs[i] for i in np.linspace(0, len(runs) - 1, n_runs).astype(int)]
    combined = combine_staged_runs(staging_dir, [(t, b) for t in runs for b in range(N_BINS)])
    combined = _drop_columns_and_verify(combined, [v for v in combined.data_vars if str(v).startswith("uh_")])
    X = np.nan_to_num(np.stack([combined[n].values for n in feature_names], axis=1).astype("float32"), nan=0.0)
    return X, combined["label"].values.astype("float32")


def member_probabilities(bundle, X: np.ndarray, batch: int = 16) -> np.ndarray:
    """(members, N, H, W) sigmoid probabilities."""
    out = []
    with torch.no_grad():
        for model in bundle.models:
            chunks = [torch.sigmoid(model(torch.from_numpy(X[i:i + batch]))).squeeze(1).numpy() for i in range(0, len(X), batch)]
            out.append(np.concatenate(chunks))
    return np.stack(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_BUNDLE_MANIFEST_PATH)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_STAGING_DIR / "manifest.json")
    parser.add_argument("--n-runs", type=int, default=200, help="evenly spaced runs to score (x2 bins)")
    parser.add_argument("--out", type=Path, default=None, help="where to write the JSON (default: next to the manifest)")
    args = parser.parse_args()

    bundle = load_bundle(args.manifest)
    members = bundle.manifest["members"]
    print(f"bundle {bundle.bundle_id}: {len(members)} members, trained on "
          f"{bundle.manifest['training_data']['n_samples']} samples\n")

    cv_losses = cv_final_losses()
    print("1. final training loss (CV reference, same recipe: "
          f"mean {np.mean(cv_losses):.6f}, range {min(cv_losses):.6f}-{max(cv_losses):.6f}, n={len(cv_losses)})")
    for m in members:
        print(f"   seed {m['seed']}: {m['final_train_loss']:.6f}")

    X, y = load_sample(args.staging_dir, args.source_manifest, args.n_runs, bundle.feature_names)
    probs = member_probabilities(bundle, X)  # (members, N, H, W)
    mean, spread = probs.mean(axis=0), probs.std(axis=0, ddof=0)
    us = load_us_trusted_mask()

    print(f"\n2. agreement on {len(X)} samples")
    corrs = {f"{members[i]['seed']}-{members[j]['seed']}": float(np.corrcoef(probs[i].ravel(), probs[j].ravel())[0, 1])
             for i, j in itertools.combinations(range(len(members)), 2)}
    print(f"   pairwise correlation of probability maps: min {min(corrs.values()):.3f}, "
          f"mean {np.mean(list(corrs.values())):.3f}, max {max(corrs.values()):.3f}")
    print(f"   spread across members: mean {spread.mean():.5f}, max {spread.max():.4f} (nonzero => members differ)")

    us_mean = float(mean[:, us].mean())
    lo, hi = CV_US_MEAN_PROBA_RANGE
    print(f"\n3. mean ensemble probability over US-trusted cells: {us_mean:.4f} "
          f"(CV EMA models on the default {200}-run sample: {lo}-{hi}; only comparable with --n-runs 200)")

    print("\n4. TRAINING-SET FIT, NOT SKILL (every run is in the training set)")
    y_flat = y[:, us].ravel()
    results = {"ensemble": compute_metrics(y_flat, mean[:, us].ravel(), include_curves=False)}
    for i, m in enumerate(members):
        results[f"seed{m['seed']}"] = compute_metrics(y_flat, probs[i][:, us].ravel(), include_curves=False)
    print(f"   no-skill AUC-PR (positive rate): {y_flat.mean():.2e}")
    for name, metrics in results.items():
        print(f"   {name:10s} in-sample AUC-PR {metrics['auc_pr']:.4f}  AUC-ROC {metrics['auc_roc']:.4f}")

    out = args.out or args.manifest.with_name("mechanics_check.json")
    out.write_text(json.dumps({
        "bundle_id": bundle.bundle_id,
        "note": "mechanics checks only; in-sample metrics are training-set fit, not skill",
        "cv_reference_final_loss": {"mean": float(np.mean(cv_losses)), "min": min(cv_losses), "max": max(cv_losses), "n": len(cv_losses)},
        "member_final_loss": {str(m["seed"]): m["final_train_loss"] for m in members},
        "pairwise_correlation": corrs,
        "spread_mean": float(spread.mean()),
        "spread_max": float(spread.max()),
        "us_mean_ensemble_proba": us_mean,
        "n_samples_scored": len(X),
        "in_sample": {k: {m: float(v[m]) for m in ("auc_pr", "auc_roc")} for k, v in results.items()},
        "no_skill_auc_pr": float(y_flat.mean()),
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
