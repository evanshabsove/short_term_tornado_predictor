"""
Trains sklearn.ensemble.HistGradientBoostingClassifier (an HGBT --
Histogram Gradient-Boosted Trees, matching the literature's own
terminology) as a genuinely different model family from the CNN/U-Net
line, motivated by a closely analogous study finding an HGBT beat a
deep U-Net on a CAM-grid severe-weather task.

Uses the 3-way split (training_dataset_{train,test,val}_3way.nc) so the
headline comparison -- against the U-Net's own 0.0836 test AUC-PR -- is
on the same never-touched-by-any-model-selection-decision test set, not
the repeatedly-checked old val.

Per-cell downsampling (gbt.py) is the key difference from the CNN
pipeline: a spatial CNN can't drop arbitrary pixels without breaking
its convolutional receptive field (see training.QuietMapDownsampler's
map-level-only downsampling), but a tabular per-cell model has no such
constraint. Every positive cell is kept; negative cells are downsampled
to a configurable ratio (default 20:1) -- real numbers: train has 9,992
positive cells out of 46,656,972 total (pos rate 2.14e-04), so this
keeps ~210K rows, not 46.7M. val/test are evaluated on their full, real,
non-downsampled cell population, matching training.evaluate()'s own
"a val set should be evaluated on in full" principle.

Usage:
    python scripts/train_gbt.py
    python scripts/train_gbt.py --negative-ratio 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from tornado_predictor.evaluate import compute_metrics
from tornado_predictor.gbt import downsample_negative_cells, flatten_for_gbt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_train_3way.nc"
DEFAULT_VAL_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_val_3way.nc"
DEFAULT_TEST_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_test_3way.nc"
DEFAULT_OUT_DIR = REPO_ROOT / "models" / "gbt"
DEFAULT_MODEL_PATH = DEFAULT_OUT_DIR / "tornado_gbt.joblib"
DEFAULT_RESULTS_PATH = DEFAULT_OUT_DIR / "results.json"

NEGATIVE_RATIO = 20.0
SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--negative-ratio", type=float, default=NEGATIVE_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--feasibility-check-only", action="store_true", help="fit on a 10%% subsample and time it, then exit")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and flattening datasets...")
    train_ds = xr.open_dataset(args.train)
    val_ds = xr.open_dataset(args.val)
    test_ds = xr.open_dataset(args.test)

    X_train_full, y_train_full, feature_names = flatten_for_gbt(train_ds)
    X_val, y_val, val_feature_names = flatten_for_gbt(val_ds)
    X_test, y_test, test_feature_names = flatten_for_gbt(test_ds)
    assert feature_names == val_feature_names == test_feature_names, "feature order mismatch across splits"

    X_train, y_train = downsample_negative_cells(X_train_full, y_train_full, negative_ratio=args.negative_ratio, seed=args.seed)
    print(f"train: {len(y_train):,} rows after downsampling ({int(y_train.sum()):,} positive, ratio={args.negative_ratio})")
    print(f"val:   {len(y_val):,} rows (full, not downsampled)")
    print(f"test:  {len(y_test):,} rows (full, not downsampled)")

    if args.feasibility_check_only:
        rng = np.random.default_rng(args.seed)
        subsample_idx = rng.choice(len(y_train), size=len(y_train) // 10, replace=False)
        t0 = time.time()
        HistGradientBoostingClassifier(class_weight="balanced", random_state=args.seed).fit(
            X_train[subsample_idx], y_train[subsample_idx]
        )
        elapsed = time.time() - t0
        print(f"10% subsample ({len(subsample_idx):,} rows) fit time: {elapsed:.1f}s -> "
              f"estimated full fit: ~{elapsed * 10:.0f}s")
        return

    print("Fitting HistGradientBoostingClassifier...")
    t0 = time.time()
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=args.seed)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"fit time: {elapsed:.1f}s")

    results = {"negative_ratio": args.negative_ratio, "seed": args.seed, "elapsed_sec": elapsed, "n_train_rows": len(y_train)}

    for split_name, X_split, y_split in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_score = model.predict_proba(X_split)[:, 1]
        metrics = compute_metrics(y_split, y_score, include_curves=False)
        results[split_name] = metrics
        print(f"\n{split_name}: AUC-PR={metrics['auc_pr']:.4f} AUC-ROC={metrics['auc_roc']:.4f} "
              f"cohens_d={metrics['cohens_d']:.2f} f1={metrics['f1_at_best_f1']:.4f}")

    print("\nComputing permutation importance on val...")
    importance = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=args.seed, scoring="average_precision")
    ranked = sorted(zip(feature_names, importance.importances_mean, importance.importances_std), key=lambda t: -t[1])
    results["feature_importance"] = [{"feature": f, "importance_mean": float(m), "importance_std": float(s)} for f, m, s in ranked]
    for f, m, s in ranked:
        print(f"  {f:20s} {m:+.5f} +/- {s:.5f}")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_path)
    args.results.write_text(json.dumps(results, indent=2))
    print(f"\nSaved model -> {args.model_path}")
    print(f"Saved results -> {args.results}")


if __name__ == "__main__":
    main()
