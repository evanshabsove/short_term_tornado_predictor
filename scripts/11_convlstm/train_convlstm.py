"""
Train one ConvLSTM-U-Net on one held-out-year fold (docs/convlstm_spec.md, ticket T3).

Same split as every year-holdout study here (train = every other year, 24 h buffer; test = the
held-out year), reading the hourly store directly and the whole-bin labels. Recipe mirrors arm A
(focal loss alpha 0.25 gamma 2.0, lr 1e-3, quiet_keep_fraction 0.2, EMA 0.9999) except batch_size is
1 RUN (2 bin-maps per step, matching A's batch of 2 maps).

Writes to --out-dir: raw and EMA checkpoints, float16 held-out scores in arm A's (run, bin) order,
`results.json` (metrics, per-epoch seconds, run-list hash) and `train_runs.json` (the exact training
run list, so a matched U-Net control can use the identical runs).

Usage (repo root, project venv):
    python scripts/11_convlstm/train_convlstm.py --fold 2020 --seed 10 --epochs 10 --out-dir models/convlstm/snippet
    python scripts/11_convlstm/train_convlstm.py ... --max-train-runs 800    # proportional subset (dev scale)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tornado_predictor.build_scaled import load_manifest
from tornado_predictor.sequence_dataset import DenseSequenceDataset, runs_from_run_bins
from tornado_predictor.sequence_study import run_list_hash, stratified_run_indices, train_and_evaluate
from tornado_predictor.split import split_run_bins_year_holdout
from tornado_predictor.time_bins import N_BINS

REPO = Path(__file__).resolve().parents[2]
BUFFER_HOURS = 24.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True, help="held-out year")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1, help="runs per step (default 1 = 2 bin-maps, like arm A's batch of 2)")
    ap.add_argument("--max-train-runs", type=int, default=None, help="proportional active/quiet subset of the training runs")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--hidden-channels", type=int, default=64)
    ap.add_argument("--ema-decays", type=float, nargs="*", default=[0.9999])
    ap.add_argument("--hourly-dir", type=Path, default=REPO / "data/interim/scaled_build_hourly")
    ap.add_argument("--label-dir", type=Path, default=REPO / "data/interim/scaled_build_wholebin")
    ap.add_argument("--scores-meta-dir", type=Path, default=REPO / "data/interim/diag_scores",
                    help="holds arm A's meta_<year>.npz; the test sample order is asserted against it when present")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    manifest = load_manifest(args.label_dir / "manifest.json")
    run_bins = [(pd.Timestamp(k), b) for k, v in manifest.items() if v["status"] == "done" for b in range(N_BINS)]
    train_rb, test_rb = split_run_bins_year_holdout(run_bins, args.fold, buffer_hours=BUFFER_HOURS)
    train_runs, test_runs = runs_from_run_bins(train_rb), runs_from_run_bins(test_rb)

    print(f"fold {args.fold}: {len(train_runs)} train runs, {len(test_runs)} test runs; reading labels...", flush=True)
    train_ds = DenseSequenceDataset(args.hourly_dir, args.label_dir, train_runs)
    test_ds = DenseSequenceDataset(args.hourly_dir, args.label_dir, test_runs)
    if args.max_train_runs is not None:
        idx = stratified_run_indices(train_ds.is_active, args.max_train_runs, args.subset_seed)
        train_ds = train_ds.subset(idx)
        print(f"subset: {len(train_ds)} runs ({int(train_ds.is_active.sum())} active)", flush=True)

    meta_path = args.scores_meta_dir / f"meta_{args.fold}.npz"
    order_checked = False
    if meta_path.exists():
        meta = np.load(meta_path)
        init = np.repeat(np.array([r.to_datetime64() for r in test_runs]), N_BINS).astype("datetime64[s]").astype("int64")
        assert np.array_equal(init, meta["init_time"]) and np.array_equal(np.tile(np.arange(N_BINS), len(test_runs)), meta["bin_index"]), \
            "test-sample order differs from arm A's cached scores"
        order_checked = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_runs.json").write_text(json.dumps({"hash": run_list_hash(train_ds.runs), "runs": [r.isoformat() for r in train_ds.runs]}))
    res = train_and_evaluate(
        train_ds, test_ds, out_dir=args.out_dir, held_out_year=args.fold, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, base_channels=args.base_channels, hidden_channels=args.hidden_channels,
        ema_decays=tuple(args.ema_decays),
    )
    res["test_order_matches_arm_a_scores"] = order_checked
    res["max_train_runs"] = args.max_train_runs
    (args.out_dir / "results.json").write_text(json.dumps(res, indent=2, default=str))
    for name, m in res["metrics"].items():
        print(f"{name:12s} AUC-PR={m['auc_pr']:.4f} AUC-ROC={m['auc_roc']:.4f} best-F1={m['f1_at_best_f1']:.4f}")
    print(f"epoch seconds: {[round(s) for s in res['epoch_seconds']]}  val/train loss {res['val_over_train_loss']:.2f}")
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
