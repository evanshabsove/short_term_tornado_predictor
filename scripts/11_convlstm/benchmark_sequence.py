"""
Timing and memory benchmark for the ConvLSTM study (docs/convlstm_spec.md, ticket T4).

Measures real wall-clock seconds per training epoch and per scored sample, and peak resident memory,
for the ConvLSTM-U-Net (batch 1 run) and the matched whole-bin U-Net (batch 2 maps, arm A's recipe) on
fold 2020, at a given number of training runs. Do not extrapolate from FLOPs: dilated convs ran 2x
slower than FLOPs predicted on this CPU.

One measurement per invocation (a fresh process, so peak RSS is that configuration's own):
    python scripts/11_convlstm/benchmark_sequence.py --model convlstm --n-runs 100 --out models/convlstm/benchmark_raw.jsonl
    python scripts/11_convlstm/benchmark_sequence.py --model unet     --n-runs 100 --out models/convlstm/benchmark_raw.jsonl
Then project full-scale costs from the measurements:
    python scripts/11_convlstm/benchmark_sequence.py --summarize models/convlstm/benchmark_raw.jsonl --out models/convlstm/benchmark.json

Units: the ConvLSTM trains/scores RUNS (2 bin-maps each); the U-Net trains/scores bin MAPS. The
per-epoch work is the active samples plus quiet_keep_fraction (0.2) of the quiet ones, as in training.
"""
import argparse
import json
import platform
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tornado_predictor.build_scaled import combine_staged_runs, load_manifest
from tornado_predictor.convlstm_unet import TornadoConvLSTMUNet
from tornado_predictor.evaluate import get_proba_labels
from tornado_predictor.sequence_dataset import DenseSequenceDataset, runs_from_run_bins
from tornado_predictor.sequence_study import score_runs, stratified_run_indices
from tornado_predictor.split import split_run_bins_year_holdout
from tornado_predictor.time_bins import N_BINS
from tornado_predictor.training import DenseGridDataset, train_model
from tornado_predictor.unet import TornadoUNet

REPO = Path(__file__).resolve().parents[2]
HOURLY = REPO / "data/interim/scaled_build_hourly"
WHOLEBIN = REPO / "data/interim/scaled_build_wholebin"
FOLD, SEED, QUIET_KEEP = 2020, 10, 0.2
N_SCORE_RUNS = 40  # runs timed for the scoring (forward-only) rate


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024**2 if platform.system() == "Darwin" else rss / 1024  # macOS reports bytes, Linux KiB


def measure(model_name: str, n_runs: int, epochs: int) -> dict:
    manifest = load_manifest(WHOLEBIN / "manifest.json")
    run_bins = [(pd.Timestamp(k), b) for k, v in manifest.items() if v["status"] == "done" for b in range(N_BINS)]
    train_rb, test_rb = split_run_bins_year_holdout(run_bins, FOLD, buffer_hours=24.0)
    full_train = DenseSequenceDataset(HOURLY, WHOLEBIN, runs_from_run_bins(train_rb))
    full_test_runs = runs_from_run_bins(test_rb)
    n_active_runs, n_active_bins = int(full_train.is_active.sum()), int(full_train.is_active_by_bin.sum())
    n_runs_full, n_bins_full = len(full_train), len(full_train) * N_BINS
    units_full = {
        "convlstm": n_active_runs + QUIET_KEEP * (n_runs_full - n_active_runs),
        "unet": n_active_bins + QUIET_KEEP * (n_bins_full - n_active_bins),
    }[model_name]

    sub = full_train.subset(stratified_run_indices(full_train.is_active, n_runs, 0))
    score_runs_list = full_test_runs[:N_SCORE_RUNS]
    torch.manual_seed(SEED)
    out = {
        "model": model_name, "n_runs": len(sub), "n_active_runs": int(sub.is_active.sum()), "epochs": epochs,
        "torch_threads": torch.get_num_threads(), "fold": FOLD,
        "full_fold": {"train_runs": n_runs_full, "units_per_epoch": units_full, "test_runs": len(full_test_runs)},
    }
    if model_name == "convlstm":
        model = TornadoConvLSTMUNet(in_channels=14, base_channels=16, hidden_channels=64)
        out["parameters"] = sum(p.numel() for p in model.parameters())
        units_measured = int(sub.is_active.sum() + QUIET_KEEP * (len(sub) - int(sub.is_active.sum())))
        res = train_model(sub, model, epochs=epochs, batch_size=1, seed=SEED, quiet_keep_fraction=QUIET_KEEP)
        test_sub = DenseSequenceDataset(HOURLY, WHOLEBIN, score_runs_list)
        t0 = time.perf_counter()
        score_runs(model, test_sub)
        score_seconds, score_units = time.perf_counter() - t0, len(test_sub)
        full_test_units = len(full_test_runs)
    else:
        model = TornadoUNet(in_channels=14, base_channels=16)
        out["parameters"] = sum(p.numel() for p in model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            run_bins_sub = [(r, b) for r in sub.runs for b in range(N_BINS)]
            path = Path(tmp) / "train.nc"
            combine_staged_runs(WHOLEBIN, run_bins_sub).to_netcdf(path)
            train_ds = DenseGridDataset(str(path))
            units_measured = int(train_ds.is_active.sum() + QUIET_KEEP * (len(train_ds) - int(train_ds.is_active.sum())))
            res = train_model(train_ds, model, epochs=epochs, batch_size=2, seed=SEED, quiet_keep_fraction=QUIET_KEEP)
            test_path = Path(tmp) / "test.nc"
            combine_staged_runs(WHOLEBIN, [(r, b) for r in score_runs_list for b in range(N_BINS)]).to_netcdf(test_path)
            test_ds = DenseGridDataset(str(test_path))
            t0 = time.perf_counter()
            get_proba_labels(model, test_ds)
            score_seconds, score_units = time.perf_counter() - t0, len(test_ds)
        full_test_units = len(full_test_runs) * N_BINS
    out.update({
        "units_per_epoch_measured": units_measured,
        "epoch_seconds": res["epoch_seconds"],
        "train_seconds_per_unit": float(np.mean(res["epoch_seconds"]) / units_measured),
        "score_seconds_per_unit": score_seconds / score_units,
        "score_units_timed": score_units,
        "full_test_units": full_test_units,
        "final_loss": res["loss_history"][-1],
        "peak_rss_mb": peak_rss_mb(),
    })
    return out


def summarize(raw_path: Path) -> dict:
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    summary = {"measurements": rows, "projections": {}}
    for model in ("convlstm", "unet"):
        mine = sorted((r for r in rows if r["model"] == model), key=lambda r: r["n_runs"])
        if not mine:
            continue
        best = mine[-1]  # largest subset: least fixed-overhead share
        train_unit, score_unit = best["train_seconds_per_unit"], best["score_seconds_per_unit"]
        ff = best["full_fold"]
        epoch_train = train_unit * ff["units_per_epoch"]
        epoch_val = score_unit * best["full_test_units"]  # per-epoch validation pass over the held-out year
        final_scoring = 2 * epoch_val  # raw + EMA model, once at the end
        per_epoch = epoch_train + epoch_val
        hours = lambda s: s / 3600
        summary["projections"][model] = {
            "based_on_n_runs": best["n_runs"],
            "train_seconds_per_unit_by_size": {r["n_runs"]: r["train_seconds_per_unit"] for r in mine},
            "peak_rss_mb_by_size": {r["n_runs"]: r["peak_rss_mb"] for r in mine},
            "full_fold_epoch_train_min": epoch_train / 60,
            "full_fold_epoch_val_min": epoch_val / 60,
            "snippet_10_epochs_full_fold_hours": hours(10 * per_epoch + final_scoring),
            "single_run_50_epochs_hours": hours(50 * per_epoch + final_scoring),
            "full_study_9_runs_50_epochs_hours": hours(9 * (50 * per_epoch + final_scoring)),
            "note": "fold 2020's train size is used for all three folds (2016/2024 differ by a few percent); "
                    "per-epoch validation over the held-out year is included, as train_model does it",
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["convlstm", "unet"])
    ap.add_argument("--n-runs", type=int)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summarize", type=Path, help="raw jsonl to project from; --out is then the summary json")
    args = ap.parse_args()
    if args.summarize:
        args.out.write_text(json.dumps(summarize(args.summarize), indent=2))
        print(json.dumps(json.loads(args.out.read_text())["projections"], indent=2))
        return
    if not (args.model and args.n_runs):
        ap.error("--model and --n-runs are required")
    row = measure(args.model, args.n_runs, args.epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: row[k] for k in ("model", "n_runs", "units_per_epoch_measured", "epoch_seconds", "train_seconds_per_unit",
                                          "score_seconds_per_unit", "peak_rss_mb", "parameters")}))


if __name__ == "__main__":
    main()
