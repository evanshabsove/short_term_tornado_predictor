"""
Pre-flight diagnostic for the repaired ConvLSTM (arm C): does the saturation appear within the first epochs of the ORIGINAL configuration, and do
the repairs prevent it? Trains each configuration for a few epochs on a stratified subset of fold 2020's training runs (A's recipe: focal loss 0.25/2,
Adam 1e-3, batch 1 run, quiet_keep 0.2, seed 10) and logs every `--log-every` steps: loss, global gradient norm (before clipping), and the std of the
LSTM gate pre-activations during the forward pass. Not part of any pre-registered analysis.

Usage: python scripts/11_convlstm/diagnose_training.py --n-runs 800 --epochs 2      Output: models/convlstm/diagnostics/training_diagnostic.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tornado_predictor.build_scaled import load_manifest
from tornado_predictor.convlstm_unet import TornadoConvLSTMUNet
from tornado_predictor.sequence_dataset import DenseSequenceDataset, runs_from_run_bins
from tornado_predictor.sequence_study import stratified_run_indices
from tornado_predictor.split import split_run_bins_year_holdout
from tornado_predictor.time_bins import N_BINS
from tornado_predictor.training import FocalLoss, QuietMapDownsampler

REPO = Path(__file__).resolve().parents[2]
CONFIGS = {
    "original": dict(model={}, clip=None),
    "norm": dict(model=dict(norm_bottleneck=True), clip=None),
    "norm+clip": dict(model=dict(norm_bottleneck=True), clip=1.0),
    "C3 (norm+pooled+clip)": dict(model=dict(norm_bottleneck=True, pooled_skips=True), clip=1.0),
}


def run(name, cfg, train_ds, epochs, log_every, seed=10):
    torch.manual_seed(seed)
    model = TornadoConvLSTMUNet(in_channels=14, base_channels=16, hidden_channels=64, **cfg["model"])
    gate_std = []
    orig = model.lstm.forward
    model.lstm.forward = lambda x, st: (gate_std.append((model.lstm.x_gates(x) + model.lstm.h_gates(st[0])).detach().std().item()), orig(x, st))[1]
    sampler = QuietMapDownsampler(train_ds.is_active, quiet_keep_fraction=0.2, seed=seed)
    loader = DataLoader(train_ds, batch_size=1, sampler=sampler)
    criterion, opt = FocalLoss(alpha=0.25, gamma=2.0), torch.optim.Adam(model.parameters(), lr=1e-3)
    log, step = [], 0
    for epoch in range(epochs):
        for X, y in loader:
            gate_std.clear()
            loss = criterion(model(X).squeeze(1), y)
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"] if cfg["clip"] else float("inf")).item()
            opt.step()
            step += 1
            if step % log_every == 0:
                log.append({"step": step, "epoch": epoch, "loss": loss.item(), "grad_norm": gnorm, "gate_preact_std_max_over_hours": float(max(gate_std))})
        print(f"{name:24s} epoch {epoch}: last logged {log[-1]}", flush=True)
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-runs", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--out", type=Path, default=REPO / "models/convlstm/diagnostics/training_diagnostic.json")
    args = ap.parse_args()
    H, W = REPO / "data/interim/scaled_build_hourly", REPO / "data/interim/scaled_build_wholebin"
    manifest = load_manifest(W / "manifest.json")
    run_bins = [(pd.Timestamp(k), b) for k, v in sorted(manifest.items()) if v["status"] == "done" for b in range(N_BINS)]
    train_rb, _ = split_run_bins_year_holdout(run_bins, 2020, buffer_hours=24.0)
    full = DenseSequenceDataset(H, W, runs_from_run_bins(train_rb))
    sub = full.subset(stratified_run_indices(full.is_active, args.n_runs, 0))
    out = {name: run(name, CONFIGS[name], sub, args.epochs, args.log_every) for name in args.configs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_runs": len(sub), "epochs": args.epochs, "logs": out}, indent=1))
    for name, log in out.items():
        g, s = np.array([r["grad_norm"] for r in log]), np.array([r["gate_preact_std_max_over_hours"] for r in log])
        print(f"{name:24s} grad-norm median {np.median(g):9.2f} max {g.max():10.1f} | gate preact std first {s[0]:8.2f} last {s[-1]:10.2f} max {s.max():10.2f}")


if __name__ == "__main__":
    main()
