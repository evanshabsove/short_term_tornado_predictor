"""
Post-hoc diagnostic (NOT part of the pre-registered analysis): are the ConvLSTM's gates saturated in the trained full-study models?
For each EMA checkpoint, runs 44 held-out runs and reports the std of the gate pre-activations (x_gates + h_gates) and the share with |value| > 4
(sigmoid/tanh are flat there). Untrained models sit at std ~0.5, share 0.
Usage: python scripts/11_convlstm/diagnose_gates.py [--study-dir models/convlstm/full_study]     Output: <study-dir>/gate_saturation.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tornado_predictor.build_scaled import load_manifest
from tornado_predictor.inference import load_checkpoint
from tornado_predictor.sequence_dataset import DenseSequenceDataset, runs_from_run_bins
from tornado_predictor.split import split_run_bins_year_holdout
from tornado_predictor.time_bins import N_BINS

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--study-dir", default="models/convlstm/full_study")
STUDY = REPO / ap.parse_args().study_dir
H, W = REPO / "data/interim/scaled_build_hourly", REPO / "data/interim/scaled_build_wholebin"
manifest = load_manifest(W / "manifest.json")
run_bins = [(pd.Timestamp(k), b) for k, v in sorted(manifest.items()) if v["status"] == "done" for b in range(N_BINS)]
out = {}
for year in (2016, 2020, 2024):
    _, te = split_run_bins_year_holdout(run_bins, year, buffer_hours=24.0)
    ds = DenseSequenceDataset(H, W, runs_from_run_bins(te)[::10])
    x = torch.from_numpy(np.stack([ds[i][0] for i in range(len(ds))]))
    for seed in (10, 11, 12):
        model, _ = load_checkpoint(str(STUDY / f"seed{seed}/fold{year}/convlstm_heldout_{year}_s{seed}_ema0.9999.pt"))
        model.eval()
        cell, orig, pre = model.lstm, model.lstm.forward, []

        def hook(xx, state, cell=cell, orig=orig, pre=pre):
            pre.append((cell.x_gates(xx) + cell.h_gates(state[0])).detach())
            return orig(xx, state)

        cell.forward = hook
        with torch.no_grad():
            model(x)
        g = torch.stack(pre)
        out[f"{year}_s{seed}"] = {"preact_std": float(g.std()), "frac_abs_gt4": float((g.abs() > 4).float().mean()), "n_runs": len(ds)}
        print(year, seed, out[f"{year}_s{seed}"], flush=True)
(STUDY / "gate_saturation.json").write_text(json.dumps(out, indent=1))
