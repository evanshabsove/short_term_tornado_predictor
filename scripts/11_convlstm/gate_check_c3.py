"""
Repair-arm gate check (docs/convlstm_repair_plan.md), fold 2020 seed 10, PLAIN final weights.
Checks 1 and 2: gate saturation on held-out runs, and AUC-PR on 500 stratified TRAIN runs vs. the matched U-Net control.
Checks 3 and 4 come from each snippet's results.json. Usage: python scripts/11_convlstm/gate_check_c3.py [snippet_dir ...]
Output: models/convlstm/repair_gate_check.json
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from tornado_predictor.build_scaled import combine_staged_runs, load_manifest
from tornado_predictor.evaluate import get_proba_labels
from tornado_predictor.inference import load_checkpoint
from tornado_predictor.sequence_dataset import DenseSequenceDataset, runs_from_run_bins
from tornado_predictor.sequence_study import score_runs, stratified_run_indices
from tornado_predictor.split import split_run_bins_year_holdout
from tornado_predictor.time_bins import N_BINS
from tornado_predictor.training import DenseGridDataset

REPO = Path(__file__).resolve().parents[2]
H, W = REPO / "data/interim/scaled_build_hourly", REPO / "data/interim/scaled_build_wholebin"
snips = sys.argv[1:] or ["snippet", "snippet_c1", "snippet_c2", "snippet_c3"]
manifest = load_manifest(W / "manifest.json")
run_bins = [(pd.Timestamp(k), b) for k, v in sorted(manifest.items()) if v["status"] == "done" for b in range(N_BINS)]
tr, te = split_run_bins_year_holdout(run_bins, 2020, buffer_hours=24.0)
train = DenseSequenceDataset(H, W, runs_from_run_bins(tr))
sub = train.subset(stratified_run_indices(train.is_active, 500, 0))
test = DenseSequenceDataset(H, W, runs_from_run_bins(te)[::10])
xt = torch.from_numpy(np.stack([test[i][0] for i in range(len(test))]))
out = {}

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "sub.nc"
    combine_staged_runs(W, [(r, b) for r in sub.runs for b in range(N_BINS)]).to_netcdf(p)
    ds = DenseGridDataset(str(p))
    m, _ = load_checkpoint(str(REPO / "models/convlstm/snippet_control/tornado_unet_heldout_2020.pt"))
    proba, labels = get_proba_labels(m, ds)
    out["control"] = {"train_subset_auc_pr": float(average_precision_score(np.asarray(labels).ravel(), np.asarray(proba).ravel()))}
    print("control", out["control"], flush=True)

for s in snips:
    d = REPO / "models/convlstm" / s
    model, _ = load_checkpoint(str(d / "convlstm_heldout_2020_s10.pt"))
    model.eval()
    cell, orig, pre = model.lstm, model.lstm.forward, []

    def hook(xx, state, cell=cell, orig=orig, pre=pre):
        pre.append((cell.x_gates(xx) + cell.h_gates(state[0])).detach())
        return orig(xx, state)

    cell.forward = hook
    with torch.no_grad():
        model(xt)
    g = torch.stack(pre)
    cell.forward = orig
    pr = score_runs(model, sub)
    y = np.stack([sub[i][1] for i in range(len(sub))]).reshape(len(sub) * N_BINS, *pr.shape[1:]) if False else None
    from tornado_predictor.sequence_dataset import flatten_run_bins
    lab = flatten_run_bins(np.stack([sub[i][1] for i in range(len(sub))]))
    res = json.load(open(d / "results.json"))
    out[s] = {
        "preact_std": float(g.std()), "frac_abs_gt4": float((g.abs() > 4).float().mean()),
        "train_subset_auc_pr": float(average_precision_score(lab.ravel() > 0.5, pr.ravel())),
        "heldout_auc_pr_plain": res["metrics"]["raw"]["auc_pr"], "val_over_train_loss": res["val_over_train_loss"],
    }
    print(s, out[s], flush=True)
(REPO / "models/convlstm/repair_gate_check.json").write_text(json.dumps(out, indent=1))
