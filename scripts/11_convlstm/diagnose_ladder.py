"""
Exploratory diagnostic ladder for the ConvLSTM (docs/convlstm_diagnostic_ladder.md). Fold 2020, seed 10. Not a pre-registered analysis.

  python scripts/11_convlstm/diagnose_ladder.py overfit              # step 1: memorize 20 active runs
  python scripts/11_convlstm/diagnose_ladder.py full early_fusion    # step 2
  python scripts/11_convlstm/diagnose_ladder.py full c3_norecur      # step 3
  python scripts/11_convlstm/diagnose_ladder.py full c3_b4_lr2e3     # step 4a
  python scripts/11_convlstm/diagnose_ladder.py full c3_lr1e4        # step 4b
Output: models/convlstm/diagnostics/ladder/<name>.json
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from tornado_predictor.build_scaled import combine_staged_runs, load_manifest
from tornado_predictor.convlstm_unet import TornadoConvLSTMUNet
from tornado_predictor.evaluate import compute_metrics
from tornado_predictor.sequence_dataset import DenseSequenceDataset, flatten_run_bins, runs_from_run_bins
from tornado_predictor.sequence_study import score_runs, stratified_run_indices
from tornado_predictor.split import split_run_bins_year_holdout
from tornado_predictor.time_bins import N_BINS
from tornado_predictor.training import DenseGridDataset, FocalLoss, train_model
from tornado_predictor.unet import TornadoUNet

REPO = Path(__file__).resolve().parents[2]
H, W = REPO / "data/interim/scaled_build_hourly", REPO / "data/interim/scaled_build_wholebin"
OUT = REPO / "models/convlstm/diagnostics/ladder"
SEED = 10


class EarlyFusionUNet(nn.Module):
    """All hours stacked as channels into the plain U-Net, two output channels (bin 0, bin 1)."""

    def __init__(self, in_channels=14, steps=8):
        super().__init__()
        self.net = TornadoUNet(in_channels * steps, 16)
        self.net.head = nn.Conv2d(16, N_BINS, kernel_size=1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        return self.net(x.reshape(b, t * c, h, w))


class NoRecurrence(nn.Module):
    """Drop-in for the ConvLSTM cell with no memory: each hour -> 1x1 conv + tanh."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.proj = nn.Conv2d(in_channels, hidden_channels, 1)

    def forward(self, x, state):
        return torch.tanh(self.proj(x)), state[1]


def c3(norecur=False):
    m = TornadoConvLSTMUNet(14, 16, 64, norm_bottleneck=True, pooled_skips=True)
    if norecur:
        m.lstm = NoRecurrence(m.lstm.x_gates.in_channels, 64)
    return m


FACTORIES = {
    "original": lambda: TornadoConvLSTMUNet(14, 16, 64),
    "c3": lambda: c3(),
    "early_fusion": lambda: EarlyFusionUNet(),
    "c3_norecur": lambda: c3(True),
    "c3_b4_lr2e3": lambda: c3(),
    "c3_lr1e4": lambda: c3(),
    "c3_lr1e4_s11": lambda: c3(),
    "c3_norecur_lr1e4": lambda: c3(True),
}
FULL = {  # name -> train_model kwargs
    "early_fusion": dict(batch_size=1, lr=1e-3, grad_clip_norm=None),
    "c3_norecur": dict(batch_size=1, lr=1e-3, grad_clip_norm=1.0),
    "c3_b4_lr2e3": dict(batch_size=4, lr=2e-3, grad_clip_norm=1.0),
    "c3_lr1e4": dict(batch_size=1, lr=1e-4, grad_clip_norm=1.0),
    "c3_lr1e4_s11": dict(batch_size=1, lr=1e-4, grad_clip_norm=1.0, seed=11),  # round 2: another seed
    "c3_norecur_lr1e4": dict(batch_size=1, lr=1e-4, grad_clip_norm=1.0),  # round 2: no recurrence at the better lr
}


def fold_data(fold=2020):
    manifest = load_manifest(W / "manifest.json")
    run_bins = [(pd.Timestamp(k), b) for k, v in sorted(manifest.items()) if v["status"] == "done" for b in range(N_BINS)]
    tr, te = split_run_bins_year_holdout(run_bins, fold, buffer_hours=24.0)
    return DenseSequenceDataset(H, W, runs_from_run_bins(tr)), DenseSequenceDataset(H, W, runs_from_run_bins(te))


def gate_stats(model, x):
    if not hasattr(model, "lstm") or not hasattr(model.lstm, "x_gates"):
        return None
    cell, orig, pre = model.lstm, model.lstm.forward, []

    def hook(xx, state):
        pre.append((cell.x_gates(xx) + cell.h_gates(state[0])).detach())
        return orig(xx, state)

    cell.forward = hook
    model.eval()
    with torch.no_grad():
        model(x)
    cell.forward = orig
    g = torch.stack(pre)
    return {"preact_std": float(g.std()), "frac_abs_gt4": float((g.abs() > 4).float().mean())}


def seq_auc_pr(model, ds):
    pr = score_runs(model, ds)
    return float(average_precision_score(flatten_run_bins(ds.y).ravel() > 0.5, pr.ravel()))


def run_overfit():
    train, _ = fold_data()
    rng = np.random.default_rng(0)
    idx = sorted(int(i) for i in rng.choice(np.where(train.is_active)[0], 20, replace=False))
    sub = train.subset(idx)
    epochs = 300
    out = {"runs": [r.isoformat() for r in sub.runs], "epochs": epochs, "models": {}}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "s.nc"
        combine_staged_runs(W, [(r, b) for r in sub.runs for b in range(N_BINS)]).to_netcdf(p)
        map_ds = DenseGridDataset(str(p))
        for name in ("original", "c3", "early_fusion", "unet"):
            torch.manual_seed(SEED)
            is_unet = name == "unet"
            model = TornadoUNet(14, 16) if is_unet else FACTORIES[name]()
            ds, bs = (map_ds, 2) if is_unet else (sub, 1)
            g = torch.Generator().manual_seed(SEED)
            loader = DataLoader(ds, batch_size=bs, shuffle=True, generator=g)
            crit = FocalLoss(alpha=0.25, gamma=2.0)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            clip = 1.0 if name == "c3" else None
            losses, t0 = [], time.time()
            for ep in range(epochs):
                model.train()
                tot = 0.0
                for X, y in loader:
                    loss = crit(model(X).squeeze(1) if is_unet else model(X), y)
                    opt.zero_grad()
                    loss.backward()
                    if clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    opt.step()
                    tot += loss.item() * len(X)
                losses.append(tot / len(ds))
                if (ep + 1) % 50 == 0:
                    print(f"overfit {name:12s} epoch {ep + 1} loss {losses[-1]:.3e}", flush=True)
            model.eval()
            with torch.no_grad():
                if is_unet:
                    pr = np.concatenate([torch.sigmoid(model(X)).squeeze(1).numpy() for X, _ in DataLoader(map_ds, batch_size=2)])
                    lab = np.concatenate([y.numpy() for _, y in DataLoader(map_ds, batch_size=2)])
                else:
                    pr, lab = score_runs(model, sub), flatten_run_bins(sub.y)
            ap = float(average_precision_score(lab.ravel() > 0.5, pr.ravel()))
            out["models"][name] = {"final_train_auc_pr": ap, "loss_every_10": losses[9::10], "loss_first": losses[0], "loss_last": losses[-1],
                                   "seconds": time.time() - t0}
            print(f"overfit {name:12s} train AUC-PR on the 20 runs {ap:.4f}  loss {losses[0]:.2e} -> {losses[-1]:.2e}", flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "overfit.json").write_text(json.dumps(out, indent=1))


def run_full(name):
    train, test = fold_data()
    sub = train.subset(stratified_run_indices(train.is_active, 500, 0))
    cfg = dict(FULL[name])
    seed = cfg.pop("seed", SEED)
    torch.manual_seed(seed)
    model = FACTORIES[name]()
    t0 = time.time()
    res = train_model(train, model, val_dataset=test, epochs=10, seed=seed, quiet_keep_fraction=0.2, ema_decays=(0.9999,), **cfg)
    y_true = flatten_run_bins(test.y)
    xt = torch.from_numpy(np.stack([test[i][0] for i in range(0, len(test), 10)]))
    out = {"name": name, "config": cfg, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()), "train_seconds": time.time() - t0,
           "epoch_seconds": res["epoch_seconds"], "loss_history": res["loss_history"], "val_loss_history": res["val_loss_history"],
           "val_over_train_loss": res["val_loss_history"][-1] / res["loss_history"][-1], "variants": {}}
    ema = FACTORIES[name]()
    ema.load_state_dict(res["ema_state_dicts"][0.9999])
    for vname, m in (("plain", model), ("ema", ema)):
        scores = score_runs(m, test)
        met = compute_metrics(y_true.ravel(), scores.ravel(), include_curves=False)
        out["variants"][vname] = {"heldout_auc_pr": met["auc_pr"], "heldout_best_f1": met["f1_at_best_f1"], "auc_roc": met["auc_roc"],
                                  "train_subset_auc_pr": seq_auc_pr(m, sub), "gates": gate_stats(m, xt)}
        print(name, vname, out["variants"][vname], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "overfit":
        run_overfit()
    else:
        run_full(sys.argv[2])
