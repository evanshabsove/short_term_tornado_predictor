"""
Class-imbalance handling for training a CNN on the dense grid: focal
loss + quiet-map downsampling, in preference to per-cell downsampling
(which isn't really executable for a spatial CNN -- you can't drop
arbitrary pixels out of a grid map without breaking the convolutional
receptive field).

Grounded in this project's real numbers (from tornado_labels_2012_2022.csv):
if every hourly HRRR run 2012-2022 were used, the positive rate would be
~4.3e-05 (~1 in 23,000) at the per-cell level, but ~15% of (run, bin)
grid-maps contain at least one positive cell -- i.e. most of the
imbalance is "positives are rare within an active map," not "active
maps are rare." That's why the strategy below only downsamples at the
map level (quiet vs. active), never at the per-cell level.

Note: the current pilot dataset (training_dataset_pilot.nc) has only 6
samples, all of which are "active" (it was deliberately built around a
known outbreak). QuietMapDownsampler is a no-op at that scale -- there
are no quiet maps yet to downsample. It only starts doing real work
once the training set is scaled beyond a single outbreak.

Also holds train_model(), the training loop that wires FocalLoss,
DenseGridDataset, and QuietMapDownsampler together with a given model
(see model.py for the architecture) -- used by scripts/03_architecture_search/train_model.py.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset, Sampler


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017) over a dense (row, col) grid.
    alpha=0.25, gamma=2.0 are RetinaNet's defaults -- starting points,
    not tuned for this dataset's imbalance.

    alpha_t below is `alpha` for the positive class and `1 - alpha`
    for the negative class -- so a HIGHER alpha gives MORE weight to
    the rare positive class, not less. (An earlier version of this
    docstring said the opposite -- "expect to need a lower alpha" --
    which was backwards; corrected after scripts/03_architecture_search/sweep_alpha.py
    empirically confirmed the direction on the full-scale dataset:
    mean_proba_positive rose monotonically with alpha across 0.1-0.9,
    and alpha=0.9 gave the best held-out Cohen's d (3.08) and miss
    rate (1.7%) of the values tested. See
    notebooks/alpha_sweep_results.ipynb and CLAUDE.md "Alpha sweep".)"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """mask (optional, broadcastable to logits, e.g. (row, col)): 1 where a
        cell's label is trustworthy, 0 where it should not contribute. Masked
        cells are zeroed and the sum is divided by the TOTAL cell count (not
        mask.sum()), so the loss scale -- and hence the effective learning rate --
        matches the unmasked loss; only the excluded cells' contribution is
        removed. mask=None is exactly the original behavior."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        if mask is None:
            return loss.mean()
        return (loss * mask).sum() / loss.numel()


class DenseGridDataset(Dataset):
    """Wraps a dataset.py-built netCDF (e.g. training_dataset_pilot.nc)
    directly, flagging which samples are "active" (>=1 positive cell)
    vs. "quiet" for the downsampler below."""

    def __init__(self, nc_path: str):
        ds = xr.open_dataset(nc_path)
        self.feature_names = [v for v in ds.data_vars if v != "label"]
        # (sample, channel, row, col) -- channel-first, as torch conv layers expect
        self.X = np.stack([ds[v].values for v in self.feature_names], axis=1).astype("float32")
        # 0-2km/0-3km updraft helicity are NaN before 2018-07-13 (HRRR
        # didn't output them yet -- see features.py's module docstring);
        # a no-op for every dataset without that gap (pilot, scaled,
        # split-demo, the original full-scale build). Imputed to 0.0
        # here, at model-input time, not in the stored .nc file, so the
        # dataset itself stays honest about what's real vs. missing --
        # the model can still condition on missingness explicitly via
        # the always-present uh_layers_available feature.
        self.X = np.nan_to_num(self.X, nan=0.0)
        self.y = ds["label"].values.astype("float32")  # (sample, row, col)
        self.is_active = self.y.sum(axis=(1, 2)) > 0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def standardize_channels(dataset: "DenseGridDataset", names: list[str], stats: dict | None = None, min_std: float = 1e-6) -> dict:
    """Z-scores the named channels of dataset.X in place and returns the statistics used.

    stats=None computes mean/std per channel over every sample and cell of `dataset`
    (pass the TRAINING set); a returned stats dict is then passed back for the test
    set, so no test information reaches the scaling. Channels not named are left
    raw -- the base 14 features have never been normalized in this project and
    experiment B deliberately keeps it that way. std is floored at min_std so a
    constant channel cannot divide by zero.

    The dict ({"names", "mean", "std"}, plain lists) is what checkpoints store as
    `input_scaling`; inference.apply_input_scaling replays it."""
    if stats is None:
        mean, std = [], []
        for name in names:
            channel = dataset.X[:, dataset.feature_names.index(name)]
            mean.append(float(channel.mean(dtype=np.float64)))
            std.append(max(float(channel.std(dtype=np.float64)), min_std))
        stats = {"names": list(names), "mean": mean, "std": std}
    for name, m, s in zip(stats["names"], stats["mean"], stats["std"]):
        i = dataset.feature_names.index(name)
        dataset.X[:, i] = (dataset.X[:, i] - m) / s
    return stats


def masked_is_active(dataset: "DenseGridDataset", cell_mask: np.ndarray | None) -> np.ndarray:
    """Per-sample 'active' flags counting only positives inside cell_mask (a map
    whose only positive lies in an excluded cell is quiet for sampling purposes).
    cell_mask=None returns dataset.is_active unchanged."""
    if cell_mask is None:
        return dataset.is_active
    return (dataset.y * np.asarray(cell_mask, dtype="float32")).sum(axis=(1, 2)) > 0


class QuietMapDownsampler(Sampler):
    """Keeps every active map but only a fraction of quiet ones,
    re-drawn each epoch (not a fixed subset) so the model sees
    different quiet examples over training rather than overfitting to
    one arbitrary slice."""

    def __init__(self, is_active: np.ndarray, quiet_keep_fraction: float = 0.2, seed: int = 0):
        self.active_idx = np.where(is_active)[0]
        self.quiet_idx = np.where(~is_active)[0]
        self.quiet_keep_fraction = quiet_keep_fraction
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        n_keep = int(len(self.quiet_idx) * self.quiet_keep_fraction)
        kept_quiet = self.rng.choice(self.quiet_idx, size=n_keep, replace=False)
        indices = np.concatenate([self.active_idx, kept_quiet])
        self.rng.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self):
        return len(self.active_idx) + int(len(self.quiet_idx) * self.quiet_keep_fraction)


# Held-out metrics recorded every epoch when train_model(epoch_metrics_fn=...) is used (a slim subset of evaluate.compute_metrics).
EPOCH_METRIC_KEYS = (
    "auc_pr", "auc_roc", "precision_at_best_f1", "recall_at_best_f1", "f1_at_best_f1", "best_f1_threshold",
    "mean_proba_positive", "mean_proba_negative", "cohens_d",
)


def slim_epoch_metrics(metrics: dict) -> dict:
    """The per-epoch subset of a compute_metrics dict (plain floats, JSON-safe)."""
    return {k: float(metrics[k]) for k in EPOCH_METRIC_KEYS}


def evaluate(dataset: DenseGridDataset, model: nn.Module, criterion: nn.Module, batch_size: int = 2, cell_mask: torch.Tensor | None = None) -> float:
    """Mean loss over every sample in dataset, no gradient updates, no
    QuietMapDownsampler (a val set should be evaluated on in full, not
    downsampled -- downsampling exists only to manage training-time
    imbalance/cost). Restores the model's prior training/eval mode."""
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for X, y in DataLoader(dataset, batch_size=batch_size):
            logits = model(X).squeeze(1)
            losses.append((criterion(logits, y) if cell_mask is None else criterion(logits, y, cell_mask)).item())
    model.train(was_training)
    return float(np.mean(losses))


def train_model(
    dataset: DenseGridDataset,
    model: nn.Module,
    *,
    val_dataset: DenseGridDataset | None = None,
    epochs: int = 50,
    batch_size: int = 2,
    lr: float = 1e-3,
    alpha: float = 0.25,
    gamma: float = 2.0,
    quiet_keep_fraction: float = 0.2,
    seed: int = 0,
    ema_decays: tuple[float, ...] = (),
    cell_mask: np.ndarray | None = None,
    grad_clip_norm: float | None = None,
    epoch_metrics_fn=None,
    early_stop_fn=None,
    early_stop_patience: int = 8,
    early_stop_min_epochs: int = 10,
    early_stop_decay: float | None = None,
) -> dict:
    """Trains model on dataset with FocalLoss + QuietMapDownsampler
    (re-drawing its quiet-map subset each epoch, since DataLoader calls
    iter(sampler) fresh every time the loader is iterated). Returns
    {"loss_history": [...], "hyperparameters": {...}}, plus
    "val_loss_history" if val_dataset is given; saving
    model.state_dict() is the caller's responsibility.

    ema_decays (default empty = off, behavior and results unchanged): for
    each decay d, keeps an exponential moving average of the model's
    parameters, updated after every optimizer step (ema = d*ema +
    (1-d)*param, ~1/(1-d)-step horizon), and returns the final averaged
    weights as result["ema_state_dicts"][d] (loadable into a fresh model
    of the same architecture). Averaging weights rather than predictions
    targets the seed-to-seed noise of the final iterate; it needs no
    extra forward passes, so it adds negligible cost. Only parameters
    are averaged -- correct for the models here, which have no
    batch-norm running statistics or other buffers.

    cell_mask (default None = off, behavior and results unchanged): a (row, col)
    boolean array, True where labels are trustworthy (see cell_mask.py). False
    cells contribute nothing to the training or validation loss, and positives
    there do not make a map "active" for QuietMapDownsampler. Inputs are not
    masked -- the model still sees the whole grid.

    grad_clip_norm (default None = off, behavior and results unchanged): clip the global
    gradient norm to this value after backward and before each optimizer step.

    epoch_metrics_fn (default None = off, training and results unchanged): a callable `model -> dict` of held-out metrics (e.g. slim_epoch_metrics of
    evaluate.compute_metrics on the held-out scores). When given, it is called after every epoch on the current weights and, for each EMA decay, on the
    averaged weights (temporarily swapped in, then restored), and the results come back as result["epoch_metrics"] = {"raw": [per-epoch dicts],
    "ema<decay>": [...]}. It only reads the model, so training is unaffected (tested); it adds one held-out scoring pass per weight set per epoch.
    The held-out set used here is the year being scored, as everywhere in these studies, so these curves describe training; choosing an epoch from
    them would be selecting on the test year.

    early_stop_fn (default None = off, training and results unchanged): a callable `model -> dict` that must contain "auc_pr" (e.g. slim_epoch_metrics of
    evaluate.compute_metrics on a VALIDATION set, never the test year -- stopping on the test year would select on the test data). When given, it is called
    after every epoch on the monitored weights (the EMA weights of `early_stop_decay`, temporarily swapped in, or the raw weights when `early_stop_decay`
    is None; a decay must be one of `ema_decays`). The best epoch is the one with the highest AUC-PR (strictly greater than any earlier epoch), and its raw
    weights AND every EMA weight set are snapshotted. Training stops once at least `early_stop_min_epochs` epochs are done and `early_stop_patience` epochs
    have passed since the best one; the loop never exceeds `epochs`. At the end the model and the EMA weights are restored to the best epoch, so everything
    downstream (checkpoints, test scoring) sees best-epoch weights. result["early_stopping"] records the settings, best_epoch, stopped_epoch, stopped_early
    and the per-epoch validation metrics. It only reads the model, so training up to the stopping epoch is identical to a run without it (tested).

    val_dataset should come from split.split_run_bins -- a leakage-safe
    split, not an arbitrary held-out set -- and is evaluated on in full
    each epoch (see evaluate()), never used for gradient updates.

    Model weight initialization is the only training-relevant source of
    torch-level randomness this function doesn't control -- call
    torch.manual_seed(seed) before constructing model for full
    end-to-end determinism. The sampler's randomness (which quiet maps
    get kept, and batch shuffling) is controlled independently via the
    seed argument, decoupled from torch's global RNG."""
    mask_t = None if cell_mask is None else torch.as_tensor(np.asarray(cell_mask), dtype=torch.float32)
    sampler = QuietMapDownsampler(masked_is_active(dataset, cell_mask), quiet_keep_fraction=quiet_keep_fraction, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ema_params = {d: [p.detach().clone() for p in model.parameters()] for d in ema_decays}

    loss_history = []
    epoch_seconds = []
    val_loss_history = [] if val_dataset is not None else None
    epoch_metrics = {"raw": [], **{f"ema{d}": [] for d in ema_decays}} if epoch_metrics_fn is not None else None
    if early_stop_fn is not None and early_stop_decay is not None and early_stop_decay not in ema_decays:
        raise ValueError(f"early_stop_decay {early_stop_decay} is not one of ema_decays {tuple(ema_decays)}")
    stop_history, best_epoch, best_auc_pr, best_raw, best_ema, stopped_early = [], 0, -float("inf"), None, None, False
    for _ in range(epochs):
        stop_now = False
        epoch_start = time.perf_counter()
        epoch_losses = []
        for X, y in loader:
            logits = model(X).squeeze(1)
            loss = criterion(logits, y) if mask_t is None else criterion(logits, y, mask_t)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            if ema_params:
                with torch.no_grad():
                    current = [p.detach() for p in model.parameters()]
                    for d, ema in ema_params.items():
                        torch._foreach_mul_(ema, d)
                        torch._foreach_add_(ema, current, alpha=1.0 - d)
            epoch_losses.append(loss.item())
        loss_history.append(float(np.mean(epoch_losses)))
        if val_dataset is not None:
            val_loss_history.append(evaluate(val_dataset, model, criterion, batch_size=batch_size, cell_mask=mask_t))
        if epoch_metrics_fn is not None:
            epoch_metrics["raw"].append(epoch_metrics_fn(model))
            for d, ema in ema_params.items():
                backup = [p.detach().clone() for p in model.parameters()]
                with torch.no_grad():
                    for p, e in zip(model.parameters(), ema):
                        p.copy_(e)
                try:
                    epoch_metrics[f"ema{d}"].append(epoch_metrics_fn(model))
                finally:
                    with torch.no_grad():
                        for p, b in zip(model.parameters(), backup):
                            p.copy_(b)
        if early_stop_fn is not None:
            if early_stop_decay is None:
                monitored = early_stop_fn(model)
            else:
                backup = [p.detach().clone() for p in model.parameters()]
                with torch.no_grad():
                    for p, e in zip(model.parameters(), ema_params[early_stop_decay]):
                        p.copy_(e)
                try:
                    monitored = early_stop_fn(model)
                finally:
                    with torch.no_grad():
                        for p, b in zip(model.parameters(), backup):
                            p.copy_(b)
            stop_history.append(monitored)
            if monitored["auc_pr"] > best_auc_pr:
                best_auc_pr, best_epoch = monitored["auc_pr"], len(loss_history)
                best_raw = [p.detach().clone() for p in model.parameters()]
                best_ema = {d: [t.clone() for t in ema] for d, ema in ema_params.items()}
            print(f"  epoch {len(loss_history)}/{epochs} train {loss_history[-1]:.3e} | monitored val AUC-PR {monitored['auc_pr']:.4f} (best {best_auc_pr:.4f} @ {best_epoch}) | {time.perf_counter() - epoch_start:.0f}s", flush=True)
            stop_now = len(loss_history) >= early_stop_min_epochs and len(loss_history) - best_epoch >= early_stop_patience
        if epoch_metrics_fn is not None and all({"auc_pr", "recall_at_best_f1", "precision_at_best_f1"} <= v[-1].keys() for v in epoch_metrics.values()):  # progress line (these runs take hours; the log is otherwise silent until the end)
            line = " | ".join(f"{k} AUC-PR {v[-1]['auc_pr']:.4f} R {v[-1]['recall_at_best_f1']:.3f} P {v[-1]['precision_at_best_f1']:.3f}" for k, v in epoch_metrics.items())
            print(f"  epoch {len(loss_history)}/{epochs} train {loss_history[-1]:.3e}" + (f" val {val_loss_history[-1]:.3e}" if val_dataset is not None else "") + f" | {line} | {time.perf_counter() - epoch_start:.0f}s", flush=True)
        epoch_seconds.append(time.perf_counter() - epoch_start)  # train + per-epoch validation, wall clock
        if stop_now:
            stopped_early = True
            break

    if early_stop_fn is not None:
        with torch.no_grad():
            for p, b in zip(model.parameters(), best_raw):
                p.copy_(b)
        ema_params = best_ema

    result = {
        "loss_history": loss_history,
        "epoch_seconds": epoch_seconds,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "alpha": alpha,
            "gamma": gamma,
            "quiet_keep_fraction": quiet_keep_fraction,
            "seed": seed,
        },
    }
    if grad_clip_norm is not None:
        result["hyperparameters"]["grad_clip_norm"] = grad_clip_norm
    if cell_mask is not None:
        result["hyperparameters"]["cell_mask_kept_cells"] = int(np.asarray(cell_mask).sum())
        result["hyperparameters"]["cell_mask_total_cells"] = int(np.asarray(cell_mask).size)
    if val_dataset is not None:
        result["val_loss_history"] = val_loss_history
    if epoch_metrics is not None:
        result["epoch_metrics"] = epoch_metrics
    if early_stop_fn is not None:
        result["early_stopping"] = {
            "monitor": "raw" if early_stop_decay is None else f"ema{early_stop_decay}", "metric": "auc_pr",
            "patience": early_stop_patience, "min_epochs": early_stop_min_epochs, "max_epochs": epochs,
            "best_epoch": best_epoch, "stopped_epoch": len(loss_history), "stopped_early": stopped_early, "history": stop_history,
        }
    if ema_params:
        names = [n for n, _ in model.named_parameters()]
        base = {k: v.detach().clone() for k, v in model.state_dict().items()}
        result["ema_state_dicts"] = {d: {**base, **{n: t.clone() for n, t in zip(names, ema)}} for d, ema in ema_params.items()}
        result["hyperparameters"]["ema_decays"] = list(ema_decays)
    return result
