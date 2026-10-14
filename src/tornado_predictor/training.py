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
    val_loss_history = [] if val_dataset is not None else None
    for _ in range(epochs):
        epoch_losses = []
        for X, y in loader:
            logits = model(X).squeeze(1)
            loss = criterion(logits, y) if mask_t is None else criterion(logits, y, mask_t)
            optimizer.zero_grad()
            loss.backward()
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

    result = {
        "loss_history": loss_history,
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
    if cell_mask is not None:
        result["hyperparameters"]["cell_mask_kept_cells"] = int(np.asarray(cell_mask).sum())
        result["hyperparameters"]["cell_mask_total_cells"] = int(np.asarray(cell_mask).size)
    if val_dataset is not None:
        result["val_loss_history"] = val_loss_history
    if ema_params:
        names = [n for n, _ in model.named_parameters()]
        base = {k: v.detach().clone() for k, v in model.state_dict().items()}
        result["ema_state_dicts"] = {d: {**base, **{n: t.clone() for n, t in zip(names, ema)}} for d, ema in ema_params.items()}
        result["hyperparameters"]["ema_decays"] = list(ema_decays)
    return result
