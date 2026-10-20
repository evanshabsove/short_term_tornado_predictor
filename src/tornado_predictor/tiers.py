"""
Risk tiers (Low / Elevated / High) on a neighborhood-smoothed ensemble score,
plus the statistics that say what each tier measurably means.

Two layers:

1. Serving primitives, applied to one forecast's `(bins, H, W)` probability
   array: `smooth`, `assign_tiers`, `tier_set_id`. Tier codes follow
   api/README.md: -1 no model coverage, 0 Low, 1 Elevated, 2 High.

2. Study machinery (scripts/09_serving/tier_study.py,
   docs/tier_study_preregistration.md): scores are quantized to float16 and
   every statistic is computed from per-score-level count tables instead of
   from ~10^8 individual cells. A probability in [0, 1] has only 15,361
   distinct float16 values, so a table of (day x level) cell counts holds
   everything needed for AUC-PR, operating-point cutoffs, per-tier area /
   capture / hit rate, isotonic calibration and day-block bootstrap draws
   (a draw is a day-weight vector times the table). Cutoffs are therefore
   float16 levels, and a cell is in a tier when its quantized score is at or
   above the cutoff.

Everything here is evaluation or post-processing on top of the model: nothing
retrains, and no label definition changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.ndimage import maximum_filter, uniform_filter
from sklearn.isotonic import IsotonicRegression

TIER_NO_COVERAGE, TIER_LOW, TIER_ELEVATED, TIER_HIGH = -1, 0, 1, 2
TIER_NAMES = {TIER_NO_COVERAGE: "No model coverage", TIER_LOW: "Low", TIER_ELEVATED: "Elevated", TIER_HIGH: "High"}

# float16 keys 0..15360 cover [0.0, 1.0]; key order equals value order for non-negative floats.
N_LEVELS = 15361
LEVEL_VALUES = np.arange(N_LEVELS, dtype=np.uint16).view(np.float16).astype(np.float32)

OPERATORS = ("max", "mean")


# --- serving primitives ---


def smooth(scores: np.ndarray, operator: str, radius: int) -> np.ndarray:
    """Neighborhood smoothing over the last two axes (a `(2r+1) x (2r+1)` window).

    max:  window maximum; at the grid edge the window is clipped to the grid.
    mean: average of the in-grid cells of the window only (no zero padding).
    radius 0 returns a copy. Works on `(H, W)` or `(N, H, W)` arrays."""
    if operator not in OPERATORS:
        raise ValueError(f"operator must be one of {OPERATORS}, got {operator!r}")
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    x = np.asarray(scores, dtype=np.float32)
    if radius == 0:
        return x.copy()
    w = 2 * radius + 1
    size = (1,) * (x.ndim - 2) + (w, w)
    if operator == "max":
        return maximum_filter(x, size=size, mode="nearest")
    ones = np.ones(x.shape[-2:], dtype=np.float32)
    count = uniform_filter(ones, size=(w, w), mode="constant")  # fraction of the window that is in-grid
    return uniform_filter(x, size=size, mode="constant") / count


def dilate_labels(labels: np.ndarray, k: int) -> np.ndarray:
    """A cell is positive if it or any cell within the `(2k+1)^2` window is.
    Zero padding outside the grid. k=0 returns a copy."""
    y = np.asarray(labels)
    if k == 0:
        return y.copy()
    return maximum_filter(y, size=(1,) * (y.ndim - 2) + (2 * k + 1, 2 * k + 1), mode="constant", cval=0)


def assign_tiers(smoothed: np.ndarray, cutoffs: dict, coverage: np.ndarray) -> np.ndarray:
    """int8 tier codes. `cutoffs` = {"elevated": e, "high": h} on the smoothed
    score; a score exactly at a cutoff goes up a tier. `coverage` is a boolean
    `(H, W)` mask; cells outside it are -1 whatever their score."""
    e, h = float(cutoffs["elevated"]), float(cutoffs["high"])
    if not e < h:
        raise ValueError(f"elevated cutoff {e} must be below the high cutoff {h}")
    s = np.asarray(smoothed)
    tiers = np.full(s.shape, TIER_LOW, dtype=np.int8)
    tiers[s >= e] = TIER_ELEVATED
    tiers[s >= h] = TIER_HIGH
    tiers[..., ~np.asarray(coverage, dtype=bool)] = TIER_NO_COVERAGE
    return tiers


def tier_forecast(
    mean: np.ndarray, operator: str, radius: int, cutoffs: dict, coverage: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The single serving path from raw ensemble probabilities to tiers:
    smooth, quantize to float16, then compare with the cutoffs. This is exactly
    the order the tier study measured (scores are quantized after smoothing and
    a cell is in a tier when its quantized score is at or above the cutoff), so
    a served tier means what the study's table says it means.

    mean: `(H, W)` or `(N, H, W)` probabilities. Returns `(tiers, smoothed)`:
    int8 tier codes and the quantized smoothed score as float32."""
    smoothed = quantize(smooth(mean, operator, radius)).astype(np.float32)
    return assign_tiers(smoothed, cutoffs, coverage), smoothed


def tier_set_id(operator: str, radius: int, cutoffs: dict) -> str:
    """`ts-` + 8 hex chars; changes whenever the operator, radius or either
    cutoff (to 6 significant figures) does."""
    payload = json.dumps(
        {
            "operator": operator,
            "radius": int(radius),
            "cutoffs": {k: float(f"{float(v):.6g}") for k, v in sorted(cutoffs.items())},
        },
        sort_keys=True,
    )
    return "ts-" + hashlib.sha256(payload.encode()).hexdigest()[:8]


# --- quantization ---


def quantize(scores: np.ndarray) -> np.ndarray:
    """Clip to [0, 1] and round to float16. Raises on NaN."""
    s = np.asarray(scores, dtype=np.float32)
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    return np.clip(s, 0.0, 1.0).astype(np.float16)


def level_keys(scores16: np.ndarray) -> np.ndarray:
    """int64 level index (0..15360) of float16 scores in [0, 1]."""
    return np.ascontiguousarray(scores16).view(np.uint16).astype(np.int64)


def key_of(value: float) -> int:
    """Level index of a float16-representable cutoff value."""
    return int(np.float16(value).view(np.uint16))


# --- count tables ---


@dataclass(frozen=True)
class LevelCounts:
    """Weighted cell counts per score level, as 1-D arrays of length N_LEVELS
    indexed by key (ascending score). total: coverage cells; pos: cells with a
    tornado (undilated label); dpos: dilated-positive cells."""

    total: np.ndarray
    pos: np.ndarray
    dpos: np.ndarray

    def __add__(self, other: "LevelCounts") -> "LevelCounts":
        return LevelCounts(self.total + other.total, self.pos + other.pos, self.dpos + other.dpos)


class LevelTableBuilder:
    """Accumulates per-day sparse rows; `build()` gives a `LevelTable`."""

    def __init__(self, n_days: int):
        self.n_days = n_days
        self._rows: list[int] = []
        self._cols: list[np.ndarray] = []
        self._data = {"total": [], "pos": [], "dpos": []}

    def add(self, day: int, keys: np.ndarray, pos: np.ndarray, dpos: np.ndarray, weights=1.0) -> None:
        """One day's cells: parallel arrays of level keys and 0/1 labels.
        `weights` is a scalar or an array parallel to keys."""
        keys = np.asarray(keys, dtype=np.int64).ravel()
        if keys.size == 0:
            return
        if keys.min() < 0 or keys.max() >= N_LEVELS:
            raise ValueError("keys outside [0, N_LEVELS)")
        w = np.broadcast_to(np.asarray(weights, dtype=np.float64), keys.shape)
        uniq, inv = np.unique(keys, return_inverse=True)
        n = uniq.size
        self._rows.append(np.full(n, day, dtype=np.int64))
        self._cols.append(uniq)
        self._data["total"].append(np.bincount(inv, weights=w, minlength=n))
        self._data["pos"].append(np.bincount(inv, weights=w * np.asarray(pos).ravel(), minlength=n))
        self._data["dpos"].append(np.bincount(inv, weights=w * np.asarray(dpos).ravel(), minlength=n))

    def build(self) -> "LevelTable":
        shape = (self.n_days, N_LEVELS)
        if not self._rows:
            empty = sparse.csr_matrix(shape)
            return LevelTable(empty, empty.copy(), empty.copy())
        rows, cols = np.concatenate(self._rows), np.concatenate(self._cols)
        mats = {
            name: sparse.coo_matrix((np.concatenate(parts).astype(np.float32), (rows, cols)), shape=shape).tocsr()  # sums duplicates
            for name, parts in self._data.items()
        }
        return LevelTable(**mats)


@dataclass(frozen=True, eq=False)
class LevelTable:
    """(n_days x N_LEVELS) sparse weighted counts. A bootstrap draw or a
    subset of days is a day-weight vector: `counts(w)`."""

    total: sparse.csr_matrix
    pos: sparse.csr_matrix
    dpos: sparse.csr_matrix

    def counts(self, day_weights: np.ndarray | None = None) -> LevelCounts:
        w = np.ones(self.total.shape[0]) if day_weights is None else np.asarray(day_weights, dtype=np.float64)
        return LevelCounts(*(np.asarray(m.T @ w).ravel() for m in (self.total, self.pos, self.dpos)))


# --- curves and operating points (all on LevelCounts) ---


def curves(c: LevelCounts) -> dict:
    """Cumulative quantities walking the levels from the highest score down.
    Index j is the j-th highest level; values are the level's score."""
    t, p, d = c.total[::-1], c.pos[::-1], c.dpos[::-1]
    ct, cp, cd = np.cumsum(t), np.cumsum(p), np.cumsum(d)
    T, P, D = ct[-1], cp[-1], cd[-1]
    if T <= 0:
        raise ValueError("no coverage cells in this sample")
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(ct > 0, cd / ct, 0.0)
        recall = cd / D if D > 0 else np.zeros_like(cd)
        capture = cp / P if P > 0 else np.zeros_like(cp)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {
        "values": LEVEL_VALUES[::-1],
        "occupied": t > 0,
        "area": ct / T,
        "capture": capture,
        "precision": precision,
        "dilated_recall": recall,
        "f1": f1,
        "totals": {"total": float(T), "pos": float(P), "dpos": float(D)},
    }


def average_precision(c: LevelCounts) -> float:
    """AUC-PR of score vs dilated label (ties grouped by level, as sklearn)."""
    cv = curves(c)
    D = cv["totals"]["dpos"]
    if D <= 0:
        return float("nan")
    return float(np.sum(c.dpos[::-1] / D * cv["precision"]))


def best_f1_cutoff(c: LevelCounts) -> tuple[float, dict]:
    """Level maximizing F1 of (score >= level) against the dilated label;
    on a tie the higher score wins. Returns (cutoff, stats at the cutoff)."""
    cv = curves(c)
    f1 = np.where(cv["occupied"], cv["f1"], -1.0)
    j = int(np.argmax(f1))
    return float(cv["values"][j]), {
        "f1": float(cv["f1"][j]),
        "precision": float(cv["precision"][j]),
        "dilated_recall": float(cv["dilated_recall"][j]),
        "area": float(cv["area"][j]),
        "capture": float(cv["capture"][j]),
    }


def capture_cutoff(c: LevelCounts, target: float) -> float:
    """Highest level whose cumulative capture share (undilated tornado cells
    at or above it) reaches `target`."""
    cv = curves(c)
    ok = cv["occupied"] & (cv["capture"] >= target)
    if not ok.any():
        raise ValueError(f"capture {target} is never reached")
    return float(cv["values"][int(np.argmax(ok))])


def area_cutoff(c: LevelCounts, share: float) -> float:
    """Lowest occupied level whose cumulative area share is still at most
    `share` (if even the top level exceeds it, the top occupied level)."""
    cv = curves(c)
    occ = np.flatnonzero(cv["occupied"])
    within = occ[cv["area"][occ] <= share]
    j = int(within[-1]) if within.size else int(occ[0])
    return float(cv["values"][j])


def nearest_area_cutoff(c: LevelCounts, share: float) -> float:
    """Occupied level whose cumulative area share is closest to `share`
    (higher level on a tie)."""
    cv = curves(c)
    occ = np.flatnonzero(cv["occupied"])
    return float(cv["values"][int(occ[np.argmin(np.abs(cv["area"][occ] - share))])])


def area_at(c: LevelCounts, cutoff: float) -> float:
    """Share of coverage cells with score at or above the cutoff."""
    k = key_of(cutoff)
    return float(c.total[k:].sum() / c.total.sum())


def elevated_cutoff(
    c: LevelCounts, high: float, capture_target: float = 0.6, area_cap: float = 0.15, high_multiple: float = 3.0
) -> tuple[float, str]:
    """The pre-registered Elevated rule: highest level reaching `capture_target`
    cumulative capture; (a) if that puts more than `area_cap` of the map at or
    above it, use the area_cap quantile; (b) if the result is not strictly
    below `high`, use the level whose area is `high_multiple` x High's.
    Returns (cutoff, which rule produced it)."""
    e, rule = capture_cutoff(c, capture_target), "capture"
    if area_at(c, e) > area_cap:
        e, rule = area_cutoff(c, area_cap), "area_cap"
    if e >= high:
        e, rule = nearest_area_cutoff(c, high_multiple * area_at(c, high)), "high_multiple"
    if not e < high:
        raise ValueError(f"cannot place the Elevated cutoff strictly below High ({e} vs {high})")
    return e, rule


def tier_counts(c: LevelCounts, cutoffs: dict) -> dict:
    """Per-tier statistics from a count table: exclusive tiers (low, elevated,
    high) and cumulative ones (elevated_up = Elevated or High, high_up = High).
    area_share: fraction of coverage cells; capture_share: fraction of
    undilated tornado cells; hit_rate: fraction of the tier's cells that are
    dilated-positive; undilated_hit_rate: fraction that are tornado cells."""
    ek, hk = key_of(cutoffs["elevated"]), key_of(cutoffs["high"])
    T, P = c.total.sum(), c.pos.sum()

    def block(lo: int, hi: int) -> dict:
        t, p, d = c.total[lo:hi].sum(), c.pos[lo:hi].sum(), c.dpos[lo:hi].sum()
        return {
            "area_share": float(t / T) if T else float("nan"),
            "capture_share": float(p / P) if P else float("nan"),
            "hit_rate": float(d / t) if t else float("nan"),
            "undilated_hit_rate": float(p / t) if t else float("nan"),
            "n_cells": float(t),
            "n_tornado_cells": float(p),
        }

    return {
        "low": block(0, ek),
        "elevated": block(ek, hk),
        "high": block(hk, N_LEVELS),
        "elevated_up": block(ek, N_LEVELS),
    }


# --- calibration ---


def isotonic_map(c: LevelCounts) -> np.ndarray:
    """Monotone non-decreasing calibrated probability for every level, fit by
    isotonic regression of the dilated-positive fraction on the score
    (weighted by cell counts). Returns an array of length N_LEVELS."""
    occ = c.total > 0
    if not occ.any() or c.dpos.sum() <= 0:
        raise ValueError("cannot fit a calibration map without positives")
    iso = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(LEVEL_VALUES[occ], c.dpos[occ] / c.total[occ], sample_weight=c.total[occ])
    return iso.predict(LEVEL_VALUES).astype(np.float64)


def quantile_bins(c: LevelCounts, edges=(0.0, 0.5, 0.9, 0.97, 0.99, 0.997, 0.999, 1.0)) -> list[np.ndarray]:
    """Boolean level masks (length N_LEVELS) for the score-quantile bins
    [e0,e1), ..., [e_{n-1}, e_n], listed lowest bin first. A level goes to the
    bin of its cumulative area share counted from the top (first edge at least
    that share), so ties never split a level. Unoccupied levels are in no bin."""
    top_share = np.empty(N_LEVELS)
    top_share[::-1] = curves(c)["area"]  # share of cells at or above each level, indexed by key
    n_bins = len(edges) - 1
    # ascending top-shares bounding each bin from the top: (0, 1-e_{n-1}], ..., (1-e_1, 1]
    bounds = np.asarray([1.0 - e for e in edges[-2:0:-1]] + [1.0])
    bin_from_top = np.searchsorted(bounds, top_share, side="left")
    occupied = c.total > 0
    return [occupied & (bin_from_top == n_bins - 1 - b) for b in range(n_bins)]


def ratio_over(c: LevelCounts, calibrated: np.ndarray, level_mask: np.ndarray) -> dict:
    """Calibration check over a set of levels: mean predicted probability,
    observed dilated-positive rate, and their ratio (predicted / observed)."""
    t = c.total[level_mask].sum()
    if t <= 0:
        return {"n_cells": 0.0, "predicted": float("nan"), "observed": float("nan"), "ratio": float("nan"), "n_positive": 0.0}
    pred = float((c.total[level_mask] * calibrated[level_mask]).sum() / t)
    d = float(c.dpos[level_mask].sum())
    obs = d / float(t)
    return {
        "n_cells": float(t),
        "predicted": pred,
        "observed": obs,
        "ratio": pred / obs if obs > 0 else float("nan"),
        "n_positive": d,
    }


def match_levels(src: LevelCounts, ref: LevelCounts) -> np.ndarray:
    """Quantile matching: for every occupied level of `src`, the level of `ref`
    whose cumulative area share (from the top) is nearest. Returns an array of
    length N_LEVELS of ref-scale scores (NaN where `src` has no cells)."""
    a, b = curves(src), curves(ref)
    ref_occ = np.flatnonzero(b["occupied"])
    ref_area, ref_vals = b["area"][ref_occ], b["values"][ref_occ]  # area is non-decreasing along j
    out = np.full(N_LEVELS, np.nan)
    src_occ = np.flatnonzero(a["occupied"])
    pos = np.clip(np.searchsorted(ref_area, a["area"][src_occ]), 0, len(ref_area) - 1)
    lower = np.clip(pos - 1, 0, len(ref_area) - 1)
    pick = np.where(np.abs(ref_area[lower] - a["area"][src_occ]) <= np.abs(ref_area[pos] - a["area"][src_occ]), lower, pos)
    out[N_LEVELS - 1 - src_occ] = ref_vals[pick]
    return out
