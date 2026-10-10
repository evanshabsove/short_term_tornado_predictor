"""
Is the seed-to-seed spread in AUC-PR model instability, or just test-set sampling noise in the metric?

Day-block bootstrap over cached per-cell scores (data/interim/diag_scores/; no retraining). The test unit is
the UTC calendar date of init_time (all samples from one date are correlated and are resampled together).

Speed trick: scores are cached as float16, so each model-year is compressed once into a (day x score-level)
table of positive and total cell counts. A bootstrap draw is then a day-weight vector w; AP is computed from
w @ counts and a cumulative sum over score levels (ties grouped, same definition as sklearn's average_precision).
Scores of ensembles are quantized to float16 too, so the point AP differs from the earlier float32-based numbers
only in the 4th decimal.

Per fold: bootstrap SD / 95% interval of AP for each model; paired-difference intervals for seed pairs, for old
vs. v4 seed 0 (all 12 folds), and for the old+v4 ensemble vs each member; seed spread under resampling.
Usage: python scripts/bootstrap_stability.py [--boots 2000]
Output: models/seed_study/bootstrap_stability.json
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_sample_mix as dsm  # noqa: E402

OUT = dsm.REPO / "models" / "seed_study" / "bootstrap_stability.json"
SEED_FOLDS = (2016, 2020, 2024)


def build_table(score, y, day_idx, n_days):
    """(n_days x K) positive and total counts per float16 score level, columns ordered by DESCENDING score."""
    key = score.astype(np.float16).view(np.uint16).astype(np.int64)          # monotone in value for non-negative floats
    n = key.shape[0]
    flat = (day_idx[:, None, None] * 65536 + key).ravel()
    tot = np.bincount(flat, minlength=n_days * 65536).reshape(n_days, 65536)
    pos = np.bincount(flat, weights=y.ravel(), minlength=n_days * 65536).reshape(n_days, 65536)
    keep = np.where(tot.sum(0) > 0)[0][::-1]                                   # occupied levels, high score first
    return pos[:, keep].astype(np.float64), tot[:, keep].astype(np.float64)


def ap_from_weights(pos, tot, w):
    p, t = w @ pos, w @ tot
    P = p.sum()
    if P <= 0:
        return np.nan
    tp, n = np.cumsum(p), np.cumsum(t)
    prec = np.divide(tp, n, out=np.zeros_like(tp), where=n > 0)   # levels with no resampled cells contribute 0 (their p is 0 too)
    return float(np.sum(p / P * prec))


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boots", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    results = []
    for year in range(2014, 2026):
        meta = np.load(dsm.CACHE / f"meta_{year}.npz")
        y = meta["label"].astype(np.float32)
        days, day_idx = np.unique(meta["init_time"] // 86400, return_inverse=True)
        D = len(days)
        S = {"old": np.load(dsm.CACHE / f"old_{year}.npy").astype(np.float32), "seed0": np.load(dsm.CACHE / f"v4_{year}.npy").astype(np.float32)}
        if year in SEED_FOLDS:
            for s in (1, 2):
                S[f"seed{s}"] = np.load(dsm.CACHE / f"seed{s}_{year}.npy").astype(np.float32)
            v4 = [k for k in S if k.startswith("seed")]
            S["ens_seeds"] = np.mean([S[k] for k in v4], axis=0)
            S["ens_all"] = np.mean([S["old"]] + [S[k] for k in v4], axis=0)
        S["ens_old_v4"] = 0.5 * (S["old"] + S["seed0"])
        T = {k: build_table(v, y, day_idx, D) for k, v in S.items()}
        names = list(S)
        W = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(args.boots)]).astype(np.float64)
        pt = {k: ap_from_weights(*T[k], np.ones(D)) for k in names}
        boot = {k: np.array([ap_from_weights(*T[k], w) for w in W]) for k in names}
        row = {"year": year, "n_days": int(D), "n_samples": int(len(y)),
               "models": {k: {"ap": pt[k], "boot_sd": float(boot[k].std(ddof=1)), "boot_rel_sd": float(boot[k].std(ddof=1) / pt[k]),
                              "ci95": ci(boot[k]), "boot_mean": float(boot[k].mean())} for k in names}}
        pairs = [("seed0", "old")] + [("ens_old_v4", "old"), ("ens_old_v4", "seed0")]
        if year in SEED_FOLDS:
            pairs += [("seed1", "seed0"), ("seed2", "seed0"), ("seed2", "seed1"), ("ens_seeds", "seed0"), ("ens_seeds", "seed1"), ("ens_seeds", "seed2")]
        row["paired"] = {}
        for a, b in pairs:
            d = boot[a] - boot[b]
            row["paired"][f"{a}-{b}"] = {"point": pt[a] - pt[b], "ci95": ci(d), "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
                                         "rel_point": (pt[a] - pt[b]) / pt[b], "p_same_sign": float(np.mean(np.sign(d) == np.sign(pt[a] - pt[b])))}
        if year in SEED_FOLDS:
            M = np.stack([boot[f"seed{i}"] for i in range(3)])             # seeds x boots
            cv_b = M.std(axis=0, ddof=1) / M.mean(axis=0)
            row["seed_spread_under_resampling"] = {"point_cv": float(np.std([pt[f"seed{i}"] for i in range(3)], ddof=1) / np.mean([pt[f"seed{i}"] for i in range(3)])),
                                                   "mean_cv_over_boots": float(cv_b.mean()), "ci95_cv": ci(cv_b)}
        results.append(row)
        m = row["models"]
        print(f"[{year}] days={D} old AP={m['old']['ap']:.4f} (boot rel SD {100*m['old']['boot_rel_sd']:.0f}%, CI {m['old']['ci95'][0]:.4f}-{m['old']['ci95'][1]:.4f}) | "
              f"seed0 {m['seed0']['ap']:.4f} (rel SD {100*m['seed0']['boot_rel_sd']:.0f}%) | ens_old_v4 {m['ens_old_v4']['ap']:.4f}", flush=True)
        OUT.write_text(json.dumps({"boots": args.boots, "folds": results}, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
