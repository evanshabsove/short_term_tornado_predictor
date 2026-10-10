"""
Evaluation-only follow-up to the seed study (see notebooks/sample_mix_diagnostics.ipynb, Part 2).
Uses the cached per-cell scores in data/interim/diag_scores/ (no retraining):
  A) all 12 folds: mean-probability ensemble of the old model and the v4 seed-0 model vs. each member alone.
  B) the 3 seed-study folds (2016/2020/2024): every 2-member ensemble -- same-recipe seed pairs (0,1),(0,2),(1,2)
     and old+seed pairs (old,0),(old,1),(old,2) -- vs. the mean of its members, to test whether old and v4 models are
     exchangeable (an old+v4 pair should gain about as much as a seed+seed pair if so).
Output: models/seed_study/ensemble_followup.json
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_sample_mix as dsm  # noqa: E402

OUT = dsm.REPO / "models" / "seed_study" / "ensemble_followup.json"
YEARS = list(range(2014, 2026))
SEED_FOLDS = (2016, 2020, 2024)


def load(tag, year):
    return np.load(dsm.CACHE / f"{tag}_{year}.npy").astype("float32")


def metrics(y, s):
    m = dsm.cell_metrics(y, s)
    return {k: m[k] for k in ("auc_pr", "f1_at_best_f1", "precision_at_best_f1", "recall_at_best_f1", "pos_rate")}


res = {"all_folds_old_v4": [], "pairs": []}
for year in YEARS:
    y = np.load(dsm.CACHE / f"meta_{year}.npz")["label"].astype("float32")
    old, v4 = load("old", year), load("v4", year)
    row = {"year": year, "old": metrics(y, old), "v4_seed0": metrics(y, v4), "ens": metrics(y, 0.5 * (old + v4))}
    res["all_folds_old_v4"].append(row)
    print(f"[A] {year}: old {row['old']['auc_pr']:.4f} seed0 {row['v4_seed0']['auc_pr']:.4f} ens {row['ens']['auc_pr']:.4f}", flush=True)
    if year in SEED_FOLDS:
        S = {"old": old, "s0": v4, "s1": load("seed1", year), "s2": load("seed2", year)}
        for a, b in combinations(S, 2):
            kind = "seed+seed" if "old" not in (a, b) else "old+seed"
            ma, mb, me = metrics(y, S[a]), metrics(y, S[b]), metrics(y, 0.5 * (S[a] + S[b]))
            res["pairs"].append({"year": year, "pair": f"{a}+{b}", "kind": kind, "members": [ma["auc_pr"], mb["auc_pr"]],
                                 "ens_auc_pr": me["auc_pr"], "members_f1": [ma["f1_at_best_f1"], mb["f1_at_best_f1"]], "ens_f1": me["f1_at_best_f1"]})
            print(f"[B] {year} {a}+{b} ({kind}): members {ma['auc_pr']:.4f},{mb['auc_pr']:.4f} -> ens {me['auc_pr']:.4f}", flush=True)
OUT.write_text(json.dumps(res, indent=1))
print("DONE")
