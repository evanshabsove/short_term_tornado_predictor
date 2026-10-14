"""
EXPLORATORY (post hoc, not pre-registered) follow-up to scripts/07_updraft_helicity/uh_eval.py: with only three UH models per fold, one weak member can dominate a
3-member ensemble. Scores every UH 2-member ensemble and each UH single against the 15 possible 2-member baseline ensembles (six baseline
averaged models) per fold. Cached scores only. Output: models/uh_study/uh_pairs_exploratory.json
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_year_cv_diagnostics"))
import bootstrap_stability as bs  # noqa: E402
import diagnose_sample_mix as dsm  # noqa: E402

OUT = dsm.REPO / "models" / "uh_study" / "uh_pairs_exploratory.json"


def ap(score, y, di, D):
    return bs.ap_from_weights(*bs.build_table(score, y, di, D), np.ones(D))


res = []
for year in bs.SEED_FOLDS:
    meta = np.load(dsm.CACHE / f"meta_{year}.npz")
    y = meta["label"].astype(np.float32)
    days, di = np.unique(meta["init_time"] // 86400, return_inverse=True)
    D = len(days)
    UH = {s: np.load(dsm.CACHE / f"uhstudy_ema0.9999_s{s}_{year}.npy").astype(np.float32) for s in (10, 11, 12)}
    BA = {s: np.load(dsm.CACHE / f"emastudy_ema0.9999_s{s}_{year}.npy").astype(np.float32) for s in range(10, 16)}
    uh_pairs = {"+".join(map(str, c)): ap(np.mean([UH[s] for s in c], axis=0), y, di, D) for c in combinations((10, 11, 12), 2)}
    ba_pairs = np.array([ap(np.mean([BA[s] for s in c], axis=0), y, di, D) for c in combinations(range(10, 16), 2)])
    res.append({"year": year, "uh_singles": {str(s): ap(UH[s], y, di, D) for s in UH}, "uh_pairs": uh_pairs,
                "base_pairs_mean": float(ba_pairs.mean()), "base_pairs_sd": float(ba_pairs.std(ddof=1)),
                "base_pairs_p10": float(np.percentile(ba_pairs, 10)), "base_pairs_p90": float(np.percentile(ba_pairs, 90))})
    print(year, res[-1]["uh_pairs"], "baseline pair mean", round(res[-1]["base_pairs_mean"], 4), flush=True)
OUT.write_text(json.dumps(res, indent=1))
