"""
Are ensembles stable? On the three seed-study folds we have 4 independently trained models (old recipe, v4 seeds 0/1/2).
Split them into two DISJOINT 2-member ensembles in all 3 possible ways and compare the two ensembles' AP (with day-bootstrap
paired intervals), versus the disagreement between disjoint single models (6 model pairs). Evaluation only; reuses the
day x score-level table machinery from bootstrap_stability.py.
Output: models/seed_study/ensemble_stability.json
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap_stability as bs  # noqa: E402
import diagnose_sample_mix as dsm  # noqa: E402

OUT = dsm.REPO / "models" / "seed_study" / "ensemble_stability.json"
SPLITS = [(("old", "s0"), ("s1", "s2")), (("old", "s1"), ("s0", "s2")), (("old", "s2"), ("s0", "s1"))]
rng = np.random.default_rng(1)
B = 2000
res = []
for year in bs.SEED_FOLDS:
    meta = np.load(dsm.CACHE / f"meta_{year}.npz")
    y = meta["label"].astype(np.float32)
    days, day_idx = np.unique(meta["init_time"] // 86400, return_inverse=True)
    D = len(days)
    M = {"old": np.load(dsm.CACHE / f"old_{year}.npy").astype(np.float32), "s0": np.load(dsm.CACHE / f"v4_{year}.npy").astype(np.float32),
         "s1": np.load(dsm.CACHE / f"seed1_{year}.npy").astype(np.float32), "s2": np.load(dsm.CACHE / f"seed2_{year}.npy").astype(np.float32)}
    W = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(B)]).astype(np.float64)
    def stats(score):
        t = bs.build_table(score, y, day_idx, D)
        return bs.ap_from_weights(*t, np.ones(D)), np.array([bs.ap_from_weights(*t, w) for w in W])
    single = {k: stats(v) for k, v in M.items()}
    row = {"year": year, "single_pairs": [], "ens_pairs": []}
    for a, b in combinations(M, 2):
        pa, ba = single[a]; pb, bb = single[b]; d = ba - bb
        row["single_pairs"].append({"pair": f"{a}|{b}", "ap": [pa, pb], "rel_gap": abs(pa - pb) / np.mean([pa, pb]), "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)})
    for (a1, a2), (b1, b2) in SPLITS:
        pa, ba = stats(0.5 * (M[a1] + M[a2])); pb, bb = stats(0.5 * (M[b1] + M[b2])); d = ba - bb
        row["ens_pairs"].append({"split": f"({a1}+{a2})|({b1}+{b2})", "ap": [pa, pb], "rel_gap": abs(pa - pb) / np.mean([pa, pb]), "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
                                 "ci95_diff": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]})
    res.append(row)
    print(year, "single rel gaps", [round(100 * p["rel_gap"]) for p in row["single_pairs"]], "| ens rel gaps", [round(100 * p["rel_gap"]) for p in row["ens_pairs"]], "| ens APs", [[round(x, 4) for x in p["ap"]] for p in row["ens_pairs"]], flush=True)
OUT.write_text(json.dumps(res, indent=1))
sg = [p["rel_gap"] for r in res for p in r["single_pairs"]]; eg = [p["rel_gap"] for r in res for p in r["ens_pairs"]]
se = [p["excludes_zero"] for r in res for p in r["single_pairs"]]; ee = [p["excludes_zero"] for r in res for p in r["ens_pairs"]]
print(f"\nmean |gap|/mean: single-model pairs {100*np.mean(sg):.0f}% (n={len(sg)}; differ beyond resampling in {sum(se)}/{len(se)}), disjoint 2-member ensembles {100*np.mean(eg):.0f}% (n={len(eg)}; differ beyond resampling in {sum(ee)}/{len(ee)})")
