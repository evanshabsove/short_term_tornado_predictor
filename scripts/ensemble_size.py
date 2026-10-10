"""
Ensemble-size study. With up to 6 independently seeded v4 models per seed-study fold (seed 0 = existing CV model, seeds 1-5 from
models/seed_study/), measure how ensemble size changes (a) mean AUC-PR, (b) how much two DISJOINT ensembles of the same size disagree
(the stability test), and (c) whether those disagreements exceed test-day sampling noise (day-block bootstrap, paired).

Scores any new seed checkpoints on the cached test samples first (reuses seed_study_eval.score_new_seeds), then for every non-empty
subset of the available seeds builds the mean-probability ensemble and computes AP (float16-quantized scores, same machinery as
bootstrap_stability.py) with bootstrap draws. Disjoint-split gaps are computed for k = 1, 2, 3 when 2k <= number of seeds.

Usage: python scripts/ensemble_size.py [--boots 500] [--years 2016 2020 2024] [--out PATH]
Output: models/seed_study/ensemble_size.json
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap_stability as bs  # noqa: E402
import diagnose_sample_mix as dsm  # noqa: E402
import seed_study_eval as sse  # noqa: E402

from tornado_predictor.build_scaled import load_manifest  # noqa: E402
from tornado_predictor.time_bins import N_BINS  # noqa: E402

ALL_SEEDS = (1, 2, 3, 4, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boots", type=int, default=500)
    ap.add_argument("--years", type=int, nargs="+", default=list(bs.SEED_FOLDS))
    ap.add_argument("--out", type=Path, default=dsm.REPO / "models" / "seed_study" / "ensemble_size.json")
    ap.add_argument("--extra-raw-seeds", type=int, nargs="*", default=[], help="also pool the plain (raw) models of these EMA-study seeds (scores cached by scripts/ema_eval.py)")
    ap.add_argument("--exclude-s0", action="store_true", help="drop seed 0 from the pool (its folds were selected for a large v4-vs-old gap)")
    ap.add_argument("--emastudy-variant", default=None, help="pool ONLY the EMA-study models of this variant (raw | ema0.9999 | ema0.999), seeds given by --emastudy-seeds (scores cached by scripts/ema_eval.py)")
    ap.add_argument("--emastudy-seeds", type=int, nargs="*", default=[10, 11, 12, 13, 14, 15])
    ap.add_argument("--max-k", type=int, default=99, help="only build ensembles up to this size (plus the full pool)")
    args = ap.parse_args()
    sse.SEEDS = ALL_SEEDS
    manifest = load_manifest(dsm.STAGING / "manifest.json")
    run_bins = [(pd.Timestamp(k), b) for k, v in sorted(manifest.items()) if v["status"] == "done" for b in range(N_BINS)]
    rng = np.random.default_rng(2)
    out = []
    for year in args.years:
        if not args.emastudy_variant:
            sse.score_new_seeds(year, run_bins)
        meta = np.load(dsm.CACHE / f"meta_{year}.npz")
        y = meta["label"].astype(np.float32)
        days, day_idx = np.unique(meta["init_time"] // 86400, return_inverse=True)
        D = len(days)
        if args.emastudy_variant:
            M = {f"s{s}": np.load(dsm.CACHE / f"emastudy_{args.emastudy_variant}_s{s}_{year}.npy").astype(np.float32)
                 for s in args.emastudy_seeds if (dsm.CACHE / f"emastudy_{args.emastudy_variant}_s{s}_{year}.npy").exists()}
        else:
            M = {"s0": np.load(dsm.CACHE / f"v4_{year}.npy").astype(np.float32)}
            for s in ALL_SEEDS:
                p = dsm.CACHE / f"seed{s}_{year}.npy"
                if p.exists():
                    M[f"s{s}"] = np.load(p).astype(np.float32)
        for s_ in args.extra_raw_seeds:
            M[f"s{s_}"] = np.load(dsm.CACHE / f"emastudy_raw_s{s_}_{year}.npy").astype(np.float32)
        if args.exclude_s0:
            M.pop("s0", None)
        names = sorted(M, key=lambda x: int(x[1:]))
        n = len(names)
        ks = list(range(1, min(args.max_k, n) + 1)) + ([n] if n > args.max_k else [])
        W = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(args.boots)]).astype(np.float64)
        subs = {}
        for k in ks:
            for c in combinations(names, k):
                tab = bs.build_table(np.mean([M[x] for x in c], axis=0), y, day_idx, D)
                point = bs.ap_from_weights(*tab, np.ones(D))
                boot = np.array([bs.ap_from_weights(*tab, w) for w in W])
                subs["+".join(c)] = {"k": k, "ap": point, "boot": boot}
        print(f"[{year}] {n} seeds ({','.join(names)}) -> {len(subs)} ensembles; full-ensemble AP {subs['+'.join(names)]['ap']:.4f}", flush=True)
        row = {"year": year, "n_seeds": n, "seeds": names, "n_days": int(D), "size_curve": {}, "splits": {}, "full_ensemble": {}}
        for k in ks:
            aps = np.array([v["ap"] for v in subs.values() if v["k"] == k])
            brs = np.array([v["boot"].std(ddof=1) / v["ap"] for v in subs.values() if v["k"] == k])
            row["size_curve"][str(k)] = {"n_subsets": int(len(aps)), "mean_ap": float(aps.mean()), "sd_across_subsets": float(aps.std(ddof=1)) if len(aps) > 1 else None,
                                         "rel_sd_across_subsets": float(aps.std(ddof=1) / aps.mean()) if len(aps) > 1 else None, "mean_boot_rel_sd": float(brs.mean())}
        full = subs["+".join(names)]
        row["full_ensemble"] = {"ap": full["ap"], "ci95": [float(np.percentile(full["boot"], 2.5)), float(np.percentile(full["boot"], 97.5))],
                                "boot_rel_sd": float(full["boot"].std(ddof=1) / full["ap"])}
        for k in (1, 2, 3, 4):
            if 2 * k > n or k not in ks:
                continue
            gaps = []
            seen = set()
            for a in combinations(names, k):
                rest = [x for x in names if x not in a]
                for b in combinations(rest, k):
                    key = frozenset([a, b])
                    if key in seen:
                        continue
                    seen.add(key)
                    A, B = subs["+".join(a)], subs["+".join(b)]
                    d = A["boot"] - B["boot"]
                    gaps.append({"split": f"{'+'.join(a)}|{'+'.join(b)}", "ap": [A["ap"], B["ap"]], "rel_gap": abs(A["ap"] - B["ap"]) / np.mean([A["ap"], B["ap"]]),
                                 "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)})
            row["splits"][str(k)] = {"n_splits": len(gaps), "mean_rel_gap": float(np.mean([g["rel_gap"] for g in gaps])),
                                     "max_rel_gap": float(np.max([g["rel_gap"] for g in gaps])),
                                     "frac_beyond_noise": float(np.mean([g["excludes_zero"] for g in gaps])), "detail": gaps}
        out.append(row)
        msg = " | ".join(f"k={k}: gap {100*v['mean_rel_gap']:.0f}% (max {100*v['max_rel_gap']:.0f}%), beyond-noise {100*v['frac_beyond_noise']:.0f}% of {v['n_splits']}" for k, v in row["splits"].items())
        print(f"   disjoint-split stability: {msg}", flush=True)
        print("   mean AP by size:", {k: round(v["mean_ap"], 4) for k, v in row["size_curve"].items()}, flush=True)
        args.out.write_text(json.dumps(out, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
