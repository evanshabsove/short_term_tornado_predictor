"""
Evaluate the weight-averaging (EMA) study (scripts/run_ema_study.sh): for each fold and seed, score the plain final model ("raw") and
the EMA models (decay 0.9999 = primary, 0.999 = exploratory) on the same cached v4 test year, then compare:
  * seed-to-seed spread of AUC-PR per variant (CV across seeds)         -> does averaging weights reduce instability?
  * paired raw -> EMA change per run, with a day-bootstrap interval     -> same training run, so a clean paired comparison
  * disjoint-pair disagreement among seeds (+ beyond test-noise flag)   -> same stability test as the ensemble study
  * 3-member ensembles of each variant
Scores are cached (float16) in data/interim/diag_scores/. Evaluation only (the trainings are done by run_ema_study.sh).
Usage: python scripts/ema_eval.py [--years 2016 2020 2024] [--seeds 10 11 12] [--boots 1000] [--out PATH]
Output: models/ema_study/ema_eval.json
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

from tornado_predictor.build_scaled import combine_staged_runs, load_manifest  # noqa: E402
from tornado_predictor.evaluate import get_proba_labels  # noqa: E402
from tornado_predictor.inference import assert_feature_order_matches, load_checkpoint  # noqa: E402
from tornado_predictor.split import split_run_bins_year_holdout  # noqa: E402
from tornado_predictor.time_bins import N_BINS  # noqa: E402
from tornado_predictor.training import DenseGridDataset  # noqa: E402

EMA_DIR = dsm.REPO / "models" / "ema_study"
VARIANTS = {"raw": "", "ema0.9999": "_ema0.9999", "ema0.999": "_ema0.999"}


def ckpt(year, seed, suffix):
    return EMA_DIR / f"seed{seed}" / f"tornado_unet_heldout_{year}{suffix}.pt"


def score_missing(year, seeds, run_bins):
    todo = [(v, s) for v, suf in VARIANTS.items() for s in seeds if ckpt(year, s, suf).exists() and not (dsm.CACHE / f"emastudy_{v}_s{s}_{year}.npy").exists()]
    if not todo:
        return
    print(f"[score] {year}: {len(todo)} models", flush=True)
    _, test_runs = split_run_bins_year_holdout(run_bins, year, buffer_hours=24.0)
    combined = combine_staged_runs(dsm.STAGING, test_runs)
    meta = np.load(dsm.CACHE / f"meta_{year}.npz")
    init = pd.to_datetime(combined["init_time"].values).values.astype("datetime64[s]").astype("int64")
    assert np.array_equal(init, meta["init_time"]), "test-sample order differs from cached diag scores"
    path = dsm.SCRATCH / f"emastudy_test_{year}.nc"
    try:
        combined.drop_vars(dsm.UH, errors="ignore").to_netcdf(path)
        ds = DenseGridDataset(str(path))
        for v, s in todo:
            model, c = load_checkpoint(str(ckpt(year, s, VARIANTS[v])))
            assert_feature_order_matches(ds.feature_names, c)
            sc, _ = get_proba_labels(model, ds)
            np.save(dsm.CACHE / f"emastudy_{v}_s{s}_{year}.npy", sc.reshape(len(ds), 81, 138).astype("float16"))
    finally:
        path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(bs.SEED_FOLDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=EMA_DIR / "ema_eval.json")
    args = ap.parse_args()
    manifest = load_manifest(dsm.STAGING / "manifest.json")
    run_bins = [(pd.Timestamp(k), b) for k, v in sorted(manifest.items()) if v["status"] == "done" for b in range(N_BINS)]
    rng = np.random.default_rng(3)
    out = []
    for year in args.years:
        score_missing(year, args.seeds, run_bins)
        meta = np.load(dsm.CACHE / f"meta_{year}.npz")
        y = meta["label"].astype(np.float32)
        days, day_idx = np.unique(meta["init_time"] // 86400, return_inverse=True)
        D = len(days)
        W = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(args.boots)]).astype(np.float64)
        S = {}
        for v in VARIANTS:
            for s in args.seeds:
                p = dsm.CACHE / f"emastudy_{v}_s{s}_{year}.npy"
                if p.exists():
                    S[(v, s)] = np.load(p).astype(np.float32)
        def stats(score):
            t = bs.build_table(score, y, day_idx, D)
            return bs.ap_from_weights(*t, np.ones(D)), np.array([bs.ap_from_weights(*t, w) for w in W])
        res = {k: stats(sc) for k, sc in S.items()}
        row = {"year": year, "n_days": int(D), "variants": {}, "paired_raw_vs_ema": []}
        for v in VARIANTS:
            ss = [s for s in args.seeds if (v, s) in res]
            if not ss:
                continue
            aps = np.array([res[(v, s)][0] for s in ss])
            brs = np.array([res[(v, s)][1].std(ddof=1) / res[(v, s)][0] for s in ss])
            pairs = []
            for a, b in combinations(ss, 2):
                d = res[(v, a)][1] - res[(v, b)][1]
                pairs.append({"pair": f"{a}|{b}", "rel_gap": abs(res[(v, a)][0] - res[(v, b)][0]) / np.mean([res[(v, a)][0], res[(v, b)][0]]),
                              "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)})
            ens = stats(np.mean([S[(v, s)] for s in ss], axis=0)) if len(ss) > 1 else None
            row["variants"][v] = {"seeds": ss, "aps": aps.tolist(), "mean_ap": float(aps.mean()), "cv": float(aps.std(ddof=1) / aps.mean()) if len(ss) > 1 else None,
                                  "mean_boot_rel_sd": float(brs.mean()), "pairs": pairs,
                                  "mean_pair_gap": float(np.mean([p["rel_gap"] for p in pairs])) if pairs else None,
                                  "pairs_beyond_noise": int(sum(p["excludes_zero"] for p in pairs)), "n_pairs": len(pairs),
                                  "ens": None if ens is None else {"ap": ens[0], "ci95": [float(np.percentile(ens[1], 2.5)), float(np.percentile(ens[1], 97.5))], "boot_rel_sd": float(ens[1].std(ddof=1) / ens[0])}}
        for v in ("ema0.9999", "ema0.999"):
            for s in args.seeds:
                if ("raw", s) in res and (v, s) in res:
                    d = res[(v, s)][1] - res[("raw", s)][1]
                    row["paired_raw_vs_ema"].append({"variant": v, "seed": s, "raw_ap": res[("raw", s)][0], "ema_ap": res[(v, s)][0],
                                                     "rel_change": res[(v, s)][0] / res[("raw", s)][0] - 1, "ci95_diff": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                                     "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)})
        out.append(row)
        msg = " | ".join(f"{v}: AP {np.round(r['aps'], 4).tolist()} CV {None if r['cv'] is None else round(100 * r['cv'])}%" for v, r in row["variants"].items())
        print(f"[{year}] {msg}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
