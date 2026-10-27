"""
Evaluate experiment A (whole-bin, 14 features) or B (whole-bin + extras) against the rules in
docs/hourly_features_preregistration.md. Evaluation only; scores are cached (float16) in data/interim/diag_scores/.

For each fold (2016/2020/2024) scores the arm's EMA-0.9999 models on the held-out year and reports:
  * the arm's k-member ensemble AUC-PR vs. the distribution of all k-member ensembles of the six BASELINE averaged models
    (models/ema_study/, seeds 10-15, cached scores `emastudy_ema0.9999_*`), the matched baseline ensemble (seeds 10..), the
    effect E = arm / mean of baseline ensembles - 1, the percentile within them, and a paired day-bootstrap interval;
  * for --arm extras, the same against experiment A's ensemble (E_vs_A): the extras' marginal effect;
  * secondary metrics (best-F1, precision/recall, Cohen's d, map-level AUC-ROC, 1-cell-dilated AUC-PR) and the
    hypothesis-specific false-alarm rate on quiet bins of active runs at 50% / 80% recall;
  * the pre-registered verdict (pooled E and number of positive folds) per comparison.

Usage:
    python scripts/10_hourly_features/hourly_eval.py --arm wholebin
    python scripts/10_hourly_features/hourly_eval.py --arm extras           # needs the wholebin arm scored first
    python scripts/10_hourly_features/hourly_eval.py --arm wholebin --seeds 10 11 12 13 14 15   # pre-registered extension
Output: models/hourly_study/<arm>/hourly_eval.json
"""
import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "05_year_cv_diagnostics"))
sys.path.insert(0, str(HERE.parents[0] / "07_updraft_helicity"))
import bootstrap_stability as bs  # noqa: E402
import diagnose_sample_mix as dsm  # noqa: E402
from uh_eval import BASE_SEEDS, full_metrics  # noqa: E402

from tornado_predictor.build_scaled import combine_staged_runs, load_manifest  # noqa: E402
from tornado_predictor.evaluate import get_proba_labels  # noqa: E402
from tornado_predictor.inference import apply_input_scaling, assert_feature_order_matches, load_checkpoint  # noqa: E402
from tornado_predictor.split import split_run_bins_year_holdout  # noqa: E402
from tornado_predictor.time_bins import N_BINS  # noqa: E402
from tornado_predictor.training import DenseGridDataset  # noqa: E402

REPO = dsm.REPO
STAGING = {"wholebin": REPO / "data/interim/scaled_build_wholebin", "extras": REPO / "data/interim/scaled_build_wholebin_extra"}
STUDY = {"wholebin": REPO / "models/hourly_study/wholebin", "extras": REPO / "models/hourly_study/extras"}
PREREG = REPO / "docs/hourly_features_preregistration.md"
HELPS, HURTS, EXTEND_LO = 0.08, -0.08, 0.04  # pre-registered thresholds


def tag(arm):
    return f"hourly_{arm}"


# "ema0.9999" is the pre-registered scoring variant. "raw" (the plain final weights) exists only to validate the scoring path against the
# training script's own in-script AUC-PR (the EMA of a 1-epoch smoke run is still mostly the random init).
VARIANT = "ema0.9999"


def score_missing(arm, year, seeds, run_bins):
    cache = lambda s: dsm.CACHE / f"{tag(arm)}_{VARIANT}_s{s}_{year}.npy"
    ckpt = lambda s: STUDY[arm] / f"seed{s}" / f"tornado_unet_heldout_{year}{'' if VARIANT == 'raw' else '_' + VARIANT}.pt"
    todo = [s for s in seeds if ckpt(s).exists() and not cache(s).exists()]
    if not todo:
        return
    print(f"[score] {arm} {year}: seeds {todo}", flush=True)
    _, test_runs = split_run_bins_year_holdout(run_bins, year, buffer_hours=24.0)
    combined = combine_staged_runs(STAGING[arm], test_runs)
    meta = np.load(dsm.CACHE / f"meta_{year}.npz")
    init = pd.to_datetime(combined["init_time"].values).values.astype("datetime64[s]").astype("int64")
    assert np.array_equal(init, meta["init_time"]), "test-sample order differs from the cached baseline scores"
    assert np.array_equal(combined["bin_index"].values, meta["bin_index"]), "bin order differs from the cached baseline scores"
    path = dsm.SCRATCH / f"{tag(arm)}_test_{year}.nc"
    dsm.SCRATCH.mkdir(parents=True, exist_ok=True)
    try:
        combined.to_netcdf(path)
        for s in todo:
            model, c = load_checkpoint(str(ckpt(s)))
            ds = DenseGridDataset(str(path))  # fresh copy per model: scaling is applied in place
            if ds.feature_names != c["feature_names"]:
                assert set(ds.feature_names) == set(c["feature_names"]), "feature sets differ"
                idx = [ds.feature_names.index(n) for n in c["feature_names"]]
                ds.X = ds.X[:, idx]
                ds.feature_names = list(c["feature_names"])
            assert_feature_order_matches(ds.feature_names, c)
            apply_input_scaling(ds, c)
            sc, _ = get_proba_labels(model, ds)
            np.save(cache(s), sc.reshape(len(ds), 81, 138).astype("float16"))
    finally:
        path.unlink(missing_ok=True)


def metrics(y, s, quiet_run, y_dil1):
    m = full_metrics(y, s, quiet_run, y_dil1)
    ml = dsm.map_level(s, y, quiet_run)
    for r in dsm.RECALLS:
        m[f"far_quiet_bins_of_active_runs@recall{r}"] = float(ml.get(f"far_quiet_bins_of_active_runs@recall{r}", np.nan))
    return m


def verdict(effects):
    pooled = float(np.mean(effects))
    pos, neg = sum(e > 0 for e in effects), sum(e < 0 for e in effects)
    if pooled >= HELPS and pos >= 2:
        v = "helps"
    elif pooled <= HURTS and neg >= 2:
        v = "hurts"
    elif EXTEND_LO <= pooled < HELPS and pos >= 2:
        v = "ambiguous: extend to seeds 13-15 (pre-registered)"
    else:
        v = "no detectable effect"
    return {"pooled_E": pooled, "folds_positive": int(pos), "folds_negative": int(neg), "per_fold_E": [float(e) for e in effects], "verdict": v}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["wholebin", "extras"], required=True)
    ap.add_argument("--years", type=int, nargs="+", default=list(bs.SEED_FOLDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--variant", choices=["ema0.9999", "raw"], default="ema0.9999", help="raw is for validating the scoring path only; the pre-registered analysis uses ema0.9999")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    global VARIANT
    VARIANT = args.variant
    out_path = args.out or STUDY[args.arm] / ("hourly_eval.json" if len(args.seeds) == 3 else f"hourly_eval_{len(args.seeds)}seeds.json")

    manifest = load_manifest(STAGING[args.arm] / "manifest.json")
    run_bins = [(pd.Timestamp(k), b) for k, v in sorted(manifest.items()) if v["status"] == "done" for b in range(N_BINS)]
    rng = np.random.default_rng(4)
    folds, k = [], len(args.seeds)
    for year in args.years:
        score_missing(args.arm, year, args.seeds, run_bins)
        meta = np.load(dsm.CACHE / f"meta_{year}.npz")
        y = meta["label"].astype(np.float32)
        act = y.sum((1, 2)) > 0
        quiet_run = ~pd.Series(act).groupby(meta["init_time"]).transform("any").values
        y_dil1 = maximum_filter(y, size=(1, 3, 3), mode="constant")
        days, day_idx = np.unique(meta["init_time"] // 86400, return_inverse=True)
        D = len(days)
        W = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(args.boots)]).astype(np.float64)

        load = lambda t, s: np.load(dsm.CACHE / f"{t}_{VARIANT}_s{s}_{year}.npy").astype(np.float32)
        ARM = {s: load(tag(args.arm), s) for s in args.seeds}
        BASE = {s: load("emastudy", s) for s in BASE_SEEDS}

        def ap_boot(score):
            t = bs.build_table(score, y, day_idx, D)
            return bs.ap_from_weights(*t, np.ones(D)), np.array([bs.ap_from_weights(*t, w) for w in W])

        arm_ens = np.mean([ARM[s] for s in args.seeds], axis=0)
        arm_ap, arm_boot = ap_boot(arm_ens)
        matched_seeds = tuple(range(10, 10 + k))
        base_matched = np.mean([BASE[s] for s in matched_seeds], axis=0)
        m_ap, m_boot = ap_boot(base_matched)
        tv = np.array([ap_boot(np.mean([BASE[s] for s in c], axis=0))[0] for c in combinations(BASE_SEEDS, k)])
        d = arm_boot - m_boot
        row = {
            "year": int(year), "n_days": int(D), "arm_seeds": list(args.seeds),
            "arm_single_aps": {str(s): ap_boot(ARM[s])[0] for s in args.seeds},
            "base_single_aps": {str(s): ap_boot(BASE[s])[0] for s in BASE_SEEDS},
            "vs_baseline": {
                "k": k, "arm_ens_ap": arm_ap, "arm_ens_ci95": [float(np.percentile(arm_boot, 2.5)), float(np.percentile(arm_boot, 97.5))],
                "matched_base_ap": m_ap, "base_ensembles_n": len(tv), "base_ensembles_mean": float(tv.mean()),
                "base_ensembles_sd": float(tv.std(ddof=1)) if len(tv) > 1 else None,
                "base_ensembles_p10": float(np.percentile(tv, 10)), "base_ensembles_p90": float(np.percentile(tv, 90)),
                "arm_percentile_in_base": float(100 * np.mean(tv < arm_ap)),
                "E_vs_mean": arm_ap / tv.mean() - 1, "E_vs_matched": arm_ap / m_ap - 1,
                "paired_diff_vs_matched_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            },
            "arm_ens_metrics": metrics(y, arm_ens, quiet_run, y_dil1),
            "base_matched_ens_metrics": metrics(y, base_matched, quiet_run, y_dil1),
        }
        if args.arm == "extras":
            a_seeds = [s for s in args.seeds if (dsm.CACHE / f"{tag('wholebin')}_{VARIANT}_s{s}_{year}.npy").exists()]
            assert a_seeds == list(args.seeds), f"score experiment A for seeds {args.seeds} first (have {a_seeds})"
            a_ens = np.mean([load(tag("wholebin"), s) for s in args.seeds], axis=0)
            a_ap, a_boot = ap_boot(a_ens)
            dba = arm_boot - a_boot
            row["vs_wholebin"] = {
                "wholebin_ens_ap": a_ap, "E": arm_ap / a_ap - 1,
                "paired_diff_ci95": [float(np.percentile(dba, 2.5)), float(np.percentile(dba, 97.5))],
            }
            row["wholebin_ens_metrics"] = metrics(y, a_ens, quiet_run, y_dil1)
        folds.append(row)
        e = row["vs_baseline"]
        print(f"[{year}] {args.arm} ens {e['arm_ens_ap']:.4f} | baseline {k}-member mean {e['base_ensembles_mean']:.4f} matched {e['matched_base_ap']:.4f} "
              f"| E {100 * e['E_vs_mean']:+.1f}% | pctile {e['arm_percentile_in_base']:.0f}", flush=True)

    summary = {"vs_baseline": verdict([f["vs_baseline"]["E_vs_mean"] for f in folds])}
    # Pre-registered strength qualifiers (they never change the verdict category, only how strongly it may be worded):
    # folds where the arm ensemble beats the baseline ensembles' 90th percentile / folds whose paired day-bootstrap interval excludes zero.
    summary["vs_baseline"]["folds_above_baseline_p90"] = int(sum(f["vs_baseline"]["arm_ens_ap"] > f["vs_baseline"]["base_ensembles_p90"] for f in folds))
    summary["vs_baseline"]["folds_paired_ci_excludes_zero"] = int(sum(np.prod(f["vs_baseline"]["paired_diff_vs_matched_ci95"]) > 0 for f in folds))
    if args.arm == "extras":
        summary["vs_wholebin"] = verdict([f["vs_wholebin"]["E"] for f in folds])
        summary["vs_wholebin"]["folds_paired_ci_excludes_zero"] = int(sum(np.prod(f["vs_wholebin"]["paired_diff_ci95"]) > 0 for f in folds))
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest() if PREREG.exists() else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"arm": args.arm, "seeds": list(args.seeds), "preregistration_sha256": prereg_sha, "summary": summary, "folds": folds}, indent=1))
    for name, v in summary.items():
        extra = {k: v[k] for k in ("folds_above_baseline_p90", "folds_paired_ci_excludes_zero") if k in v}
        print(f"{name}: pooled E {100 * v['pooled_E']:+.1f}% | folds positive {v['folds_positive']}/3 | {v['verdict']} | {extra}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
