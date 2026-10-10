"""
Evaluate the UH re-test (scripts/run_uh_study.sh). Scores the UH25 models (14 base features + uh_2_5km mean/max, EMA 0.9999 and plain) on the
cached v4 test years and compares them with the BASELINE averaged models (14 features, models/ema_study/, EMA 0.9999, seeds 10-15), which were
trained on the same runs and splits:
  * 3-member ensemble AUC-PR (UH seeds 10-12) vs. the distribution of all 20 three-member baseline ensembles drawn from six baseline models,
    and vs. the matched baseline triple (seeds 10-12); effect = UH ensemble / mean of baseline triples - 1; percentile of the UH ensemble
    within the baseline-triple distribution; paired day-bootstrap interval for UH ensemble minus the matched baseline triple.
  * single-model means (UH vs. baseline), and for the two ensembles: best-F1, precision/recall, Cohen's d, map-level AUC-ROC, 1-cell-dilated AUC-PR.
Scores are cached (float16) in data/interim/diag_scores/ (uhstudy_*). Evaluation only.
Usage: python scripts/uh_eval.py [--years 2016 2020 2024] [--seeds 10 11 12] [--boots 1000] [--out PATH]
Output: models/uh_study/uh_eval.json
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap_stability as bs  # noqa: E402
import diagnose_sample_mix as dsm  # noqa: E402

from tornado_predictor.build_scaled import combine_staged_runs, load_manifest  # noqa: E402
from tornado_predictor.evaluate import compute_metrics, get_proba_labels  # noqa: E402
from tornado_predictor.inference import assert_feature_order_matches, load_checkpoint  # noqa: E402
from tornado_predictor.split import split_run_bins_year_holdout  # noqa: E402
from tornado_predictor.time_bins import N_BINS  # noqa: E402
from tornado_predictor.training import DenseGridDataset  # noqa: E402

REPO = dsm.REPO
UH_DIR = REPO / "models" / "uh_study"
MERGED = REPO / "data" / "interim" / "scaled_build_uh_merged"
DROP = ["uh_0_2km_mean", "uh_0_2km_max", "uh_0_3km_mean", "uh_0_3km_max", "uh_layers_available"]
VARIANTS = {"raw": "", "ema0.9999": "_ema0.9999"}
BASE_SEEDS = (10, 11, 12, 13, 14, 15)


def score_missing(year, seeds, run_bins):
    todo = [(v, s) for v, suf in VARIANTS.items() for s in seeds
            if (UH_DIR / f"seed{s}" / f"tornado_unet_heldout_{year}{suf}.pt").exists() and not (dsm.CACHE / f"uhstudy_{v}_s{s}_{year}.npy").exists()]
    if not todo:
        return
    print(f"[score] {year}: {len(todo)} UH models", flush=True)
    _, test_runs = split_run_bins_year_holdout(run_bins, year, buffer_hours=24.0)
    combined = combine_staged_runs(MERGED, test_runs)
    meta = np.load(dsm.CACHE / f"meta_{year}.npz")
    init = pd.to_datetime(combined["init_time"].values).values.astype("datetime64[s]").astype("int64")
    assert np.array_equal(init, meta["init_time"]), "test-sample order differs from cached diag scores"
    path = dsm.SCRATCH / f"uhstudy_test_{year}.nc"
    try:
        combined.drop_vars(DROP, errors="ignore").to_netcdf(path)
        ds = DenseGridDataset(str(path))
        for v, s in todo:
            model, c = load_checkpoint(str(UH_DIR / f"seed{s}" / f"tornado_unet_heldout_{year}{VARIANTS[v]}.pt"))
            if ds.feature_names != c["feature_names"]:  # channel order depends on the first staged run's source file; align to training order
                assert set(ds.feature_names) == set(c["feature_names"]), "feature sets differ"
                idx = [ds.feature_names.index(n) for n in c["feature_names"]]
                ds.X = ds.X[:, idx]
                ds.feature_names = list(c["feature_names"])
            assert_feature_order_matches(ds.feature_names, c)
            sc, _ = get_proba_labels(model, ds)
            np.save(dsm.CACHE / f"uhstudy_{v}_s{s}_{year}.npy", sc.reshape(len(ds), 81, 138).astype("float16"))
    finally:
        path.unlink(missing_ok=True)


def full_metrics(y, s, quiet_run, y_dil1):
    m = compute_metrics(y.ravel(), s.ravel(), include_curves=False)
    ml = dsm.map_level(s, y, quiet_run)
    return {"auc_pr": float(m["auc_pr"]), "auc_roc": float(m["auc_roc"]), "f1": float(m["f1_at_best_f1"]), "precision": float(m["precision_at_best_f1"]),
            "recall": float(m["recall_at_best_f1"]), "cohens_d": float(m["cohens_d"]), "map_auc_roc": float(ml.get("map_auc_roc", np.nan)),
            "auc_pr_k1": float(dsm.cell_metrics(y_dil1, s)["auc_pr"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(bs.SEED_FOLDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=UH_DIR / "uh_eval.json")
    args = ap.parse_args()
    manifest = load_manifest(MERGED / "manifest.json")
    run_bins = [(pd.Timestamp(k), b) for k, v in sorted(manifest.items()) if v["status"] == "done" for b in range(N_BINS)]
    rng = np.random.default_rng(4)
    out = []
    for year in args.years:
        score_missing(year, args.seeds, run_bins)
        meta = np.load(dsm.CACHE / f"meta_{year}.npz")
        y = meta["label"].astype(np.float32)
        act = y.sum((1, 2)) > 0
        quiet_run = ~pd.Series(act).groupby(meta["init_time"]).transform("any").values
        y_dil1 = maximum_filter(y, size=(1, 3, 3), mode="constant")
        days, day_idx = np.unique(meta["init_time"] // 86400, return_inverse=True)
        D = len(days)
        W = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(args.boots)]).astype(np.float64)
        UH = {s: np.load(dsm.CACHE / f"uhstudy_ema0.9999_s{s}_{year}.npy").astype(np.float32) for s in args.seeds if (dsm.CACHE / f"uhstudy_ema0.9999_s{s}_{year}.npy").exists()}
        UHraw = {s: np.load(dsm.CACHE / f"uhstudy_raw_s{s}_{year}.npy").astype(np.float32) for s in args.seeds if (dsm.CACHE / f"uhstudy_raw_s{s}_{year}.npy").exists()}
        BASE = {s: np.load(dsm.CACHE / f"emastudy_ema0.9999_s{s}_{year}.npy").astype(np.float32) for s in BASE_SEEDS}
        def ap_boot(score):
            t = bs.build_table(score, y, day_idx, D)
            return bs.ap_from_weights(*t, np.ones(D)), np.array([bs.ap_from_weights(*t, w) for w in W])
        uh_ids = sorted(UH)
        row = {"year": year, "n_days": int(D), "uh_seeds": uh_ids}
        # singles
        row["uh_single_aps"] = {str(s): ap_boot(UH[s])[0] for s in uh_ids}
        row["uh_raw_single_aps"] = {str(s): ap_boot(UHraw[s])[0] for s in sorted(UHraw)}
        row["base_single_aps"] = {str(s): ap_boot(BASE[s])[0] for s in BASE_SEEDS}
        # ensembles
        uh_ens = np.mean([UH[s] for s in uh_ids], axis=0)
        k = len(uh_ids)
        uh_ap, uh_boot = ap_boot(uh_ens)
        matched = tuple(range(10, 10 + k))
        m_ap, m_boot = ap_boot(np.mean([BASE[s] for s in matched], axis=0))
        triples = {}
        for c in combinations(BASE_SEEDS, k):
            triples[c] = ap_boot(np.mean([BASE[s] for s in c], axis=0))[0]
        tv = np.array(list(triples.values()))
        d = uh_boot - m_boot
        row["ensemble"] = {"k": k, "uh_ens_ap": uh_ap, "uh_ens_ci95": [float(np.percentile(uh_boot, 2.5)), float(np.percentile(uh_boot, 97.5))],
                           "matched_base_ap": m_ap, "base_triples_n": len(tv), "base_triples_mean": float(tv.mean()), "base_triples_sd": float(tv.std(ddof=1)),
                           "base_triples_min": float(tv.min()), "base_triples_max": float(tv.max()), "base_triples_p10": float(np.percentile(tv, 10)), "base_triples_p90": float(np.percentile(tv, 90)),
                           "uh_percentile_in_base_triples": float(100 * np.mean(tv < uh_ap)), "effect_vs_triple_mean": uh_ap / tv.mean() - 1, "effect_vs_matched": uh_ap / m_ap - 1,
                           "paired_diff_vs_matched_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "paired_excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}
        row["uh_ens_metrics"] = full_metrics(y, uh_ens, quiet_run, y_dil1)
        row["base_matched_ens_metrics"] = full_metrics(y, np.mean([BASE[s] for s in matched], axis=0), quiet_run, y_dil1)
        out.append(row)
        e = row["ensemble"]
        print(f"[{year}] UH ens {e['uh_ens_ap']:.4f} (95% {e['uh_ens_ci95'][0]:.4f}-{e['uh_ens_ci95'][1]:.4f}) | baseline triples mean {e['base_triples_mean']:.4f} (p10-p90 {e['base_triples_p10']:.4f}-{e['base_triples_p90']:.4f}) matched {e['matched_base_ap']:.4f} "
              f"| effect {100*e['effect_vs_triple_mean']:+.1f}% | UH percentile {e['uh_percentile_in_base_triples']:.0f}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
