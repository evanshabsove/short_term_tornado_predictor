"""
Evaluate the US-only baseline (scripts/08_canada_transfer/run_us_only_study.sh): models trained with Canada/Mexico/other-land cells excluded
from the loss. Everything is scored on the SAME cells -- US land + water (cell_mask.us_trusted_mask) -- for both sides:

  * US-ONLY models: models/us_only_study/seed{10,11,12}/ (EMA 0.9999), 3-member ensemble.
  * BASELINE: the existing unmasked-trained averaged models models/ema_study/ (EMA 0.9999, seeds 10-15; cached scores emastudy_ema0.9999_*),
    trained on the same runs/splits. The US-only ensemble is compared with all 20 three-member baseline ensembles and the matched triple
    (seeds 10-12), exactly as scripts/07_updraft_helicity/uh_eval.py does for UH.

Pre-registered rule (written before any US-only model was trained), on pooled-over-folds effect = US-only ensemble AUC-PR / mean of the
20 baseline triples - 1, masked cells only:
  "helps"  : pooled effect >= +8% AND positive in >= 2 of 3 folds;
  "hurts"  : pooled effect <= -8% AND negative in >= 2 of 3 folds;
  otherwise: no detectable effect.
Secondary / descriptive (no rule): best-F1, precision/recall at best-F1, Cohen's d, and mean predicted probability over Canadian / Mexican / US
cells for the US-only vs matched-baseline ensembles (does unmasked training suppress probability over cells it was always told were negative?).

Scores are cached (float16) in data/interim/diag_scores/ (usonly_*). Evaluation only.
Usage: python scripts/08_canada_transfer/us_only_eval.py [--years 2016 2020 2024] [--seeds 10 11 12] [--boots 1000] [--out PATH]
Output: models/us_only_study/us_only_eval.json
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_year_cv_diagnostics"))
import bootstrap_stability as bs  # noqa: E402
import diagnose_sample_mix as dsm  # noqa: E402

from tornado_predictor.build_scaled import combine_staged_runs, load_manifest  # noqa: E402
from tornado_predictor.cell_mask import CA, MX, US, load_country, us_trusted_mask  # noqa: E402
from tornado_predictor.evaluate import compute_metrics, get_proba_labels  # noqa: E402
from tornado_predictor.inference import assert_feature_order_matches, load_checkpoint  # noqa: E402
from tornado_predictor.split import split_run_bins_year_holdout  # noqa: E402
from tornado_predictor.time_bins import N_BINS  # noqa: E402
from tornado_predictor.training import DenseGridDataset  # noqa: E402

REPO = dsm.REPO
STUDY_DIR = REPO / "models" / "us_only_study"
VARIANTS = {"raw": "", "ema0.9999": "_ema0.9999"}
BASE_SEEDS = (10, 11, 12, 13, 14, 15)
TAG = "usonly"


def score_missing(year, seeds, run_bins):
    todo = [(v, s) for v, suf in VARIANTS.items() for s in seeds
            if (STUDY_DIR / f"seed{s}" / f"tornado_unet_heldout_{year}{suf}.pt").exists() and not (dsm.CACHE / f"{TAG}_{v}_s{s}_{year}.npy").exists()]
    if not todo:
        return
    print(f"[score] {year}: {len(todo)} US-only models", flush=True)
    _, test_runs = split_run_bins_year_holdout(run_bins, year, buffer_hours=24.0)
    combined = combine_staged_runs(dsm.STAGING, test_runs)
    meta = np.load(dsm.CACHE / f"meta_{year}.npz")
    init = pd.to_datetime(combined["init_time"].values).values.astype("datetime64[s]").astype("int64")
    assert np.array_equal(init, meta["init_time"]), "test-sample order differs from cached diag scores"
    path = dsm.SCRATCH / f"{TAG}_test_{year}.nc"
    try:
        combined.drop_vars(dsm.UH, errors="ignore").to_netcdf(path)
        ds = DenseGridDataset(str(path))
        for v, s in todo:
            model, c = load_checkpoint(str(STUDY_DIR / f"seed{s}" / f"tornado_unet_heldout_{year}{VARIANTS[v]}.pt"))
            assert_feature_order_matches(ds.feature_names, c)
            sc, _ = get_proba_labels(model, ds)  # full grid scored; the mask is applied at evaluation time, below
            np.save(dsm.CACHE / f"{TAG}_{v}_s{s}_{year}.npy", sc.reshape(len(ds), 81, 138).astype("float16"))
    finally:
        path.unlink(missing_ok=True)


def masked_table(score, y, valid, day_idx, n_days):
    """bs.build_table over valid cells only. Shapes are (n, 1, K) so build_table's day_idx[:, None, None] broadcast stays correct."""
    return bs.build_table(score[:, valid][:, None, :], y[:, valid][:, None, :], day_idx, n_days)


def summary_metrics(y, s, valid):
    m = compute_metrics(y[:, valid].ravel(), s[:, valid].ravel(), include_curves=False)
    return {"auc_pr": float(m["auc_pr"]), "auc_roc": float(m["auc_roc"]), "f1": float(m["f1_at_best_f1"]),
            "precision": float(m["precision_at_best_f1"]), "recall": float(m["recall_at_best_f1"]), "cohens_d": float(m["cohens_d"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(bs.SEED_FOLDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=STUDY_DIR / "us_only_eval.json")
    args = ap.parse_args()
    country = load_country()
    valid = us_trusted_mask(country)
    manifest = load_manifest(dsm.STAGING / "manifest.json")
    run_bins = [(pd.Timestamp(k), b) for k, v in sorted(manifest.items()) if v["status"] == "done" for b in range(N_BINS)]
    rng = np.random.default_rng(5)
    out, effects = [], []
    for year in args.years:
        score_missing(year, args.seeds, run_bins)
        meta = np.load(dsm.CACHE / f"meta_{year}.npz")
        y = meta["label"].astype(np.float32)
        days, day_idx = np.unique(meta["init_time"] // 86400, return_inverse=True)
        D = len(days)
        W = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(args.boots)]).astype(np.float64)
        US_ = {s: np.load(dsm.CACHE / f"{TAG}_ema0.9999_s{s}_{year}.npy").astype(np.float32) for s in args.seeds}
        BASE = {s: np.load(dsm.CACHE / f"emastudy_ema0.9999_s{s}_{year}.npy").astype(np.float32) for s in BASE_SEEDS}

        def ap_boot(score):
            t = masked_table(score, y, valid, day_idx, D)
            return bs.ap_from_weights(*t, np.ones(D)), np.array([bs.ap_from_weights(*t, w) for w in W])

        us_ids = sorted(US_)
        k = len(us_ids)
        row = {"year": year, "n_days": int(D), "us_only_seeds": us_ids,
               "us_only_single_aps": {str(s): ap_boot(US_[s])[0] for s in us_ids},
               "base_single_aps": {str(s): ap_boot(BASE[s])[0] for s in BASE_SEEDS}}
        us_ens = np.mean([US_[s] for s in us_ids], axis=0)
        matched = tuple(range(10, 10 + k))
        base_ens = np.mean([BASE[s] for s in matched], axis=0)
        us_ap, us_boot = ap_boot(us_ens)
        m_ap, m_boot = ap_boot(base_ens)
        tv = np.array([ap_boot(np.mean([BASE[s] for s in c], axis=0))[0] for c in combinations(BASE_SEEDS, k)])
        d = us_boot - m_boot
        row["ensemble"] = {"k": k, "us_only_ens_ap": us_ap, "us_only_ens_ci95": bs.ci(us_boot), "matched_base_ap": m_ap,
                           "base_triples_n": len(tv), "base_triples_mean": float(tv.mean()), "base_triples_sd": float(tv.std(ddof=1)),
                           "base_triples_p10": float(np.percentile(tv, 10)), "base_triples_p90": float(np.percentile(tv, 90)),
                           "us_only_percentile_in_base_triples": float(100 * np.mean(tv < us_ap)),
                           "effect_vs_triple_mean": us_ap / tv.mean() - 1, "effect_vs_matched": us_ap / m_ap - 1,
                           "paired_diff_vs_matched_ci95": bs.ci(d), "paired_excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}
        row["us_only_ens_metrics"] = summary_metrics(y, us_ens, valid)
        row["base_matched_ens_metrics"] = summary_metrics(y, base_ens, valid)
        row["mean_p_by_country"] = {name: {"us_only": float(us_ens[:, country == code].mean()), "baseline": float(base_ens[:, country == code].mean())}
                                    for name, code in (("US", US), ("CA", CA), ("MX", MX))}
        out.append(row)
        effects.append(row["ensemble"]["effect_vs_triple_mean"])
        e = row["ensemble"]
        mp = row["mean_p_by_country"]
        print(f"[{year}] US-only ens {e['us_only_ens_ap']:.4f} (95% {e['us_only_ens_ci95'][0]:.4f}-{e['us_only_ens_ci95'][1]:.4f}) | baseline triples mean "
              f"{e['base_triples_mean']:.4f} (p10-p90 {e['base_triples_p10']:.4f}-{e['base_triples_p90']:.4f}) matched {e['matched_base_ap']:.4f} | effect "
              f"{100 * e['effect_vs_triple_mean']:+.1f}% | percentile {e['us_only_percentile_in_base_triples']:.0f} | mean p CA {mp['CA']['us_only']:.5f} vs "
              f"{mp['CA']['baseline']:.5f}, MX {mp['MX']['us_only']:.5f} vs {mp['MX']['baseline']:.5f}, US {mp['US']['us_only']:.5f} vs {mp['US']['baseline']:.5f}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
    pooled = float(np.mean(effects))
    n_pos, n_neg = int(sum(e > 0 for e in effects)), int(sum(e < 0 for e in effects))
    need = int(np.ceil(2 * len(effects) / 3))
    verdict = "helps" if pooled >= 0.08 and n_pos >= need else "hurts" if pooled <= -0.08 and n_neg >= need else "no detectable effect"
    print(f"\nPooled effect {100 * pooled:+.1f}% ({n_pos} folds positive, {n_neg} negative of {len(effects)}) -> pre-registered rule: {verdict}")
    args.out.write_text(json.dumps({"folds": out, "pooled_effect": pooled, "n_positive": n_pos, "n_negative": n_neg, "verdict": verdict}, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
