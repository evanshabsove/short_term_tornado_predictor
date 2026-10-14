"""
Stage A of the US-only baseline ticket: evaluation-only rescore with non-US
cells (Canada, Mexico, other land) excluded from the metrics.

Uses the cached float16 per-cell scores in data/interim/diag_scores/ (no model
inference, no training). These models were trained WITH the unmasked loss, so
this measures a metric-only exclusion, not the true US-only baseline.

Prediction written before running: masked AUC-PR >= unmasked in >= 9/12 folds
(dropping only negatives raises precision), mean gain 0-10% (mostly inside the
~19% seed noise), best-F1 moves less than AUC-PR.

Also reports where predicted probability sits (mean p over US / Canada / Mexico
cells, by month) -- descriptive only.

Usage: python scripts/08_canada_transfer/rescore_with_mask.py
Output: models/us_only_study/stage_a_rescore.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tornado_predictor.cell_mask import CA, MX, US, load_country, us_trusted_mask
from tornado_predictor.evaluate import compute_metrics

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data" / "interim" / "diag_scores"
OUT = REPO / "models" / "us_only_study" / "stage_a_rescore.json"
YEARS = list(range(2014, 2026))
SEED_FOLDS = (2016, 2020, 2024)
KEEP = ["auc_pr", "auc_roc", "f1_at_best_f1", "precision_at_best_f1", "recall_at_best_f1", "pos_rate"]


def metrics(y, s, valid):
    """y, s: (n, 81, 138). valid None -> all cells."""
    if valid is not None:
        y, s = y[:, valid], s[:, valid]
    m = compute_metrics(y.ravel(), s.ravel().astype("float32"), include_curves=False)
    return {k: float(m[k]) for k in KEEP} | {"n_cells": int(y.size), "n_pos": int(y.sum())}


def paired(path_tag, year, y, valid):
    s = np.load(CACHE / f"{path_tag}_{year}.npy")
    return {"unmasked": metrics(y, s, None), "masked": metrics(y, s, valid)}


def summarize(rows, label):
    d = np.array([r["masked"]["auc_pr"] / r["unmasked"]["auc_pr"] - 1 for r in rows])
    f = np.array([r["masked"]["f1_at_best_f1"] / r["unmasked"]["f1_at_best_f1"] - 1 for r in rows])
    print(f"  {label:22s} n={len(rows):2d}  AUC-PR unmasked {np.mean([r['unmasked']['auc_pr'] for r in rows]):.4f} -> "
          f"masked {np.mean([r['masked']['auc_pr'] for r in rows]):.4f}  (mean rel {100 * d.mean():+.1f}%, "
          f"masked>=unmasked in {int((d >= 0).sum())}/{len(rows)});  best-F1 mean rel {100 * f.mean():+.1f}%")
    return {"mean_rel_auc_pr": float(d.mean()), "n_improved": int((d >= 0).sum()), "n": len(rows),
            "mean_rel_best_f1": float(f.mean())}


def main():
    country = load_country()
    valid = us_trusted_mask(country)
    print(f"valid (US land+water) cells: {int(valid.sum())} of {valid.size}; dropped {int((~valid).sum())}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = {"n_valid_cells": int(valid.sum()), "folds": {}, "summary": {}}

    # --- all-True sanity: masked-with-everything must equal unmasked exactly ---
    meta0 = np.load(CACHE / "meta_2016.npz")
    y0 = meta0["label"].astype("float32")
    s0 = np.load(CACHE / "v4_2016.npy")
    a, b = metrics(y0, s0, None), metrics(y0, s0, np.ones_like(valid))
    assert a == b, "all-True mask must reproduce the unmasked metrics"
    print(f"sanity ok: all-True mask reproduces unmasked (v4 2016 AUC-PR {a['auc_pr']:.4f})\n")

    # --- 12-fold old / v4 models ---
    tags = {"old": [], "v4": []}
    mass = {"US": [], "CA": [], "MX": []}
    by_month = {}
    for year in YEARS:
        meta = np.load(CACHE / f"meta_{year}.npz")
        y = meta["label"].astype("float32")
        res["folds"][year] = {}
        for tag in tags:
            r = paired(tag, year, y, valid)
            res["folds"][year][tag] = r
            tags[tag].append(r)
        # where does the (v4) probability sit?
        s = np.load(CACHE / f"v4_{year}.npy").astype("float32")
        month = np.array(meta["init_time"]).astype("datetime64[s]").astype("datetime64[M]").astype(int) % 12 + 1
        for name, code in (("US", US), ("CA", CA), ("MX", MX)):
            cells = country == code
            mass[name].append(float(s[:, cells].mean()))
            for mth in range(1, 13):
                sel = month == mth
                if sel.any():
                    by_month.setdefault((name, mth), []).append(float(s[sel][:, cells].mean()))
        print(f"{year}: v4 AUC-PR {res['folds'][year]['v4']['unmasked']['auc_pr']:.4f} -> "
              f"{res['folds'][year]['v4']['masked']['auc_pr']:.4f}   "
              f"old {res['folds'][year]['old']['unmasked']['auc_pr']:.4f} -> "
              f"{res['folds'][year]['old']['masked']['auc_pr']:.4f}")
    print("\nSummary (12 folds, single-seed models; metric-only mask):")
    for tag, rows in tags.items():
        res["summary"][tag] = summarize(rows, f"{tag} (12 folds)")

    # --- seed-study EMA models on the 3 seed folds: per-model + 6-seed ensembles ---
    for variant in ("raw", "ema0.9999"):
        rows = []
        for year in SEED_FOLDS:
            y = np.load(CACHE / f"meta_{year}.npz")["label"].astype("float32")
            scores = [np.load(CACHE / f"emastudy_{variant}_s{sd}_{year}.npy").astype("float32") for sd in range(10, 16)]
            for sd, s in zip(range(10, 16), scores):
                rows.append({"unmasked": metrics(y, s, None), "masked": metrics(y, s, valid)})
            ens = np.mean(scores, axis=0)
            e = {"unmasked": metrics(y, ens, None), "masked": metrics(y, ens, valid)}
            res["folds"][year][f"ens6_{variant}"] = e
            print(f"  {variant} 6-seed ensemble {year}: AUC-PR {e['unmasked']['auc_pr']:.4f} -> {e['masked']['auc_pr']:.4f}")
        res["summary"][f"emastudy_{variant}_singles"] = summarize(rows, f"emastudy {variant} singles")

    print("\nMean predicted probability by country (v4 models, pooled over 12 folds):")
    print({k: round(float(np.mean(v)), 5) for k, v in mass.items()})
    print("by month (US / CA / MX):")
    for mth in range(1, 13):
        print(f"  {mth:2d}: " + " / ".join(f"{np.mean(by_month[(n, mth)]):.5f}" if (n, mth) in by_month else "  n/a  " for n in ("US", "CA", "MX")))
    res["mean_p_by_country"] = {k: float(np.mean(v)) for k, v in mass.items()}
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
