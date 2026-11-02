"""
Evaluate the ConvLSTM full study against docs/convlstm_preregistration.md. Evaluation only; scores are read from the study's float16 files.

Per fold (2016/2020/2024): the 3-member ensemble (EMA 0.9999, seeds 10-12) AUC-PR vs. arm A's 3-member ensemble from the cached scores
(data/interim/diag_scores/hourly_wholebin_ema0.9999_s*_<year>.npy), E = ConvLSTM / A - 1, a paired day-bootstrap 95% interval, the position
against A's single models, secondary metrics (best-F1 with precision/recall, Cohen's d, map-level AUC-ROC, 1-cell-dilated AUC-PR, AUC-PR per bin,
false-alarm rate on quiet bins of active runs at 50%/80% recall), and the pre-registered verdict. Step 0 reproduces A's ensemble AUC-PR from the
cached scores and asserts it equals models/hourly_study/wholebin/hourly_eval.json before any ConvLSTM score is read.

Usage:  python scripts/11_convlstm/convlstm_eval.py
Output: models/convlstm/full_study/convlstm_eval.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "05_year_cv_diagnostics"))
sys.path.insert(0, str(HERE.parent / "07_updraft_helicity"))
import bootstrap_stability as bs  # noqa: E402
import diagnose_sample_mix as dsm  # noqa: E402
from uh_eval import full_metrics  # noqa: E402

from tornado_predictor.build_scaled import load_manifest  # noqa: E402
from tornado_predictor.sequence_dataset import runs_from_run_bins  # noqa: E402
from tornado_predictor.sequence_study import run_list_hash  # noqa: E402
from tornado_predictor.split import split_run_bins_year_holdout  # noqa: E402
from tornado_predictor.time_bins import N_BINS  # noqa: E402

REPO = dsm.REPO
PREREG = REPO / "docs/convlstm_preregistration.md"
A_EVAL = REPO / "models/hourly_study/wholebin/hourly_eval.json"
WHOLEBIN = REPO / "data/interim/scaled_build_wholebin"
HELPS, HURTS = 0.08, -0.08  # pre-registered thresholds
SEEDS = (10, 11, 12)


def verdict(effects):
    pooled = float(np.mean(effects))
    pos, neg = sum(e > 0 for e in effects), sum(e < 0 for e in effects)
    v = "helps" if pooled >= HELPS and pos >= 2 else "hurts" if pooled <= HURTS and neg >= 2 else "no detectable effect"
    return {"pooled_E": pooled, "folds_positive": int(pos), "folds_negative": int(neg), "per_fold_E": [float(e) for e in effects], "verdict": v}


def metrics(y, s, quiet_run, y_dil1, bins):
    m = full_metrics(y, s, quiet_run, y_dil1)
    ml = dsm.map_level(s, y, quiet_run)
    for r in dsm.RECALLS:
        m[f"far_quiet_bins_of_active_runs@recall{r}"] = float(ml.get(f"far_quiet_bins_of_active_runs@recall{r}", np.nan))
    for b in range(N_BINS):
        sel = bins == b
        m[f"auc_pr_bin{b}"] = float(average_precision_score(y[sel].ravel() > 0, s[sel].ravel()))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study-dir", type=Path, default=REPO / "models/convlstm/full_study")
    ap.add_argument("--years", type=int, nargs="+", default=[2016, 2020, 2024])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--prereg", type=Path, default=PREREG, help="pre-registration whose sha256 is recorded in the output")
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_path = args.out or args.study_dir / "convlstm_eval.json"
    a_ref = {f["year"]: f for f in json.loads(A_EVAL.read_text())["folds"]}
    manifest = load_manifest(WHOLEBIN / "manifest.json")
    run_bins = [(pd.Timestamp(k), b) for k, v in sorted(manifest.items()) if v["status"] == "done" for b in range(N_BINS)]
    rng = np.random.default_rng(4)
    folds = []
    for year in args.years:
        meta = np.load(dsm.CACHE / f"meta_{year}.npz")
        y = meta["label"].astype(np.float32)
        bins = meta["bin_index"]
        act = y.sum((1, 2)) > 0
        quiet_run = ~pd.Series(act).groupby(meta["init_time"]).transform("any").values
        y_dil1 = maximum_filter(y, size=(1, 3, 3), mode="constant")
        days, day_idx = np.unique(meta["init_time"] // 86400, return_inverse=True)
        D = len(days)
        W = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(args.boots)]).astype(np.float64)

        def ap_boot(score):
            t = bs.build_table(score, y, day_idx, D)
            return bs.ap_from_weights(*t, np.ones(D)), np.array([bs.ap_from_weights(*t, w) for w in W])

        # Step 0: A from its cached scores, checked against the recorded value before any ConvLSTM score is loaded.
        A = {s: np.load(dsm.CACHE / f"hourly_wholebin_ema0.9999_s{s}_{year}.npy").astype(np.float32) for s in args.seeds}
        a_ens = np.mean([A[s] for s in args.seeds], axis=0)
        a_ap, a_boot = ap_boot(a_ens)
        if tuple(args.seeds) == SEEDS:
            ref = a_ref[year]["vs_baseline"]["arm_ens_ap"]
            assert abs(a_ap - ref) < 1e-6, f"fold {year}: could not reproduce A's ensemble AUC-PR ({a_ap} vs {ref})"

        # Checks: every run used A's training runs and A's test order.
        train_rb, _ = split_run_bins_year_holdout(run_bins, year, buffer_hours=24.0)
        expected_hash = run_list_hash(runs_from_run_bins(train_rb))
        runs_meta = {}
        for s in args.seeds:
            r = json.loads((args.study_dir / f"seed{s}/fold{year}/results.json").read_text())
            assert r["train_run_list_hash"] == expected_hash, f"fold {year} seed {s}: training runs differ from A's"
            assert r["test_order_matches_arm_a_scores"] is True, f"fold {year} seed {s}: test order not checked against A"
            runs_meta[str(s)] = {k: r[k] for k in ("final_train_loss", "final_val_loss", "val_over_train_loss", "train_seconds")}
            runs_meta[str(s)]["any_nan_loss"] = bool(not np.isfinite([r["final_train_loss"], r["final_val_loss"]]).all())

        C = {s: np.load(args.study_dir / f"seed{s}/fold{year}/convlstm_heldout_{year}_s{s}_ema0.9999_scores.npy").astype(np.float32) for s in args.seeds}
        assert all(v.shape == a_ens.shape for v in C.values()), "score shape differs from A's"
        c_ens = np.mean([C[s] for s in args.seeds], axis=0)
        c_ap, c_boot = ap_boot(c_ens)
        d = c_boot - a_boot
        row = {
            "year": int(year), "n_days": int(D), "seeds": list(args.seeds), "train_run_list_hash": expected_hash, "runs": runs_meta,
            "convlstm_single_aps": {str(s): ap_boot(C[s])[0] for s in args.seeds},
            "a_single_aps": {str(s): ap_boot(A[s])[0] for s in args.seeds},
            "convlstm_ens_ap": c_ap, "convlstm_ens_ci95": [float(np.percentile(c_boot, 2.5)), float(np.percentile(c_boot, 97.5))],
            "a_ens_ap": a_ap, "E": c_ap / a_ap - 1,
            "paired_diff_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "convlstm_ens_metrics": metrics(y, c_ens, quiet_run, y_dil1, bins),
            "a_ens_metrics": metrics(y, a_ens, quiet_run, y_dil1, bins),
        }
        folds.append(row)
        print(f"[{year}] ConvLSTM ens {c_ap:.4f} | A ens {a_ap:.4f} | E {100 * row['E']:+.1f}% | paired CI {row['paired_diff_ci95']}", flush=True)

    summary = verdict([f["E"] for f in folds])
    summary["folds_paired_ci_excludes_zero"] = int(sum(np.prod(f["paired_diff_ci95"]) > 0 for f in folds))
    summary["confirmed"] = bool(summary["verdict"] == "helps" and summary["folds_paired_ci_excludes_zero"] >= 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"preregistration_sha256": hashlib.sha256(args.prereg.read_bytes()).hexdigest(), "summary": summary, "folds": folds}, indent=1))
    print(f"pooled E {100 * summary['pooled_E']:+.1f}% | folds positive {summary['folds_positive']}/{len(folds)} | {summary['verdict']} | "
          f"paired CI excludes 0 in {summary['folds_paired_ci_excludes_zero']} | confirmed={summary['confirmed']}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
