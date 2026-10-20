"""
Forecast API ticket 4: the tier study. Implements docs/tier_study_preregistration.md exactly as written
(including its Errata); read that file for every definition, rule and pass/fail criterion.

Evaluation only: no retraining, no new HRRR pulls. Reads the cached held-out scores in
data/interim/diag_scores/ (A = mean of the `old` and `v4` leave-one-year-out models, all 12 years;
B = six EMA-0.9999 seeds on 2016/2020/2024) and writes models/tiers/tier_study.json, which records the
pre-registration file's sha256 so an edit after the fact is visible.

Stages: 1 count tables -> 2 operator decision -> 3 cutoffs -> 4 tier table + bootstrap -> 5 map-level and
US-land-only figures -> 6 transfer check on B -> 7 stability -> 8 calibration -> 9 prediction scoring.

Usage: python scripts/09_serving/tier_study.py [--boots 2000] [--quick]
(--quick shrinks the bootstraps for a smoke run and writes to a different file; never cite its numbers.)
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tornado_predictor import tiers as T
from tornado_predictor.cell_mask import load_country, us_trusted_mask

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data" / "interim" / "diag_scores"
PREREG = REPO / "docs" / "tier_study_preregistration.md"
OUT_DIR = REPO / "models" / "tiers"
YEARS = tuple(range(2014, 2026))
B_YEARS = (2016, 2020, 2024)
SEEDS = tuple(range(10, 16))
VARIANTS = {"raw": ("max", 0, 1), "max1": ("max", 1, 1), "mean1": ("mean", 1, 1), "max2": ("max", 2, 2), "mean2": ("mean", 2, 2)}
PRIMARY_HOUR = 20
ACTIVE_TARGET = 0.17  # active-map share of an all-hourly-runs stream (CLAUDE.md "Class imbalance strategy")
CAL_EDGES = (0.0, 0.5, 0.9, 0.97, 0.99, 0.997, 0.999, 1.0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean(x):
    """JSON-safe: numpy scalars -> python, NaN/inf -> None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


# ---------------------------------------------------------------------------- data


class Data:
    """Metadata and labels for every year, plus the global day index and sample weights."""

    def __init__(self):
        self.year = {}
        dates = []
        for y in YEARS:
            m = np.load(CACHE / f"meta_{y}.npz")
            d = pd.to_datetime(m["init_time"], unit="s")
            self.year[y] = dict(
                label=m["label"].astype("uint8"), hour=m["init_hour"], bin=m["bin_index"],
                init=d, date=d.normalize(),
            )
            dates.extend(d.normalize())
        self.dates = np.array(sorted(set(dates)))
        lookup = {d: i for i, d in enumerate(self.dates)}
        self.n_days = len(self.dates)
        n_active = n_all = 0
        for y in YEARS:
            d = self.year[y]
            d["day"] = np.array([lookup[x] for x in d["date"]])
            d["active"] = d["label"].reshape(len(d["label"]), -1).sum(1) > 0
            d["primary"] = d["hour"] == PRIMARY_HOUR
            n_active += int(d["active"].sum())
            n_all += len(d["active"])
        self.active_frac = n_active / n_all
        wa, wq = ACTIVE_TARGET / self.active_frac, (1 - ACTIVE_TARGET) / (1 - self.active_frac)
        for y in YEARS:
            d = self.year[y]
            d["weight"] = np.where(d["active"], wa, wq)
        self.weights = {"active": wa, "quiet": wq}
        self.year_days = {y: np.unique(self.year[y]["day"]) for y in YEARS}

    def scores_A(self, y):
        o = np.load(CACHE / f"old_{y}.npy").astype("float32")
        v = np.load(CACHE / f"v4_{y}.npy").astype("float32")
        return (o + v) / 2.0

    def scores_B(self, y, seeds, sel=None):
        acc = None
        for s in seeds:
            a = np.load(CACHE / f"emastudy_ema0.9999_s{s}_{y}.npy", mmap_mode="r")
            a = np.asarray(a[sel] if sel is not None else a, dtype="float32")
            acc = a if acc is None else acc + a
        return acc / len(seeds)


# ---------------------------------------------------------------------------- table building


def add_samples(builder, day_idx, keys_c, lab_c, dl_c, weights=None):
    """keys_c/lab_c/dl_c: (n_samples, n_coverage_cells); day_idx per sample."""
    for d in np.unique(day_idx):
        idx = np.flatnonzero(day_idx == d)
        w = 1.0 if weights is None else np.repeat(weights[idx], keys_c.shape[1])
        builder.add(int(d), keys_c[idx].ravel(), lab_c[idx].ravel(), dl_c[idx].ravel(), w)


def stage1_tables(data, cov):
    """A tables for every variant (primary sample) and for the max/mean radius-1 variants (weighted full mix)."""
    prim = {v: T.LevelTableBuilder(data.n_days) for v in VARIANTS}
    wfull = {v: T.LevelTableBuilder(data.n_days) for v in ("max1", "mean1")}
    for y in YEARS:
        d = data.year[y]
        A = data.scores_A(y)
        lab_c = d["label"][:, cov]
        dl_c = {k: T.dilate_labels(d["label"], k)[:, cov] for k in (1, 2)}
        for name, (op, r, k) in VARIANTS.items():
            keys = T.level_keys(T.quantize(T.smooth(A, op, r)))[:, cov]
            p = d["primary"]
            add_samples(prim[name], d["day"][p], keys[p], lab_c[p], dl_c[k][p])
            if name in wfull:
                add_samples(wfull[name], d["day"], keys, lab_c, dl_c[k], d["weight"])
        log(f"stage 1: {y} done")
    return {v: b.build() for v, b in prim.items()}, {v: b.build() for v, b in wfull.items()}


def days_with(table):
    return np.flatnonzero(np.asarray(table.total.sum(axis=1)).ravel() > 0)


def draw(rng, days, n_days):
    return np.bincount(rng.choice(days, size=len(days), replace=True), minlength=n_days).astype(np.float64)


def ci(x):
    x = np.asarray(x, dtype="float64")
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def flat_stats(stats: dict) -> dict:
    return {f"{t}.{k}": v for t, blk in stats.items() for k, v in blk.items()}


def unflat(flat: dict) -> dict:
    out = {}
    for key, v in flat.items():
        t, k = key.split(".", 1)
        out.setdefault(t, {})[k] = v
    return out


# ---------------------------------------------------------------------------- stages


def stage2_operator(prim, days, n_days, boots, rng):
    out = {"radius1": {}, "reference": {}}
    ap = {v: T.average_precision(prim[v].counts()) for v in VARIANTS}
    diffs = []
    for _ in range(boots):
        w = draw(rng, days, n_days)
        cm, cx = prim["mean1"].counts(w), prim["max1"].counts(w)
        diffs.append(T.average_precision(cm) - T.average_precision(cx))
    lo, hi = ci(diffs)
    chosen = "mean" if lo > 0 else "max" if hi < 0 else "max"
    out["radius1"] = {
        "ap_max": ap["max1"], "ap_mean": ap["mean1"], "diff_mean_minus_max": ap["mean1"] - ap["max1"],
        "diff_ci95": [lo, hi], "chosen_operator": chosen,
        "rule": "mean only if the paired 95% interval is entirely > 0; max if entirely < 0; otherwise max (default)",
    }
    out["reference"] = {
        "ap_raw_vs_k1_labels": ap["raw"], "ap_max2_vs_k2_labels": ap["max2"], "ap_mean2_vs_k2_labels": ap["mean2"],
        "note": "AUC-PR is prevalence dependent; values across different label radii are not comparable",
    }
    return chosen, out


def stage3_cutoffs(prim, chosen):
    c = prim[f"{chosen}1"].counts()
    high, high_stats = T.best_f1_cutoff(c)
    elev, rule = T.elevated_cutoff(c, high)
    cut = {"elevated": elev, "high": high}
    return cut, {
        "cutoffs": cut,
        "tier_set_id": T.tier_set_id(chosen, 1, cut),
        "high_rule": "best F1 vs k=1 dilated labels", "high_at_cutoff": high_stats,
        "elevated_rule_used": rule,
        "cum_area_share": {"elevated": T.area_at(c, elev), "high": T.area_at(c, high)},
    }


def stage4_tables(prim_t, wfull_t, cutoffs, days_p, days_w, n_days, boots, rng, data):
    res = {}
    for label, table, days in (("primary_20Z", prim_t, days_p), ("prior_shift_weighted_full_mix", wfull_t, days_w)):
        point = T.tier_counts(table.counts(), cutoffs)
        flats = []
        for _ in range(boots):
            w = draw(rng, days, n_days)
            flats.append(flat_stats(T.tier_counts(table.counts(w), cutoffs)))
        keys = flats[0].keys()
        cis = {k: ci([f[k] for f in flats]) for k in keys}
        res[label] = {"point": point, "ci95": unflat(cis)}
        log(f"stage 4: {label} bootstrap done")
    per_year = {}
    for y in YEARS:
        w = np.zeros(n_days)
        w[data.year_days[y]] = 1.0
        per_year[y] = T.tier_counts(prim_t.counts(w), cutoffs)
    res["primary_per_year"] = per_year
    return res


def tier_arrays_pass(data, cov, land, op, cutoffs, n_days):
    """Map-level and US-land-only figures need the per-cell tier arrays."""
    keys = ("n", "n_high", "n_high_tor")
    acc = {s: {k: np.zeros(n_days) for k in keys} for s in ("primary", "wfull")}
    land_counts = {s: np.zeros(3) for s in ("primary", "wfull")}
    for y in YEARS:
        d = data.year[y]
        A = data.scores_A(y)
        s16 = T.quantize(T.smooth(A, op, 1)).astype("float32")
        tiers = T.assign_tiers(s16, cutoffs, cov)
        has_high = (tiers == T.TIER_HIGH).any(axis=(1, 2))
        has_tor = (d["label"].astype(bool) & cov).any(axis=(1, 2))
        for s, sel, w in (("primary", d["primary"], np.ones(len(has_high))), ("wfull", np.ones(len(has_high), bool), d["weight"])):
            idx = np.flatnonzero(sel)
            np.add.at(acc[s]["n"], d["day"][idx], w[idx])
            np.add.at(acc[s]["n_high"], d["day"][idx], w[idx] * has_high[idx])
            np.add.at(acc[s]["n_high_tor"], d["day"][idx], w[idx] * (has_high & has_tor)[idx])
            lt = tiers[idx][:, land]
            for t in (0, 1, 2):
                land_counts[s][t] += float((w[idx][:, None] * (lt == t)).sum())
        log(f"stage 5: {y} done")
    return acc, land_counts


def stage5_summary(acc, land_counts, days_p, days_w, n_days, boots, rng):
    out = {}
    for s, days in (("primary", days_p), ("wfull", days_w)):
        a = acc[s]

        def stat(w):
            n, nh, nht = (float((w * a[k]).sum()) for k in ("n", "n_high", "n_high_tor"))
            return {"frac_maps_with_high": nh / n if n else float("nan"), "frac_of_those_with_tornado": nht / nh if nh else float("nan")}

        point = stat(np.ones(n_days))
        draws = [stat(draw(rng, days, n_days)) for _ in range(boots)]
        out[s] = {
            "point": point,
            "ci95": {k: ci([d[k] for d in draws]) for k in point},
            "us_land_only_area_share": dict(zip(("low", "elevated", "high"), (land_counts[s] / land_counts[s].sum()).tolist())),
        }
    return out


def b_tables(data, cov, ops=("max", "mean")):
    """Per five-member subset (drop one seed) and operator: table over B's three folds, primary sample."""
    subsets = [tuple(s for s in SEEDS if s != drop) for drop in SEEDS]
    builders = {(i, op): T.LevelTableBuilder(data.n_days) for i in range(len(subsets)) for op in ops}
    for y in B_YEARS:
        d = data.year[y]
        sel = np.flatnonzero(d["primary"])
        lab_c = d["label"][sel][:, cov]
        dl_c = T.dilate_labels(d["label"][sel], 1)[:, cov]
        for i, seeds in enumerate(subsets):
            B = data.scores_B(y, seeds, sel)
            for op in ops:
                keys = T.level_keys(T.quantize(T.smooth(B, op, 1)))[:, cov]
                add_samples(builders[(i, op)], d["day"][sel], keys, lab_c, dl_c)
        log(f"stage 6: B year {y} done")
    return subsets, {k: b.build() for k, b in builders.items()}


def a_counts_on_years(table, data, years):
    w = np.zeros(data.n_days)
    for y in years:
        w[data.year_days[y]] = 1.0
    return table.counts(w)


def stage6_transfer(data, chosen, cutoffs_A, prim_t, b_tabs, b_subsets, days_b, boots, rng):
    opn = chosen
    cA = a_counts_on_years(prim_t[f"{opn}1"], data, B_YEARS)
    ref_area = {t: T.area_at(cA, cutoffs_A[t]) for t in ("elevated", "high")}
    per_subset = []
    for i, seeds in enumerate(b_subsets):
        cB = b_tabs[(i, opn)].counts()
        areas = {t: T.area_at(cB, cutoffs_A[t]) for t in ("elevated", "high")}
        per_subset.append({"dropped_seed": [s for s in SEEDS if s not in seeds][0], "area_at_A_cutoffs": areas,
                           "ratio_to_A_reference": {t: areas[t] / ref_area[t] for t in areas}})
    med = {t: float(np.median([p["ratio_to_A_reference"][t] for p in per_subset])) for t in ("elevated", "high")}
    holds = all(0.75 <= med[t] <= 1.33 for t in med)
    if holds:
        serving = dict(cutoffs_A)
        per_subset_cut = None
    else:
        per_subset_cut = [
            {t: T.nearest_area_cutoff(b_tabs[(i, opn)].counts(), ref_area[t]) for t in ("elevated", "high")}
            for i in range(len(b_subsets))
        ]
        serving = {t: float(np.median([c[t] for c in per_subset_cut])) for t in ("elevated", "high")}
        serving = {t: float(np.float16(v)) for t, v in serving.items()}
    # operator ordering on B (reported only)
    ap_b = {op: [T.average_precision(b_tabs[(i, op)].counts()) for i in range(len(b_subsets))] for op in ("max", "mean")}
    # table at the serving cutoffs on B (per subset; bootstrap of the subset-median)
    def b_stat(w):
        per = [flat_stats(T.tier_counts(b_tabs[(i, opn)].counts(w), serving if per_subset_cut is None else
                                        {t: float(np.float16(per_subset_cut[i][t])) for t in serving}))
               for i in range(len(b_subsets))]
        return {k: float(np.median([p[k] for p in per])) for k in per[0]}

    point = b_stat(np.ones(data.n_days))
    flats = [b_stat(draw(rng, days_b, data.n_days)) for _ in range(boots)]
    return holds, serving, per_subset_cut, {
        "reference_area_share_A_on_B_years": ref_area,
        "per_subset": per_subset, "median_ratio": med, "criterion": "median ratio in [0.75, 1.33] for both tiers",
        "transfer_holds": holds, "serving_cutoffs": serving,
        "quantile_matched_cutoffs_per_subset": per_subset_cut,
        "operator_ordering_on_B": {"ap_max_per_subset": ap_b["max"], "ap_mean_per_subset": ap_b["mean"],
                                   "mean_beats_max_in_subsets": int(sum(m > x for m, x in zip(ap_b["mean"], ap_b["max"])))},
        "B_table_at_serving_cutoffs": {"point_median_over_subsets": unflat(point),
                                       "ci95": unflat({k: ci([f[k] for f in flats]) for k in point})},
    }


def stage7_stability(data, cov, chosen, cutoffs_A, serving, area_targets, boots_unused=None):
    # S1: disjoint 3-vs-3 splits of the six B members
    combos = list(itertools.combinations(SEEDS, 3))
    builders = {c: T.LevelTableBuilder(data.n_days) for c in combos}
    held = {c: {} for c in combos}
    for y in B_YEARS:
        d = data.year[y]
        sel = np.flatnonzero(d["primary"])
        lab_c = d["label"][sel][:, cov]
        dl_c = T.dilate_labels(d["label"][sel], 1)[:, cov]
        for c in combos:
            s16 = T.quantize(T.smooth(data.scores_B(y, c, sel), chosen, 1))
            held[c][y] = s16[:, cov]
            add_samples(builders[c], d["day"][sel], T.level_keys(s16)[:, cov], lab_c, dl_c)
    cuts = {}
    for c in combos:
        cnt = builders[c].build().counts()
        cuts[c] = {t: float(T.nearest_area_cutoff(cnt, area_targets[t])) for t in ("elevated", "high")}
    splits = [(a, tuple(s for s in SEEDS if s not in a)) for a in combos if a[0] == SEEDS[0]]  # 10 disjoint pairs
    rows = []
    for a, b in splits:
        inter = {"high": 0.0, "elev_up": 0.0}
        union = {"high": 0.0, "elev_up": 0.0}
        agree_maps = both_maps = either_maps = n_maps = 0
        for y in B_YEARS:
            for key, thr in (("high", "high"), ("elev_up", "elevated")):
                fa, fb = held[a][y] >= np.float16(cuts[a][thr]), held[b][y] >= np.float16(cuts[b][thr])
                inter[key] += float((fa & fb).sum())
                union[key] += float((fa | fb).sum())
            ha = (held[a][y] >= np.float16(cuts[a]["high"])).any(axis=1)
            hb = (held[b][y] >= np.float16(cuts[b]["high"])).any(axis=1)
            agree_maps += int((ha == hb).sum()); both_maps += int((ha & hb).sum()); either_maps += int((ha | hb).sum()); n_maps += len(ha)
        rows.append({
            "split": [list(a), list(b)],
            "jaccard_high": inter["high"] / union["high"] if union["high"] else float("nan"),
            "jaccard_elevated_up": inter["elev_up"] / union["elev_up"] if union["elev_up"] else float("nan"),
            "map_any_high_agreement": agree_maps / n_maps,
            "map_any_high_both_given_either": both_maps / either_maps if either_maps else float("nan"),
        })
    s1 = {
        "n_splits": len(rows), "splits": rows,
        "mean": {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k != "split"},
        "note": "3-member ensembles are noisier than the served 5-member one: an upper bound on model-noise instability",
    }
    # S2: anecdotal consecutive-hour pairs inside B's folds, five-member mean (seeds 10-14) at the serving cutoffs
    pair_rows = []
    for y in B_YEARS:
        d = data.year[y]
        B = T.quantize(T.smooth(data.scores_B(y, SEEDS[:5]), chosen, 1)).astype("float32")
        tiers = T.assign_tiers(B, serving, cov)
        init = pd.Series(d["init"])
        for i in range(len(init)):
            for j in range(i + 1, len(init)):
                if d["bin"][i] == d["bin"][j] and init[j] - init[i] == pd.Timedelta(hours=1):
                    a, b = tiers[i][cov], tiers[j][cov]
                    ha, hb = a == 2, b == 2
                    pair_rows.append({"year": y, "init_a": str(init[i]), "bin": int(d["bin"][i]),
                                      "frac_cells_changed_tier": float((a != b).mean()),
                                      "jaccard_high": float((ha & hb).sum() / (ha | hb).sum()) if (ha | hb).any() else None})
    s2 = {"n_pairs": len(pair_rows), "anecdotal": True, "pairs": pair_rows,
          "note": "20 pairs at most; descriptive only. The real measurement is deferred to the live archive."}
    return {"S1_model_noise": s1, "S2_hour_to_hour": s2}


def stage8_calibration(data, chosen, cutoffs_A, prim_t, wfull_t, days_p, b_subsets, b_tabs, transfer_holds, serving, boots, rng, ref_B_matching):
    pt = prim_t[f"{chosen}1"]
    wt = wfull_t[f"{chosen}1"]
    n_days = data.n_days
    rows = {y: data.year_days[y] for y in YEARS}
    subs = {id(t): {y: tuple(m[rows[y]] for m in (t.total, t.pos, t.dpos)) for y in YEARS} for t in (pt, wt)}

    def year_counts(table, w):
        # per-year row blocks: one pass over the table's non-zeros per draw instead of twelve
        return {y: T.LevelCounts(*(np.asarray(m.T @ w[rows[y]]).ravel() for m in subs[id(table)][y])) for y in YEARS}

    pooled = pt.counts()
    bins = T.quantile_bins(pooled, CAL_EDGES)
    hk = T.key_of(cutoffs_A["high"])
    high_mask = np.zeros(T.N_LEVELS, bool)
    high_mask[hk:] = True

    def loyo(yc):
        total = None
        for c in yc.values():
            total = c if total is None else total + c
        cal = {}
        for y, c in yc.items():
            fit = T.LevelCounts(total.total - c.total, total.pos - c.pos, total.dpos - c.dpos)
            cal[y] = T.isotonic_map(fit)
        return cal

    def bin_and_tier(yc, cal, ycw=None):
        pred = np.zeros(len(bins)); obs = np.zeros(len(bins)); n = np.zeros(len(bins))
        for y, c in yc.items():
            for b, m in enumerate(bins):
                pred[b] += float((c.total[m] * cal[y][m]).sum())
                obs[b] += float(c.dpos[m].sum())
                n[b] += float(c.total[m].sum())
        return pred, obs, n

    ones = np.ones(n_days)
    yc = year_counts(pt, ones)
    cal = loyo(yc)
    pred, obs, n = bin_and_tier(yc, cal)
    with np.errstate(divide="ignore", invalid="ignore"):
        point_ratio = np.where(obs > 0, pred / obs, np.nan)
    ratios = []
    for _ in range(boots):
        w = draw(rng, days_p, n_days)
        ycb = year_counts(pt, w)
        calb = loyo(ycb)
        p, o, _n = bin_and_tier(ycb, calb)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios.append(np.where(o > 0, p / o, np.nan))
    ratios = np.array(ratios)
    bin_rows = []
    for b in range(len(bins)):
        lo, hi = ci(ratios[:, b])
        bin_rows.append({"bin": b, "quantile_range": [CAL_EDGES[b], CAL_EDGES[b + 1]], "n_cells": n[b], "observed_dilated_rate": obs[b] / n[b] if n[b] else None,
                         "predicted": pred[b] / n[b] if n[b] else None, "ratio": point_ratio[b], "ratio_ci95": [lo, hi], "n_dilated_positive": obs[b]})
    top = bin_rows[3:]
    c1 = all(0.7 <= r["ratio"] <= 1.4 and r["ratio_ci95"][0] <= 1.0 <= r["ratio_ci95"][1] for r in top if r["ratio"] == r["ratio"]) \
        and all(r["ratio"] == r["ratio"] for r in top)

    # C2: High tier, per held-out year
    yr = {}
    for y, c in yc.items():
        r = T.ratio_over(c, cal[y], high_mask & (c.total > 0))
        yr[y] = r
    eligible = {y: r for y, r in yr.items() if r["n_positive"] >= 10}
    in_range = [y for y, r in eligible.items() if 0.5 <= r["ratio"] <= 2.0]
    c2 = len(eligible) >= 9 and len(in_range) >= 9

    # C3: A's map fit on the 9 non-B years, applied to B's five-member ensembles
    nonB = [y for y in YEARS if y not in B_YEARS]
    fit = None
    for y in nonB:
        fit = yc[y] if fit is None else fit + yc[y]
    calA = T.isotonic_map(fit)
    cA3 = a_counts_on_years(pt, data, B_YEARS)
    c3_rows = []
    for i in range(len(b_subsets)):
        cB = b_tabs[(i, chosen)].counts()
        if transfer_holds:
            level_cal = calA
        else:
            mapped = T.match_levels(cB, cA3)
            level_cal = np.zeros(T.N_LEVELS)
            ok = ~np.isnan(mapped)
            level_cal[ok] = calA[np.array([T.key_of(v) for v in mapped[ok]])]
        cut = serving if ref_B_matching is None else {t: float(np.float16(ref_B_matching[i][t])) for t in serving}
        ek, hk_b = T.key_of(cut["elevated"]), T.key_of(cut["high"])
        occ = cB.total > 0
        mh = np.zeros(T.N_LEVELS, bool); mh[hk_b:] = True
        me = np.zeros(T.N_LEVELS, bool); me[ek:hk_b] = True
        c3_rows.append({"dropped_seed": [s for s in SEEDS if s not in b_subsets[i]][0],
                        "high": T.ratio_over(cB, level_cal, mh & occ), "elevated": T.ratio_over(cB, level_cal, me & occ)})
    med = {t: float(np.nanmedian([r[t]["ratio"] for r in c3_rows])) for t in ("high", "elevated")}
    c3 = all(0.7 <= med[t] <= 1.4 for t in med)

    # C4: weighted full mix, maps from the primary LOYO fit
    ycw = year_counts(wt, ones)
    pr = ob = 0.0
    for y, c in ycw.items():
        m = high_mask & (c.total > 0)
        pr += float((c.total[m] * cal[y][m]).sum())
        ob += float(c.dpos[m].sum())
    c4_ratio = pr / ob if ob > 0 else float("nan")
    c4 = 0.5 <= c4_ratio <= 2.0
    return {
        "bins": bin_rows,
        "C1": {"pass": bool(c1), "criterion": "top 4 bins: ratio in [0.7,1.4] and bootstrap 95% interval contains 1"},
        "C2": {"pass": bool(c2), "eligible_years": len(eligible), "years_in_range": in_range, "per_year": yr,
               "criterion": ">= 9 eligible years and >= 9 of them with High-tier ratio in [0.5, 2.0]"},
        "C3": {"pass": bool(c3), "median_ratio": med, "per_subset": c3_rows,
               "criterion": "median over subsets: High and Elevated ratio in [0.7, 1.4]"},
        "C4": {"pass": bool(c4), "high_ratio_weighted_mix": c4_ratio, "criterion": "High ratio in [0.5, 2.0]"},
        "calibration_defensible": bool(c1 and c2 and c3 and c4),
        "calibrated_percent": None,
    }


def score_predictions(res):
    op = res["operator"]["radius1"]
    cum = res["cutoffs"]["cum_area_share"]
    prim = res["tier_table_A"]["primary_20Z"]["point"]
    p = {}
    p["P1_no_significant_max_vs_mean_so_max_chosen"] = {"predicted": True, "observed": op["chosen_operator"] == "max" and op["diff_ci95"][0] <= 0 <= op["diff_ci95"][1]}
    p["P2_high"] = {"predicted": "area 0.3-3%, hit rate 12-35%, capture 10-35%",
                    "observed": {"area": prim["high"]["area_share"], "hit_rate": prim["high"]["hit_rate"], "capture": prim["high"]["capture_share"]},
                    "met": bool(0.003 <= prim["high"]["area_share"] <= 0.03 and 0.12 <= prim["high"]["hit_rate"] <= 0.35 and 0.10 <= prim["high"]["capture_share"] <= 0.35)}
    p["P3_elevated_area_cap_binds"] = {"predicted": True, "observed": res["cutoffs"]["elevated_rule_used"] == "area_cap"}
    p["P4_fixed_cutoffs_do_not_transfer"] = {"predicted": True, "observed": not res["transfer_check"]["transfer_holds"]}
    p["P5_calibration_fails"] = {"predicted": True, "observed": not res["calibration"]["calibration_defensible"]}
    j = res["stability"]["S1_model_noise"]["mean"]["jaccard_high"]
    p["P6_S1_high_jaccard_0.35_to_0.70"] = {"predicted": [0.35, 0.70], "observed": j, "met": bool(0.35 <= j <= 0.70)}
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    boots = 20 if args.quick else args.boots
    rng = np.random.default_rng(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / ("tier_study_QUICK_DO_NOT_CITE.json" if args.quick else "tier_study.json")

    country = load_country()
    cov = us_trusted_mask(country)
    land = country == 1
    data = Data()
    log(f"loaded: {data.n_days} days, active fraction {data.active_frac:.4f}, weights {data.weights}")

    res = {
        "preregistration_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "bootstrap": {"draws": boots, "seed": 0, "unit": "UTC date"},
        "coverage_cells": int(cov.sum()), "us_land_cells": int(land.sum()),
        "prior_shift_weights": {"active_fraction_in_data": data.active_frac, **data.weights},
    }
    prim, wfull = stage1_tables(data, cov)
    days_p, days_w = days_with(prim["max1"]), days_with(wfull["max1"])
    res["sample"] = {"primary_samples": int(sum(data.year[y]["primary"].sum() for y in YEARS)),
                     "primary_days": int(len(days_p)), "full_mix_days": int(len(days_w))}
    chosen, res["operator"] = stage2_operator(prim, days_p, data.n_days, boots, rng)
    log(f"stage 2: chosen operator = {chosen}")
    cutoffs, res["cutoffs"] = stage3_cutoffs(prim, chosen)
    log(f"stage 3: cutoffs {cutoffs}, rule {res['cutoffs']['elevated_rule_used']}")
    res["tier_table_A"] = stage4_tables(prim[f"{chosen}1"], wfull[f"{chosen}1"], cutoffs, days_p, days_w, data.n_days, boots, rng, data)
    acc, land_counts = tier_arrays_pass(data, cov, land, chosen, cutoffs, data.n_days)
    res["map_level_and_land"] = stage5_summary(acc, land_counts, days_p, days_w, data.n_days, boots, rng)
    b_subsets, b_tabs = b_tables(data, cov)
    days_b = days_with(b_tabs[(0, chosen)])
    holds, serving, per_subset_cut, res["transfer_check"] = stage6_transfer(data, chosen, cutoffs, prim, b_tabs, b_subsets, days_b, boots, rng)
    log(f"stage 6: transfer holds = {holds}; serving cutoffs {serving}")
    area_targets = res["cutoffs"]["cum_area_share"]
    res["stability"] = stage7_stability(data, cov, chosen, cutoffs, serving, area_targets)
    log("stage 7 done")
    res["calibration"] = stage8_calibration(data, chosen, cutoffs, prim, wfull, days_p, b_subsets, b_tabs, holds, serving, boots, rng, per_subset_cut)
    log(f"stage 8: calibration defensible = {res['calibration']['calibration_defensible']}")
    res["serving"] = {"operator": chosen, "radius_cells": 1, "cutoffs": serving, "tier_set_id": T.tier_set_id(chosen, 1, serving),
                      "cutoffs_source": "A fixed scores" if holds else "quantile-matched on B (median of 5-member subsets)"}
    res["predictions_scored"] = score_predictions(res)
    out_path.write_text(json.dumps(clean(res), indent=1))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
