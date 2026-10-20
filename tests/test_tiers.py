import numpy as np
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve

from tornado_predictor import tiers
from tornado_predictor.tiers import (
    LEVEL_VALUES,
    N_LEVELS,
    LevelCounts,
    LevelTableBuilder,
    area_at,
    area_cutoff,
    assign_tiers,
    average_precision,
    best_f1_cutoff,
    capture_cutoff,
    curves,
    dilate_labels,
    elevated_cutoff,
    isotonic_map,
    key_of,
    level_keys,
    match_levels,
    nearest_area_cutoff,
    quantile_bins,
    quantize,
    ratio_over,
    smooth,
    tier_counts,
    tier_set_id,
)

# --- smoothing ---


def test_max_smoothing_3x3_and_edges():
    x = np.zeros((4, 5), "float32")
    x[0, 0] = 1.0
    x[2, 3] = 2.0
    out = smooth(x, "max", 1)
    expected = np.zeros((4, 5), "float32")
    expected[0:2, 0:2] = 1.0  # corner cell: window clipped to the grid
    expected[1:4, 2:5] = 2.0
    np.testing.assert_array_equal(out, expected)


def test_mean_smoothing_uses_only_in_grid_cells():
    x = np.ones((3, 3), "float32")
    np.testing.assert_allclose(smooth(x, "mean", 1), 1.0, atol=1e-6)  # no zero-padding dilution at edges
    y = np.zeros((3, 3), "float32")
    y[1, 1] = 9.0
    out = smooth(y, "mean", 1)
    assert out[1, 1] == pytest.approx(1.0)  # 9 / 9 cells
    assert out[0, 0] == pytest.approx(9.0 / 4)  # corner window holds 4 in-grid cells
    assert out[0, 1] == pytest.approx(9.0 / 6)


def test_smoothing_radius_zero_is_identity_copy_and_batch_axes_are_independent():
    x = np.random.default_rng(0).random((3, 6, 7)).astype("float32")
    out = smooth(x, "max", 0)
    np.testing.assert_array_equal(out, x)
    assert out is not x
    batched = smooth(x, "max", 1)
    for i in range(3):
        np.testing.assert_array_equal(batched[i], smooth(x[i], "max", 1))
    batched_mean = smooth(x, "mean", 2)
    for i in range(3):
        np.testing.assert_allclose(batched_mean[i], smooth(x[i], "mean", 2), atol=1e-6)


def test_smoothing_rejects_bad_arguments():
    with pytest.raises(ValueError):
        smooth(np.zeros((3, 3)), "median", 1)
    with pytest.raises(ValueError):
        smooth(np.zeros((3, 3)), "max", -1)


def test_max_smoothing_never_lowers_a_score_and_mean_stays_in_range():
    x = np.random.default_rng(1).random((2, 8, 9)).astype("float32")
    assert (smooth(x, "max", 1) >= x).all()
    m = smooth(x, "mean", 1)
    assert m.min() >= x.min() - 1e-6 and m.max() <= x.max() + 1e-6


def test_dilate_labels():
    y = np.zeros((2, 5, 5), "uint8")
    y[0, 2, 2] = 1
    y[1, 0, 4] = 1
    d1 = dilate_labels(y, 1)
    assert d1[0].sum() == 9 and d1[0, 1:4, 1:4].all()
    assert d1[1].sum() == 4  # corner: clipped, zero padded outside
    assert dilate_labels(y, 0).sum() == 2
    assert dilate_labels(y, 2)[0].all()


# --- tier assignment ---


def test_assign_tiers_boundaries_coverage_and_order_check():
    s = np.array([[0.01, 0.10, 0.2], [0.5, 0.0, 0.9]], dtype="float32")
    cov = np.array([[True, True, True], [True, True, False]])
    t = assign_tiers(s, {"elevated": 0.10, "high": 0.5}, cov)
    np.testing.assert_array_equal(t, [[0, 1, 1], [2, 0, -1]])  # equal-to-cutoff goes up; masked cell is -1 even at 0.9
    assert t.dtype == np.int8
    with pytest.raises(ValueError):
        assign_tiers(s, {"elevated": 0.5, "high": 0.5}, cov)


def test_assign_tiers_is_monotone_in_score():
    rng = np.random.default_rng(2)
    s = np.sort(rng.random(500)).reshape(1, 500).astype("float32")
    t = assign_tiers(s, {"elevated": 0.3, "high": 0.7}, np.ones(500, bool))[0]
    assert (np.diff(t) >= 0).all()


def test_assign_tiers_broadcasts_over_bins():
    s = np.random.default_rng(3).random((2, 4, 6)).astype("float32")
    cov = np.ones((4, 6), bool)
    cov[0, 0] = False
    t = assign_tiers(s, {"elevated": 0.4, "high": 0.8}, cov)
    assert t.shape == (2, 4, 6) and (t[:, 0, 0] == -1).all() and (t[:, 1:, :] >= 0).all()


def test_tier_set_id_stable_and_sensitive():
    a = tier_set_id("max", 1, {"elevated": 0.0123456789, "high": 0.2})
    assert a == tier_set_id("max", 1, {"high": 0.2, "elevated": 0.0123456789})
    assert a.startswith("ts-") and len(a) == 11
    assert a != tier_set_id("mean", 1, {"elevated": 0.0123456789, "high": 0.2})
    assert a != tier_set_id("max", 2, {"elevated": 0.0123456789, "high": 0.2})
    assert a != tier_set_id("max", 1, {"elevated": 0.0123456789, "high": 0.21})
    assert a == tier_set_id("max", 1, {"elevated": 0.01234568, "high": 0.2})  # 6 significant figures


# --- quantization ---


def test_quantize_and_keys_round_trip():
    s = np.array([0.0, 1e-3, 0.5, 1.0, 1.7, -0.2], "float32")
    q = quantize(s)
    assert q.dtype == np.float16 and q.max() == 1.0 and q.min() == 0.0
    keys = level_keys(q)
    np.testing.assert_array_equal(LEVEL_VALUES[keys], q.astype("float32"))
    assert key_of(0.5) == int(level_keys(np.float16([0.5]))[0])
    with pytest.raises(ValueError):
        quantize(np.array([np.nan]))


def test_keys_are_monotone_in_value():
    vals = np.sort(np.random.default_rng(4).random(2000)).astype("float32")
    k = level_keys(quantize(vals))
    assert (np.diff(k) >= 0).all() and k.max() < N_LEVELS


# --- count tables vs brute force ---


def make_data(seed=0, n=40, cells=300, tied=True):
    rng = np.random.default_rng(seed)
    score = rng.random((n, cells)) ** 6
    if tied:
        score[rng.random((n, cells)) < 0.3] = 0.0
    q = quantize(score)
    dlab = (rng.random((n, cells)) < (0.02 + 0.3 * score)).astype("uint8")
    lab = (dlab * (rng.random((n, cells)) < 0.5)).astype("uint8")  # tornado cells are a subset of dilated positives
    return q, lab, dlab


def table_from(q, lab, dlab, days=None, weights=None):
    n = q.shape[0]
    days = np.arange(n) // 2 if days is None else days
    b = LevelTableBuilder(int(days.max()) + 1)
    for d in np.unique(days):
        idx = np.flatnonzero(days == d)
        w = None if weights is None else np.repeat(weights[idx], q.shape[1])
        b.add(int(d), level_keys(q[idx]).ravel(), lab[idx].ravel(), dlab[idx].ravel(), 1.0 if w is None else w)
    return b.build()


def test_average_precision_matches_sklearn():
    q, lab, dlab = make_data()
    c = table_from(q, lab, dlab).counts()
    expected = average_precision_score(dlab.ravel(), q.astype("float64").ravel())
    assert average_precision(c) == pytest.approx(expected, rel=1e-9)


def test_best_f1_matches_brute_force():
    q, lab, dlab = make_data(seed=1)
    c = table_from(q, lab, dlab).counts()
    cutoff, stats = best_f1_cutoff(c)
    prec, rec, thr = precision_recall_curve(dlab.ravel(), q.astype("float64").ravel())
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)[:-1]
    assert stats["f1"] == pytest.approx(f1.max(), rel=1e-9)
    flagged = q.astype("float32") >= cutoff
    p = dlab[flagged].sum() / flagged.sum()
    r = dlab[flagged].sum() / dlab.sum()
    assert stats["precision"] == pytest.approx(p) and stats["dilated_recall"] == pytest.approx(r)
    assert stats["area"] == pytest.approx(flagged.mean())
    assert stats["capture"] == pytest.approx(lab[flagged].sum() / lab.sum())


def test_capture_cutoff_is_highest_level_reaching_target():
    q, lab, dlab = make_data(seed=2)
    c = table_from(q, lab, dlab).counts()
    for target in (0.2, 0.6, 0.95):
        cutoff = capture_cutoff(c, target)
        cap = lab[q.astype("float32") >= cutoff].sum() / lab.sum()
        assert cap >= target
        above = np.unique(q.astype("float32")[q.astype("float32") > cutoff])
        if above.size:  # the next level up must not already reach it
            nxt = above.min()
            assert lab[q.astype("float32") >= nxt].sum() / lab.sum() < target
    with pytest.raises(ValueError):
        capture_cutoff(LevelCounts(c.total, np.zeros(N_LEVELS), c.dpos), 0.5)


def test_area_cutoffs():
    q, lab, dlab = make_data(seed=3, tied=False)
    c = table_from(q, lab, dlab).counts()
    for share in (0.01, 0.05, 0.3):
        cut = area_cutoff(c, share)
        assert area_at(c, cut) <= share + 1e-12
        assert abs(area_at(c, nearest_area_cutoff(c, share)) - share) <= abs(area_at(c, cut) - share) + 1e-12
        assert area_at(c, cut) == pytest.approx((q.astype("float32") >= cut).mean())


def test_area_cutoff_when_top_level_already_exceeds_share():
    q = np.zeros((4, 10), "float16")
    q[:, :8] = 0.5  # 80% of cells share the single top level
    lab = np.zeros((4, 10), "uint8")
    c = table_from(q, lab, lab).counts()
    assert area_cutoff(c, 0.1) == pytest.approx(0.5)


def test_elevated_rule_capture_branch_and_guards():
    q, lab, dlab = make_data(seed=4)
    c = table_from(q, lab, dlab).counts()
    high, _ = best_f1_cutoff(c)
    e, rule = elevated_cutoff(c, high=1.0, capture_target=0.5)  # a high cutoff above everything: capture rule applies
    assert rule == "capture" and e < 1.0
    # guard (a): force the cap to bind
    e2, rule2 = elevated_cutoff(c, high=1.0, capture_target=0.9, area_cap=0.01)
    assert rule2 == "area_cap" and area_at(c, e2) <= 0.01 + 1e-12
    # guard (b): High already captures more than the target -> fall back to a multiple of High's area
    h = capture_cutoff(c, 0.8)
    e3, rule3 = elevated_cutoff(c, high=h, capture_target=0.5, area_cap=1.0, high_multiple=3.0)
    if rule3 == "high_multiple":
        assert e3 < h
    assert e3 < h


def test_elevated_rule_raises_when_it_cannot_sit_below_high():
    q = np.full((2, 10), 0.5, "float16")  # one level only
    lab = np.zeros((2, 10), "uint8")
    lab[0, 0] = 1
    c = table_from(q, lab, lab).counts()
    with pytest.raises(ValueError, match="strictly below"):
        elevated_cutoff(c, high=0.5)


def test_tier_counts_match_direct_masks():
    q, lab, dlab = make_data(seed=5)
    c = table_from(q, lab, dlab).counts()
    cut = {"elevated": float(np.float16(0.02)), "high": float(np.float16(0.2))}
    out = tier_counts(c, cut)
    s = q.astype("float32")
    masks = {"low": s < cut["elevated"], "elevated": (s >= cut["elevated"]) & (s < cut["high"]), "high": s >= cut["high"],
             "elevated_up": s >= cut["elevated"]}
    for name, m in masks.items():
        assert out[name]["area_share"] == pytest.approx(m.mean())
        assert out[name]["capture_share"] == pytest.approx(lab[m].sum() / lab.sum())
        assert out[name]["hit_rate"] == pytest.approx(dlab[m].sum() / m.sum())
        assert out[name]["undilated_hit_rate"] == pytest.approx(lab[m].sum() / m.sum())
    assert sum(out[k]["area_share"] for k in ("low", "elevated", "high")) == pytest.approx(1.0)


def test_weights_scale_counts_and_day_weights_select_days():
    q, lab, dlab = make_data(seed=6, n=20)
    w = np.where(np.arange(20) % 2 == 0, 2.0, 0.5)
    t = table_from(q, lab, dlab, weights=w)
    c = t.counts()
    assert c.total.sum() == pytest.approx(w.sum() * q.shape[1])
    assert c.dpos.sum() == pytest.approx((w[:, None] * dlab).sum())
    keep = np.zeros(t.total.shape[0])
    keep[:3] = 1  # first three days = first six samples (2 per day)
    sub = t.counts(keep)
    assert sub.total.sum() == pytest.approx(w[:6].sum() * q.shape[1])
    boot = t.counts(np.full(t.total.shape[0], 2.0))
    assert boot.total.sum() == pytest.approx(2 * c.total.sum())


def test_empty_table_and_zero_positive_handling():
    b = LevelTableBuilder(3)
    assert b.build().counts().total.sum() == 0
    with pytest.raises(ValueError):
        curves(b.build().counts())
    q = np.full((2, 5), 0.1, "float16")
    z = np.zeros((2, 5), "uint8")
    c = table_from(q, z, z).counts()
    assert np.isnan(average_precision(c))
    with pytest.raises(ValueError):
        isotonic_map(c)


# --- calibration ---


def test_isotonic_map_is_monotone_bounded_and_recovers_a_calibrated_score():
    rng = np.random.default_rng(7)
    n, cells = 400, 400
    p = quantize(rng.random((n, cells)) ** 3 * 0.5)
    y = (rng.random((n, cells)) < p.astype("float64")).astype("uint8")  # perfectly calibrated by construction
    c = table_from(p, y, y, days=np.arange(n)).counts()
    cal = isotonic_map(c)
    assert (np.diff(cal) >= -1e-12).all() and cal.min() >= 0 and cal.max() <= 1
    occ = c.total > 0
    mid = occ & (c.total > 2000)
    np.testing.assert_allclose(cal[mid], LEVEL_VALUES[mid], atol=0.05)
    overall = ratio_over(c, cal, occ)
    assert overall["ratio"] == pytest.approx(1.0, abs=1e-6)  # isotonic fit preserves the overall rate


def test_quantile_bins_partition_occupied_levels_and_respect_area_edges():
    q, lab, dlab = make_data(seed=8, n=200, cells=500, tied=False)
    c = table_from(q, lab, dlab).counts()
    masks = quantile_bins(c)
    assert len(masks) == 7
    stacked = np.stack(masks)
    assert (stacked.sum(0)[c.total > 0] == 1).all()  # each occupied level in exactly one bin
    areas = [c.total[m].sum() / c.total.sum() for m in masks]
    assert sum(areas) == pytest.approx(1.0)
    assert areas[-1] <= 0.001 + 1e-9 or masks[-1].sum() == 1  # top bin: at most 0.1% of cells (or one tied level)
    # bins are ordered by score
    mins = [LEVEL_VALUES[m].min() for m in masks if m.any()]
    assert mins == sorted(mins)


def test_ratio_over_empty_mask():
    q, lab, dlab = make_data(seed=9)
    c = table_from(q, lab, dlab).counts()
    out = ratio_over(c, np.zeros(N_LEVELS), np.zeros(N_LEVELS, bool))
    assert out["n_cells"] == 0 and np.isnan(out["ratio"])


# --- quantile matching ---


def test_match_levels_identity_and_monotone():
    q, lab, dlab = make_data(seed=10, tied=False)
    c = table_from(q, lab, dlab).counts()
    m = match_levels(c, c)
    occ = c.total > 0
    np.testing.assert_allclose(m[occ], LEVEL_VALUES[occ])
    assert np.isnan(m[~occ]).all()
    # a monotone transform of the same ranking maps back onto the original scale
    q2 = quantize(q.astype("float32") * 0.5)
    c2 = table_from(q2, lab, dlab).counts()
    mapped = match_levels(c2, c)
    vals = mapped[c2.total > 0]
    assert (np.diff(vals) >= -1e-9).all()


def test_match_levels_scale_shift_recovers_area_shares():
    rng = np.random.default_rng(11)
    ref = quantize(rng.random((100, 400)) ** 4)
    src = quantize(rng.random((100, 400)) ** 4 * 0.6)  # same shape of distribution, compressed scale
    z = np.zeros(ref.shape, "uint8")
    cr, cs = table_from(ref, z, z).counts(), table_from(src, z, z).counts()
    m = match_levels(cs, cr)
    k = level_keys(src)
    mapped = m[k]
    # after matching, the share above a reference cutoff is close to the reference's own share
    cutoff = float(np.quantile(ref.astype("float32"), 0.95))
    assert abs((mapped >= cutoff).mean() - (ref.astype("float32") >= cutoff).mean()) < 0.02
