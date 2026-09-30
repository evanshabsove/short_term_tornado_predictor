"""
Standardized model-evaluation metrics: AUC-ROC, AUC-PR (average
precision), accuracy/precision/recall/F1 (at threshold=0.5 and at each
model's own best-F1 threshold), and Cohen's d, computed per-cell over
every sample in a dataset.

Extracted from notebooks/model_comparison_summary.ipynb, which found
that accuracy is meaningless at this project's ~1e-4 to 1e-3 per-cell
positive rate (every model scores >99.8%, including an implicit
always-negative baseline) and that AUC-ROC is misleadingly high
(0.92-0.98) under the same imbalance -- AUC-PR (~0.02-0.03 for every
model tested so far) is the metric that actually reflects operating-
point usefulness. That notebook also found no model's raw probability
ever crosses the conventional 0.5 threshold for a real positive cell,
which is why every threshold-based metric here is reported both at 0.5
and at each model's own best-F1 threshold -- comparing different
configurations at a fixed 0.5 cutoff is not a fair comparison once
alpha (or any other calibration-shifting hyperparameter) varies.

Use evaluate_model() when a model is already in memory (e.g. right
after training, in scripts/sweep_alpha.py-style experiments) and
evaluate_checkpoint() when starting from a saved checkpoint path (e.g.
scripts/evaluate_model.py, or comparing many past runs).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from tornado_predictor.inference import assert_feature_order_matches, load_checkpoint, predict_proba
from tornado_predictor.training import DenseGridDataset


def get_proba_labels(model, dataset: DenseGridDataset) -> tuple[np.ndarray, np.ndarray]:
    """Runs model over every sample in dataset, flattening and
    concatenating predicted probabilities and true labels across every
    cell in every sample into one (y_score, y_true) pair."""
    probs, labels = [], []
    for i in range(len(dataset)):
        X, y = dataset[i]
        proba = predict_proba(model, X)
        probs.append(proba.ravel())
        labels.append(y.ravel())
    return np.concatenate(probs), np.concatenate(labels)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, include_curves: bool = True) -> dict:
    """Full per-cell metric set for one model's predictions against
    ground truth. y_true with no positive cells is a valid input (e.g.
    an all-quiet sample set) -- AUC fields come back nan rather than
    raising, matching inference.sanity_check_against_labels's existing
    nan-on-no-positives convention.

    roc_curve/pr_curve are returned as plain Python lists (not numpy
    arrays) when include_curves=True, so the whole dict round-trips
    through json.dumps with no extra conversion -- needed for
    accumulating results across many runs (see scripts/sweep_alpha.py's
    results.json for the existing pattern this follows). Pass
    include_curves=False to omit them when comparing many models and
    plotting data isn't needed."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos_mask = y_true > 0
    has_positives = bool(pos_mask.any())
    has_negatives = bool((~pos_mask).any())

    if has_positives and has_negatives:
        auc_roc = float(roc_auc_score(y_true, y_score))
        auc_pr = float(average_precision_score(y_true, y_score))
    else:
        auc_roc = float("nan")
        auc_pr = float("nan")

    pred_05 = y_score >= 0.5
    acc_05 = float((pred_05 == pos_mask).mean())
    tp05 = int((pred_05 & pos_mask).sum())
    fp05 = int((pred_05 & ~pos_mask).sum())
    fn05 = int((~pred_05 & pos_mask).sum())
    recall_05 = tp05 / (tp05 + fn05) if (tp05 + fn05) > 0 else float("nan")
    precision_05 = tp05 / (tp05 + fp05) if (tp05 + fp05) > 0 else float("nan")

    if has_positives and has_negatives:
        prec, rec, thr = precision_recall_curve(y_true, y_score)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        best_i = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
        best_thr = float(thr[best_i]) if len(thr) else float("nan")
        precision_best_f1 = float(prec[best_i])
        recall_best_f1 = float(rec[best_i])
        f1_best_f1 = float(f1[best_i])
    else:
        prec = rec = thr = np.array([])
        best_thr = float("nan")
        precision_best_f1 = recall_best_f1 = f1_best_f1 = float("nan")

    if has_positives and has_negatives:
        pooled_std = np.sqrt((y_score[pos_mask].std() ** 2 + y_score[~pos_mask].std() ** 2) / 2)
        cohens_d = float((y_score[pos_mask].mean() - y_score[~pos_mask].mean()) / pooled_std) if pooled_std > 0 else float("nan")
    else:
        cohens_d = float("nan")

    metrics = {
        "n_cells": int(len(y_true)),
        "pos_rate": float(pos_mask.mean()),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "acc_at_0.5": acc_05,
        "recall_at_0.5": recall_05,
        "precision_at_0.5": precision_05,
        "best_f1_threshold": best_thr,
        "precision_at_best_f1": precision_best_f1,
        "recall_at_best_f1": recall_best_f1,
        "f1_at_best_f1": f1_best_f1,
        "cohens_d": cohens_d,
        "mean_proba_positive": float(y_score[pos_mask].mean()) if has_positives else float("nan"),
        "mean_proba_negative": float(y_score[~pos_mask].mean()) if has_negatives else float("nan"),
    }

    if include_curves:
        if has_positives and has_negatives:
            fpr, tpr, roc_thr = roc_curve(y_true, y_score)
            metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thr.tolist()}
            metrics["pr_curve"] = {"precision": prec.tolist(), "recall": rec.tolist(), "thresholds": thr.tolist()}
        else:
            metrics["roc_curve"] = {"fpr": [], "tpr": [], "thresholds": []}
            metrics["pr_curve"] = {"precision": [], "recall": [], "thresholds": []}

    return metrics


def evaluate_model(model, dataset: DenseGridDataset, include_curves: bool = True) -> dict:
    """Evaluates an already-in-memory model against dataset -- no
    checkpoint save/reload round-trip, for use right after training
    (e.g. an alpha or architecture sweep)."""
    y_score, y_true = get_proba_labels(model, dataset)
    return compute_metrics(y_true, y_score, include_curves=include_curves)


def evaluate_checkpoint(checkpoint_path: str, dataset_path: str, include_curves: bool = True) -> dict:
    """Loads a saved checkpoint and dataset from disk and evaluates.
    Raises via assert_feature_order_matches if the dataset's feature
    channel order doesn't match the checkpoint's training-time order --
    same guard scripts/run_inference.py already relies on, since a
    silent mismatch would produce plausible-looking but meaningless
    metrics. Merges in the checkpoint's own final_train_loss/
    final_val_loss (nan if the checkpoint has no val_loss_history, e.g.
    the pilot checkpoint) alongside the computed metrics."""
    model, checkpoint = load_checkpoint(checkpoint_path)
    dataset = DenseGridDataset(dataset_path)
    assert_feature_order_matches(dataset.feature_names, checkpoint)

    metrics = evaluate_model(model, dataset, include_curves=include_curves)
    metrics["final_train_loss"] = checkpoint["loss_history"][-1]
    val_loss_history = checkpoint.get("val_loss_history")
    metrics["final_val_loss"] = val_loss_history[-1] if val_loss_history else float("nan")
    return metrics
