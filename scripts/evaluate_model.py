"""
Loads a trained checkpoint and a dataset, computes the standardized
metric set (evaluate.py: AUC-ROC, AUC-PR, precision/recall/F1 at
threshold=0.5 and at the model's own best-F1 threshold, Cohen's d),
prints a summary, and optionally saves the full metrics (including
ROC/PR curve data) as JSON.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --checkpoint models/alpha_sweep/tornado_cnn_alpha_0.9.pt --dataset data/processed/training_dataset_val_full.nc
    python scripts/evaluate_model.py --out outputs/eval_full_alpha025.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tornado_predictor.evaluate import evaluate_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "models" / "tornado_cnn_full.pt"
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_val_full.nc"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out", type=Path, default=None, help="optional path to save full metrics (incl. curves) as JSON")
    parser.add_argument("--no-curves", action="store_true", help="omit ROC/PR curve data from the saved JSON")
    args = parser.parse_args()

    metrics = evaluate_checkpoint(str(args.checkpoint), str(args.dataset), include_curves=not args.no_curves)

    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset:    {args.dataset}")
    print(f"n_cells={metrics['n_cells']:,}  pos_rate={metrics['pos_rate']:.2e}")
    print(f"final_train_loss={metrics['final_train_loss']:.5f}  final_val_loss={metrics['final_val_loss']:.5f}")
    print(f"AUC-ROC={metrics['auc_roc']:.4f}  AUC-PR={metrics['auc_pr']:.4f}  Cohen's d={metrics['cohens_d']:.2f}")
    print(f"acc@0.5={metrics['acc_at_0.5']:.5f}  recall@0.5={metrics['recall_at_0.5']:.4f}  precision@0.5={metrics['precision_at_0.5']:.4f}")
    print(
        f"best-F1 threshold={metrics['best_f1_threshold']:.4f}  "
        f"precision={metrics['precision_at_best_f1']:.4f}  "
        f"recall={metrics['recall_at_best_f1']:.4f}  "
        f"f1={metrics['f1_at_best_f1']:.4f}"
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2))
        print(f"\nSaved metrics to {args.out}")


if __name__ == "__main__":
    main()
