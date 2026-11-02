#!/usr/bin/env bash
# Ladder round 3 (docs/convlstm_diagnostic_ladder.md): arm A's U-Net at lr 1e-4 on fold 2020, seeds 10 and 11. Resumable.
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
for S in 10 11; do
  D=models/convlstm/diagnostics/ladder/unet_lr1e4_s$S
  [ -f $D/results.json ] && { echo "skip s$S"; continue; }
  mkdir -p $D
  python -u -W ignore scripts/05_year_cv_diagnostics/cross_validate_year.py --seed $S --years 2020 --epochs 10 --lr 1e-4 --ema-decays 0.9999 \
    --staging-dir data/interim/scaled_build_wholebin --manifest data/interim/scaled_build_wholebin/manifest.json \
    --scratch-dir data/interim/cv_scratch_ladder3 --out-dir $D --results $D/results.json 2>&1 | tee -a $D/run.log
done
echo LADDER3_DONE | tee -a models/convlstm/diagnostics/ladder/run3.log
