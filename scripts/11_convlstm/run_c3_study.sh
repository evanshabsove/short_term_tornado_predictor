#!/usr/bin/env bash
# Repaired ConvLSTM (C3, lr 1e-4) full study (docs/convlstm_c3_preregistration.md): folds 2016/2020/2024 x seeds 10-12, 50 epochs, EMA 0.9999.
# Resumable: a run whose results.json exists is skipped. Never run two studies at once. Plug in; usage: caffeinate -i bash scripts/11_convlstm/run_c3_study.sh
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
OUT=models/convlstm/full_study_c3
mkdir -p $OUT
shasum -a 256 docs/convlstm_c3_preregistration.md >> $OUT/prereg_sha_at_launch.txt
for FOLD in ${FOLDS:-2016 2020 2024}; do
  for SEED in ${SEEDS:-10 11 12}; do
    D=$OUT/seed$SEED/fold$FOLD
    if [ -f $D/results.json ]; then echo "skip fold $FOLD seed $SEED (done)"; continue; fi
    mkdir -p $D
    python -u -W ignore scripts/11_convlstm/train_convlstm.py --fold $FOLD --seed $SEED --epochs 50 --ema-decays 0.9999 \
      --norm-bottleneck --pooled-skips --grad-clip-norm 1.0 --lr 1e-4 --out-dir $D 2>&1 | tee -a $D/run.log
  done
done
echo C3_STUDY_DONE | tee -a $OUT/run.log
