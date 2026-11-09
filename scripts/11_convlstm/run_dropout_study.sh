#!/usr/bin/env bash
# Stage 2 of docs/convlstm_dropout_preregistration.md: C3 + dropout 0.1 (rate chosen in Stage 1), folds 2016/2020/2024 x seeds 10-12, 50 epochs, EMA 0.9999.
# Three parallel lanes (one per seed, 4 threads each). Resumable: a run whose results.json exists is skipped. Plug in.
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
OUT=models/convlstm/full_study_c3_dropout
P=${DROPOUT:-0.1}
mkdir -p $OUT
shasum -a 256 docs/convlstm_dropout_preregistration.md >> $OUT/prereg_sha_at_launch.txt
lane() { SEED=$1; for FOLD in 2016 2020 2024; do D=$OUT/seed$SEED/fold$FOLD; [ -f $D/results.json ] && continue; mkdir -p $D
  OMP_NUM_THREADS=4 python -u -W ignore scripts/11_convlstm/train_convlstm.py --fold $FOLD --seed $SEED --epochs 50 --ema-decays 0.9999 \
    --norm-bottleneck --pooled-skips --grad-clip-norm 1.0 --lr 1e-4 --dropout $P --out-dir $D 2>&1 | tee -a $D/run.log; done; }
lane 10 & lane 11 & lane 12 &
wait
echo DROPOUT_STUDY_DONE | tee -a $OUT/run.log
