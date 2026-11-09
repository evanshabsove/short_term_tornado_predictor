#!/usr/bin/env bash
# Stage 1 of docs/convlstm_dropout_preregistration.md: dropout rate selection on fold 2018, seeds 20 and 21 (two lanes in parallel, one per seed). Resumable.
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
OUT=models/convlstm/dropout_select
mkdir -p $OUT
shasum -a 256 docs/convlstm_dropout_preregistration.md >> $OUT/prereg_sha_at_launch.txt
lane() {
  SEED=$1
  for P in 0.0 0.1 0.3 0.5; do
    D=$OUT/p$P/seed$SEED; [ -f $D/results.json ] && continue
    mkdir -p $D
    OMP_NUM_THREADS=6 python -u -W ignore scripts/11_convlstm/train_convlstm.py --fold 2018 --seed $SEED --epochs 50 --ema-decays 0.9999 \
      --norm-bottleneck --pooled-skips --grad-clip-norm 1.0 --lr 1e-4 --dropout $P --out-dir $D 2>&1 | tee -a $D/run.log
  done
}
lane 20 & lane 21 &
wait
echo DROPOUT_SELECT_DONE | tee -a $OUT/run.log
