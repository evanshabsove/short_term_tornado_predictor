#!/usr/bin/env bash
# ConvLSTM full study (docs/convlstm_preregistration.md): folds 2016/2020/2024 x seeds 10-12, 50 epochs, EMA 0.9999, whole train set.
# ~2.4 h per run x 9 = ~22 h on this CPU. Resumable: a run whose results.json exists is skipped; a crashed run is redone from scratch
# with the same seed (rerun the script). Never run two studies at once (CPU contention distorts the timing).
# Usage (repo root):  caffeinate -i bash scripts/11_convlstm/run_full_study.sh
set -eo pipefail   # pipefail: tee must not hide a failing python
cd "$(dirname "$0")/../.."
source .venv/bin/activate   # bare `python` hits a broken pyenv shim outside the venv
OUT=models/convlstm/full_study
mkdir -p $OUT
# Record the pre-registration's hash at launch, so an edit after the first run started is visible.
sha256sum docs/convlstm_preregistration.md 2>/dev/null >> $OUT/prereg_sha_at_launch.txt || shasum -a 256 docs/convlstm_preregistration.md >> $OUT/prereg_sha_at_launch.txt
for FOLD in ${FOLDS:-2016 2020 2024}; do
  for SEED in ${SEEDS:-10 11 12}; do
    D=$OUT/seed$SEED/fold$FOLD
    if [ -f $D/results.json ]; then echo "skip fold $FOLD seed $SEED (done)"; continue; fi
    mkdir -p $D
    python -u -W ignore scripts/11_convlstm/train_convlstm.py --fold $FOLD --seed $SEED --epochs 50 --ema-decays 0.9999 \
      --out-dir $D 2>&1 | tee -a $D/run.log
  done
done
