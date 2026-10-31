#!/usr/bin/env bash
# Repair-arm snippets (docs/convlstm_repair_plan.md): fold 2020, seed 10, 10 epochs, full train set, EMA 0.9999, three variants one after another.
# ~35-50 min each. C1 = bottleneck GroupNorm + grad clip; C2 = pooled skips only; C3 = all. Resumable (a variant with results.json is skipped).
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
run() { # name, flags...
  local name=$1; shift
  local D=models/convlstm/snippet_$name
  if [ -f $D/results.json ]; then echo "skip $name"; return; fi
  mkdir -p $D
  python -u -W ignore scripts/11_convlstm/train_convlstm.py --fold 2020 --seed 10 --epochs 10 "$@" --out-dir $D 2>&1 | tee -a $D/run.log
}
run c1 --norm-bottleneck --grad-clip-norm 1.0
run c2 --pooled-skips
run c3 --norm-bottleneck --pooled-skips --grad-clip-norm 1.0
