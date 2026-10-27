#!/usr/bin/env bash
# Full hourly HRRR pull: base-14 store first, then the extra candidate fields,
# one retry pass each, then derive whole-bin staging (base, and base+extra).
# Resumable: every step skips what is already in its manifest, so re-running
# after an interruption just continues. Intended to be launched detached:
#
#   nohup caffeinate -i bash scripts/02_dataset_build/run_full_hourly_pull.sh \
#       > outputs/logs/full_hourly_pull.log 2>&1 &
#
# Progress: grep -c '): OK' outputs/logs/full_hourly_pull.log ; finished when
# outputs/logs/full_hourly_pull.DONE exists.

set -u
cd "$(dirname "$0")/../.."
source .venv/bin/activate

WORKERS="${WORKERS:-4}"
BUILD="python scripts/02_dataset_build/build_hourly_dataset.py"
rm -f outputs/logs/full_hourly_pull.DONE

for fs in base extra; do
  echo "=== $(date -u +%FT%TZ) pull: $fs ==="
  $BUILD --feature-set "$fs" --workers "$WORKERS" --skip-derive
  echo "=== $(date -u +%FT%TZ) retry-failed: $fs ==="
  $BUILD --feature-set "$fs" --workers "$WORKERS" --skip-derive --retry-failed
done

echo "=== $(date -u +%FT%TZ) derive whole-bin staging ==="
$BUILD --derive-only
$BUILD --derive-only --include-extra

echo "=== $(date -u +%FT%TZ) done ==="
touch outputs/logs/full_hourly_pull.DONE
