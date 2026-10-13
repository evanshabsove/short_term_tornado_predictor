#!/usr/bin/env bash
# Sequential driver: (A) UH 2-5 km seeds 13-15, then (B) full 21-feature UH seeds 10-12. Both resumable.
set -e
SEEDS="13 14 15" bash scripts/run_uh_study.sh
SEEDS="10 11 12" bash scripts/run_uh21_study.sh
