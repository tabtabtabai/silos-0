#!/bin/bash
# Run one benchmark slot through the policy server. Args: one line of scripts/jobs.py
#   slot benchmark benchmark_dir idx horizon_s left,right,wrist
# Env: MOLMOSPACES_SRC (MolmoSpaces v0.2.9 checkout), MOLMOSPACES_PYTHON (its venv's python), MLSPACES_ASSETS_DIR,
#      RUN_OUT (one slot-NNN folder per slot), HARNESS_URL (default ws://127.0.0.1:8001).
# A slot with a DONE marker is skipped, so rerunning the same command resumes.
set -u
slot=$1; name=$2; dir=$3; idx=$4; hsec=$5; cams=$6
HERE=$(cd "$(dirname "$0")/.." && pwd)
out=${RUN_OUT:?}/slot-$slot
[ -f "$out/DONE" ] && exit 0
mkdir -p "$out"
cd "${MOLMOSPACES_SRC:?}" || exit 1
export PYTHONPATH="$HERE/src:$HERE${PYTHONPATH:+:$PYTHONPATH}" MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:?}" \
  HARNESS_CAMS="$cams" HARNESS_MOTION_LIMIT_S="$hsec" HARNESS_BENCH="molmospaces-hundred/$name" HARNESS_SLOT="$slot"
if "${MOLMOSPACES_PYTHON:?}" molmo_spaces/evaluation/eval_main.py client.policy:HarnessEvalConfig \
    --benchmark_dir "$dir" --idx "$idx" --no_wandb --output_dir "$out/eval" > "$out/eval.log" 2>&1; then
  date -u +%FT%TZ > "$out/DONE"
else
  echo "FAIL slot $slot"; tail -3 "$out/eval.log"
fi
