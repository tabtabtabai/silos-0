#!/bin/bash
# Run one episode through the policy server. Args: one line of scripts/jobs.py
#   job set benchmark slot benchmark_dir episode_index horizon_s left,right,wrist
# Env: MOLMOSPACES_SRC (MolmoSpaces v0.2.9 checkout), MOLMOSPACES_PYTHON (its venv's python), MLSPACES_ASSETS_DIR,
#      RUN_OUT (one folder per job), HARNESS_URL (default ws://127.0.0.1:8001).
# Writes $RUN_OUT/<job>/status.json at start and end (see scripts/episode_status.py) and a DONE marker on success.
# A job with DONE is skipped, so rerunning the same command resumes. A job that failed is run again; the failed
# attempt's eval.log and status.json are kept under attempts/<n>/ and its partial output is removed, so the folder
# never holds two trajectories.
set -u
job=$1; set_name=$2; name=$3; slot=$4; dir=$5; idx=$6; hsec=$7; cams=$8
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${MOLMOSPACES_PYTHON:?}
out=${RUN_OUT:?}/$job
[ -f "$out/DONE" ] && exit 0
if [ -f "$out/status.json" ]; then
  n=1; while [ -d "$out/attempts/$n" ]; do n=$((n + 1)); done
  mkdir -p "$out/attempts/$n"
  mv "$out/status.json" "$out/attempts/$n/"
  [ -f "$out/eval.log" ] && mv "$out/eval.log" "$out/attempts/$n/"
  rm -rf "$out/eval"
fi
mkdir -p "$out"
"$PY" "$HERE/scripts/episode_status.py" start "$out" "$job" "$set_name" "$name" "$slot" "$idx"
cd "${MOLMOSPACES_SRC:?}" || exit 1
export PYTHONPATH="$HERE/src:$HERE${PYTHONPATH:+:$PYTHONPATH}" MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:?}" \
  HARNESS_CAMS="$cams" HARNESS_MOTION_LIMIT_S="$hsec" HARNESS_BENCH="$set_name/$name" HARNESS_SLOT="$slot"
"$PY" molmo_spaces/evaluation/eval_main.py client.policy:HarnessEvalConfig \
  --benchmark_dir "$dir" --idx "$idx" --no_wandb --output_dir "$out/eval" > "$out/eval.log" 2>&1
code=$?
"$PY" "$HERE/scripts/episode_status.py" finish "$out" "$code"
if grep -q '"state": "done"' "$out/status.json"; then
  date -u +%FT%TZ > "$out/DONE"
else
  echo "FAIL $job (exit $code): $(grep -o '"error": .*' "$out/status.json" | cut -c1-200)"
fi
