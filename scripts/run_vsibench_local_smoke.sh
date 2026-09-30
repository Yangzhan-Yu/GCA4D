#!/bin/bash
# Run a handful of VSI-Bench question types against the LOCAL planner.
#
# Prerequisites:
#   1. vLLM is serving the planner (scripts/serve_planner_local.sh)
#   2. The perception GPU is free
#
# Usage:
#   PERCEPTION_GPU=7 bash scripts/run_vsibench_local_smoke.sh
#
# Override the planner endpoint if it is not on localhost:8000.
set -u

cd "$(dirname "$0")/.."
source scripts/gca_env.sh

export AGENT_PLANNER_MODEL="${AGENT_PLANNER_MODEL:-qwen3-30b-a3b}"
export AGENT_PLANNER_BASE_URL="${AGENT_PLANNER_BASE_URL:-http://127.0.0.1:8000/v1}"
export AGENT_PLANNER_API_KEY="${AGENT_PLANNER_API_KEY:-local}"
export GCA_LLM_TIMEOUT="${GCA_LLM_TIMEOUT:-120}"
# Hybrid Qwen3 thinks by default; the planner only needs a small JSON decision.
# NOTE: do not fold this into ${VAR:-...}; bash consumes the JSON's closing
# brace as the end of the expansion and the value comes out malformed.
if [ -z "${GCA_LLM_EXTRA_BODY:-}" ]; then
    export GCA_LLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
fi

PERCEPTION_GPU="${PERCEPTION_GPU:-0}"
DATASET="${DATASET:-arkitscenes}"
SCENE="${SCENE:-41069025}"
# Local inference has no marginal cost, so the budget is set for correctness
# rather than to save API spend.  The repeat guard (a third identical tool call
# is refused) is what actually stops a loop, not the step cap - raising the cap
# only gives a wandering Planner room to recover.
# Deliberately generous: on free local inference the budget must never be the
# binding constraint.  A run that genuinely stalls is stopped by the loop's
# progress watchdog ("No measurable progress for N steps"), which reports a
# reasoning problem rather than a truncation.
MAX_STEPS="${MAX_STEPS:-40}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
MAX_TOOL_FAILURES="${MAX_TOOL_FAILURES:-5}"
PYTHON="${PYTHON:-/data2/conda_envs/gca/bin/python}"
ROOT="results/VSI-Bench-Local"
mkdir -p logs

# type                       question_id  ground_truth  metric
runs=(
  "object_counting            1            2              MRA"
  "object_counting            0            4              MRA"
  "object_abs_distance        680          2.9            MRA"
  "object_size_estimation     167          62             MRA"
  "object_rel_direction_easy  1236         A              MCA"
  "object_rel_direction_medium 1100        C              MCA"
  "object_rel_direction_hard  957          A              MCA"
  "room_size_estimation       530          26.4           MRA"
)

echo "planner   : $AGENT_PLANNER_BASE_URL ($AGENT_PLANNER_MODEL)"
echo "perception: GPU $PERCEPTION_GPU"
echo "budget    : rounds=$MAX_ROUNDS steps=$MAX_STEPS tool_failures=$MAX_TOOL_FAILURES"
echo "results   : $ROOT"
echo

for row in "${runs[@]}"; do
  read -r qtype qid gt metric <<<"$row"
  log="logs/local_${qtype}_${qid}.log"
  # One results root PER QUESTION.  Sharing a scene root between questions
  # (e.g. both counting questions) makes them share evidence/agent_memory.json,
  # so one question's observations leak into the next and the accumulated
  # memory blows the prompt past the model's context limit.
  qroot="$ROOT/$qtype/$qid"

  echo "============================================================"
  echo ">>> $qtype  id=$qid  GT=$gt  ($metric)"
  echo "    results: $qroot"
  echo "============================================================"

  # Start from nothing.  Re-running with the same root would otherwise reload
  # the cached question plan, agent memory, 4D memory (frames/observations/
  # geometry), the visibility scan, any detection overrides and the extracted
  # frames - i.e. almost the whole previous run.
  rm -rf "$qroot"
  CUDA_VISIBLE_DEVICES="$PERCEPTION_GPU" "$PYTHON" -m entrypoints.run_vsibench_agent \
      --dataset "$DATASET" \
      --scene-name "$SCENE" \
      --question-id "$qid" \
      --results-root "$qroot" \
      --device cuda \
      --reset-agent-memory \
      --max-rounds "$MAX_ROUNDS" \
      --max-total-steps "$MAX_STEPS" \
      --max-tool-failures "$MAX_TOOL_FAILURES" \
      --max-api-calls 200 >"$log" 2>&1
  rc=$?
  result="$qroot/$DATASET/$SCENE/evidence/agent_result.json"
  if [ -f "$result" ]; then
    "$PYTHON" - "$result" "$gt" "$qid" <<'PY'
import json, sys
path, gt, qid = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(path))
pred = r.get('final_answer')
print(f"  done={r.get('done')} steps={r.get('steps')} pred={pred!r} gt={gt!r}")
if r.get('error'):
    print(f"  error: {str(r['error'])[:160]}")
ops = r.get('operation_results') or {}
for k, v in ops.items():
    print(f"  {k}: value={v.get('value')!r} flags={v.get('quality_flags')} "
          f"status={v.get('verification_status')}")
PY
  else
    echo "  no agent_result.json (exit=$rc) - see $log"
  fi
  echo
done

echo "============================================================"
echo "logs      : logs/local_*.log"
echo "per-question results: $ROOT/<question_type>/<question_id>/"
echo
echo "Each question ran from an empty directory, so nothing was reused."
