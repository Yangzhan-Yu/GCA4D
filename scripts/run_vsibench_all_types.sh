#!/usr/bin/env bash
set -u

cd /data3/Agentic-Spatial-Reasoning/gca-main
source scripts/gca_env.sh
# Planner model (text-only) and VLM model (vision) may be configured
# separately; both fall back to the legacy AGENT_COT_REASONER_* vars.
eval "$(grep -E '^export AGENT_(PLANNER|VLM|COT_REASONER)_(BASE_URL|API_KEY|MODEL)=' API.txt)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export MAX_API_CALLS="${MAX_API_CALLS:-120}"
mkdir -p logs

runs=(
  "object_counting arkitscenes 41069025 1"
  "object_size_estimation arkitscenes 41069025 168"
  "object_abs_distance arkitscenes 41069025 680"
  "object_rel_distance arkitscenes 42446103 1334"
  "object_rel_direction_easy arkitscenes 41069025 1236"
  "object_rel_direction_medium arkitscenes 41069025 1100"
  "object_rel_direction_hard arkitscenes 41069025 957"
  "room_size_estimation arkitscenes 41069025 530"
  "obj_appearance_order scannetpp 45b0dac5e3 2713"
  "route_planning arkitscenes 42446167 4959"
)

for entry in "${runs[@]}"; do
  read -r task_type dataset scene_name question_id <<< "${entry}"
  result_root="results/VSI-Bench-TypeSmoke/${task_type}"
  log_path="logs/type_smoke_${task_type}.log"
  echo "============================================================"
  echo "RUN ${task_type}: ${dataset}/${scene_name} q${question_id}"
  echo "RESULT_ROOT ${result_root}"
  echo "LOG ${log_path}"
  echo "============================================================"

  rm -rf "${result_root}"
  /data2/conda_envs/gca/bin/python -m entrypoints.run_vsibench_agent \
    --dataset "${dataset}" \
    --scene-name "${scene_name}" \
    --question-id "${question_id}" \
    --data-root /data3/Agentic-Spatial-Reasoning/gca-main/data/vsibench \
    --results-root "/data3/Agentic-Spatial-Reasoning/gca-main/${result_root}" \
    --device cuda \
    --max-api-calls "${MAX_API_CALLS}" \
    --max-rounds 5 \
    --max-total-steps 25 \
    --max-tool-failures 2 2>&1 | tee "${log_path}"

  result_path="${result_root}/${dataset}/${scene_name}/evidence/agent_result.json"
  if [ -f "${result_path}" ]; then
    echo "RESULT ${task_type}:"
    cat "${result_path}"
    echo
  else
    echo "RESULT ${task_type}: missing agent_result.json"
  fi
done
