#!/bin/bash
set -x # Print commands and their arguments as they are executed

AGENT_DIR=./
EXP_ID=gosou
CUSTOM_EXP_NAME="goso10"
dataset_dir=/home/donaldzy/文档/data
MEMORY_INDEX=0

code_model=deepseek-chat
code_temp=0.5
code_base_url="https://api.deepseek.com/beta"
code_api_key=${DEEPSEEK_API_KEY}

feedback_model=deepseek-chat
feedback_temp=0.5
feedback_base_url="https://api.deepseek.com/beta"
feedback_api_key=${DEEPSEEK_API_KEY}

start_cpu=0
CPUS_PER_TASK=4
end_cpu=$((start_cpu + CPUS_PER_TASK - 1))

TIME_LIMIT_SECS=3600

cd ${AGENT_DIR}
export MEMORY_INDEX
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)
export STEP_LIMIT=500

mkdir -p ${AGENT_DIR}/logs

# use the mirror if needed
# export HF_ENDPOINT=https://hf-mirror.com

DEFAULT_EXP_NAME="${EXP_ID}_mcts_comp_validcheck_[cpu-${start_cpu}-${end_cpu}]"

# 如果 CUSTOM_EXP_NAME 不为空，则覆盖默认命名
FINAL_EXP_NAME=${CUSTOM_EXP_NAME:-$DEFAULT_EXP_NAME}

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} timeout $TIME_LIMIT_SECS python main_mcts.py \
  dataset_dir="${dataset_dir}" \
  data_dir="${dataset_dir}/${EXP_ID}/public" \
  template_file="./instruction/instruction_template.txt" \
  exp_name="${FINAL_EXP_NAME}" \
  start_cpu_id="${start_cpu}" \
  cpu_number="${CPUS_PER_TASK}" \
  agent.time_limit=${TIME_LIMIT_SECS} \
  agent.check_format=false\
  agent.steps=${STEP_LIMIT} \
  agent.code.model=$code_model \
  agent.code.temp=$code_temp \
  agent.code.base_url=$code_base_url \
  agent.code.api_key=$code_api_key \
  agent.feedback.model=$feedback_model \
  agent.feedback.temp=$feedback_temp \
  agent.feedback.base_url=$feedback_base_url \
  agent.feedback.api_key=$feedback_api_key \
  agent.search.parallel_search_num=1
  # agent.steerable_reasoning=false

if [ $? -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi
