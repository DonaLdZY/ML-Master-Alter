#!/bin/bash
set -x # Print commands and their arguments as they are executed

# the same dataset_dir with run.sh
dataset_dir=/home/donaldzy/文档/data

# launch a server which tells agent whether the submission is valid or not, allowed by MLE-Bench rules
nohup python -u grading_server.py \
  dataset_dir="${dataset_dir}" \
  data_dir="none" \
  desc_file="none" > grading_server.out 2>&1 &


