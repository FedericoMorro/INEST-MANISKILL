#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_path> <dataset/trajs_name_suffix>"
    exit 1
fi

python scripts/eval_policy.py \
    $1

model_name=$(basename "$1")
model_no_ext="${model_name%.*}"
dir_name=$(dirname "$1")
parent_dir=$(dirname "$dir_name")

python inest_irl/maniskill3/replay_trajectory.py \
    --traj-path ${parent_dir}/eval_results/${model_no_ext}/trajectories.h5 \
    --save-traj \
    --obs-mode rgb \
    --output-path ../data/inest-maniskill/experiment_data-trajs/trajs_$2 \
    --save-unsuccessful \
    --subtask-json

python inest_irl/dataset_utils/h5_to_dataset.py \
    --h5_path ../data/inest-maniskill/experiment_data-trajs/trajs_$2/trajectories.rgb.pd_ee_delta_pose.physx_cpu.h5 \
    --dataset_path ../data/inest-maniskill/experiment_data-trajs/dataset_$2 \
    --config inest_irl/dataset_utils/h5_to_dataset_configs/sb3-sac_trajs.yaml

