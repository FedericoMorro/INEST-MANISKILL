#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_path> [--save_video] [--eval_learned_return_model <dir>]"
    exit 1
fi

# parse arguments
model_path="$1"
save_video=false
eval_learned_return_model=""

# parse optional flags
shift 1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --save_video)
            save_video=true
            shift
            ;;
        --eval_learned_return_model)
            if [ -z "$2" ]; then
                echo "Error: --eval_learned_return_model requires a directory argument"
                exit 1
            fi
            eval_learned_return_model="$2"
            shift 2
            ;;
        *)
            echo "Unknown flag: $1"
            echo "Usage: $0 <model_path> [--save_video] [--eval_learned_return_model <dir>]"
            exit 1
            ;;
    esac
done

if [ ! -f "$model_path" ]; then
    echo "Error: Model checkpoint file not found at path: $model_path"
    exit 1
fi

model_step=$(basename "$model_path")
model_step_no_ext="${model_step%.*}"
checkpoint_path=$(dirname "$model_path")
exp_path=$(dirname "$checkpoint_path")
exp_seed=$(basename "$exp_path")
exp_name=$(basename "$(dirname "$exp_path")")


print_with_border() {
    local message="$1"
    local border=$(printf '%*s' "${#message}" '' | tr ' ' '=')
    echo
    echo "+=$border=+"
    echo "| $message |"
    echo "+=$border=+"
}

print_with_border "SETTINGS"
echo "Model checkpoint path: $model_path"
echo "Experiment name: $exp_name"
echo "Experiment seed: $exp_seed"
echo "Model step: $model_step_no_ext"
echo "Save video: $save_video"
if [ -n "$eval_learned_return_model" ]; then
    echo Model directory for learned return evaluation: $eval_learned_return_model
else
    echo "Not evaluating learned return model. To evaluate, rerun with --eval_learned_return_model <dir> flag."
fi



# Check if evaluation results already exist for this model, if not run evaluation
print_with_border "EVALUATION"

eval_path="${exp_path}/eval_results/${model_step_no_ext}"

if [ -d "$eval_path" ]; then
    echo "Evaluation results for model: $exp_name - $exp_seed - $model_step_no_ext, already exist at: $eval_path"
else
    echo "Evaluation results for model: $exp_name - $exp_seed - $model_step_no_ext, not found. Running policy evaluation with default settings..."
    python scripts/eval_policy.py \
        "$model_path"
fi



# Replay trajectories for the eval trajs
print_with_border "TRAJECTORY REPLAY"

save_folder_name=${exp_name}_${exp_seed}_${model_step_no_ext}

input_traj_h5="${exp_path}/eval_results/${model_step_no_ext}/trajectories.h5"
output_path="../data/inest-maniskill/experiment_data-trajs/${save_folder_name}"
output_traj_dir="${output_path}/trajs"

python inest_irl/maniskill3/replay_trajectory.py \
    --traj-path ${input_traj_h5} \
    --save-traj \
    --obs-mode rgb \
    --output-path ${output_traj_dir} \
    --allow-failure \
    --subtask-json \
    --num-envs 10



# Create dataset from replayed trajectories
print_with_border "DATASET CREATION"

output_traj_h5="${output_traj_dir}/trajectories.rgb.pd_ee_delta_pose.physx_cpu.h5"
output_dataset_dir="${output_path}/dataset"

python inest_irl/dataset_utils/h5_to_dataset.py \
    --h5_path ${output_traj_h5} \
    --dataset_path ${output_dataset_dir} \
    --config inest_irl/dataset_utils/configs_h5_to_dataset/sb3-sac_trajs.yaml



# Optionally save videos of the replayed trajectories
if [ "$save_video" = true ]; then
    echo "Saving videos of replayed trajectories..."

    python inest_irl/dataset_utils/h5_analyzer.py \
        ${output_traj_h5} \
        --output_path ${output_traj_dir}/videos
else
    echo "Not saving videos of trajectories. To save videos, rerun the script with the --save_video flag."
fi



# Optionally evaluate learned return model on the replayed trajectories
if [ -n "$eval_learned_return_model" ]; then
    print_with_border "EVALUATING LEARNED RETURNS"

    python inest_irl/utils/compute_learned_return.py \
        --experiment_path "$eval_learned_return_model" \
        --output_dir "${output_path}/learned_return_evaluation" \
        --diff_trajs_dataset ${output_dataset_dir} \
        --plot_subgoal_dists
else
    echo "Not evaluating learned returns. To evaluate returns, rerun the script with the --eval_learned_return_model <dir> flag."
fi
    