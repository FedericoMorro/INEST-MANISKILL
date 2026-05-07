#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_path> [--count N] [--lr_model_path lr_model_path] [--lr_data_dir lr_data_dir] [--overwrite] [--eval_learned_return_model <dir>[,<data_dir>]]"
    exit 1
fi

# parse arguments
model_path="$1"
count=100
overwrite=false
lr_data_dir="None"
lr_model_path="None"
eval_learned_return_model=""
eval_learned_return_data_dir=""

# parse optional flags
shift 1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --overwrite)
            overwrite=true
            shift
            ;;
        --count)
            if [ -z "$2" ]; then
                echo "Error: --count flag requires a numeric argument"
                exit 1
            fi
            count="$2"
            shift 2
            ;;
        --lr_model_path)
            if [ -z "$2" ]; then
                echo "Error: --lr_model_path requires a path argument"
                exit 1
            fi
            lr_model_path="$2"
            shift 2
            ;;
        --lr_data_dir)
            if [ -z "$2" ]; then
                echo "Error: --lr_data_dir requires a directory argument"
                exit 1
            fi
            lr_data_dir="$2"
            shift 2
            ;;
        --eval_learned_return_model)
            if [ -z "$2" ]; then
                echo "Error: --eval_learned_return_model requires a directory argument"
                exit 1
            fi
            IFS=',' read -r eval_learned_return_model eval_learned_return_data_dir <<< "$2"

            shift 2
            ;;
        *)
            echo "Unknown flag: $1"
            echo "Usage: $0 <model_path> [--lr_model_path <lr_model_path>] [--lr_data_dir <lr_data_dir>] [--overwrite] [--eval_learned_return_model <dir>[,<data_dir>]]"
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

eval_path="${exp_path}/out_eval-policy-py/${model_step_no_ext}"

if [[ -d "$eval_path" && "$overwrite" == false ]]; then
    echo "Evaluation results for model: $exp_name - $exp_seed - $model_step_no_ext, already exist at: $eval_path"
else
    echo "Evaluation results for model: $exp_name - $exp_seed - $model_step_no_ext, not found. Running policy evaluation with default settings..."
    echo "Custom Learned Reward Model Path specified: $lr_model_path"
    echo "Custom Learned Reward Data Dir specified: $lr_data_dir"

    python scripts/eval_policy.py \
        "$model_path" \
        --save_rgb \
        --num_episodes "$count" \
        --learned_reward_model_path "$lr_model_path" \
        --learned_reward_data_dir "$lr_data_dir"
fi



# Create dataset directly from evaluation trajectories (already contain RGB observations)
print_with_border "DATASET CREATION"

save_folder_name="${exp_name}.${exp_seed}.${model_step_no_ext}"
input_traj_h5="${eval_path}/trajectories.h5"
output_path="../data/inest-maniskill/experiment_data-trajs/${save_folder_name}"
output_dataset_dir="${output_path}/dataset"

python inest_irl/dataset_utils/h5_to_dataset.py \
    --h5_path ${input_traj_h5} \
    --dataset_path ${output_dataset_dir} \
    --config inest_irl/dataset_utils/configs_h5_to_dataset/sb3-sac_trajs.yaml



# Optionally evaluate learned return model on the trajectories dataset
if [ -n "$eval_learned_return_model" ]; then
    print_with_border "EVALUATING LEARNED RETURNS"

    if [ -z "$eval_learned_return_data_dir" ]; then
        python inest_irl/utils/compute_learned_return.py \
            --experiment_path "$eval_learned_return_model" \
            --output_dir "${output_path}/learned_return_evaluation" \
            --diff_trajs_dataset "$eval_learned_return_data_dir" \
            --plot_subgoal_dists
    else
        python inest_irl/utils/compute_learned_return.py \
            --experiment_path "$eval_learned_return_model" \
            --output_dir "${output_path}/learned_return_evaluation" \
            --diff_trajs_dataset "$eval_learned_return_data_dir" \
            --data_root "$eval_learned_return_data_dir" \
            --plot_subgoal_dists
    fi
else
    echo "Not evaluating learned returns. To evaluate returns, rerun the script with the --eval_learned_return_model <dir> flag."
fi
    