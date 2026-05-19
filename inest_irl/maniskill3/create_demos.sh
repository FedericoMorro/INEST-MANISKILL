#!/bin/bash

DATA_BASE_DIR="../data"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <env_randomization> <suffix> [--count N] [--demos-h5 H5_PATH] [--create-dataset] [--create-videos <cam0,cam1,...>]"
    exit 1
fi

# parse arguments
env_randomization="$1"
suffix="$2"
count=100
demos_h5=""
create_dataset=false
create_videos=""

# parse optional flags
shift 2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --count)
            if [ -z "$2" ]; then
                echo "Error: --count flag requires a numeric argument"
                exit 1
            fi
            count="$2"
            shift 2
            ;;
        --demos-h5)
            if [ -z "$2" ]; then
                echo "Error: --demos-h5 flag requires a path argument"
                exit 1
            fi
            demos_h5="$2"
            shift 2
            ;;
        --create-dataset)
            create_dataset=true
            shift
            ;;
        --create-videos)
            if [ -z "$2" ]; then
                echo "Error: --create-videos flag requires a comma-separated list of camera names"
                exit 1
            fi
            create_videos="$2"
            shift 2
            ;;
        *)
            echo "Unknown flag: $1"
            echo "Usage: $0 <model_path> [--lr_model_path <lr_model_path>] [--lr_data_dir <lr_data_dir>] [--overwrite] [--eval_learned_return_model <dir>[,<data_dir>]]"
            exit 1
            ;;
    esac
done


# generate motion planning trajectories if not provided
if [[ -z "$demos_h5" ]]; then
    python inest_irl/maniskill3/panda_motionplanner.py \
        --num-traj $count \
        --only-count-success \
        --traj-name trajectory \
        --record-dir ../.maniskill/demos_${suffix} \
        --num-procs 1 \
        --env-randomization $env_randomization

    demos_h5="../.maniskill/demos_${suffix}/StackPyramid-v1custom/motionplanning/trajectory.h5"
fi


# replay trajectories to generate RGB observations and save new h5 file
replay_output_dir="${DATA_BASE_DIR}/maniskill/StackPyramid-v1_data-${suffix}"

python inest_irl/maniskill3/replay_trajectory.py \
    --traj-path $demos_h5 \
    --save-traj \
    --obs-mode rgb+state_dict \
    --output_path $replay_output_dir \
    --use-env-states \
    --record-rewards \
    --num-envs 10 \
    --base-camera base_camera \
    --cam-width 128 \
    --cam-height 128
    #--count $count \

replay_traj_h5="${replay_output_dir}/trajectory.rgb+state_dict.pd_joint_pos.physx_cpu.h5"


# create dataset from replayed trajectories
dataset_output_dir="${DATA_BASE_DIR}/inest-maniskill/datasets/dataset-${suffix}"

if [[ "$create_dataset" == true ]]; then
    if [ -d "$dataset_output_dir" ]; then
        echo "Dataset output directory $dataset_output_dir already exists. Overriding it..."
        rm -rf "$dataset_output_dir"
    fi

    python inest_irl/dataset_utils/h5_to_dataset.py \
        --h5_path ${replay_traj_h5} \
        --dataset_path ${dataset_output_dir} \
        --config inest_irl/dataset_utils/configs_h5_to_dataset/maniskill_demos_merged.yaml
fi


# create videos from replayed trajectories
if [ -n "$create_videos" ]; then
    OLD_IFS="$IFS"
    IFS=','
    set -- $create_videos
    IFS="$OLD_IFS"

    for cam in "$@"; do
        video_output_dir="${DATA_BASE_DIR}/inest-maniskill/videos/video-${suffix}_${cam}"

        python inest_irl/dataset_utils/h5_analyzer.py \
            ${replay_traj_h5} \
            --vis ${cam} \
            --output_path ${video_output_dir} \
            --subgoals ${dataset_output_dir}/subgoal_frames.json

        python inest_irl/viz/merge_videos.py \
            --video_dir ${video_output_dir} \
            --output_name ${suffix}_${cam}
    done
fi