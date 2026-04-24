#!/bin/bash

export SLURM_EXCLUDE_NODES=compute-3-11

QUEUE_TIME=$(date +%Y.%m.%d-%H.%M.%S)

mkdir -p ${HOME}/logs

if [ $# -ne 2 ]; then
    echo "Usage: $0 <script_name> <experiment_name>"
    echo "Where: <script_name> in [pretrain, inest]"
    echo "Example: $0 inest 32_no-rng_v##"
    exit 1
fi


export EXPERIMENT_NAME="$2"
export RND_SEED="22"


manage_existing_exp_folder() {
    EXP_DIR="$1"
    if [ -d $EXP_DIR ]; then
        echo "WARNING: Directory ${EXP_DIR} already exists!"
        echo "If you do NOT wish to resume a previous run, delete the existing directory or choose a different experiment name or seed."
        read -p "Do you want to continue (c), delete the existing directory (d), or exit (e)? [c/d/e]: " choice
        case "$choice" in
            c|C ) echo "Continuing with existing directory...";;
            d|D ) rm -rf $EXP_DIR; echo "Deleted existing directory. Continuing...";;
            e|E ) echo "Exiting..."; exit 0;;
            * ) echo "Invalid choice. Exiting..."; exit 1;;
        esac
    fi
}


if [[ "$1" == "inest" ]]; then
    echo "Submitting INEST MANISKILL RL training job..."

    export REWARD_WRAPPER_TYPE="subgoal_dist"
    export REWARD_SCALING="2.0"
    export ENV_RANDOMIZATION="True"
    export REPLAY_BUFFER_CAPACITY="1_000_000"
    export ACTOR_LR="3e-4"
    export CRITIC_LR="1e-4"
    export ALPHA_LR="3e-4"
    export DISCOUNT="0.9"
    export TARGET_ENTROPY="-3.5"
    export STD_ACTION_NOISE="0.0"
    export ANNEAL_TARGET_ENTROPY="False"
    export REWARD_WRAPPER_PRETRAINED_PATH="/home/fmorro/INEST-MANISKILL/experiments/pretrain/rc1000-b32/"

    manage_existing_exp_folder "${HOME}/INEST-MANISKILL/experiments/sb3/${EXPERIMENT_NAME}/${RND_SEED}"
    manage_existing_exp_folder "${HOME}/INEST-MANISKILL/experiments/sb3-lr/${EXPERIMENT_NAME}/${RND_SEED}"

    sbatch --job-name=inest_rl_train \
        --ntasks-per-node=1 \
        --cpus-per-task=36 \
        --mem=64GB \
        --mail-type=ALL \
        --mail-user=federico.morro@polito.it \
        --partition=gpu_a40 \
        --gres=gpu:1 \
        --output=${HOME}/logs/%j_${EXPERIMENT_NAME}_inest.out \
        --error=${HOME}/logs/%j_${EXPERIMENT_NAME}_inest.err \
        --exclude=$SLURM_EXCLUDE_NODES \
        /home/fmorro/INEST-MANISKILL/scripts/submit_inest_train.sh

elif [[ "$1" == "pretrain" ]]; then
    echo "Submitting INEST MANISKILL pretraining job..."

    DATASET_PATH="/home/fmorro/data/inest-maniskill/dataset-rc-1000"

    # copy dataset to SCRATCH_FLASH for faster access during training
    echo "Copying dataset to SCRATCH_FLASH for faster access during training"
    TO_BE_COPIED=$(rsync -avh --dry-run ${DATASET_PATH} ${SCRATCH_FLASH}/ | grep -v "/$" | wc -l)
    echo "Progress: cat ${HOME}/logs/rsync_${EXPERIMENT_NAME}_pretrain_${QUEUE_TIME}.log | grep -v '/$' | wc -l | awk -v total=$TO_BE_COPIED '{printf("%.2f%%\n", ($1/total)*100)}'"
    echo "This may take a while..."
    rsync -avzh ${DATASET_PATH} ${SCRATCH_FLASH} > ${HOME}/logs/rsync_${EXPERIMENT_NAME}_pretrain_${QUEUE_TIME}.log 2>&1
    export SCRATCH_FLASH_DATASET_PATH="${SCRATCH_FLASH}/$(basename ${DATASET_PATH})"
    echo "Dataset copied to ${SCRATCH_FLASH_DATASET_PATH}"

    export BATCH_SIZE="32"
    export TRAIN_MAX_ITERS="20_000"

    manage_existing_exp_folder "${HOME}/INEST-MANISKILL/experiments/pretrain/${EXPERIMENT_NAME}"

    sbatch --job-name=inest_pretrain \
        --ntasks-per-node=1 \
        --cpus-per-task=16 \
        --mem=32GB \
        --mail-type=ALL \
        --mail-user=federico.morro@polito.it \
        --partition=gpu_a40 \
        --gres=gpu:1 \
        --output=${HOME}/logs/%j_${EXPERIMENT_NAME}_pretrain.out \
        --error=${HOME}/logs/%j_${EXPERIMENT_NAME}_pretrain.err \
        --exclude=$SLURM_EXCLUDE_NODES \
        /home/fmorro/INEST-MANISKILL/scripts/submit_pretrain.sh
    
else
    echo "Unknown argument: $1"
fi