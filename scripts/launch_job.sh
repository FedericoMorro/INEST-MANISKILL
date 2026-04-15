#!/bin/bash

QUEUE_TIME=$(date +%Y.%m.%d-%H.%M.%S)
export QUEUE_TIME

mkdir -p ${HOME}/logs/${QUEUE_TIME}

if [ $# -ne 2 ]; then
    echo "Usage: $0 <script_name> <experiment_name>"
    echo "Example: $0 inest 32_no-rng_v##"
    exit 1
fi

export EXPERIMENT_NAME="$2"
export RND_SEED="22"
export REPLAY_BUFFER_CAPACITY="1_000_000"
export ACTOR_LR="3e-4"
export CRITIC_LR="1e-4"
export ALPHA_LR="3e-4"
export TARGET_ENTROPY="-3.5"
export STD_ACTION_NOISE="0.0"
export ANNEAL_TARGET_ENTROPY="False"

if [[ "$1" == "inest" ]]; then
    echo "Submitting INEST MANISKILL RL training job..."

    EXPERIMENT_DIR="${HOME}/INEST-MANISKILL/experiments/sb3/${EXPERIMENT_NAME}/${RND_SEED}"
    if [ -d $EXPERIMENT_DIR ]; then
        echo "WARNING: Directory ${EXPERIMENT_DIR} already exists!"
        echo "If you do NOT wish to resume a previous run, delete the existing directory or choose a different experiment name or seed."
        read -p "Do you want to continue (c), delete the existing directory (d), or exit (e)? [c/d/e]: " choice
        case "$choice" in
            c|C ) echo "Continuing with existing directory...";;
            d|D ) rm -rf $EXPERIMENT_DIR; echo "Deleted existing directory. Continuing...";;
            e|E ) echo "Exiting..."; exit 0;;
            * ) echo "Invalid choice. Exiting..."; exit 1;;
        esac
    fi

    sbatch --job-name=inest_maniskill_rl_training \
        --ntasks-per-node=1 \
        --cpus-per-task=36 \
        --mem=50GB \
        --mail-type=ALL \
        --mail-user=federico.morro@polito.it \
        --partition=gpu_a40 \
        --gres=gpu:1 \
        --output=${HOME}/logs/${QUEUE_TIME}/inest_maniskill_rl_training_%j.out \
        --error=${HOME}/logs/${QUEUE_TIME}/inest_maniskill_rl_training_%j.err \
        /home/fmorro/INEST-MANISKILL/scripts/submit_inest_train.sh
else
    echo "Unknown argument: $1. Please specify 'inest' to submit the INEST MANISKILL RL training job."
fi