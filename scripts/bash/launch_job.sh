#!/bin/bash

export SLURM_EXCLUDE_NODES=compute-3-11,compute-5-11,compute-5-14
# compute-3-11: [date not known] seems to have a faulty GPU
# compute-5-11,14: [18/05/2026] does not see $SCRATCH data partition

QUEUE_TIME=$(date +%Y.%m.%d-%H.%M.%S)
TRAIN_DATA_PARTITION=$SCRATCH
TRAIN_NODE_PARTITION=gpu_a40

mkdir -p ${HOME}/logs

if [ $# -ne 2 ]; then
    echo "Usage: $0 <script_name> <experiment_name>"
    echo "Where: <script_name> in [rl, pretrain, opt-pretrain]"
    echo "Example: $0 rl 32_no-rng_v##"
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

manage_scratch_flash_folder() {
    TRAIN_DATASET_PATH="$1"
    DATASET_PATH="$2"
    EXPERIMENT_NAME="$3"

    TRAIN_DATASET_PATH="${TRAIN_DATA_PARTITION}/$(basename ${DATASET_PATH})"
    if [ -d $TRAIN_DATASET_PATH ]; then
        echo "Directory ${TRAIN_DATASET_PATH} already exists on TRAIN_DATA_PARTITION."
        read -p "Do you want to perform copy anyway (y/N)? [y/N]: " choice
        case "$choice" in
            y|Y ) echo "Performing copy...";;
            * ) echo "Skipping copy..."; return;;
        esac
    fi

    echo "Copying dataset to $TRAIN_DATA_PARTITION for faster access during training..."
    TO_BE_COPIED=$(rsync -avh --dry-run ${DATASET_PATH} ${TRAIN_DATA_PARTITION}/ | grep -v "/$" | wc -l)
    #echo "To track the copy progress, run the following command in another terminal:"
    #echo "cat ${HOME}/logs/rsync_${EXPERIMENT_NAME}_pretrain_${QUEUE_TIME}.log | grep -v '/$' | wc -l | awk -v total=$TO_BE_COPIED '{printf(\"%.2f%%\\n\", (\$1/total)*100)}'"
    #echo "This may take a while..."
    #rsync -avzh ${DATASET_PATH} ${TRAIN_DATA_PARTITION}/ > ${HOME}/logs/rsync_${EXPERIMENT_NAME}_pretrain_${QUEUE_TIME}.log 2>&1

    CHECKPOINT_NUM_FILES=$(TO_BE_COPIED / 10)   # check every 10%

    echo "Compressing input dataset for faster copying..."
    tar -czf --checkpoint=${CHECKPOINT_NUM_FILES} ${DATASET_PATH}.tar.gz -C $(dirname ${DATASET_PATH}) $(basename ${DATASET_PATH})
    echo "Copying compressed dataset to TRAIN_DATA_PARTITION..."
    rsync -avzh --progress ${DATASET_PATH}.tar.gz ${TRAIN_DATASET_PATH}.tar.gz
    echo "Extracting dataset on TRAIN_DATA_PARTITION..."
    tar -xzf --checkpoint=${CHECKPOINT_NUM_FILES} ${TRAIN_DATASET_PATH}.tar.gz -C $(dirname ${TRAIN_DATASET_PATH})
    echo "If you want to clean up the compressed files, run the following commands:"
    echo "rm ${DATASET_PATH}.tar.gz"
    echo "rm ${TRAIN_DATASET_PATH}.tar.gz"
}


if [[ "$1" == "rl" ]]; then
    echo "Submitting INEST MANISKILL RL training job..."

    export REWARD_WRAPPER_TYPE="goal_dist_subgoals_flat"
    export REWARD_SCALING="1.0"
    export ENV_RANDOMIZATION="minimal"
    export REPLAY_BUFFER_CAPACITY="1_000_000"
    export ACTOR_LR="3e-4"
    export CRITIC_LR="1e-4"
    export ALPHA_LR="3e-4"
    export DISCOUNT="0.95"
    export TARGET_ENTROPY="-3.5"
    export STD_ACTION_NOISE="0.0"
    export ANNEAL_TARGET_ENTROPY="False"
    export REWARD_WRAPPER_PRETRAINED_PATH="/home/fmorro/INEST-MANISKILL/experiments/pretrain/min_mc_b2_frM_vis"

    manage_existing_exp_folder "${HOME}/INEST-MANISKILL/experiments/sb3/${EXPERIMENT_NAME}/${RND_SEED}"
    manage_existing_exp_folder "${HOME}/INEST-MANISKILL/experiments/lr-sb3/${EXPERIMENT_NAME}/${RND_SEED}"

    sbatch --job-name=inest_rl_train \
        --ntasks-per-node=1 \
        --cpus-per-task=36 \
        --mem=64GB \
        --mail-type=ALL \
        --mail-user=federico.morro@polito.it \
        --partition=${TRAIN_NODE_PARTITION} \
        --gres=gpu:1 \
        --output=${HOME}/logs/%j_${EXPERIMENT_NAME}_inest.out \
        --error=${HOME}/logs/%j_${EXPERIMENT_NAME}_inest.err \
        --exclude=$SLURM_EXCLUDE_NODES \
        /home/fmorro/INEST-MANISKILL/scripts/bash/submit_inest_train.sh


elif [[ "$1" == "pretrain" || "$1" == "opt-pretrain" ]]; then
    echo "Submitting INEST MANISKILL pretraining job..."

    DATASET_PATH="/home/fmorro/data/inest-maniskill/dataset-min-rand_vis"

    # copy dataset to SCRATCH_FLASH for faster access during training (if already present ask user)
    export TRAIN_DATASET_PATH="${TRAIN_DATA_PARTITION}/$(basename ${DATASET_PATH})"
    manage_scratch_flash_folder "${TRAIN_DATASET_PATH}" "${DATASET_PATH}" "${EXPERIMENT_NAME}"
    echo "Dataset path: ${TRAIN_DATASET_PATH}"


    if [[ "$1" == "pretrain" ]]; then

        export BATCH_SIZE="8"
        export TRAIN_MAX_ITERS="10_000"
        export NUM_FRAMES_PER_SEQUENCE="50"

        manage_existing_exp_folder "${HOME}/INEST-MANISKILL/experiments/pretrain/${EXPERIMENT_NAME}"

        sbatch --job-name=inest_pretrain \
            --ntasks-per-node=1 \
            --cpus-per-task=16 \
            --mem=32GB \
            --mail-type=ALL \
            --mail-user=federico.morro@polito.it \
            --partition=${TRAIN_NODE_PARTITION} \
            --gres=gpu:1 \
            --output=${HOME}/logs/%j_${EXPERIMENT_NAME}_pretrain.out \
            --error=${HOME}/logs/%j_${EXPERIMENT_NAME}_pretrain.err \
            --exclude=$SLURM_EXCLUDE_NODES \
            /home/fmorro/INEST-MANISKILL/scripts/bash/submit_pretrain.sh


    else # opt-pretrain
        echo "Submitting optuna hyperparameter optimization job..."

        manage_existing_exp_folder "${HOME}/INEST-MANISKILL/experiments/opt_pretrain/${EXPERIMENT_NAME}"

        sbatch --job-name=inest_opt-pretrain \
            --ntasks-per-node=1 \
            --cpus-per-task=24 \
            --mem=48GB \
            --mail-type=ALL \
            --mail-user=federico.morro@polito.it \
            --partition=${TRAIN_NODE_PARTITION} \
            --gres=gpu:1 \
            --output=${HOME}/logs/%j_${EXPERIMENT_NAME}_opt-pretrain.out \
            --error=${HOME}/logs/%j_${EXPERIMENT_NAME}_opt-pretrain.err \
            --exclude=$SLURM_EXCLUDE_NODES \
            /home/fmorro/INEST-MANISKILL/scripts/bash/submit_opt-pretrain.sh

    fi
    
else
    echo "Unknown argument: $1"
fi