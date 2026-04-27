#!/bin/bash

echo "====================================="
echo "TRAINING HYPERPARAMETERS"
echo " Experiment Name: $EXPERIMENT_NAME"
echo " Dataset Path: $SCRATCH_FLASH_DATASET_PATH"
echo " Random Seed: $RND_SEED"
echo " Batch Size: $BATCH_SIZE"
echo " Train Max Iters: $TRAIN_MAX_ITERS"
echo " Num Frames per Sequence: $NUM_FRAMES_PER_SEQUENCE"
echo "====================================="

# Load necessary modules (adjust based on your cluster)
module purge
# module load cuda/11.8
# module load python/3.8
# module load anaconda3

# Activate your conda environment (adjust path and env name)
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate inest

# Change to the project directory
cd ${HOME}/INEST-MANISKILL

# Set environment variables for optimal performance
if [[ "${PARTITION:-gpu_a40}" == "fair_gpu" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
elif [[ "${PARTITION:-gpu_a40}" == "gpu_a40_ext" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
elif [[ "${PARTITION:-gpu_a40}" == "gpu_a40" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26
else
    # Default to 2 GPUs for unknown partitions
    export CUDA_VISIBLE_DEVICES=0,1
fi

# Performance optimization environment variables
export WANDB_MODE=online
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0

# Memory and performance optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_MEMORY_FRACTION=0.95

# Disable unnecessary logging for better performance
export TF_CPP_MIN_LOG_LEVEL=2

# Additional performance optimizations
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH=${HOME}/tmp/${SLURM_JOB_ID}/cuda_cache
export TORCH_USE_CUDA_DSA=1

# Fix checkpoint saving cross-device link issue
export TMPDIR=${HOME}/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR

# Fix NFS multiprocessing issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MPLCONFIGDIR=$TMPDIR
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Print job information
echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Working Directory: $(pwd)"
echo "=============================="

# Check GPU availability and memory
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo "======================"

python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA version: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

export MPLBACKEND=Agg

CMD="python /home/fmorro/INEST-MANISKILL/scripts/pretrain.py \
    --experiment_name ${EXPERIMENT_NAME} \
    --seed ${RND_SEED} \
    --wandb \
    --config.data.root=${SCRATCH_FLASH_DATASET_PATH} \
    --config.data.batch_size=${BATCH_SIZE} \
    --config.optim.train_max_iters=${TRAIN_MAX_ITERS} \
    --config.frame_sampler.num_frames_per_sequence=${NUM_FRAMES_PER_SEQUENCE}"

# Execute the command
echo "Executing: $CMD"
eval $CMD

echo "INEST RL training completed!"
echo "Job finished at: $(date)"

