#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

if [ "$#" -eq 2 ]; then
    echo "Setting seed to $2"
    seed=$2
else
    echo "Using default seed of 22. To specify a different seed, run: $0 <experiment_name> <seed>"
    seed=22
fi

TMPDIR=/home/fmorro/tmp MPLBACKEND=Agg python /home/fmorro/INEST-MANISKILL/scripts/pretrain.py \
    --experiment_name $1 \
    --seed $seed \
    --wandb