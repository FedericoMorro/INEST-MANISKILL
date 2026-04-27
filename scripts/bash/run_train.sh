#!/bin/bash

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Usage: $0 <experiment_name> [multi-env]"
    exit 1  
fi

if [ "$2" == "multi-env" ]; then
    echo "Running with multi-env configuration"
    TMPDIR=/home/fmorro/tmp MPLBACKEND=Agg python /home/fmorro/INEST-MANISKILL/scripts/train_policy_vect-env.py --experiment_name $1 --wandb
else
    echo "Running with single-env configuration"
    TMPDIR=/home/fmorro/tmp MPLBACKEND=Agg python /home/fmorro/INEST-MANISKILL/scripts/train_policy.py --experiment_name $1 --wandb
fi