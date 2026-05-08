#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [flags...]"
    exit 1
fi

if [ $1 == "test" ]; then
    echo "Running test run -> removing existing experiment artifacts"
    if [ -d "/home/fmorro/INEST-MANISKILL/experiments/sb3/test" ]; then
        rm -r /home/fmorro/INEST-MANISKILL/experiments/sb3/test
    elif [ -d "/home/fmorro/INEST-MANISKILL/experiments/lr-sb3/test" ]; then
        rm -r /home/fmorro/INEST-MANISKILL/experiments/lr-sb3/test
    fi
fi

TMPDIR=/home/fmorro/tmp MPLBACKEND=Agg python /home/fmorro/INEST-MANISKILL/scripts/sb3_sac.py \
    --experiment_name "$1" \
    "${@:2}"