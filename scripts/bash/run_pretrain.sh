#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [flags...]"
    exit 1
fi

if [ $1 == "test" ]; then
    echo "Running test run -> removing existing experiment artifacts"
    rm -r /home/fmorro/INEST-MANISKILL/experiments/pretrain/test
fi

TMPDIR=/home/fmorro/tmp MPLBACKEND=Agg python /home/fmorro/INEST-MANISKILL/scripts/pretrain.py \
    --experiment_name "$1" \
    "${@:2}"