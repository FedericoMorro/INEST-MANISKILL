#!/bin/bash

python ~/INEST-MANISKILL/maniskill3/replay_trajectory.py \
    --traj-path $1 \
    --output_path $2 \
    --obs-mode rgb \
    --save-traj \
    --count 100 \
    --num-envs 10 \
    --cam-width 128 \
    --cam-height 128 \
    --subtask-json