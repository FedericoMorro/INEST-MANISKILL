#!/bin/bash

python /home/fmorro/INEST-MANISKILL/inest_irl/maniskill3/replay_trajectory.py \
    --traj-path $1 \
    --output_path $2 \
    --obs-mode rgb \
    --save-traj \
    --count 1000 \
    --num-envs 10 \
    --cam-width 128 \
    --cam-height 128 \
    --subtask-json