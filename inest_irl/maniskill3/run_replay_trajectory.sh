#!/bin/bash

python /home/fmorro/INEST-MANISKILL/inest_irl/maniskill3/replay_trajectory.py \
    --traj-path $1 \
    --save-traj \
    --obs-mode rgb+state_dict \
    --output_path $2 \
    --use-env-states \
    --count 1000 \
    --num-envs 10 \
    --render-camera base_camera \
    --cam-width 128 \
    --cam-height 128