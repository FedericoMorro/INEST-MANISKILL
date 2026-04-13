#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <suffix_for_output_folder>"
    exit 1
fi

if [ "$#" -eq 2 ] && [ "$2" == "not-all" ]; then
    echo "Running with --no_plot_all_rew: reward curves for individual demos will not be plotted, only summary stats and mean reward curve will be computed."
    PLOT_FLAG="--no_plot_all_rew"
else
    PLOT_FLAG=""
fi

OUT_DIR="/data/fmorro/maniskill/StackPyramid-v1_$1"

echo "The re-rolled demos will be saved to: $OUT_DIR"
    

echo "+=====================================================+"
echo "| Replaying trajectory to extract environment rewards |"
echo "+=====================================================+"

python /home/fmorro/INEST-MANISKILL/inest_irl/maniskill3/replay_trajectory.py \
    --traj-path /home/fmorro/.maniskill/demos/StackPyramid-v1/motionplanning/trajectory.h5 \
    --save-traj \
    --obs-mode state_dict \
    --output-path $OUT_DIR \
    --record-rewards \
    --count 100 \
    --num-envs 10 \
    --subtask-json

echo "+========================================================+"
echo "| Analyzing trajectory to plot rewards and compute stats |"
echo "+========================================================+"

python /home/fmorro/INEST-MANISKILL/inest_irl/dataset_utils/h5_analyzer.py \
	$OUT_DIR/trajectory.state_dict.pd_joint_pos.physx_cpu.h5 \
	--rewards $PLOT_FLAG