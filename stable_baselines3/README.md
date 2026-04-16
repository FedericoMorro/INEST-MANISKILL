# Copied from [DLR-RM/stable-baselines3 - Release v2.8.0](https://github.com/DLR-RM/stable-baselines3/commit/5e00d26428f8633141b841a51be473971e2a6ea4)

## Modifications
1. Added support for target entropy annealing in SAC.
2. Added support for returning episode subgoals during evaluation.
3. Log rollout metrics to W&B.

## Edited Files
- `sac/sac.py`: 1
- `common/evaluation.py`: 2
- `common/off_policy_algorithm.py`: 3