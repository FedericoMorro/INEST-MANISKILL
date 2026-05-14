# Experiments Naming

## Common
- `bc` / `rc` / `hc` / `mc`
  - `bc`: base camera
  - `rc`: render camera
  - `hc`: hand camera
  - `mc`: multi-camera (usually base + hand)
- `min`
  - `minimal randomization` applied to the environment

## Pretraining
- `s1k`
  - `s`: state replay used
  - `1k`: 1000 trajectories used for pretraining
- `fr50` / `maxfr`
  - `config.frame_sampler.num_frames_per_sequence = 50`
  - `config.frame_sampler.max_frames_per_sequence = -1` => use max length of training trajectories
- `b16`
  - `config.batch_size = 16`
