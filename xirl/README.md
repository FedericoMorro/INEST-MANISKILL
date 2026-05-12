# Copied from [google-research/google-research/xirl](https://github.com/google-research/google-research/tree/master/xirl)

## Modifications
1. Eval output embeddings returned by `eval_manager.evaluate()` to compute learned reward evaluation metrics.
2. Problem with NFS data management, when data loaders use `num_workers > 0`.
3. Enable multi-camera support: different classes interpreted as different camera views of the same trajectory, consequently modify dataset handling and loading, and create model with two separate encoders and a shared fusion module.

## Edited Files
- `evaluators/manager.py`: 1
    - `evaluate()`: 1
- `common.py`: 2, 3
  - `get_pretraining_dataloaders()`: 2, 3
  - `get_downstream_dataloaders()`: 2, 3
  - `get_factories()`: 3
  - `get_model()`: 3
- `multiple_cameras.py` (created): 3
- `dataset.py`: 3
  - `VideoDataset.__init__()`: 3
- `factory.py`: 3
  - `dataset_from_config()`: 3