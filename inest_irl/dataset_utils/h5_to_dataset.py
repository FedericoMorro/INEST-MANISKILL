"""
Converts h5 files to a dataset to be used for training and validation.
Configuration is loaded from a yaml file (default: configs_h5_to_dataset/maniskill_demos.yaml)

Sample h5 file structure assumed:
  traj_0:
    Keys: ['obs', 'actions', 'terminated', 'truncated', 'success', 'env_states']
      obs: Group with keys ['agent', 'extra', 'sensor_param', 'sensor_data']
        agent: Group with keys ['qpos', 'qvel']
          qpos: shape=(177, 9), dtype=float32
          qvel: shape=(177, 9), dtype=float32
        extra: Group with keys ['tcp_pose']
          tcp_pose: shape=(177, 7), dtype=float32
        sensor_param: Group with keys ['base_camera', 'hand_camera']
          base_camera: Group with keys ['extrinsic_cv', 'cam2world_gl', 'intrinsic_cv']
            extrinsic_cv: shape=(177, 3, 4), dtype=float32
            cam2world_gl: shape=(177, 4, 4), dtype=float32
            intrinsic_cv: shape=(177, 3, 3), dtype=float32
          hand_camera: Group with keys ['extrinsic_cv', 'cam2world_gl', 'intrinsic_cv']
            extrinsic_cv: shape=(177, 3, 4), dtype=float32
            cam2world_gl: shape=(177, 4, 4), dtype=float32
            intrinsic_cv: shape=(177, 3, 3), dtype=float32
        sensor_data: Group with keys ['base_camera', 'hand_camera']
          base_camera: Group with keys ['rgb']
            rgb: shape=(177, 128, 128, 3), dtype=uint8
          hand_camera: Group with keys ['rgb']
            rgb: shape=(177, 128, 128, 3), dtype=uint8
      actions: shape=(176, 8), dtype=float32
      terminated: shape=(176,), dtype=bool
      truncated: shape=(176,), dtype=bool
      success: shape=(176,), dtype=bool
      env_states: Group with keys ['actors', 'articulations']
        actors: Group with keys ['table-workspace', 'cubeA', 'cubeB', 'cubeC']
          table-workspace: shape=(177, 13), dtype=float32
          cubeA: shape=(177, 13), dtype=float32
          cubeB: shape=(177, 13), dtype=float32
          cubeC: shape=(177, 13), dtype=float32
        articulations: Group with keys ['panda_wristcam']
          panda_wristcam: shape=(177, 31), dtype=float32
  traj_1: ...

The structure of the dataset is as follows:
  dataset/
    subgoal_frames.json               (if found in the same folder as the h5 file)
    train/
    valid/
      name/                           (by default, name of obs_key)
        traj-idx(s)/
          frame-idx(s).png
          traj-idx_element(s).json -> list of lists
"""

"""
Example usage:

python inest_irl/dataset_utils/h5_to_dataset.py
    --h5_path ../data/maniskill/StackPyramid-v1_data.../trajectory...h5
    --dataset_path ../data/inest-maniskill/dataset...

# for negative trajs
python inest_irl/dataset_utils/h5_to_dataset.py
    --h5_path ../data/inest-maniskill/experiments_data-trajs/tajs_.../trajectory...h5
    --dataset_path ../data/inest-maniskill/experiments_data-trajs/dataset...
    --config inest_irl/dataset_utils/configs_h5_to_dataset/sb3-sac_trajs.yaml
"""


import argparse
import h5py
import json
import numpy as np
import os
import shutil
import yaml
from tqdm import tqdm
from PIL import Image


def _access_nested_group(group, key):
  keys = key.split('/')
  for k in keys:
    group = group[k]
  return group


def _save_data_as_json(group, keys, key_type, path, traj_idx):
  if keys is None:
    return
  
  for key in keys:
    data = _access_nested_group(group, key)
    data_list = np.array(data).tolist()  # convert to list for json serialization
    key_prefix = f'{key_type}-' if key_type else ''
    json_path = os.path.join(path, f'{traj_idx}_{key_prefix}{key.split("/")[-1]}.json')
    with open(json_path, 'w') as f:
      json.dump(data_list, f)


def handle_traj(group, path, idx, obs_key, action_keys, robot_state_keys, objects_state_keys):
  # save a frame-idx.png for each frame in the trajectory
  if obs_key is not None:
    try:
      obs = _access_nested_group(group, obs_key)
      for i in range(obs.shape[0]):
        frame_path = os.path.join(path, f'{i}.png')
        Image.fromarray(obs[i]).save(frame_path)
    except KeyError:
      # Handle case where obs_key doesn't exist in this trajectory
      pass
    
  # save a traj-idx_element.json for each trajectory element
  _save_data_as_json(group, action_keys, None, path, idx)
  _save_data_as_json(group, robot_state_keys, 'robot', path, idx)
  _save_data_as_json(group, objects_state_keys, 'objects', path, idx)



def main(args):
  # set random seed for reproducibility
  np.random.seed(args.random_seed)

  # load configuration from yaml file
  with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
  
  obs_key = config['obs_key']
  action_keys = config['action_keys']
  robot_state_keys = config['robot_state_keys']
  objects_state_keys = config['objects_state_keys']

  # handle dataset path
  if args.dataset_path is None:
    args.dataset_path = args.h5_path.replace('.h5', '_dataset')
  os.makedirs(args.dataset_path, exist_ok=True)
  print(f'Dataset will be saved to: {args.dataset_path}')

  # create needed nesting
  if obs_key is not None:
    name = f'{obs_key.split("/")[-2]}-{obs_key.split("/")[-1]}'
  else:
    name = 'data'

  train_path = os.path.join(args.dataset_path, 'train')
  train_path = os.path.join(train_path, name)
  os.makedirs(train_path, exist_ok=True)

  valid_path = os.path.join(args.dataset_path, 'valid')
  valid_path = os.path.join(valid_path, name)
  os.makedirs(valid_path, exist_ok=True)

  # open and read the h5 file
  h5_file = h5py.File(args.h5_path, 'r')
  traj_names = list(h5_file.keys())
  print(f'Found {len(traj_names)} trajectories in the h5 file.')
    
  for traj_name in tqdm(h5_file, desc='Processing trajectories'):
    traj_group = h5_file[traj_name]

    # create trajectory folder
    traj_idx = traj_name.split('_')[-1]
    if np.random.rand() < config['train_split']:
      traj_path = os.path.join(train_path, traj_idx)
    else:
      traj_path = os.path.join(valid_path, traj_idx)
    os.makedirs(traj_path, exist_ok=True)

    # handle trajectory data
    handle_traj(traj_group, traj_path, traj_idx, obs_key, action_keys, robot_state_keys, objects_state_keys)

  h5_file.close()

  # if find subgoal_frames.json file, copy it in the dataset folder
  subgoals_path = os.path.dirname(args.h5_path) + '/subgoal_frames.json'
  if os.path.exists(subgoals_path):
    shutil.copy(subgoals_path, args.dataset_path)
    print(f'Found subgoal_frames.json file, copied to dataset folder')
  else:
    print(f'No subgoal_frames.json file found, skipping copy')


if __name__ == '__main__':
  arg_pars = argparse.ArgumentParser()
  arg_pars.add_argument('--h5_path', type=str, required=True,
                        help='Path to the h5 file')
  arg_pars.add_argument('--dataset_path', type=str, default=None,
                        help='Path to the dataset folder (if not specified, it will be the saame as the h5 file)')
  arg_pars.add_argument('--config', type=str, 
                        default=os.path.join(os.path.dirname(__file__), 'configs_h5_to_dataset', 'maniskill_demos.yaml'),
                        help='Path to the configuration yaml file')
  arg_pars.add_argument('--random_seed', type=int, default=22,
                        help='Random seed for reproducibility')
  args = arg_pars.parse_args()

  main(args)