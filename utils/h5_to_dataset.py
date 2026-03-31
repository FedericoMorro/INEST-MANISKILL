"""
Converts h5 files to a dataset to be used for training and validation.
! check global variables to handle keys to be saved

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
        train/
        valid/
            name/                           (by default, name of obs_key)
                traj-idx(s)/
                    frame-idx(s).png
                    traj-idx_element(s).json -> list of lists
"""


import argparse
import h5py
import json
import numpy as np
import os
from tqdm import tqdm
from PIL import Image


OBS_KEY = 'obs/sensor_data/base_camera/rgb'
ACTION_KEYS = ['actions']
ROBOT_STATE_KEYS = ['obs/agent/qpos', 'obs/agent/qvel']
OBJECTS_STATE_KEYS = ['env_states/actors/cubeA', 'env_states/actors/cubeB', 'env_states/actors/cubeC']


def _access_nested_group(group, key):
    keys = key.split('/')
    for k in keys:
        group = group[k]
    return group


def _save_data_as_json(group, keys, key_type, path, traj_idx):
    for key in keys:
        data = _access_nested_group(group, key)
        data_list = np.array(data).tolist()  # convert to list for json serialization
        key_type = f'{key_type}-' if key_type else ''
        json_path = os.path.join(path, f'{traj_idx}_{key_type}{key.split("/")[-1]}.json')
        with open(json_path, 'w') as f:
            json.dump(data_list, f)


def handle_traj(group, path, idx):
    # save a frame-idx.png for each frame in the trajectory
    obs = _access_nested_group(group, OBS_KEY)
    for i in range(obs.shape[0]):
        frame_path = os.path.join(path, f'{i}.png')
        Image.fromarray(obs[i]).save(frame_path)
    
    # save a traj-idx_element.json for each trajectory element
    _save_data_as_json(group, ACTION_KEYS, None, path, idx)
    _save_data_as_json(group, ROBOT_STATE_KEYS, 'robot', path, idx)
    _save_data_as_json(group, OBJECTS_STATE_KEYS, 'objects', path, idx)



def main(args):

    # handle dataset path
    if args.dataset_path is None:
        args.dataset_path = args.h5_path.replace('.h5', '_dataset')
    os.makedirs(args.dataset_path, exist_ok=True)
    print(f'Dataset will be saved to: {args.dataset_path}')

    # create needed nesting
    name = f'{OBS_KEY.split("/")[-2]}-{OBS_KEY.split("/")[-1]}'

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
        if np.random.rand() < args.train_split:
            traj_path = os.path.join(train_path, traj_idx)
        else:
            traj_path = os.path.join(valid_path, traj_idx)
        os.makedirs(traj_path, exist_ok=True)

        # handle trajectory data
        handle_traj(traj_group, traj_path, traj_idx)

    h5_file.close()


if __name__ == '__main__':
    arg_pars = argparse.ArgumentParser()
    arg_pars.add_argument('--h5_path', type=str, required=True,
                          help='Path to the h5 file')
    arg_pars.add_argument('--dataset_path', type=str, default=None,
                          help='Path to the dataset folder (if not specified, it will be the saame as the h5 file)')
    arg_pars.add_argument('--train_split', type=float, default=0.9,
                          help='Percentage of the data to be used for training (the rest will be used for validation)')
    arg_pars.add_argument('--random_seed', type=int, default=22,
                          help='Random seed for reproducibility')
    args = arg_pars.parse_args()

    main(args)
