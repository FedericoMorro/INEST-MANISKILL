# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Useful methods shared by all scripts."""

import json
import json
import os
import pickle
import typing
from typing import Any, Dict, Optional

from click import Path
from click import Path
from absl import logging
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import matplotlib.pyplot as plt
from ml_collections import config_dict
import numpy as np
import torch
from torchkit import CheckpointManager
from torchkit.experiment import git_revision_hash
import mani_skill
import mani_skill.envs.tasks as mani_envs
import mani_skill
import mani_skill.envs.tasks as mani_envs
import importlib

from inest_irl.sac import replay_buffer, wrappers
from xirl import common

import yaml

# Global variable to store reference flattened size
_FLATTEN_REF_SIZE = None

def flatten_observation(obs):
    """Robust flattening of nested ManiSkill observations into a fixed-length float32 vector."""
    def extract_arrays(item):
        arrays = []

        if isinstance(item, np.ndarray):
            if item.dtype == np.object_:
                for sub_item in item.flat:
                    arrays.extend(extract_arrays(sub_item))
            elif np.issubdtype(item.dtype, (np.floating, np.integer)):
                arr = item.astype(np.float32).ravel()
                # Replace NaNs or infs
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                arrays.append(arr)

        elif isinstance(item, torch.Tensor):
            # Safely handle GPU tensors
            arr = item.detach().cpu().numpy().astype(np.float32).ravel()
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arrays.append(arr)

        elif isinstance(item, (list, tuple)):
            for sub_item in item:
                arrays.extend(extract_arrays(sub_item))

        elif isinstance(item, dict):
            for v in item.values():
                arrays.extend(extract_arrays(v))

        elif isinstance(item, (int, float, bool, np.number)):
            arrays.append(np.array([float(item)], dtype=np.float32))

        return arrays

    float32_arrays = extract_arrays(obs)

    if not float32_arrays:
        flat = np.zeros((1,), dtype=np.float32)
    else:
        flat = np.concatenate(float32_arrays, dtype=np.float32)

    # Enforce consistent size across timesteps
    global _FLATTEN_REF_SIZE
    if _FLATTEN_REF_SIZE is None:
        _FLATTEN_REF_SIZE = flat.shape[0]
        # print(f"[flatten_observation] reference size set to {_FLATTEN_REF_SIZE}")
    else:
        if flat.shape[0] != _FLATTEN_REF_SIZE:
            diff = _FLATTEN_REF_SIZE - flat.shape[0]
            if diff > 0:
                # Pad missing elements with zeros
                flat = np.pad(flat, (0, diff))
                # print(f"[flatten_observation] padded +{diff} to maintain size {_FLATTEN_REF_SIZE}")
            elif diff < 0:
                # Truncate extras (shouldn't happen)
                flat = flat[:_FLATTEN_REF_SIZE]
                # print(f"[flatten_observation] truncated {abs(diff)} to maintain size {_FLATTEN_REF_SIZE}")

    return flat
  
# pylint: disable=logging-fstring-interpolation

ConfigDict = config_dict.ConfigDict
FrozenConfigDict = config_dict.FrozenConfigDict

# ========================================= #
# Experiment utils.
# ========================================= #


def setup_experiment(exp_dir, config, resume = False):
  """Initializes a pretraining or RL experiment."""
  #  If the experiment directory doesn't exist yet, creates it and dumps the
  # config dict as a yaml file and git hash as a text file.
  # If it exists already, raises a ValueError to prevent overwriting
  # unless resume is set to True.
  if os.path.exists(exp_dir):
    if not resume:
      raise ValueError(
          "Experiment already exists. Run with --resume to continue.")
    load_config_from_dir(exp_dir, config)
  else:
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
      yaml.dump(ConfigDict.to_dict(config), fp)
    with open(os.path.join(exp_dir, "git_hash.txt"), "w") as fp:
      fp.write(git_revision_hash())


def load_config_from_dir(
  exp_dir,
  config = None,
):
  """Load experiment config."""
  with open(os.path.join(exp_dir, "config.yaml"), "r") as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)
  # Inplace update the config if one is provided.
  if config is not None:
    config.update(cfg)
    return
  return ConfigDict(cfg)


def dump_config(exp_dir, config):
  """Dump config to disk."""
  # Note: No need to explicitly delete the previous config file as "w" will
  # overwrite the file if it already exists.
  with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
    yaml.dump(ConfigDict.to_dict(config), fp)


def copy_config_and_replace(
  config,
  update_dict = None,
  freeze = False,
):
  """Makes a copy of a config and optionally updates its values."""
  # Using the ConfigDict constructor leaves the `FieldReferences` untouched
  # unlike `ConfigDict.copy_and_resolve_references`.
  new_config = ConfigDict(config)
  if update_dict is not None:
    new_config.update(update_dict)
  if freeze:
    return FrozenConfigDict(new_config)
  return new_config


def load_model_checkpoint(pretrained_path, device):
  """Load a pretrained model and optionally a precomputed goal embedding."""
  config = load_config_from_dir(pretrained_path)
  model = common.get_model(config)
  model.to(device).eval()
  checkpoint_dir = os.path.join(pretrained_path, "checkpoints")
  checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
  global_step = checkpoint_manager.restore_or_initialize()
  logging.info("Restored model from checkpoint %d.", global_step)
  return model, config, global_step


def save_pickle(experiment_path, arr, name):
  """Save an array as a pickle file."""
  filename = os.path.join(experiment_path, name)
  with open(filename, "wb") as fp:
    pickle.dump(arr, fp)
  logging.info("Saved %s to %s", name, filename)


def load_pickle(pretrained_path, name):
  """Load a pickled array."""
  filename = os.path.join(pretrained_path, name)
  with open(filename, "rb") as fp:
    arr = pickle.load(fp)
  logging.info("Successfully loaded %s from %s", name, filename)
  return arr


# ========================================= #
# RL utils.
# ========================================= #


def make_env(
  env_name,
  seed,
  reward_type = "env",
  obs_mode = "state",
  frame_stack = 1,
  action_repeat = 1,
  rank = 0,
  train_flag = False,
  exp_dir = None,
  learned_reward_pretrained_path = None,
  device = None,
  add_episode_monitor = True,
  save_video = False,
  wrap=True,
):
  """Env factory with wrapping.

  Args:
    env_name: The name of the environment.
    seed: The RNG seed.
    save_dir: Specifiy a save directory to wrap with `VideoRecorder`.
    add_episode_monitor: Set to True to wrap with `EpisodeMonitor`.
    action_repeat: A value > 1 will wrap with `ActionRepeat`.
    frame_stack: A value > 1 will wrap with `FrameStack`.
    obs_mode: ManiSkill observation mode (e.g. 'state', 'state_dict', 'rgbd').

  Returns:
    gym.Env object.
  """
  # Create ManiSkill environment 
    # This guarantees the environments (e.g. StackPyramid-v1) are registered with gym.
  try:
    importlib.import_module("mani_skill.envs.tasks.tabletop")
    print("Successfully imported mani_skill.envs.tasks.tabletop")
  except Exception:
    # If import fails, fall back to importing the top-level tasks package which
    # should still register most environments.
    print("Importing mani_skill.envs.tasks.tabletop failed, trying mani_skill.envs.tasks instead.")
    try:
      importlib.import_module("mani_skill.envs.tasks")
    except Exception:
      # If both imports fail, proceed and let gym.raise the appropriate error.
      pass

  #! create env with local StackPyramid, e.g.
  if env_name == "StackPyramid-v1custom":
    import inest_irl.maniskill3.stack_pyramid as local_stack_pyramid
    
  env = gym.make(
    env_name, # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    seed=seed,
    obs_mode=obs_mode,
    control_mode="pd_ee_delta_pose", # pd_ee_delta_pos[e], with e includes also gripper quaternion orientation control
    render_mode="rgb_array",
    env_reward_type="normalized_dense",
  )

  if add_episode_monitor:
    env = wrappers.EpisodeMonitor(env)
  if action_repeat > 1:
    env = wrappers.ActionRepeat(env, action_repeat)
  # Temporarily disable RescaleAction to debug reset issue
  # env = RescaleAction(env, -1.0, 1.0)
  if save_video and exp_dir is not None:
    env = wrappers.VideoRecorder(env, save_dir=exp_dir)
  if frame_stack > 1:
    env = wrappers.FrameStack(env, frame_stack)

  if not wrap:
    return env

  wrapped_env = wrap_env(
    env,
    reward_type,
    rank,
    train_flag,
    exp_dir,
    learned_reward_pretrained_path,
    device,
  )

  # Seed.
#   env.seed(seed)
#   env.action_space.seed(seed)
#   env.observation_space.seed(seed)

  #return wrappers.GymCompatibilityWrapper(wrapped_env)
  return wrapped_env


def wrap_env(env, reward_type, rank, train_flag, exp_dir, learned_reward_pretrained_path, device):
  """Wrap the environment with a learned reward wrapper.

  Args:
    env: A `gym.Env` to wrap with a `LearnedVisualRewardWrapper` wrapper.
    env_reward_type: The type of reward wrapper to use.
    learned_reward_pretrained_path: The path to the pretrained reward model.
    device: The device to use for the reward model. 

  Returns:
    gym.Env object.
  """
  print("Wrapping environment...")
  if reward_type in ["env", "sparse"]:
    return wrappers.EnvironmentRewardWrapper(env, rank, train_flag, exp_dir)
  elif reward_type == "env_state-intrinsic":
    return wrappers.EnvironmentRewardStateIntrinsicWrapper(env, rank, train_flag, exp_dir)
  
  if learned_reward_pretrained_path is None:
    raise ValueError(f"learned_reward_pretrained_path must be provided for learned reward wrapper types (specified: {reward_type}).")

  pretrained_path = learned_reward_pretrained_path
  model, model_config, model_step = load_model_checkpoint(pretrained_path, device)
  
  cache_path = os.path.join(pretrained_path, "checkpoints", f"cached_embeddings_step_{model_step}.pkl")
  if os.path.exists(cache_path):
    print(f"Loading precomputed goal embedding from {cache_path}")
    with open(cache_path, "rb") as fp:
      goal_emb, subgoal_embs, dist_scale = pickle.load(fp)
  else:
    print("No precomputed goal embedding found, computing now...")
    from inest_irl.utils.compute_learned_return import compute_goal_embedding
    train_loader = common.get_downstream_dataloaders(model_config)["train"]
    subgoal_frames_path = Path(model_config.data.root) / "subgoal_frames.json"
    if subgoal_frames_path.exists():
      with open(subgoal_frames_path, 'r') as f:
        subgoal_frames = json.load(f)
      print(f"Found subgoal frames file with {len(subgoal_frames)} trajectories - will compute and plot subgoal rewards")
    else:
      subgoal_frames = None
      print("No subgoal frames file found - will only compute and plot rewards to final goal")
    goal_emb, subgoal_embs, dist_scale = compute_goal_embedding(model, train_loader, device, subgoal_frames=subgoal_frames)
    with open(cache_path, "wb") as fp:
      pickle.dump((goal_emb, subgoal_embs, dist_scale), fp)
    print(f"Computed and cached goal embedding at {cache_path}")
  
  if reward_type == "goal_dist":
    return wrappers.GoalDistanceLearnedVisualRewardWrapper(
      env=env, rank=rank, train_flag=train_flag, exp_dir=exp_dir,
      model=model, device=device, #res_hw=model_config.data_augmentation.image_size,  -> should be already 128x128
      goal_emb=goal_emb, dist_scale=dist_scale,
    )
  else:
     raise NotImplementedError(f"Reward wrapper type {reward_type} not implemented yet.")
  
  if reward_type == "reds":
    print("Model loaded")
    model.load_state_dict(torch.load(
        os.path.join(pretrained_path, "reds_model.pth"),
        map_location=device,
    ))
    model.to(device).eval()

  kwargs = {
      "env": env,
      "model": model,
      "device": device,
      "res_hw": model_config.data_augmentation.image_size,
  }
  

  if reward_type == "goal_classifier":
    env = wrappers.GoalClassifierLearnedVisualReward(**kwargs)

  elif reward_type == "distance_to_goal":
    kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                           "distance_scale.pkl")
    env = wrappers.DistanceToGoalLearnedVisualReward(**kwargs)
    
  elif reward_type == "inest":
    all_means = load_pickle(pretrained_path, "subtask_means.pkl")
    selected_means_2_4_6 = [all_means[i] for i in [1]]  # elements 2,4,6
    kwargs["subtask_means"] = selected_means_2_4_6
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                           "distance_scale.pkl")
    env = wrappers.INESTIRLLearnedVisualReward(**kwargs)
    
  elif reward_type == "inest_knn":
    all_means = load_pickle(pretrained_path, "subtask_means.pkl")
    selected_means_2_4_6 = [all_means[i] for i in [0,1]]  # elements 2,4,6
    kwargs["subtask_means"] = selected_means_2_4_6
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                           "distance_scale.pkl")
    env = wrappers.KNNINESTIRLLearnedVisualReward(**kwargs)
    
  elif reward_type == "state_intrinsic":
    all_means = load_pickle(pretrained_path, "subtask_means.pkl")
    selected_means_2_4_6 = [all_means[i] for i in [0,1]]  # elements 2,4,6
    kwargs["subtask_means"] = selected_means_2_4_6
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                           "distance_scale.pkl")
    env = wrappers.STATEINTRINSICLearnedVisualReward(**kwargs)
    
  elif reward_type == "reds":
    print("AAAAAAA")
    env = wrappers.REDSLearnedVisualReward(**kwargs)

  else:
    raise ValueError(
        f"{reward_type} is not a valid reward wrapper.")

  return env

