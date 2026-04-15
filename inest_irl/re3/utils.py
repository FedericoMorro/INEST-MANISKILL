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

import os
import pickle
import typing
from typing import Any, Dict, Optional

from absl import logging
# import gym
# from gym.wrappers import RescaleAction
import matplotlib.pyplot as plt
from ml_collections import config_dict
import numpy as np
from sac.replay_buffer import (
    ReplayBuffer,
    ReplayBufferGoalClassifier,
    ReplayBufferDistanceToGoal,
    ReplayBufferDistanceToGoalVip,
    ReplayBufferDistanceToGoalBbox,
)
from sac.replay_buffer_lstm import (
    ReplayBufferLSTM,
    ReplayBufferLearnedRewardLSTM,
    ReplayBufferDistanceToGoalLSTM,
    ReplayBufferDistanceToGoalBboxLSTM,
)
from sac import wrappers
import torch
from torchkit import CheckpointManager
from torchkit.experiment import git_revision_hash
from xirl import common
import xmagical
import yaml
import wandb
from datetime import datetime
import functools
from torchkit import checkpoint
import math
from ultralytics import YOLO
from sac.state_entropy_tracker import StateEntropyTracker, IntrinsicRewardWrapper

# pylint: disable=logging-fstring-interpolation

ConfigDict = config_dict.ConfigDict
FrozenConfigDict = config_dict.FrozenConfigDict

# ========================================= #
# Experiment utils.
# ========================================= #


def setup_experiment(exp_dir, config, resume = False, use_wandb=False):
  """Initializes a pretraining or RL experiment."""
  #  If the experiment directory doesn't exist yet, creates it and dumps the
  # config dict as a yaml file and git hash as a text file.
  # If it exists already, raises a ValueError to prevent overwriting
  # unless resume is set to True.
  if os.path.exists(exp_dir):
    if not resume:
      raise ValueError(
          "Experiment already exists. Run with --resume to continue.")
    load_config_from_dir(exp_dir, config, use_wandb)
  else:
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
      yaml.dump(ConfigDict.to_dict(config), fp)
    with open(os.path.join(exp_dir, "git_hash.txt"), "w") as fp:
      fp.write(git_revision_hash())


def load_config_from_dir(
    exp_dir,
    config = None,
    use_wandb=False
):
  """Load experiment config."""
  if use_wandb:
    cfg = wandb.restore("config.yaml", exp_dir)
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
  model = common.get_model(config, device)
  model.to(device).eval()
  checkpoint_dir = os.path.join(pretrained_path, "checkpoints")
  checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
  global_step = checkpoint_manager.restore_or_initialize()
  logging.info("Restored model from checkpoint %d.", global_step)
  #wandb.restore(f"{global_step}.ckpt", pretrained_path)
  return config, model


def save_pickle(experiment_path, arr, name):
  """Save an array as a pickle file."""
  filename = os.path.join(experiment_path, name)
  with open(filename, "wb") as fp:
    pickle.dump(arr, fp)
  logging.info("Saved %s to %s", name, filename)


def load_pickle(pretrained_path, name):
  """Load a pickled array."""
  #wandb.restore(name, pretrained_path)
  filename = os.path.join(pretrained_path, name)
  with open(filename, "rb") as fp:
    arr = pickle.load(fp)
  logging.info("Successfully loaded %s from %s", name, filename)
  return arr


def load_model(pretrained_path, load_goal_emb, device):
  """Load a pretrained model and optionally a precomputed goal embedding."""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  config = load_config_from_dir(pretrained_path)
  model = common.get_model(config, device)
  checkpoint_dir = os.path.join(pretrained_path, "checkpoints")
  # checkpoint_manager = checkpoint.CheckpointManager(
  #     checkpoint.Checkpoint(model=model), checkpoint_dir, device)
  checkpoint_manager = checkpoint.CheckpointManager(checkpoint_dir, model=model)
  global_step = checkpoint_manager.restore_or_initialize()

#  model.load_state_dict(torch.load(checkpoint_dir + "/0000000000008001.ckpt"))
  if load_goal_emb:
    print("Loading goal embedding.")
    
    with open(os.path.join(pretrained_path, "goal_emb.pkl"), "rb") as fp:
      goal_emb = pickle.load(fp)
    
    with open(os.path.join(pretrained_path, "distance_scale.pkl"), "rb") as fp:
      distance_scale = pickle.load(fp)

    model.goal_emb = goal_emb
    model.distance_scale = distance_scale
  
  return config, model

# ========================================= #
# RL utils.
# ========================================= #


def make_env(
    env_name,
    seed,
    save_dir = None,
    add_episode_monitor = True,
    action_repeat = 1,
    frame_stack = 1,
    use_dense_reward=False,
    wandb_video_freq=0,
    config=None
):
  """Env factory with wrapping.

  Args:
    env_name: The name of the environment.
    seed: The RNG seed.
    save_dir: Specifiy a save directory to wrap with `VideoRecorder`.
    add_episode_monitor: Set to True to wrap with `EpisodeMonitor`.
    action_repeat: A value > 1 will wrap with `ActionRepeat`.
    frame_stack: A value > 1 will wrap with `FrameStack`.

  Returns:
    gym.Env object.
  """
  xmagical.register_envs()
  if env_name in xmagical.ALL_REGISTERED_ENVS:
    # x-magical logic
    import gym
    from gym.wrappers import RescaleAction
    env = gym.make(env_name, use_dense_reward=use_dense_reward,config=config)
    if add_episode_monitor:
      env = wrappers.EpisodeMonitor(env)
    if action_repeat > 1:
      env = wrappers.ActionRepeat(env, action_repeat)
    env = RescaleAction(env, -1.0, 1.0)
    if save_dir is not None:
      env = wrappers.VideoRecorder(env, save_dir=save_dir,wandb_video_freq=wandb_video_freq)
    if frame_stack > 1:
      env = wrappers.FrameStack(env, frame_stack)
    env.seed(seed)
  else:
    raise ValueError(f"{env_name} is not a valid environment name.")
   
  # Seed.
  env.action_space.seed(seed)
  env.observation_space.seed(seed)

  return env

def sigmoid(x, t = 1.0):
  return 1 / (1 + math.exp(-x / t))


def wrap_learned_reward(env, config, device="cpu"):
  """Wrap the environment with a learned reward wrapper.

  Args:
    env: A `gym.Env` to wrap with a `LearnedVisualRewardWrapper` wrapper.
    config: RL config dict, must inherit from base config defined in
      `configs/rl_default.py`.

  Returns:
    gym.Env object.
  """
  pretrained_path = config.reward_wrapper.pretrained_path
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  kwargs = {
      "env": env,
      "device": device,
  }
  if config.reward_wrapper.type != "distance_to_goal_vip":
    model_config, model = load_model_checkpoint(pretrained_path, device)
    kwargs["res_hw"] = model_config.data_augmentation.image_size,
    kwargs['model'] = model

  if config.reward_wrapper.type == "goal_classifier":
    env = wrappers.GoalClassifierLearnedVisualReward(**kwargs)

  elif config.reward_wrapper.type == "distance_to_goal":
    kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                           "distance_scale.pkl")
    kwargs["res_hw"]  = model_config.data_augmentation.image_size
    env = wrappers.DistanceToGoalLearnedVisualReward(**kwargs)
  elif config.reward_wrapper.type == "distance_to_goal_bbox":
    kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                          "distance_scale.pkl")
    kwargs["obj_det"]  = YOLO('obj_detect/best.pt').to(device)
    kwargs["res_hw"]  = (384,384)
    # kwargs["distance_func"] = functools.partial(
    #     sigmoid,
    #     config.reward_wrapper.distance_func_temperature,
    # )

    #kwargs["state_dims"] = config.state_dims
    #kwargs["embodiment"] = config.embodiment.capitalize()
    env = wrappers.DistanceToGoalBboxReward(**kwargs)
  elif config.reward_wrapper.type == "distance_to_goal_bbox_sweepToTop":
    kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                          "distance_scale.pkl")
    kwargs["obj_det"]  = YOLO('obj_detect/best_sweepToTop_3.pt').to(device)
    kwargs["res_hw"]  = (384,384)

    env = wrappers.DistanceToGoalBboxReward_sweepToTop(**kwargs)
  elif config.reward_wrapper.type == "distance_to_goal_vip":
    kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                          "distance_scale.pkl")
    kwargs["res_hw"]  = (384,384)
    env = wrappers.DistanceToGoalLearnedVipReward(**kwargs)
  elif config.reward_wrapper.type == "distance_to_goal_bbox_putshoesinbox":
    kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
    kwargs["distance_scale"] = load_pickle(pretrained_path, "distance_scale.pkl")
    kwargs["res_hw"]  = (256,256)
    env = wrappers.DistanceToGoalBboxReward_PutShoesInBox(**kwargs)
  else:
    raise ValueError(
        f"{config.reward_wrapper.type} is not a valid reward wrapper.")
  
  return env


def wrap_intrinsic_reward(env, config, device="cpu"):
  """Wrap the environment with an intrinsic reward wrapper based on state entropy.

  This keeps the SAC policy observing whatever the env emits (vector or image),
  but for RE3 we can *optionally* use env.render('rgb_array') frames as the
  input to the random encoder.

  Args:
    env: A `gym.Env` to wrap with an intrinsic reward wrapper.
    config: RL config dict, must inherit from base config defined in
      `configs/rl_default.py`.
    device: Device to use for computations.

  Returns:
    gym.Env object with intrinsic rewards.
  """
  if not hasattr(config, 'intrinsic_reward') or not config.intrinsic_reward.enabled:
    print("❌ Intrinsic reward is not enabled")
    return env

  method = config.intrinsic_reward.get('method', 'kmeans')

  # ---- Build tracker kwargs common to all methods ----
  tracker_kwargs = dict(
      state_dim=None,  # we won't rely on this for RE3; legacy methods ignore None
      max_states=config.intrinsic_reward.get('max_states', 10000),
      novelty_threshold=config.intrinsic_reward.get('novelty_threshold', 0.1),
      intrinsic_weight=config.intrinsic_reward.get('intrinsic_weight', 0.1),
      method=method,
      n_clusters=config.intrinsic_reward.get('n_clusters', 100),
      memory_decay=config.intrinsic_reward.get('memory_decay', 0.99),
      device=device,
  )

  # ---- RE3-specific config (with images by default) ----
  re3_cfg = None
  if method == "re3":
    # allow a nested config.intrinsic_reward.re3 block, but fall back gracefully
    re3_block = getattr(config.intrinsic_reward, 're3', None)
    re3_cfg = type("Cfg", (object,), {})()
    setattr(re3_cfg, "k", getattr(re3_block, "k", 3) if re3_block else 3)
    setattr(re3_cfg, "embed_dim", getattr(re3_block, "embed_dim", 512) if re3_block else 512)
    setattr(re3_cfg, "beta0", getattr(re3_block, "beta0", config.intrinsic_reward.get('intrinsic_weight', 0.1)) if re3_block else config.intrinsic_reward.get('intrinsic_weight', 0.1))
    setattr(re3_cfg, "beta_decay", getattr(re3_block, "beta_decay", 0.0) if re3_block else 0.0)
    setattr(re3_cfg, "memory_subsample", getattr(re3_block, "memory_subsample", 4096) if re3_block else 4096)
    setattr(re3_cfg, "use_images", getattr(re3_block, "use_images", True) if re3_block else True)
    setattr(re3_cfg, "seed", getattr(re3_block, "seed", None) if re3_block else None)
    tracker_kwargs["re3_cfg"] = re3_cfg
  
  # ---- Pretrained method config ----
  elif method == "pretrained":
    # allow a nested config.intrinsic_reward.pretrained block, but fall back gracefully
    pretrained_block = getattr(config.intrinsic_reward, 'pretrained', None)
    pretrained_cfg = type("Cfg", (object,), {})()
    setattr(pretrained_cfg, "k", getattr(pretrained_block, "k", 3) if pretrained_block else 3)
    setattr(pretrained_cfg, "embed_dim", getattr(pretrained_block, "embed_dim", 512) if pretrained_block else 512)
    setattr(pretrained_cfg, "beta0", getattr(pretrained_block, "beta0", config.intrinsic_reward.get('intrinsic_weight', 0.1)) if pretrained_block else config.intrinsic_reward.get('intrinsic_weight', 0.1))
    setattr(pretrained_cfg, "beta_decay", getattr(pretrained_block, "beta_decay", 0.0) if pretrained_block else 0.0)
    setattr(pretrained_cfg, "memory_subsample", getattr(pretrained_block, "memory_subsample", 4096) if pretrained_block else 4096)
    setattr(pretrained_cfg, "use_images", getattr(pretrained_block, "use_images", False) if pretrained_block else False)
    setattr(pretrained_cfg, "use_bboxes", getattr(pretrained_block, "use_bboxes", True) if pretrained_block else True)
    setattr(pretrained_cfg, "seed", getattr(pretrained_block, "seed", None) if pretrained_block else None)
    tracker_kwargs["re3_cfg"] = pretrained_cfg  # Use same parameter name for compatibility
    
    # Load pretrained model
    pretrained_model = getattr(config.intrinsic_reward, 'pretrained_model', None)
    if pretrained_model is None:
      raise ValueError("pretrained_model must be provided in config.intrinsic_reward for 'pretrained' method")
    tracker_kwargs["pretrained_model"] = pretrained_model

  # ---- Create tracker ----
  tracker = StateEntropyTracker(**tracker_kwargs)

  # If RE3 or pretrained method is using images, pass the env so the tracker can call env.render("rgb_array")
  if method in ["re3", "pretrained"]:
    try:
      tracker.set_image_env(env)
      cfg = re3_cfg if method == "re3" else pretrained_cfg
      if cfg.use_images:
        # quick capability check
        try:
          # Reset environment first (some environments need this before rendering)
          try:
            env.reset()
          except Exception:
            pass
          
          # Try different render call patterns
          try:
            frame = env.render()
          except Exception as e1:
            try:
              frame = env.render(mode="rgb_array")
            except Exception as e2:
              raise e2
          
          if frame is None:
            raise ValueError("Frame is None")
        except Exception as e:
          # fallback to raw state; warn once
          cfg.use_images = False
          print(f"[{method.upper()}] Warning: env.render('rgb_array') unsupported; falling back to raw-state embeddings.")

    except Exception:
      # older tracker: ignore if method not implemented
      pass
  
  # Setup bbox extraction for pretrained method
  if method == "pretrained":
    try:
      # print(f"[PRETRAINED] Environment type: {type(env)}")
      # print(f"[PRETRAINED] Environment name: {getattr(env, 'env_name', 'No env_name')}")
      # print(f"[PRETRAINED] Environment unwrapped: {getattr(env, 'unwrapped', 'No unwrapped')}")
      
      # Check if this is a ManiSkill environment (has specific attributes)
      if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'left_shoe'):
        # This is a ManiSkill environment - set up bbox extractor
        if pretrained_cfg.use_bboxes:
          try:
            # Add the put_shoes_task directory to path for bbox extraction
            import sys
            sys.path.append('/home/aprotopapa/code/maniskill/put_shoes_task')
            from bbox_extraction import PutShoesInBoxBBoxExtractor
            bbox_extractor = PutShoesInBoxBBoxExtractor(device=device)
            tracker.set_bbox_extractor(bbox_extractor)
            # print("[PRETRAINED] Set up ManiSkill bbox extractor")
          except ImportError:
            # print("[PRETRAINED] Warning: Could not import PutShoesInBoxBBoxExtractor, bbox extraction disabled")
            pretrained_cfg.use_bboxes = False
      
      # Check if this is an X-Magical environment (check both wrapper and unwrapped)
      elif (hasattr(env, 'env_name') and ('xmagical' in str(type(env)).lower() or 'MatchRegions' in str(env.env_name))) or \
           (hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'env_name') and 'MatchRegions' in str(env.unwrapped.env_name)) or \
           (hasattr(env, 'state_to_bboxes') and 'MatchRegions' in str(type(env.unwrapped))):
        # This is an X-Magical environment - set up state-to-bboxes function
        if pretrained_cfg.use_bboxes:
          if hasattr(env, 'state_to_bboxes'):
            # Use the existing state_to_bboxes method from the wrapper
            def state_to_bboxes_func(obs):
              return env.state_to_bboxes(obs)
            tracker.set_state_to_bboxes_func(state_to_bboxes_func)
            # print("[PRETRAINED] Set up X-Magical state-to-bboxes function using wrapper method")
          else:
            # Create a state-to-bboxes function similar to DistanceToGoalBboxReward
            def state_to_bboxes_func(obs):
              return tracker._create_xmagical_state_to_bboxes(obs)
            tracker.set_state_to_bboxes_func(state_to_bboxes_func)
            # print("[PRETRAINED] Set up X-Magical state-to-bboxes function using tracker method")
      else:
        print("[PRETRAINED] Environment not recognized as ManiSkill or X-Magical")
        print(f"[PRETRAINED] Available attributes: {[attr for attr in dir(env) if not attr.startswith('_')]}")
      
    except Exception as e:
      print(f"[PRETRAINED] Warning: Could not set up bbox extraction: {e}")
      pretrained_cfg.use_bboxes = False
  
  # Use a single scale: the tracker already applies beta (with decay)
  intrinsic_w = config.intrinsic_reward.get('intrinsic_weight', 0.1)
  if method in ["re3", "pretrained"]:
      intrinsic_w = 1.0

  # ---- Create wrapper ----
  env = IntrinsicRewardWrapper(
      env=env,
      state_entropy_tracker=tracker,
      intrinsic_weight=intrinsic_w,
      extrinsic_weight=config.intrinsic_reward.get('extrinsic_weight', 1.0),
  )

  return env

def wrap_custom_shoes_in_box_reward(env, config, device="cpu"):
    """Wrap the environment with a custom reward wrapper for PutShoesInBox task.
    
    Args:
        env: A `gym.Env` to wrap with the custom reward wrapper.
        config: Configuration dictionary.
        device: Device to use for computations.
        
    Returns:
        gym.Env object with custom rewards.
    """
    # Import here to avoid circular imports
    from sac.wrappers import PutShoesInBoxCustomReward
    
    # Create custom reward wrapper
    env = PutShoesInBoxCustomReward(
        env,
        reward_scale=config.get('custom_reward_scale', 1.0)
    )
    
    return env

def make_buffer(env, device, config):
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    kwargs = {
        "obs_shape": obs_shape,
        "action_shape": action_shape,
        "capacity": config.replay_buffer_capacity,
        "device": device,
    }

    pretrained_path = config.reward_wrapper.pretrained_path
    reward_type = config.reward_wrapper.type

    if not config.use_lstm:
        # ========== Non-LSTM buffer ==========
        if not pretrained_path:
            return ReplayBuffer(**kwargs)

        if reward_type != "distance_to_goal_vip":
            model_config, model = load_model_checkpoint(pretrained_path, device)
            kwargs["res_hw"] = model_config.data_augmentation.image_size
            kwargs["model"] = model

        if reward_type == "goal_classifier":
            return ReplayBufferGoalClassifier(**kwargs)
        elif reward_type == "distance_to_goal":
            kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
            kwargs["distance_scale"] = load_pickle(pretrained_path, "distance_scale.pkl")
            return ReplayBufferDistanceToGoal(**kwargs)
        elif reward_type in ["distance_to_goal_bbox", "distance_to_goal_bbox_sweepToTop"]:
            kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
            kwargs["distance_scale"] = load_pickle(pretrained_path, "distance_scale.pkl")
            kwargs["env"] = env
            kwargs["res_hw"] = (384, 384)
            kwargs["batch_size"] = config.sac.batch_size
            return ReplayBufferDistanceToGoalBbox(**kwargs)
        elif reward_type == "distance_to_goal_bbox_putshoesinbox":
            kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
            kwargs["distance_scale"] = load_pickle(pretrained_path, "distance_scale.pkl")
            kwargs["env"] = env
            kwargs["res_hw"] = (256, 256)
            kwargs["batch_size"] = config.sac.batch_size
            return ReplayBufferDistanceToGoalBbox(**kwargs)
        elif reward_type == "distance_to_goal_vip":
            kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
            kwargs["distance_scale"] = load_pickle(pretrained_path, "distance_scale.pkl")
            kwargs["res_hw"] = (384, 384)
            return ReplayBufferDistanceToGoalVip(**kwargs)
        else:
            raise ValueError(f"Unsupported reward wrapper type: {reward_type}")

    else:
        # ========== LSTM buffer ==========
        obs_dim = obs_shape[0]
        action_dim = action_shape[0]

        if not pretrained_path:
            return ReplayBufferLSTM(
                obs_shape=obs_shape,
                action_shape=action_shape,
                capacity=config.replay_buffer_capacity,
                seq_len=config.seq_len,
                device=device,
                hidden_dim=config.sac.actor.hidden_dim,
            )

        if reward_type == "distance_to_goal":
            model_config, model = load_model_checkpoint(pretrained_path, device)
            return ReplayBufferDistanceToGoalLSTM(
                obs_shape=obs_shape,
                action_shape=action_shape,
                capacity=config.replay_buffer_capacity,
                seq_len=config.seq_len,
                device=device,
                hidden_dim=config.sac.actor.hidden_dim,
                model=model,
                res_hw=(384, 384),
                goal_emb=load_pickle(pretrained_path, "goal_emb.pkl"),
                distance_scale=load_pickle(pretrained_path, "distance_scale.pkl"),
            )

        elif reward_type in ["distance_to_goal_bbox", "distance_to_goal_bbox_sweepToTop"]:
            model_config, model = load_model_checkpoint(pretrained_path, device)
            goal_emb = load_pickle(pretrained_path, "goal_emb.pkl")
            distance_scale = load_pickle(pretrained_path, "distance_scale.pkl")
            env.goal_emb = goal_emb
            env.distance_scale = distance_scale
            return ReplayBufferDistanceToGoalBboxLSTM(
                obs_shape=obs_shape,
                action_shape=action_shape,
                capacity=config.replay_buffer_capacity,
                seq_len=config.seq_len,
                device=device,
                hidden_dim=config.sac.actor.hidden_dim,
                model=model,
                res_hw=(384, 384),
                env=env,
                goal_emb=load_pickle(pretrained_path, "goal_emb.pkl"),
                distance_scale=load_pickle(pretrained_path, "distance_scale.pkl"),
            )

        else:
            raise ValueError(f"Unsupported LSTM reward wrapper type: {reward_type}")
# ========================================= #
# Misc. utils.
# ========================================= #


def plot_reward(rews):
  """Plot raw and cumulative rewards over an episode."""
  _, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
  axes[0].plot(rews)
  axes[0].set_xlabel("Timestep")
  axes[0].set_ylabel("Reward")
  axes[1].plot(np.cumsum(rews))
  axes[1].set_xlabel("Timestep")
  axes[1].set_ylabel("Cumulative Reward")
  for ax in axes:
    ax.grid(b=True, which="major", linestyle="-")
    ax.grid(b=True, which="minor", linestyle="-", alpha=0.2)
  plt.minorticks_on()
  plt.show()
  #plt.savefig("gnn_role.png")
def get_current_date():
	return str(datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))

def create_dirs(path):
	try:
		os.makedirs(os.path.join(path))
	except OSError as error:
		pass


# ========================================= #
# Convenience functions for pretrained intrinsic rewards
# ========================================= #

def create_maniskill_env_with_pretrained_intrinsic(
    pretrained_model,
    device,
    intrinsic_weight: float = 0.1,
    extrinsic_weight: float = 1.0,
    pretrained_cfg: dict = None,
    use_images: bool = False,
    use_bboxes: bool = True,
    **env_kwargs
):
    """Convenience function to create ManiSkill PutShoesInBox environment with pretrained intrinsic rewards.
    
    Args:
        pretrained_model: Pretrained model for computing embeddings
        device: Device to use for computations
        intrinsic_weight: Weight for intrinsic reward component
        extrinsic_weight: Weight for extrinsic reward component
        pretrained_cfg: Configuration for pretrained method
        use_images: Whether to use image observations
        use_bboxes: Whether to use bbox extraction (for ManiSkill)
        **env_kwargs: Additional environment creation arguments
        
    Returns:
        Wrapped ManiSkill environment with pretrained intrinsic rewards
    """
    from sac.maniskill_wrappers import create_maniskill_putshoesinbox_with_pretrained_intrinsic
    
    return create_maniskill_putshoesinbox_with_pretrained_intrinsic(
        pretrained_model=pretrained_model,
        device=device,
        intrinsic_weight=intrinsic_weight,
        extrinsic_weight=extrinsic_weight,
        pretrained_cfg=pretrained_cfg,
        use_images=use_images,
        use_bboxes=use_bboxes,
        **env_kwargs
    )


def create_xmagical_env_with_pretrained_intrinsic(
    env_name: str,
    pretrained_model,
    device,
    intrinsic_weight: float = 0.1,
    extrinsic_weight: float = 1.0,
    pretrained_cfg: dict = None,
    use_images: bool = False,
    use_bboxes: bool = True,
    **env_kwargs
):
    """Convenience function to create X-Magical environment with pretrained intrinsic rewards.
    
    Args:
        env_name: Name of the X-Magical environment
        pretrained_model: Pretrained model for computing embeddings
        device: Device to use for computations
        intrinsic_weight: Weight for intrinsic reward component
        extrinsic_weight: Weight for extrinsic reward component
        pretrained_cfg: Configuration for pretrained method
        use_images: Whether to use image observations
        use_bboxes: Whether to use bbox extraction from state (for X-Magical)
        **env_kwargs: Additional environment creation arguments
        
    Returns:
        Wrapped X-Magical environment with pretrained intrinsic rewards
    """
    from sac.wrappers import create_xmagical_env_with_pretrained_intrinsic as _create_xmagical_env_with_pretrained_intrinsic
    
    return _create_xmagical_env_with_pretrained_intrinsic(
        env_name=env_name,
        pretrained_model=pretrained_model,
        device=device,
        intrinsic_weight=intrinsic_weight,
        extrinsic_weight=extrinsic_weight,
        pretrained_cfg=pretrained_cfg,
        use_images=use_images,
        use_bboxes=use_bboxes,
        **env_kwargs
    )