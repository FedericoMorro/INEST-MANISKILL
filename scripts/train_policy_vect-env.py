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

"""Launch script for training RL policies with pretrained reward models."""

import collections
import json
import os.path as osp
from typing import Dict

from absl import app
from absl import flags
from absl import logging
import gym
from ml_collections import config_dict
from ml_collections import config_flags
import numpy as np
import torch
from torchkit import CheckpointManager
from torchkit import experiment
from torchkit import Logger
from tqdm.auto import tqdm
import wandb

from configs import validate_config
from inest_irl.sac import agent
from inest_irl.utils import utils

# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_name", None, "Experiment name.")
flags.DEFINE_string("env_name", "StackPyramid-v1custom", "The environment name.")
flags.DEFINE_integer("seed", 22, "RNG seed.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")
flags.DEFINE_boolean("resume", False, "Resume experiment from last checkpoint.")
flags.DEFINE_boolean("wandb", False, "Log on W&B.")

config_flags.DEFINE_config_file(
    "config",
    "/home/fmorro/INEST-MANISKILL/scripts/configs/rl_vect-env.py",
    "File path to the training hyperparameter configuration.",
)



# Will be re-imported inside main()
def _reset_get_obs(reset_out):
  """Extract observation from reset output for Gym and Gymnasium APIs."""
  if isinstance(reset_out, tuple) and len(reset_out) == 2:
    return reset_out[0]
  return reset_out


def _step_unpack(step_out):
  """Normalize step outputs to (obs, reward, done, info)."""
  if isinstance(step_out, tuple) and len(step_out) == 5:
    obs, reward, terminated, truncated, info = step_out
    done = np.logical_or(terminated, truncated)
    return obs, reward, done, info
  if isinstance(step_out, tuple) and len(step_out) == 4:
    return step_out
  raise ValueError(f"Unexpected step() output format: {type(step_out)}")


def _get_vector_info(infos, index):
  """Extract a per-environment info dict from vector or non-vector infos."""
  if not isinstance(infos, dict):
    if isinstance(infos, (list, tuple)) and len(infos) > index:
      return infos[index] if isinstance(infos[index], dict) else {}
    return {}
  
  extracted = {}
  for key, value in infos.items():
    if key.startswith("_") or key == "final_info":
      continue
    # Check mask if present
    mask = infos.get(f"_{key}")
    if mask is not None:
      try:
        if not np.asarray(mask)[index]:
          continue
      except Exception:
        pass
    # Extract item at index
    try:
      item = value[index] if isinstance(value, (list, tuple, np.ndarray)) else value
      if item is not None:
        extracted[key] = item
    except Exception:
      pass
  
  # Merge final_info if present
  if "final_info" in infos:
    mask = infos.get("_final_info")
    try:
      if mask is None or np.asarray(mask)[index]:
        final = infos["final_info"]
        final_item = final[index] if isinstance(final, (list, tuple, np.ndarray)) else final
        if isinstance(final_item, dict):
          extracted.update(final_item)
    except Exception:
      pass
  return extracted


def _to_float_scalar(x):
  if x is None:
    return None

  if isinstance(x, (int, float, bool, np.number)):
    return float(x)

  if isinstance(x, torch.Tensor):
    x = x.detach().cpu().numpy()

  if isinstance(x, (list, tuple)):
    vals = []
    for item in x:
      s = _to_float_scalar(item)
      if s is not None and np.isfinite(s):
        vals.append(s)
    if not vals:
      return None
    return float(np.nanmean(np.asarray(vals, dtype=np.float64)))

  try:
    arr = np.asarray(x, dtype=np.float64)
  except (TypeError, ValueError):
    return None

  if arr.size == 0:
    return None
  return float(np.nanmean(arr))


def _log_scalar_safe(logger, value, step, name, split):
  scalar = _to_float_scalar(value)
  scalar_step = _to_float_scalar(step)

  if scalar is None or scalar_step is None:
    logging.warning("Skipping non-scalar metric '%s'", name)
    return

  if not np.isfinite(scalar) or not np.isfinite(scalar_step):
    logging.warning("Skipping non-finite metric '%s'", name)
    return
  logger.log_scalar(scalar, int(scalar_step), name, split)


def _log_and_wandb(logger, value, step, name, split, wandb_prefix=None):
  """Log metric to logger and optionally to W&B."""
  _log_scalar_safe(logger, value, step, name, split)
  if FLAGS.wandb and wandb_prefix:
    wandb.log({f"{wandb_prefix}/{name}": value, "train/step": step}, step=step)


def _safe_frame_from_env(env):
  try:
    frame = env.render(mode="rgb_array")
  except TypeError:
    frame = env.render()
  if isinstance(frame, torch.Tensor):
    frame = frame.detach().cpu().numpy()
  if isinstance(frame, dict):
    frame = next(iter(frame.values()))
  elif isinstance(frame, (list, tuple)):
    frame = frame[0]
  if isinstance(frame, np.ndarray):
    # Remove batch dimensions
    while frame.ndim > 3:
      frame = frame[0]
    # Normalize to uint8
    if frame.dtype != np.uint8 and frame.size > 0:
      fmin, fmax = float(frame.min()), float(frame.max())
      if fmin >= 0.0 and fmax <= 1.0:
        frame = (frame * 255).astype(np.uint8)
      elif fmin >= -1.0 and fmax <= 1.0:
        frame = ((frame + 1.0) / 2.0 * 255).astype(np.uint8)
      else:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
  return frame if (isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[-1] == 3) else None


def _eval_vector(policy, eval_env, num_episodes, stats, step_value):
  """Evaluate policy on vector environment."""
  episode_rewards = []
  last_episode_frames, last_episode_rewards, last_episode_actions = [], [], []
  observations = _reset_get_obs(eval_env.reset())
  episode_rewards_current = np.zeros(eval_env.num_envs)
  episodes_completed = 0
  
  while episodes_completed < num_episodes:
    record_last = episodes_completed == (num_episodes - 1) and hasattr(eval_env, "envs")
    if record_last:
      frame = _safe_frame_from_env(eval_env.envs[0])
      if frame is not None:
        last_episode_frames.append(frame)
    
    actions = [policy.module.act(observations[j], sample=False) if hasattr(policy, 'module')
               else policy.act(observations[j], sample=False) for j in range(eval_env.num_envs)]
    if record_last and actions:
      last_episode_actions.append(np.asarray(actions[0]).tolist())
    
    next_observations, rewards, dones, infos = _step_unpack(eval_env.step(actions))
    episode_rewards_current += rewards
    if record_last and rewards.size > 0:
      r0 = _to_float_scalar(rewards[0])
      if np.isfinite(r0):
        last_episode_rewards.append(r0)
    
    for j in range(eval_env.num_envs):
      if dones[j] and episodes_completed < num_episodes:
        episode_rewards.append(_to_float_scalar(episode_rewards_current[j]))
        episodes_completed += 1
        info_j = _get_vector_info(infos, j)
        if isinstance(info_j, dict) and "episode" in info_j:
          for k, v in info_j["episode"].items():
            stats[k].append(v)
          if "eval_score" in info_j:
            stats["eval_score"].append(info_j["eval_score"])
        episode_rewards_current[j] = 0.0
    observations = next_observations
  return episode_rewards, last_episode_frames, last_episode_rewards, last_episode_actions


def _eval_single(policy, eval_env, num_episodes, stats):
  """Evaluate policy on single environment."""
  episode_rewards = []
  for _ in range(num_episodes):
    observation = _reset_get_obs(eval_env.reset())
    if "holdr" in FLAGS.experiment_name and hasattr(eval_env, 'reset_state'):
      eval_env.reset_state()
    episode_reward = 0
    done = False
    while not done:
      action = policy.module.act(observation, sample=False) if hasattr(policy, 'module') \
               else policy.act(observation, sample=False)
      observation, reward, done, info = _step_unpack(eval_env.step(action))
      episode_reward += reward
    if isinstance(info, dict) and "episode" in info:
      for k, v in info["episode"].items():
        stats[k].append(v)
      if "eval_score" in info:
        stats["eval_score"].append(info["eval_score"])
    episode_rewards.append(episode_reward)
  return episode_rewards, [], [], []


def evaluate(policy, eval_env, num_episodes, global_step=None):
  """Evaluate the policy and dump rollout videos to disk."""
  import numpy as np
  import collections
  policy.eval()
  stats = collections.defaultdict(list)
  step_value = int(global_step) if global_step is not None else 0
  
  # Get experiment dir for saving
  exp_dir = None
  if hasattr(eval_env, "envs") and len(eval_env.envs) > 0:
    save_dir = getattr(eval_env.envs[0], "save_dir", None)
    if save_dir:
      exp_dir = osp.dirname(osp.dirname(save_dir))
  
  # Evaluate
  if hasattr(eval_env, 'num_envs'):
    episode_rewards, last_episode_frames, last_episode_rewards, last_episode_actions = \
        _eval_vector(policy, eval_env, num_episodes, stats, step_value)
  else:
    episode_rewards, last_episode_frames, last_episode_rewards, last_episode_actions = \
        _eval_single(policy, eval_env, num_episodes, stats)

  
  # Aggregate stats
  for k in stats:
    scalars = [s for item in stats[k] if (s := _to_float_scalar(item)) is not None and np.isfinite(s)]
    stats[k] = float(np.mean(scalars)) if scalars else 0.0
  
  if exp_dir is not None:
    actions_file = osp.join(exp_dir, "last_evaluation_actions.json")
    with open(actions_file, "w") as f:
      json.dump({
          "actions": last_episode_actions,
          "total_reward": float(np.sum(last_episode_rewards))
      }, f, indent=2)

  # Log media to W&B (video, plots - only at evaluation time)
  if FLAGS.wandb and step_value is not None:
    media_dict = {}
    if last_episode_frames:
      frames = np.array([frame.transpose(2, 0, 1) for frame in last_episode_frames])
      media_dict["eval/last_eval_video"] = wandb.Video(frames, fps=30, format="mp4")
      try:
        import matplotlib.pyplot as plt
        reward_series = [r for r in last_episode_rewards if np.isfinite(r)]
        if reward_series:
          plt.figure(figsize=(10, 6))
          plt.plot(reward_series)
          plt.title("Reward Evolution - Last Evaluation Episode")
          plt.xlabel("Step")
          plt.ylabel("Reward")
          plt.tight_layout()
          media_dict["eval/last_reward_plot"] = wandb.Image(plt)
          plt.close()
      except ImportError:
        pass
    if media_dict:
      wandb.log(media_dict, step=step_value)

  return stats, episode_rewards


@experiment.pdb_fallback
def main(_):
  # Make sure we have a valid config that inherits all the keys defined in the
  # base config.
  activated_subtask_experiment = False
  validate_config(FLAGS.config, mode="rl")

  config = FLAGS.config
  exp_dir = osp.join(
      config.save_dir,
      FLAGS.experiment_name,
      str(FLAGS.seed),
  )
  utils.setup_experiment(exp_dir, config, FLAGS.resume)
  
  if FLAGS.wandb:
    wandb.init(project="StackPyramidRL", group="Multi", name=FLAGS.experiment_name, mode="online")
    wandb.config.update(FLAGS)
    wandb.run.log_code(".")
    wandb.config.update(config.to_dict(), allow_val_change=True)

  # Setup compute device.
  if torch.cuda.is_available():
    device = torch.device(FLAGS.device)
  else:
    logging.info("No GPU device found. Falling back to CPU.")
    device = torch.device("cpu")
  logging.info("Using device: %s", device)

  # Set RNG seeds.
  if FLAGS.seed is not None:
    logging.info("RL experiment seed: %d", FLAGS.seed)
    experiment.seed_rngs(FLAGS.seed)
    experiment.set_cudnn(config.cudnn_deterministic, config.cudnn_benchmark)
  else:
    logging.info("No RNG seed has been set for this RL experiment.")

# Load vector environments with different seeds for each process to ensure diversity
  num_envs_per_process = config.get("num_envs_per_process", 4)  # Number of parallel envs per DDP process
  env_seed_start = FLAGS.seed + 1000  # Different seed range for each process
  eval_seed_start = FLAGS.seed + 1000 + 500
 
  env = utils.make_vector_env(
      FLAGS.env_name,
      num_envs=num_envs_per_process,
      seed_start=env_seed_start,
      action_repeat=config.action_repeat,
      frame_stack=config.frame_stack,
  )
  eval_env = utils.make_vector_env(
      FLAGS.env_name,
      num_envs=1,  # Keep eval simple with single env
      seed_start=eval_seed_start,
      action_repeat=config.action_repeat,
      frame_stack=config.frame_stack,
      save_dir=osp.join(exp_dir, "video", "eval") ,  # Only rank 0 saves videos
  )
  
  print("env:", env)
  print("env.action_space:", getattr(env, 'action_space', None))
  print("env.envs:", getattr(env, 'envs', None))
  for i, subenv in enumerate(getattr(env, 'envs', [])):
      print(f"Subenv {i}: {subenv}, action_space: {getattr(subenv, 'action_space', None)}")
  
  
  if config.reward_wrapper.pretrained_path:
    print("Using learned reward wrapper.")
    env = utils.wrap_learned_reward(env, FLAGS.config, device=device)
    eval_env = utils.wrap_learned_reward(eval_env, FLAGS.config, device=device)


  # Dynamically set observation and action space values.
  config.sac.obs_dim = env.single_observation_space.shape[0]
  config.sac.action_dim = env.single_action_space.shape[0]
  config.sac.action_range = [
      float(env.single_action_space.low.min()),
      float(env.single_action_space.high.max()),
  ]


  # Resave the config since the dynamic values have been updated at this point
  # and make it immutable for safety :)
  utils.dump_config(exp_dir, config)
  config = config_dict.FrozenConfigDict(config)

  policy = agent.SAC(device, config.sac)

  buffer = utils.make_vect_buffer(env.envs[0], device, config)

  # Create checkpoint manager.
  checkpoint_dir = osp.join(exp_dir, "checkpoints")
  checkpoint_manager = CheckpointManager(
      checkpoint_dir,
      policy=policy,
      **policy.optim_dict(),
  )

  logger = Logger(osp.join(exp_dir, "tb"), FLAGS.resume)

  try:
    start = checkpoint_manager.restore_or_initialize()
    observations = _reset_get_obs(env.reset())
    episode_rewards = np.zeros(env.num_envs)
    
    # Debug: Print initial shapes and types
    print(f"[INFO] Initial observations shape: {observations.shape}")
    print(f"[INFO] Episode rewards shape: {episode_rewards.shape}")
    print(f"[INFO] Buffer capacity: {buffer.capacity}")
    print(f"[INFO] Number of environments: {env.num_envs}")
    print(f"[INFO] Training frequency adjusted: every {env.num_envs} steps")
    
    # Track learning statistics
    training_step_count = 0
    total_episodes_completed = 0
    episode_metrics_buffer = []  # Buffer to collect episode metrics for batch logging
    
    for i in tqdm(range(start, config.num_train_steps), initial=start):
        
      for subenv in env.envs:
        subenv.index_seed_steps = i
      # env._subtask = 1 # Reset subtask to 0 at the beginning of each step.
            
      # Subtask Exploration while in the beginning of the training.   
      
      # Block and free exploration
      # if i == 30_000 or i == 900_000 or i == 1_500_000:
      #   activated_subtask_experiment = True
          
      # if activated_subtask_experiment:
      #   if i >= 300_000 and i < 600_000:
      #       env._subtask = 1
      #   elif i >= 900_000 and i < 1_200_000:
      #       env._subtask = 2
      #   elif i >= 1_500_000 and i < 1_800_000:
      #       env._subtask = 3
      #   elif i == 600_000 or i == 1_200_000 or i == 1_800_000:
      #       activated_subtask_experiment = False
      #       env._subtask = 0
      #   else:
      #       env._subtask = 0
      
      # # ConsecutionBlocks      
      # if i == 30_000:
      #   activated_subtask_experiment = True
          
      # if activated_subtask_experiment:
      #   if i >= 300_000 and i < 600_000:
      #       env._subtask = 1
      #   elif i >= 600_000 and i < 900_000:
      #       env._subtask = 2
      #   elif i >= 900_000 and i < 1_200_000:
      #       env._subtask = 3
      #   elif i == 1_200_000:
      #       activated_subtask_experiment = False
      #       env._subtask = 0
      #   else:
      #       env._subtask = 0
      
      # Pretrained Subtask Exploration
      # if activated_subtask_experiment:
      #   if i > 25_000 and i <= 50_000:
      #       env._subtask = 1
      #   elif i > 50_000 and i <= 75_000:
      #       env._subtask = 2
      #   elif i > 75_000 and i <= 100_000:
      #       env._subtask = 3
      #   elif i > 100_000:
      #       activated_subtask_experiment = False
      #       env._subtask = 0
      #   else:
      #       env._subtask = 0
        
            
          
      if i < config.num_seed_steps:
        #Pretrain Subtask Exploration
        # activated_subtask_experiment = True
        actions = [env.single_action_space.sample() for _ in range(env.num_envs)]
      else:
        policy.eval()
        actions = []
        for j in range(env.num_envs):
          # Add noise to policy actions for better exploration in vector envs
          action = policy.act(observations[j], sample=True)
          # Add small amount of noise for diversity between environments
          if np.random.random() < 0.1:  # 10% chance of random action
            action = env.single_action_space.sample()
          actions.append(action)
      
      # Step all environments
      next_observations, rewards, dones, infos = _step_unpack(env.step(actions))
      episode_rewards += rewards
      
      # Handle automatic reset for done environments
      # Note: SyncVectorEnv should auto-reset, but let's be explicit about observation handling
      reset_indices = np.where(dones)[0]
      if len(reset_indices) > 0:
        # The next_observations already contain the reset observations for done envs
        pass  # SyncVectorEnv handles this automatically
      
      # Randomize the order of environment processing to reduce correlation
      env_indices = np.random.permutation(env.num_envs)
        
      for j in env_indices:
        observation = observations[j]
        action = actions[j]
        reward = rewards[j]
        next_observation = next_observations[j]
        done = dones[j]
        mask = 0.0 if done else 1.0
        
        # Insert into replay buffer
        if config.reward_wrapper.type == "env":
          buffer.insert(observation, action, reward, next_observation, mask)
        else:
          # For learned rewards, we need pixels from the single environment
          pixels = env.envs[j].render(mode="rgb_array") if hasattr(env.envs[j], 'render') else None
          if pixels is not None:
            buffer.insert(observation, action, reward, next_observation, mask, pixels)
          else:
            buffer.insert(observation, action, reward, next_observation, mask)
        
        # Handle episode completion for this specific environment
        if done:
          total_episodes_completed += 1
          if "holdr" in config.reward_wrapper.type:
            buffer.reset_state()
          if hasattr(env.envs[j], 'reset_state'):
              env.envs[j].reset_state()

          # Collect episode metrics for batch logging
          info_j = _get_vector_info(infos, j)
          if isinstance(info_j, dict) and "episode" in info_j:
            episode_metrics_buffer.append(info_j["episode"])
            if "eval_score" in info_j:
              episode_metrics_buffer[-1]["eval_score"] = info_j["eval_score"]
          episode_rewards[j] = 0.0
          
          # Debug: Print episode completion
          if total_episodes_completed % 100 == 0:  # Print every 100 episodes
            print(f"[INFO] Step {i}, Env {j} completed episode #{total_episodes_completed}, training steps: {training_step_count}")
      
      # Update observations for next iteration
      observations = next_observations
      
      # For vector environments, adjust training frequency to maintain same sample efficiency
      # Train every N steps where N = num_envs to match single environment sample efficiency
      should_train = (i >= config.num_seed_steps) and ((i + 1) % env.num_envs == 0)
      
      if should_train:
        policy.train()
        # For vector environments, we should train less frequently to match single env sample efficiency
        # Train only once per step, not once per environment
        train_info = policy.update(buffer, i)  # Non-DDP case
        training_step_count += 1

        if (i + 1) % config.log_frequency == 0:
          # Log training metrics
          for k, v in train_info.items():
            _log_scalar_safe(logger, v, i, k, "training")
          
          # Log episode metrics collected since last checkpoint
          if episode_metrics_buffer:
            episode_dict = collections.defaultdict(list)
            for metrics in episode_metrics_buffer:
              for k, v in metrics.items():
                episode_dict[k].append(v)
            wlog_dict = {}
            for k, v in episode_dict.items():
              scalars = [_to_float_scalar(item) for item in v]
              scalars = [s for s in scalars if s is not None and np.isfinite(s)]
              if scalars:
                wlog_dict[f"train_done/{k}"] = float(np.mean(scalars))
            episode_metrics_buffer = []
          else:
            wlog_dict = {}
          
          # Add training stats
          if FLAGS.wandb:
            wlog_dict.update({f"train/{k}": v for k, v in train_info.items()})
            wlog_dict.update({
              "train/training_step_count": training_step_count,
              "train/total_episodes_completed": total_episodes_completed,
              "train/step": i,
            })
            wandb.log(wlog_dict, step=i)
          logger.flush()
          
          # Print training progress
          #print(f"[TRAINING] Step {i}, Training steps: {training_step_count}, Episodes: {total_episodes_completed}, Avg reward: {np.mean(episode_rewards):.3f}")

      if (i + 1) % config.eval_frequency == 0:
        eval_stats, eval_episode_rewards = evaluate(policy, eval_env, config.num_eval_episodes, global_step=i)
        for k, v in eval_stats.items():
          _log_scalar_safe(logger, v, i, f"average_{k}s", "evaluation")
        if FLAGS.wandb:
          wlog_dict = {f"eval/{k}": v for k, v in eval_stats.items()}
          safe_rewards = [_to_float_scalar(r) for r in eval_episode_rewards if _to_float_scalar(r) is not None]
          if safe_rewards:
            wlog_dict["eval/mean_episode_reward"] = float(np.mean(safe_rewards))
          # Add mean_eval_score if available
          if "eval_score" in eval_stats:
            wlog_dict["eval/mean_eval_score"] = eval_stats["eval_score"]
          wlog_dict["eval/step"] = i
          wandb.log(wlog_dict, step=i)
        logger.flush()

      if (i + 1) % config.checkpoint_frequency == 0:
        checkpoint_manager.save(i)

  except KeyboardInterrupt:
    print("Caught keyboard interrupt. Saving before quitting.")

  finally:
    checkpoint_manager.save(i)  # pylint: disable=undefined-loop-variable
    logger.close()


if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_name")
  flags.mark_flag_as_required("env_name")
  app.run(main)
