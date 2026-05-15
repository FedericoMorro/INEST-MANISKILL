"""Compute and visualize learned reward signal based on goal embeddings."""

"""
Example usage:

python inest_irl/utils/compute_learned_return.py
    --experiment_path ../data/inest-maniskill/_experiments/pretrain/render-cam/
    [--cache_only]
    [--overwrite]
    [--data_root ../data/inest-maniskill/datasets/dataset-rc-1000-states]
    [--plot_subgoal_dists]


# for different trajs
python inest_irl/utils/compute_learned_return.py
    --experiment_path ../data/inest-maniskill/_experiments/pretrain/render-cam/
    --diff_trajs_dataset ../data/inest-maniskill/different-trajs/
    [--data_root ../data/inest-maniskill/datasets/dataset-rc-1000-states]
"""

import os
import typing
from pathlib import Path

import argparse
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import torch
from torchkit import CheckpointManager
from tqdm.auto import tqdm

from inest_irl.maniskill3.stack_pyramid import MAX_SUBGOAL
from inest_irl.utils.utils import load_config_from_dir
from xirl import common
from xirl.models import SelfSupervisedModel


ModelType = SelfSupervisedModel
DataLoaderType = typing.Dict[str, torch.utils.data.DataLoader]


C_VALUE = 0.25   # additional reward for reaching any subgoal
DISTANCE_THRESHOLDS = [0.5, 0.5, 0.5, 0.5]  # distance threshold for considering a subgoal reached (in embedding space)
PATIENCE_THRESHOLD = 2  # number of consecutive timesteps below distance threshold to consider subgoal reached

# report-friendly plotting defaults (compact figure size with readable text)
FIGSIZE_TRAJ = (7.0, 3.6)
FIGSIZE_MEAN = (7.0, 3.6)
FIGSIZE_HIST = (7.0, 2.8)
FS_LABEL = 12
FS_TITLE = 13
FS_LEGEND = 10
FS_TRAJ_TITLE = 12

DEBUG = 0

#DISTANCE_THRESHOLDS = [3, 3, 3, 3]
#   --experiment_path ../data/inest-maniskill/_experiments/pretrain/render-cam/

#DISTANCE_THRESHOLDS = [3.1, 10.0, 6.4, 8.2]
#   --experiment_path ../data/inest-maniskill/_experiments/pretrain/render-cam/ \
#   --diff_trajs_dataset ../data/inest-maniskill/experiment_data-trajs/full_disc-0.9_4step_high*22*best_model/dataset

#DISTANCE_THRESHOLDS = [3.0, 3.7, 3.0, 3.5]
#   --experiment_path ../data/inest-maniskill/_experiments/pretrain/rc1000-b32/


def setup_from_pretrain(experiment_path, use_cpu, diff_dataset_path=None, data_root=None):
  """Load the latest embedder checkpoint and dataloaders"""

  config = load_config_from_dir(experiment_path)
  model = common.get_model(config)
  
  device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
  model.to(device).eval()
  
  # try to load checkpoint
  checkpoint_dir = os.path.join(experiment_path, "checkpoints")
  if os.path.exists(checkpoint_dir):
    try:
      checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
      global_step = checkpoint_manager.restore_or_initialize()
      print(f"Restored model from checkpoint {global_step}")
    except Exception as e:
      print(f"Failed to load checkpoint: {e}")
  else:
    print("Skipping checkpoint restore (not found or disabled).")
    
  # override data root in config if provided as argument
  if data_root is not None:
    print(f"Overriding data root in config with provided argument: {data_root}")
    config.data.root = data_root
  
  # load data -> debug active if use_cpu, otherwise use GPU-optimized dataloader settings
  train_loader = common.get_downstream_dataloaders(config, debug=use_cpu)["train"]
  #! note batch_size=1 is enforced in the dataloader
  
  # search for subgoal_frames.json to plot also subgoal rewards
  subgoal_frames_path = Path(config.data.root) / "subgoal_frames.json"
  if subgoal_frames_path.exists():
    with open(subgoal_frames_path, 'r') as f:
      train_subgoal_frames = json.load(f)
    print(f"Found subgoal frames file with {len(train_subgoal_frames)} trajectories - will compute and plot subgoal rewards")
  else:
    train_subgoal_frames = None
    print("No subgoal frames file found - will only compute and plot rewards to final goal")

  # if a different trajectory dataset path is provided, load it instead of the validation set for evaluation
  if diff_dataset_path is not None:
    print(f"Loading different trajectory dataset from: {diff_dataset_path}")
    config.data.root = diff_dataset_path
    
    subgoal_frames_path = Path(config.data.root) / "subgoal_frames.json"
    if subgoal_frames_path.exists():
      with open(subgoal_frames_path, 'r') as f:
        valid_subgoal_frames = json.load(f)
      print(f"Found subgoal frames file for different trajectory dataset with {len(valid_subgoal_frames)} trajectories - will compute and plot subgoal rewards")
    else:
      valid_subgoal_frames = None
      
  else:
    print("No different trajectory dataset provided - using validation set for evaluation")
    valid_subgoal_frames = train_subgoal_frames
    
  valid_loader = common.get_downstream_dataloaders(config, debug=use_cpu)["valid"]
  
  return model, train_loader, valid_loader, train_subgoal_frames, valid_subgoal_frames, global_step, device


def compute_goal_embedding(model, train_loader, subgoal_frames, device):
  """Compute the mean goal embedding from the last frames of trajectories"""
  init_embs, goal_embs, subgoal_embs_list = [], [], []

  # get init, goal (final), and subgoals embeddings for each trajectory in the training set
  for class_name, class_loader in train_loader.items():
    for batch in tqdm(iter(class_loader), leave=True, desc=f"Embedding {class_name}"):
      out = model.infer(batch["frames"].to(device))   # batch_size=1 since hardcoded in downstream dataloader
      emb = out.numpy().embs  # shape: (seq_len, embedding_dim)
      
      init_embs.append(emb[0, :])   # first frame embedding
      goal_embs.append(emb[-1, :])  # last frame embedding

      if subgoal_frames is not None:
        traj_id = batch["video_name"][0].split('/')[-1]  # video name should be in format .../../video_id
        
        # skip if trajectory ID not in subgoal_frames (data mismatch)
        if traj_id not in subgoal_frames:
          print(f"Warning: Trajectory ID {traj_id} not found in subgoal frames data - skipping subgoal embedding for this trajectory")
          continue
        
        subgoal_idxs = subgoal_frames[traj_id]

        # if empty list, add empty lists inside with the length of the number of subgoals
        if len(subgoal_embs_list) == 0:
          for _ in range(MAX_SUBGOAL):
            subgoal_embs_list.append([])

        # add subgoal embeddings to the corresponding subgoal index list
        for i, idx in enumerate(subgoal_idxs):
          if idx >= emb.shape[0]:  # sanity check for subgoal index out of bounds
            print(f"Warning: Subgoal index {idx} for trajectory {traj_id} is out of bounds (trajectory length {emb.shape[0]}) - skipping this subgoal")
            continue
          subgoal_embs_list[i].append(emb[idx, :])  # subgoal frame embedding
  
  # compute mean goal embedding and distance scale
  goal_emb = np.mean(np.stack(goal_embs, axis=0), axis=0, keepdims=True)
  dist_to_goal = np.linalg.norm(np.stack(init_embs, axis=0) - goal_emb, axis=1).mean()
  dist_scale = 1.0 / (dist_to_goal + 1e-8)

  # compute mean subgoal embeddings if subgoal frames are provided
  if subgoal_frames is not None:
    subgoal_embs = []
    for traj_subgoal_embs in subgoal_embs_list:
      subgoal_embs.append(np.mean(np.stack(traj_subgoal_embs, axis=0), axis=0, keepdims=True))
  else:
    subgoal_embs = None
    print("WARNING: No subgoal embeddings computed")
  
  # add subgoal info for pickling, used by wrapper in rl training
  subgoal_info = {
    "c_value": C_VALUE,
    "distance_thresholds": DISTANCE_THRESHOLDS,
    "patience_threshold": PATIENCE_THRESHOLD,
  }
  
  return goal_emb, subgoal_embs, dist_scale, subgoal_info


def compute_reward_metrics(rewards: np.array) -> dict[str, float]:
  if len(rewards) == 0:
    return {}
  
  metrics = {
    'final_value': float(rewards[-1]),
    'cumulative': float(np.sum(rewards)),
    'mean': float(np.mean(rewards)),
    'stdev': float(np.std(rewards)),
    'min': float(np.min(rewards)),
    'max': float(np.max(rewards)),
    'range': float(np.max(rewards) - np.min(rewards))
  }
  
  # smoothness: variance of first differences (lower -> smoother)
  first_diff = np.diff(rewards)
  metrics['smoothness'] = float(np.var(first_diff))
  metrics['mean_abs_change'] = float(np.mean(np.abs(first_diff)))
  
  # monotonicity: fraction of steps where reward increases
  increases = np.sum(first_diff > 0)
  metrics['monotonicity'] = float(increases / len(first_diff))
  metrics['is_monotonic_increasing'] = bool(np.all(first_diff >= 0))
  
  # trend: linear regression slope
  x = np.arange(len(rewards))
  coeffs = np.polyfit(x, rewards, deg=1)
  metrics['trend_slope'] = float(coeffs[0])
  
  # reward improvement final-initial
  metrics['improvement'] = float(rewards[-1] - rewards[0])
  
  return metrics

def compute_avg_reward_metrics(all_rewards: list[np.array]) -> dict[str, float]:
  all_metrics = [compute_reward_metrics(rew) for rew in all_rewards]
  
  # aggregate each metric across trajectories
  avg_metrics = {}
  for key in all_metrics[0].keys():
    values = [m[key] for m in all_metrics if key in m]
    avg_metrics[key] = float(np.mean(values)) if len(values) > 0 else float('nan')
  
  return avg_metrics

def save_reward_metrics(traj_rews: dict[int, np.array], output_file: str, avg: bool = False):
  # if avg, compute avg metrics across trajs and save single dict
  if avg:
    rewards = [traj_rw for traj_rw in traj_rews.values()]
    metrics = compute_avg_reward_metrics(rewards)
    with open(output_file, 'w') as f:
      json.dump(metrics, f, indent=2)
    return
    
  # otherwise compute metrics for each traj and save dict of dicts
  metrics_dict = {}  
  for traj_id in traj_rews:
    metrics_dict[traj_id] = compute_reward_metrics(traj_rews[traj_id])
  with open(output_file, 'w') as f:
    json.dump(sorted(metrics_dict.items(), key=lambda x: x[0]), f, indent=2)
    
    
def to_json_serializable(obj):
  if isinstance(obj, np.ndarray):
    return [to_json_serializable(x) for x in obj.tolist()]
  elif isinstance(obj, dict):
    return {to_json_serializable(key): to_json_serializable(value) for key, value in obj.items()}
  elif obj == np.nan:
    return None
  elif isinstance(obj, (np.float32, np.float64)):
    return float(obj)
  elif isinstance(obj, (np.int32, np.int64)):
    return int(obj)
  else:
    return obj


class TrajectoryLearnedReward:
  def __init__(self,
               rewards: np.array,
               subgoal_rewards: np.array = None,
               subgoal_dists: list[np.array] = None,
               subgoal_reachs: list[int] = None,
               subgoal_reachs_gt: list[int] = None
  ):
    self.rewards = rewards
    self.subgoal_rewards = subgoal_rewards
    self.subgoal_dists = subgoal_dists
    self.subgoal_reachs = subgoal_reachs
    self.subgoal_reachs_gt = subgoal_reachs_gt
  
  def to_map_dict(self, keys: list[(str, str)]):
    return {new_key: getattr(self, old_key) for old_key, new_key in keys}
  
  def to_str(self):
    return f"Rewards shape: {self.rewards.shape if self.rewards is not None else 'N/A'}   " + \
           f"Subgoal Rewards shape: {self.subgoal_rewards.shape if self.subgoal_rewards is not None else 'N/A'}   " + \
           f"Subgoal Dists shape: {self.subgoal_dists.shape if self.subgoal_dists is not None else 'N/A'}   " + \
           f"Subgoal Reachs: {self.subgoal_reachs}   Subgoal Reachs GT: {self.subgoal_reachs_gt}"
           
  
class DatasetLearnedReward:
  def __init__(self):
    self.traj_lrs: list[TrajectoryLearnedReward] = {}
    
  def add_traj(self, traj_id, traj_lr):
    self.traj_lrs[traj_id] = traj_lr
    
  def __len__(self):
    return len(self.traj_lrs)
  
  def traj_lens(self):
    return [len(traj_lr.rewards) for traj_lr in self.traj_lrs.values()]
  
  
  def reward_metrics_to_file(self, out_dir):

    traj_rews = {traj_id: traj_lr.rewards for traj_id, traj_lr in self.traj_lrs.items()} 
    save_reward_metrics(traj_rews, os.path.join(out_dir, 'reward_metrics.json'), avg=True)
    save_reward_metrics(traj_rews, os.path.join(out_dir, 'reward_metrics_per_traj.json'), avg=False)
  
    if any(traj_lr.subgoal_rewards is not None for traj_lr in self.traj_lrs.values()):
      traj_subgoal_rews = {traj_id: traj_lr.subgoal_rewards for traj_id, traj_lr in self.traj_lrs.items()}
      save_reward_metrics(traj_subgoal_rews, os.path.join(out_dir, 'subgoal_reward_metrics.json'), avg=True)
      save_reward_metrics(traj_subgoal_rews, os.path.join(out_dir, 'subgoal_reward_metrics_per_traj.json'), avg=False)

      
  def subgoal_idxs_to_file(self, out_dir):
    
    def _cum_subgoal_count(traj_subgoal_idxs):
      cum_count = [0 for _ in range(MAX_SUBGOAL)]
      for traj_id in traj_subgoal_idxs:
        if traj_subgoal_idxs[traj_id] is not None:
          for i, idx in enumerate(traj_subgoal_idxs[traj_id]):
            if idx is not None and idx >= 0:
              cum_count[i] += 1
      return cum_count
    
    # compute avg difference in reaching steps between GT and detected subgoals
    avg_step_diffs, count_step_diffs = {i: 0 for i in range(MAX_SUBGOAL)}, {i: 0 for i in range(MAX_SUBGOAL)}
    for traj_id in self.traj_lrs:
      traj_lr = self.traj_lrs[traj_id]
      if traj_lr.subgoal_reachs is not None and traj_lr.subgoal_reachs_gt is not None:
        for subgoal_idx, (detected, gt) in enumerate(zip(traj_lr.subgoal_reachs, traj_lr.subgoal_reachs_gt)):
          if detected is not None and gt is not None and not np.isnan(detected) and not np.isnan(gt):
            avg_step_diffs[subgoal_idx] += abs(detected - gt)
            count_step_diffs[subgoal_idx] += 1
    for i in range(MAX_SUBGOAL):
      avg_step_diffs[i] = avg_step_diffs[i] / (count_step_diffs[i] + 1e-8)
      avg_step_diffs[f"cnt_{i}"] = count_step_diffs[i]

    # append cumulative counts, then per-traj detected and GT subgoal reaching steps in a dict and save as json
    out_data = {
      "cumulative_subgoal_count": _cum_subgoal_count({traj_id: traj_lr.subgoal_reachs for traj_id, traj_lr in self.traj_lrs.items()}),
      "cumulative_subgoal_count_gt": _cum_subgoal_count({traj_id: traj_lr.subgoal_reachs_gt for traj_id, traj_lr in self.traj_lrs.items()}),
      "avg_step_diffs": avg_step_diffs,
    }
    for traj_id in sorted(self.traj_lrs.keys()):
      out_data[traj_id] = {
        "detected": self.traj_lrs[traj_id].subgoal_reachs,
        "gt": self.traj_lrs[traj_id].subgoal_reachs_gt,
      }
    
    with open(os.path.join(out_dir, 'subgoal_idxs.json'), 'w') as f:
      json.dump(to_json_serializable(out_data), f, indent=2)
      
      
  def to_map_dict(self, keys: list[(str, str)]):
    out = {}
    for traj_id, traj_lr in self.traj_lrs.items():
      out[traj_id] = traj_lr.to_map_dict(keys)
    return out
      
  def plot_results(self, output_dir, mean=True, count=None, rewards_flag=True, subgoal_rewards_flag=True, distances_flag=True):
    if rewards_flag:
      in_dict = self.to_map_dict([("rewards", "values")])
      if mean:
        plot_mean_results(in_dict, output_dir, label="Learned Reward")
      else:
        plot_trajectory_samples(in_dict, output_dir, label="Learned Reward", count=count)
  
    if subgoal_rewards_flag and any(traj_lr.subgoal_rewards is not None for traj_lr in self.traj_lrs.values()):
      in_dict = self.to_map_dict([("subgoal_rewards", "values"), ("subgoal_reachs", "subgoal_reachs"), ("subgoal_reachs_gt", "subgoal_reachs_gt")])
      if mean:
        plot_mean_results(in_dict, output_dir, label="Subgoal Reward", show_subgoals=MAX_SUBGOAL)
      else:
        plot_trajectory_samples(in_dict, output_dir, label="Subgoal Reward", count=count, show_subgoals=MAX_SUBGOAL)
      
    if distances_flag and any(traj_lr.subgoal_dists is not None for traj_lr in self.traj_lrs.values()):
      for i in range(MAX_SUBGOAL):
        in_dict = self.to_map_dict([("subgoal_dists", "values"), ("subgoal_reachs", "subgoal_reachs"), ("subgoal_reachs_gt", "subgoal_reachs_gt")])
        for traj_id in in_dict:
          in_dict[traj_id]["values"] = in_dict[traj_id]["values"][i]  # select the distance to the i-th subgoal distance
        if mean:
          plot_mean_results(in_dict, output_dir, label=f"Distance to Subgoal {i}", show_subgoals=i, show_intersection=True)
        else:
          plot_trajectory_samples(in_dict, output_dir, label=f"Distance to Subgoal {i}", count=count, show_subgoals=i, show_intersection=True)
  
           
def compute_reward_signals(model, valid_loader, goal_emb, subgoal_embs, dist_scale, device, subgoal_frames=None):
  """Compute the negative distance to goal as reward signal for each trajectory"""
  
  dataset_lr = DatasetLearnedReward()
  
  for class_name, class_loader in valid_loader.items():
    for batch in tqdm(iter(class_loader), leave=True, desc=f"Computing rewards for {class_name}"):
      traj_id = batch["video_name"][0].split('/')[-1]
      out = model.infer(batch["frames"].to(device))
      traj_emb = out.numpy().embs  # shape: (seq_len, embedding_dim)
      
      # compute euclidean distance from each frame to goal
      dist = np.linalg.norm(goal_emb - traj_emb, axis=1)  # shape: (seq_len,)

      # learned reward is negative distance to goal, scaled by the average distance to goal in the training set
      rewards = - dist * dist_scale

      # compute subgoal rewards if subgoal embeddings are provided
      if subgoal_embs is not None:
    
        # get ground truth subgoal indices from data, or from subgoal frames if available
        gt_subgoal_idxs = []
        if subgoal_frames is not None and traj_id in subgoal_frames:
          gt_subgoal_idxs = subgoal_frames[traj_id]
        elif Path(f"{batch['video_name'][0]}/{traj_id}_subgoal_idxs.json").exists():
          with open(Path(f"{batch['video_name'][0]}/{traj_id}_subgoal_idxs.json"), 'r') as f:
            gt_subgoal_idxs = json.load(f)
            
        if len(gt_subgoal_idxs) < MAX_SUBGOAL:
          for _ in range(MAX_SUBGOAL - len(gt_subgoal_idxs)):
            gt_subgoal_idxs.append(np.nan)
            
        subgoal_dists = []
        for subgoal_emb in subgoal_embs:
          subgoal_dists.append(np.linalg.norm(subgoal_emb - traj_emb, axis=1))  # shape: (seq_len,)

        # add them progressively at task completion
        subgoal_rewards = []
        subgoal_identifs = [np.nan for _ in range(len(subgoal_embs))]
        subgoal_idx = 0
        patience = 0
        for t in range(len(dist)):
          # manage patience counter
          if subgoal_idx < len(subgoal_embs) and subgoal_dists[subgoal_idx][t] < DISTANCE_THRESHOLDS[subgoal_idx]:
            if patience >= PATIENCE_THRESHOLD:
              subgoal_idx = subgoal_idx + 1
              subgoal_identifs[subgoal_idx - 1] = t
              patience = 0
            else:
              patience = patience + 1
          else:
            patience = 0

          subgoal_rewards.append(rewards[t] + C_VALUE * subgoal_idx)   # add C_VALUE for each subgoal reached
          
      dataset_lr.add_traj(
        traj_id=traj_id,
        traj_lr=TrajectoryLearnedReward(
          rewards=np.array(rewards),
          subgoal_rewards=np.array(subgoal_rewards) if subgoal_embs is not None else None,
          subgoal_dists=np.array(subgoal_dists) if subgoal_embs is not None else None,
          subgoal_reachs=np.array(subgoal_identifs) if subgoal_embs is not None else None,
          subgoal_reachs_gt=np.array(gt_subgoal_idxs) if gt_subgoal_idxs is not None else None,
      ))
      
      if DEBUG:
        print(f"Inserted trajectory {traj_id}:\n{dataset_lr.traj_lrs[traj_id].to_str()}")
        
  return dataset_lr


def pad_trajectories(rewards):
  """Pad trajectories to same length for easier analysis"""
  lengths = np.array([len(r) for r in rewards])
  max_length = int(np.max(lengths))
  
  padded_rewards = np.full((len(rewards), max_length), np.nan)
  for i, reward in enumerate(rewards):
    padded_rewards[i, :len(reward)] = reward
  
  return padded_rewards, lengths


def _sanitize_plot_type(label):
  normalized = label.lower().replace('learned', ' ')
  normalized = normalized.replace('distance', 'dist')
  normalized = normalized.replace('subgoal', 'sub')
  normalized = normalized.replace('reward', 'rew')
  normalized = normalized.replace('negative', 'neg')
  normalized = normalized.replace('trajectory', 'traj')
  normalized = normalized.replace('(', '').replace(')', '')
  normalized = "_".join(normalized.split())
  return normalized.strip('_')


def is_not_nan_or_none(value):
  return value is not None and not np.isnan(value) and value != float('nan')


def plot_trajectory_samples(traj_infos: dict[int, dict], output_dir, label="Learned Reward", count=None, show_subgoals=-1, show_intersection=False):
  """Save one plot per trajectory in the eval set (optionally capped by count)."""
  plot_type = _sanitize_plot_type(label)
  
  num_trajs = len(traj_infos)  
  num_to_plot = num_trajs if count is None or count < 0 else min(num_trajs, count)
  
  print(f"Plotting trajectory-wise curves of {label.lower()}...")
  cnt = 0
  tqdm_bar = tqdm(total=num_to_plot, desc=f"Plotting {plot_type}")
  
  for traj_id, traj_data in traj_infos.items():
    if cnt >= num_to_plot:
      break
    
    values = traj_data["values"]
    subgoal_reachs = traj_data.get("subgoal_reachs", None)
    subgoal_reachs_gt = traj_data.get("subgoal_reachs_gt", None)
    
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_TRAJ)
    ax.plot(range(len(values)), values, 'b-', linewidth=2, label=label, marker='o', markersize=3)
    
    # show only the specified subgoal
    if 0 <= show_subgoals < MAX_SUBGOAL:
      if subgoal_reachs is not None and len(subgoal_reachs) > show_subgoals and is_not_nan_or_none(subgoal_reachs[show_subgoals]):
        ax.axvline(subgoal_reachs[show_subgoals], color='green', linestyle='--', linewidth=1.7, alpha=0.9, label=f'Reached Subgoal {show_subgoals} (Detected)')
        if show_intersection:
          ax.plot(subgoal_reachs[show_subgoals], values[int(subgoal_reachs[show_subgoals])], 'o', color='green', markersize=4, zorder=5)
      if subgoal_reachs_gt is not None and len(subgoal_reachs_gt) > show_subgoals and is_not_nan_or_none(subgoal_reachs_gt[show_subgoals]):
        ax.axvline(subgoal_reachs_gt[show_subgoals], color='purple', linestyle=':', linewidth=1.7, alpha=0.9, label=f'Reached Subgoal {show_subgoals} (GT)')
        if show_intersection:
          ax.plot(subgoal_reachs_gt[show_subgoals], values[int(subgoal_reachs_gt[show_subgoals])], 'o', color='purple', markersize=4, zorder=5)
    # show all subgoals 
    elif show_subgoals >= MAX_SUBGOAL:
      if subgoal_reachs is not None:
        for j, subgoal_step in enumerate(subgoal_reachs):
          if is_not_nan_or_none(subgoal_step):
            ax.axvline(subgoal_step, color='green', linestyle='--', linewidth=1.7, alpha=0.9, label=f'Reached Subgoal {j} (Detected)' if j == 0 else None)
      if subgoal_reachs_gt is not None:
        for j, subgoal_step in enumerate(subgoal_reachs_gt):
          if is_not_nan_or_none(subgoal_step):
            ax.axvline(subgoal_step, color='purple', linestyle=':', linewidth=1.7, alpha=0.9, label=f'Reached Subgoal {j} (GT)' if j == 0 else None)
      
    ax.set_xlabel('Timestep', fontsize=FS_LABEL)
    ax.set_ylabel(label, fontsize=FS_LABEL)
    ax.set_title(f'Trajectory {traj_id} (Length: {len(values)})', fontsize=FS_TRAJ_TITLE, fontweight='bold')
    ax.grid(True)
    ax.legend(fontsize=FS_LEGEND)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{traj_id}_{plot_type}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    tqdm_bar.update(1) 


def plot_mean_results(traj_infos: dict[int, dict], output_dir, label="Learned Reward", show_subgoals=-1, show_intersection=False):
  """Plot mean reward per timestep and return padded trajectories info."""
  
  values = [traj_data["values"] for traj_data in traj_infos.values()]
  subgoal_reachs = [traj_data.get("subgoal_reachs", None) for traj_data in traj_infos.values()]
  subgoal_reachs_gt = [traj_data.get("subgoal_reachs_gt", None) for traj_data in traj_infos.values()]
  
  # pad trajectories for analysis
  padded_values, lengths = pad_trajectories(values)
  
  # mean reward per timestep
  print(f"Plotting mean reward per timestep of {label.lower()}...")
  fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_MEAN)
  
  # calculate mean and std per timestep (ignoring NaNs)
  mean_values = np.nanmean(padded_values, axis=0)
  std_values = np.nanstd(padded_values, axis=0)
  min_values = np.nanmin(padded_values, axis=0)
  timesteps = np.arange(len(mean_values))
  
  ax.plot(timesteps, mean_values, 'b-', linewidth=2, label='Mean Reward')
  ax.fill_between(timesteps, mean_values - std_values, mean_values + std_values, alpha=0.3, label='+-1 Std Dev')
  
  # get y-range for plotting markers
  y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
  y_min, y_max = ax.get_ylim()
  y_high_mark, y_sub_high_mark = y_max - 0.1 * y_range, y_max - 0.2 * y_range
  y_low_mark, y_sub_low_mark = y_min + 0.05 * y_range, y_min + 0.15 * y_range
  
  # find and mark minimum point for distance plots
  if 0 <= show_subgoals < MAX_SUBGOAL:
    min_idx = np.nanargmin(mean_values)
    min_val = mean_values[min_idx]
    min_std = std_values[min_idx]
    ax.plot(min_idx, min_val, 'o', color='red', markersize=4, label='Min Avg Dist', zorder=5)
    ax.text(min_idx, y_low_mark, f'{min_val:.3f}\n±{min_std:.3f}', 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
    
    # also plot minimum line across timesteps for reference
    #ax.plot(min_values, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7, label='Min Distance')

  # aggregate each subgoal across trajectories and mark mean reaching timestep
  # plot GT reaching steps if available
  if any(srg is not None for srg in subgoal_reachs_gt) and len(subgoal_reachs_gt) > 0:
    subgoal_reachs_gt = np.array([srg if srg is not None else [np.nan for _ in range(MAX_SUBGOAL)] for srg in subgoal_reachs_gt])
    mean_subgoal_reachs_gt = np.nanmean(subgoal_reachs_gt, axis=0)  # shape: (num_subgoals,)
    count_subgoal_reachs_gt = np.sum(~np.isnan(subgoal_reachs_gt), axis=0)
    
    # show only the specified subgoal
    if 0 <= show_subgoals < MAX_SUBGOAL:
      ax.axvline(mean_subgoal_reachs_gt[show_subgoals], color='purple', linestyle=':', linewidth=1.7, alpha=0.95, label=f'Subgoal {show_subgoals} (GT)')
      if show_intersection:
        mean_step_int = int(np.round(mean_subgoal_reachs_gt[show_subgoals]))
        if 0 <= mean_step_int < len(mean_values):
          val_at_reach = mean_values[mean_step_int]
          ax.plot(mean_step_int, val_at_reach, 'o', color='purple', markersize=4, zorder=5)
          ax.text(mean_step_int, y_high_mark, f'{val_at_reach:.3f}\n±{std_values[mean_step_int]:.3f}', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))
    # show all subgoals
    elif show_subgoals >= MAX_SUBGOAL:
      for subgoal_i, mean_subgoal_step in enumerate(mean_subgoal_reachs_gt.T):  # iterate over subgoals
        ax.axvline(mean_subgoal_step, color='purple', linestyle=':', linewidth=1.7, alpha=0.95, label='Subgoal (GT)' if subgoal_i == 0 else None)
        ax.text(mean_subgoal_step, y_high_mark, count_subgoal_reachs_gt[subgoal_i], ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))

  # plot detected subgoal reaching steps if available
  if any(sr is not None for sr in subgoal_reachs) and len(subgoal_reachs) > 0:
    subgoal_reachs = np.array([sr if sr is not None else [np.nan for _ in range(MAX_SUBGOAL)] for sr in subgoal_reachs])
    mean_subgoal_reachs = np.nanmean(subgoal_reachs, axis=0)  # shape: (num_subgoals,)
    count_subgoal_reachs = np.sum(~np.isnan(subgoal_reachs), axis=0)
    
    # show only the specified subgoal
    if 0 <= show_subgoals < MAX_SUBGOAL:
      ax.axvline(mean_subgoal_reachs[show_subgoals], color='green', linestyle='--', linewidth=1.7, alpha=0.95, label=f'Subgoal {show_subgoals} (Detected)')
      if show_intersection:
        mean_step_int = int(np.round(mean_subgoal_reachs[show_subgoals]))
        if 0 <= mean_step_int < len(mean_values):
          val_at_reach = mean_values[mean_step_int]
          ax.plot(mean_step_int, val_at_reach, 'o', color='green', markersize=4, zorder=5)
          ax.text(mean_step_int, y_high_mark, f'{val_at_reach:.3f}\n±{std_values[mean_step_int]:.3f}', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    # show all subgoals    elif show_subgoals >= MAX_SUBGOAL:
    elif show_subgoals >= MAX_SUBGOAL:
      for subgoal_i, mean_step in enumerate(mean_subgoal_reachs.T):  # iterate over subgoals
        ax.axvline(mean_step, color='green', linestyle='--', linewidth=1.7, alpha=0.95, label='Subgoal (Detected)' if subgoal_i == 0 else None)
        ax.text(mean_step, y_sub_high_mark, count_subgoal_reachs[subgoal_i], ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

  ax.set_xlabel('Timestep', fontsize=FS_LABEL)
  ax.set_ylabel(label, fontsize=FS_LABEL)
  num_trajs = len(traj_infos)
  ax.set_title(f'Mean {label} per Timestep (N={num_trajs} trajectories)', fontsize=FS_TITLE, fontweight='bold')
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=FS_LEGEND)
  
  plt.tight_layout()
  output_path = os.path.join(output_dir, f'mean-std_{_sanitize_plot_type(label)}.png')
  plt.savefig(output_path, dpi=300, bbox_inches='tight')
  plt.close()


def plot_trajectory_lengths(lengths, output_dir):
  """Plot trajectory length histogram once per evaluation run."""
  print("Plotting trajectory lengths...")
  _fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_HIST)
  ax.hist(lengths, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
  ax.set_xlabel('Length (timesteps)', fontsize=FS_LABEL)
  ax.set_ylabel('Count', fontsize=FS_LABEL)
  ax.set_title('Trajectory Lengths', fontsize=FS_TITLE, fontweight='bold')
  ax.grid(True, alpha=0.3, axis='y')
  plt.tight_layout()
  output_path = os.path.join(output_dir, 'traj-lens.png')
  plt.savefig(output_path, dpi=300, bbox_inches='tight')
  plt.close()


def main(args):
  # set random seeds for reproducibility
  torch.manual_seed(22)
  torch.cuda.manual_seed_all(22)
  np.random.seed(22)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  exp_name = args.experiment_path.strip('/').split('/')[-1]
  out_dir = os.path.join(args.output_dir, exp_name)
  os.makedirs(out_dir, exist_ok=True)

  # if a different trajectory dataset is provided, create a subfolder for its results inside the experiment output directory
  if args.diff_trajs_dataset is not None:
    print(f"Evaluating reward signals on a different trajectory dataset: {args.diff_trajs_dataset}")
    trajs_name = args.diff_trajs_dataset.strip('/').split('/')[-1]
    out_dir = os.path.join(out_dir, trajs_name)
    os.makedirs(out_dir, exist_ok=True)

  trajs_out_dir = os.path.join(out_dir, "trajs")
  os.makedirs(trajs_out_dir, exist_ok=True)

  # setup model and data
  print(f"Loading model from: {args.experiment_path}")
  model, train_loader, valid_loader, train_subgoal_frames, valid_subgoal_frames, global_step, device = setup_from_pretrain(
    args.experiment_path, 
    args.use_cpu, 
    diff_dataset_path=args.diff_trajs_dataset,
    data_root=args.data_root,
  )

  # check for cached results in output directory
  checkpoint_dir = os.path.join(args.experiment_path, "checkpoints")
  cache_path = os.path.join(checkpoint_dir, f"cached_embeddings_step_{global_step}.pkl")
  if not os.path.exists(cache_path) or args.overwrite or args.cache_only:
    print("No cached embedddings found (or overwrite flag is set) - computing from scratch...")

    # compute goal embedding
    goal_emb, subgoal_embs, dist_scale, subgoal_info = compute_goal_embedding(model, train_loader, train_subgoal_frames, device)
    print(f"Goal embedding computed - shape: {goal_emb.shape}")
    if subgoal_embs is not None:
      print(f"Subgoals identified: {len(subgoal_embs)} - shape: {subgoal_embs[0].shape}")

    # save goal and subgoal embeddings to cache for future use
    with open(cache_path, 'wb') as f:
      pickle.dump((goal_emb, subgoal_embs, dist_scale, subgoal_info), f)
    print(f"Saved computed embeddings and distance scale to cache at: {cache_path}")
    
    if args.cache_only:
      print("Cache only flag is set - exiting after caching embeddings.")
      return

  else:
    print(f"Found cached embeddings at {cache_path} - loading...")
    with open(cache_path, 'rb') as f:
      goal_emb, subgoal_embs, dist_scale, subgoal_info = pickle.load(f)
    print(f"Goal embedding loaded - shape: {goal_emb.shape}")
    if subgoal_embs is not None:
      print(f"Subgoals identified: {len(subgoal_embs)} - shape: {subgoal_embs[0].shape}")
  
  # compute reward signals
  dataset_lr = compute_reward_signals(model, valid_loader, goal_emb, subgoal_embs, dist_scale, device, valid_subgoal_frames)
  print(f"Computed {len(dataset_lr)} reward signals")
  
  # save metrics to file
  dataset_lr.reward_metrics_to_file(out_dir)
  dataset_lr.subgoal_idxs_to_file(out_dir)

  # plot trajectory lengths once using base rewards, to understand distribution of trajectory lengths and mean-std plots better
  plot_trajectory_lengths(dataset_lr.traj_lens(), out_dir)
  
  # plot mean results
  dataset_lr.plot_results(out_dir, mean=True)
    
  # plot trajectory-wise curves
  if args.no_plot_trajs:
    print("Skipping trajectory-wise plots for learned rewards (no_plot_trajs flag is set)")
    
  if args.no_plot_subgoal_trajs:
    print("Skipping trajectory-wise plots for subgoal rewards (no_plot_subgoal_trajs flag is set)")
    
  if not args.plot_subgoal_dists:
    print("Skipping trajectory-wise curves for distances to subgoals (plot_subgoal_dists flag is not set)")
    
  dataset_lr.plot_results(trajs_out_dir, mean=False, count=args.count,
                          rewards_flag=not args.no_plot_trajs,
                          subgoal_rewards_flag=not args.no_plot_subgoal_trajs,
                          distances_flag=args.plot_subgoal_dists)

  print(f"All results saved to: {out_dir}")


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser(description="Evaluate learned reward signals based on goal embeddings")
  arg_parser.add_argument("--experiment_path", type=str, required=True,
                          help="Path to the pretraining experiment directory")
  arg_parser.add_argument("--data_root", type=str, default=None,
                          help="Optional override for the trajectory dataset root directory (if not provided, will use the one from the experiment config)")
  arg_parser.add_argument("--output_dir", type=str, default="/home/fmorro/INEST-MANISKILL/out/reward_plots",
                          help="Directory to save the generated plots")
  arg_parser.add_argument("--count", type=int, default=-1,
                          help="Maximum number of trajectory-wise plots to save per plot type (-1 plots all trajectories)")
  arg_parser.add_argument("--use_cpu", action='store_true', default=False,
                          help="Whether to force CPU usage even if GPU is available (if GPU not idle to avoid 'RuntimeError: CUDA out of memory')")
  arg_parser.add_argument("--no_plot_trajs", action='store_true', default=False,
                          help="Whether to save plots for learned rewards")
  arg_parser.add_argument("--no_plot_subgoal_trajs", action='store_true', default=False,
                          help="Whether to save plots for subgoal rewards")
  arg_parser.add_argument("--plot_subgoal_dists", action='store_true', default=False,
                          help="Whether to save plots for distances to subgoals (in embedding space)")
  arg_parser.add_argument("--diff_trajs_dataset", type=str, default=None,
                          help="Optional path to another trajectory dataset directory to check reward signal sanity")
  arg_parser.add_argument("--overwrite", action='store_true', default=False,
                          help="Wheter to overwrite cached embeddings")
  arg_parser.add_argument("--cache_only", action='store_true', default=False,
                          help="Whether to JUST update the cache (no plotting)")
  args = arg_parser.parse_args()

  main(args)
