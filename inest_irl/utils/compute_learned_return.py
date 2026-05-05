"""Compute and visualize learned reward signal based on goal embeddings."""

"""
Example usage:

python inest_irl/utils/compute_learned_return.py
    --experiment_path ../data/inest-maniskill/_experiments/pretrain/render-cam/
    [--cache_only]
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
DISTANCE_THRESHOLDS = [3, 3, 3, 3]  # distance threshold for considering a subgoal reached (in embedding space)
PATIENCE_THRESHOLD = 2  # number of consecutive timesteps below distance threshold to consider subgoal reached

# report-friendly plotting defaults (compact figure size with readable text)
FIGSIZE_TRAJ = (7.0, 3.6)
FIGSIZE_MEAN = (7.0, 3.6)
FIGSIZE_HIST = (7.0, 2.8)
FS_LABEL = 12
FS_TITLE = 13
FS_LEGEND = 10
FS_TRAJ_TITLE = 12

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
    for batch in tqdm(iter(class_loader), leave=False, desc=f"Embedding {class_name}"):
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
  
  # add subgoal info for pickling, used by wrapper in rl training
  subgoal_info = {
    "c_value": 0.25,
    "distance_thresholds": [3.0, 3.0, 3.0, 3.0],
    "patience_threshold": 0,
  }
  
  return goal_emb, subgoal_embs, dist_scale, subgoal_info


def compute_reward_signals(model, valid_loader, goal_emb, subgoal_embs, dist_scale, device, subgoal_frames=None):
  """Compute the negative distance to goal as reward signal for each trajectory"""
  rewards = []
  traj_ids = []
  subgoal_rewards = [] if subgoal_embs is not None else None
  subgoal_reachs = [ [] for _ in range(len(subgoal_embs)) ] if subgoal_embs is not None else None
  subgoal_dists = [ [] for _ in range(len(subgoal_embs)) ] if subgoal_embs is not None else None
  subgoal_reachs_gt = None
  
  for class_name, class_loader in valid_loader.items():
    for batch in tqdm(iter(class_loader), leave=False, desc=f"Computing rewards for {class_name}"):
      traj_id = batch["video_name"][0].split('/')[-1]
      traj_ids.append(traj_id)
      out = model.infer(batch["frames"].to(device))
      traj_emb = out.numpy().embs  # shape: (seq_len, embedding_dim)
      
      # compute euclidean distance from each frame to goal
      dist = np.linalg.norm(goal_emb - traj_emb, axis=1)  # shape: (seq_len,)

      # learned reward is negative distance to goal, scaled by the average distance to goal in the training set
      rewards.append(- dist * dist_scale)

      # compute subgoal rewards if subgoal embeddings are provided
      if subgoal_embs is not None:
        
        # get ground truth subgoal indices from data, or from subgoal frames if available
        subogal_reachs_gt_path = Path(f"{batch['video_name'][0]}/{traj_id}_subgoal_idxs.json")
        if subogal_reachs_gt_path.exists():
          with open(subogal_reachs_gt_path, 'r') as f:
            gt_subgoal_idxs = json.load(f)
        elif subgoal_frames is not None and traj_id in subgoal_frames:
          gt_subgoal_idxs = subgoal_frames[traj_id]
        else:
          gt_subgoal_idxs = None
          
        if gt_subgoal_idxs is not None:          
          if subgoal_reachs_gt is None:
            subgoal_reachs_gt = [ [] for _ in range(len(subgoal_embs)) ]
          for i, idx in enumerate(gt_subgoal_idxs):
            if i < len(subgoal_reachs_gt):
              subgoal_reachs_gt[i].append(idx)
            
        for i, subgoal_emb in enumerate(subgoal_embs):
          subgoal_dists[i].append( 
            np.linalg.norm(subgoal_emb - traj_emb, axis=1)  # shape: (seq_len,)
          )

        # add them progressively at task completion
        curr_subgoal_rewards = []
        curr_subgoal_identifs = [np.nan for _ in range(len(subgoal_embs))]
        subgoal_idx = 0
        patience = 0
        for t in range(len(dist)):
          # manage patience counter
          if subgoal_idx < len(subgoal_embs) and subgoal_dists[subgoal_idx][-1][t] < DISTANCE_THRESHOLDS[subgoal_idx]:
            if patience >= PATIENCE_THRESHOLD:
              subgoal_idx = subgoal_idx + 1
              curr_subgoal_identifs[subgoal_idx - 1] = t
              patience = 0
            else:
              patience = patience + 1
          else:
            patience = 0

          curr_subgoal_rewards.append(rewards[-1][t] + C_VALUE * subgoal_idx)   # add C_VALUE for each subgoal reached

        subgoal_rewards.append(np.array(curr_subgoal_rewards))  
        for i, reach_step in enumerate(curr_subgoal_identifs):
          subgoal_reachs[i].append(reach_step)

  return rewards, traj_ids, subgoal_rewards, subgoal_dists, subgoal_reachs, subgoal_reachs_gt


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
  metrics['is_monotone_increasing'] = bool(np.all(first_diff >= 0))
  
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

def save_reward_metrics(rewards: np.array, output_file: str, avg: bool = False, traj_ids: list[str] = None):
  # if avg, compute avg metrics across trajs and save single dict
  if avg:
    metrics = compute_avg_reward_metrics(rewards)
    with open(output_file, 'w') as f:
      json.dump(metrics, f, indent=2)
    return
    
  # otherwise compute metrics for each traj and save dict of dicts
  metrics_dict = {}  
  for i, rew in enumerate(rewards):
    traj_id = traj_ids[i] if traj_ids is not None and i < len(traj_ids) else f"#{i}"
    metrics_dict[traj_id] = compute_reward_metrics(rew)
    
  with open(output_file, 'w') as f:
    json.dump(sorted(metrics_dict.items(), key=lambda x: x[0]), f, indent=2)
    
    
def save_subgoal_idxs(subgoal_reachs, traj_ids, output_file, subgoal_reachs_gt=None):
  # convert from list of subgoal reaching idxs to dict with traj_id keys, computing cumulative counts for each subgoal
  def _to_dict(subgoal_reachs, traj_ids):
    traj_subgoal_idxs = defaultdict(list)
    cum_subgoal_count = defaultdict(int)
    
    for subgoal_idx, subgoal_steps in enumerate(subgoal_reachs):
      for i, subgoal_step in enumerate(subgoal_steps):
        traj_id = traj_ids[i] if i < len(traj_ids) else f"#{i}"
        traj_subgoal_idxs[traj_id].append(subgoal_step if not np.isnan(subgoal_step) else None)
        cum_subgoal_count[subgoal_idx] += 1
        
    cum_subgoal_count["num_trajs"] = len(traj_ids) if traj_ids is not None else i+1
        
    return traj_subgoal_idxs, cum_subgoal_count
  
  traj_subgoal_idxs, cum_subgoal_count = _to_dict(subgoal_reachs, traj_ids)
  traj_subgoal_idxs_gt, cum_subgoal_count_gt = _to_dict(subgoal_reachs_gt, traj_ids) if subgoal_reachs_gt is not None else None
  
  # compute avg difference in reaching steps between GT and detected subgoals
  if subgoal_reachs_gt is not None:
    avg_step_diffs = defaultdict(float)
    for traj_idx in traj_subgoal_idxs:
      detected_steps = traj_subgoal_idxs[traj_idx]
      gt_steps = traj_subgoal_idxs_gt[traj_idx] if traj_idx in traj_subgoal_idxs_gt else [None] * len(detected_steps)
      
      for i, (d, gt) in enumerate(zip(detected_steps, gt_steps)):
        if d is not None and gt is not None:
          avg_step_diffs[i] += abs(d - gt)
          avg_step_diffs[f"cnt_{i}"] += 1
    
    # compute averages after accumulating all differences
    for i in range(len(subgoal_reachs)):
      if avg_step_diffs[f"cnt_{i}"] > 0:
        avg_step_diffs[i] = avg_step_diffs[i] / int(avg_step_diffs[f"cnt_{i}"])
      else:
        avg_step_diffs[i] = None  # no data to compute avg step difference for this subgoal
        avg_step_diffs[f"cnt_{i}"] = 0
          
  else:
    avg_step_diffs = None
  
  # append cumulative counts, then per-traj detected and GT subgoal reaching steps in a dict and save as json
  out_data = {
    "cumulative_subgoal_count": cum_subgoal_count,
    "cumulative_subgoal_count_gt": cum_subgoal_count_gt,
    "avg_step_diffs": avg_step_diffs,
  }
  for traj_id, subgoal_steps in sorted(traj_subgoal_idxs.items(), key=lambda x: x[0]):
    out_data[traj_id] = {
      "detected": subgoal_steps,
      "gt": traj_subgoal_idxs_gt[traj_id] if traj_subgoal_idxs_gt is not None and traj_id in traj_subgoal_idxs_gt else None
    }
    
  with open(output_file, 'w') as f:
    json.dump(out_data, f, indent=2)


def pad_trajectories(rewards):
  """Pad trajectories to same length for easier analysis"""
  lengths = np.array([len(r) for r in rewards])
  max_length = int(np.max(lengths))
  
  padded_rewards = np.full((len(rewards), max_length), np.nan)
  for i, reward in enumerate(rewards):
    padded_rewards[i, :len(reward)] = reward
  
  return padded_rewards, lengths


def filter_by_detection(rewards, detected_flags=None):
  """Filter rewards to only include trajectories where subgoal was detected.
  
  Args:
    rewards: List of reward/distance arrays, one per trajectory
    detected_flags: List of booleans indicating if subgoal was detected for each trajectory
    
  Returns:
    Filtered rewards list. If filtering results in empty list, returns all rewards.
  """
  if detected_flags is None or len(detected_flags) == 0:
    return rewards
  
  filtered_rewards = []
  for i, reward in enumerate(rewards):
    if i < len(detected_flags) and detected_flags[i]:
      filtered_rewards.append(reward)
  
  # If filtering results in no trajectories, return all (don't filter)
  if len(filtered_rewards) == 0:
    return rewards
  
  return filtered_rewards


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


def plot_trajectory_samples(rewards, traj_ids, output_dir, subgoal_reachs, label="Learned Reward", count=None, detected_flags=None, source_label="", subgoal_reachs_gt=None, show_intersection=False):
  """Save one plot per trajectory in the eval set (optionally capped by count)."""
  plot_type = _sanitize_plot_type(label)

  num_trajs = min(len(rewards), len(traj_ids))
  num_to_plot = num_trajs if count is None or count < 0 else min(num_trajs, count)
  
  print(f"Plotting trajectory-wise curves of {label.lower()}...")
  for idx in tqdm(range(num_to_plot), desc=f"Plotting {plot_type}"):
    sample_reward = rewards[idx]
    traj_id = traj_ids[idx]

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_TRAJ)
    ax.plot(range(len(sample_reward)), sample_reward, 'b-', 
                    linewidth=2, label=label, marker='o', markersize=3)

    # Determine plot type for conditional rendering
    is_distance_plot = "distance" in label.lower() or "dist" in label.lower()
    
    # For distance plots, only show the current subgoal (index 0) with GT and detected
    if is_distance_plot:
      # Plot GT subgoal in purple if available
      if subgoal_reachs_gt is not None and len(subgoal_reachs_gt) > 0:
        subgoal_idx = 0
        if subgoal_idx < len(subgoal_reachs_gt) and idx < len(subgoal_reachs_gt[subgoal_idx]):
          step = subgoal_reachs_gt[subgoal_idx][idx]
          if not np.isnan(step) and step >= 0 and step < len(sample_reward):
            ax.axvline(step, color='purple', linestyle=':', linewidth=1.7, alpha=0.9, label='Reached Subgoal (GT)')
            # Mark intersection point
            if show_intersection:
              val_at_reach = sample_reward[int(step)]
              ax.plot(int(step), val_at_reach, 'o', color='purple', markersize=4, zorder=5)
      
      # Plot detected subgoal in green if available
      if subgoal_reachs is not None and len(subgoal_reachs) > 0:
        subgoal_idx = 0
        if subgoal_idx < len(subgoal_reachs) and idx < len(subgoal_reachs[subgoal_idx]):
          step = subgoal_reachs[subgoal_idx][idx]
          if not np.isnan(step) and step >= 0 and step < len(sample_reward):
            ax.axvline(step, color='green', linestyle='--', linewidth=1.7, alpha=0.9, label='Reached Subgoal (Detected)')
            # Mark intersection point
            if show_intersection:
              val_at_reach = sample_reward[int(step)]
              ax.plot(int(step), val_at_reach, 'o', color='green', markersize=4, zorder=5)
    else:
      # For other plots (rewards), show all subgoals with both GT and detected
      # Plot GT subgoals in purple if available
      if subgoal_reachs_gt is not None:
        for j, subgoal_steps in enumerate(subgoal_reachs_gt):
          if idx < len(subgoal_steps):
            step = subgoal_steps[idx]
            if not np.isnan(step) and step >= 0 and step < len(sample_reward):
              vline_label = "Reached Subgoal (GT)" if j == 0 else None
              ax.axvline(step, color='purple', linestyle=':', linewidth=1.7, alpha=0.9, label=vline_label)
      
      # Plot detected subgoals in green if available
      if subgoal_reachs is not None:
        for j, subgoal_steps in enumerate(subgoal_reachs):
          if idx < len(subgoal_steps):
            step = subgoal_steps[idx]
            if not np.isnan(step) and step >= 0 and step < len(sample_reward):
              vline_label = "Reached Subgoal (Detected)" if j == 0 else None
              ax.axvline(step, color='green', linestyle='--', linewidth=1.7, alpha=0.9, label=vline_label)
    
    ax.set_xlabel('Timestep', fontsize=FS_LABEL)
    ax.set_ylabel(label, fontsize=FS_LABEL)
    
    # Add note if subgoal was not detected
    title_suffix = ""
    if detected_flags is not None and idx < len(detected_flags) and not detected_flags[idx]:
      title_suffix = " [Subgoal Not Detected]"
    
    ax.set_title(f'Trajectory {traj_id} (Length: {len(sample_reward)}) {source_label}{title_suffix}', 
                 fontsize=FS_TRAJ_TITLE, fontweight='bold')
    ax.grid(True)
    ax.legend(fontsize=FS_LEGEND)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{traj_id}_{plot_type}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mean_results(rewards, output_dir, subgoal_reachs, label="Learned Reward", detected_flags=None, source_label="", subgoal_reachs_gt=None, show_intersection=False):
  """Plot mean reward per timestep and return padded trajectories info.
  
  Args:
    rewards: List of reward/distance arrays
    output_dir: Directory to save plots
    subgoal_reachs: List of subgoal reaching timesteps per trajectory (detected, or None)
    label: Label for the plot
    detected_flags: Optional list of booleans indicating which trajectories had the subgoal detected
    source_label: Optional label indicating data source (e.g. "(GT)" or "(detected)")
    subgoal_reachs_gt: Optional ground truth subgoal reaching timesteps per trajectory
    show_intersection: Whether to show intersection points (for distance plots)
  """
  os.makedirs(output_dir, exist_ok=True)
  
  # Filter rewards by detection status if provided
  filtered_rewards = filter_by_detection(rewards, detected_flags)
  if detected_flags is not None:
    detected_count = sum(detected_flags) if isinstance(detected_flags, list) else np.sum(detected_flags)
    if len(filtered_rewards) == len(rewards):
      if detected_count == 0:
        print(f"  No trajectories with detected subgoals - plotting all {len(rewards)} trajectories")
    else:
      print(f"  Filtering {label.lower()} - using only {detected_count} trajectories where subgoal was detected")
  
  # pad trajectories for analysis
  padded_rewards, lengths = pad_trajectories(filtered_rewards)
  
  # mean reward per timestep
  print(f"Plotting mean reward per timestep of {label.lower()}...")
  fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_MEAN)
  
  # calculate mean and std per timestep (ignoring NaNs)
  mean_reward = np.nanmean(padded_rewards, axis=0)
  std_reward = np.nanstd(padded_rewards, axis=0)
  min_reward = np.nanmin(padded_rewards, axis=0)
  timesteps = np.arange(len(mean_reward))
  
  ax.plot(timesteps, mean_reward, 'b-', linewidth=2, label='Mean Reward')
  ax.fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward, 
                    alpha=0.3, label='+-1 Std Dev')

  # For distance plots, add special markers
  is_distance_plot = "distance" in label.lower() or "dist" in label.lower()
  is_subgoal_reward_plot = "subgoal" in label.lower() and "reward" in label.lower() or "sub" in label.lower() and "rew" in label.lower()
  
  # get y-range for plotting markers
  y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
  y_min, y_max = ax.get_ylim()
  y_high_mark, y_sub_high_mark = y_max - 0.1 * y_range, y_max - 0.2 * y_range
  y_low_mark, y_sub_low_mark = y_min + 0.05 * y_range, y_min + 0.15 * y_range
  
  # Find and mark minimum point for distance plots
  if is_distance_plot:
    min_idx = np.nanargmin(mean_reward)
    min_val = mean_reward[min_idx]
    min_std = std_reward[min_idx]
    ax.plot(min_idx, min_val, 'o', color='red', markersize=4, label='Min Avg Dist', zorder=5)
    ax.text(min_idx, y_low_mark, f'{min_val:.3f}\n±{min_std:.3f}', 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
    
    # also plot minimum line across timesteps for reference
    #ax.plot(min_reward, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7, label='Min Distance')

  # aggregate each subgoal across trajectories and mark mean reaching timestep
  # Plot GT subgoals in purple if available
  if subgoal_reachs_gt is not None and len(subgoal_reachs_gt) > 0:
    for subgoal_i, subgoal_steps in enumerate(subgoal_reachs_gt):
      reached_steps = [step for step in subgoal_steps if not np.isnan(step) and step >= 0]
      if len(reached_steps) == 0:
        continue

      mean_step = float(np.mean(reached_steps))
      mean_label = f'Subgoal (GT)' if subgoal_i == 0 else None
      ax.axvline(mean_step, color='purple', linestyle=':', linewidth=1.7, alpha=0.95, label=mean_label)
      
      if is_subgoal_reward_plot and not is_distance_plot:
        ax.text(mean_step, y_high_mark, len(reached_steps), 
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))
      
      # For distance plots, mark intersection point with the mean curve (only for current subgoal)
      if show_intersection and is_distance_plot and subgoal_i == 0:
        mean_step_int = int(np.round(mean_step))
        if 0 <= mean_step_int < len(mean_reward):
          val_at_reach = mean_reward[mean_step_int]
          ax.plot(mean_step_int, val_at_reach, 'o', color='purple', markersize=4, zorder=5)
          ax.text(mean_step_int, y_high_mark, f'{val_at_reach:.3f}\n±{std_reward[mean_step_int]:.3f}', 
                  ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))

  # Plot detected subgoals in green if available
  if subgoal_reachs is not None and len(subgoal_reachs) > 0:
    for subgoal_i, subgoal_steps in enumerate(subgoal_reachs):
      reached_steps = [step for step in subgoal_steps if not np.isnan(step) and step >= 0]
      if len(reached_steps) == 0:
        continue

      mean_step = float(np.mean(reached_steps))
      mean_label = f'Subgoal (D)' if subgoal_i == 0 else None
      ax.axvline(mean_step, color='green', linestyle='--', linewidth=1.7, alpha=0.95, label=mean_label)
      
      if is_subgoal_reward_plot and not is_distance_plot:
        ax.text(mean_step, y_low_mark, len(reached_steps), 
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
      
      # For distance plots, mark intersection point with the mean curve (only for current subgoal)
      if show_intersection and is_distance_plot and subgoal_i == 0:
        mean_step_int = int(np.round(mean_step))
        if 0 <= mean_step_int < len(mean_reward):
          val_at_reach = mean_reward[mean_step_int]
          ax.plot(mean_step_int, val_at_reach, 'o', color='green', markersize=4, zorder=5)
          ax.text(mean_step_int, y_sub_high_mark, f'{val_at_reach:.3f}\n±{std_reward[mean_step_int]:.3f}', 
                  ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

  ax.set_xlabel('Timestep', fontsize=FS_LABEL)
  ax.set_ylabel(label, fontsize=FS_LABEL)
  num_trajs = len(filtered_rewards) if detected_flags is not None else len(rewards)
  ax.set_title(f'Mean {label} per Timestep {source_label}(N={num_trajs} trajectories)', 
                fontsize=FS_TITLE, fontweight='bold')
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=FS_LEGEND)
  
  plt.tight_layout()
  output_path = os.path.join(output_dir, f'mean-std_{_sanitize_plot_type(label)}.png')
  plt.savefig(output_path, dpi=300, bbox_inches='tight')
  plt.close()
  
  return padded_rewards, lengths, mean_reward


def plot_trajectory_lengths(lengths, output_dir):
  """Plot trajectory length histogram once per evaluation run."""
  print("Plotting trajectory lengths...")
  fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_HIST)
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
  (rewards, traj_ids, subgoal_rewards, subgoal_dists, subgoal_reachs, subgoal_reachs_gt
  ) = compute_reward_signals(model, valid_loader, goal_emb, subgoal_embs, dist_scale, device, valid_subgoal_frames)
  print(f"Computed {len(rewards)} reward signals")
  if subgoal_reachs_gt is not None:
    print(f"Using ground truth subgoal indices for distance plots")


  # plot trajectory lengths once using base rewards, to understand distribution of trajectory lengths and mean-std plots better
  _, lengths = pad_trajectories(rewards)
  plot_trajectory_lengths(lengths, out_dir)
  
  # plot mean results, and save avg reward metrics
  plot_mean_results(rewards, out_dir, None, label="Learned Reward")
  save_reward_metrics(rewards, os.path.join(out_dir, 'reward_metrics.json'), avg=True)
  
  # plot subgoal mean rewards and distances, and save avg subgoal reward metrics
  if subgoal_rewards is not None:
    plot_mean_results(subgoal_rewards, out_dir, subgoal_reachs, label="Learned Subgoal Reward", 
                     subgoal_reachs_gt=subgoal_reachs_gt, show_intersection=False)
    save_reward_metrics(subgoal_rewards, os.path.join(out_dir, 'subgoal_reward_metrics.json'), avg=True)
    
    for i, subgoal_dist in enumerate(subgoal_dists):
      # For distance plots, plot both GT and detected with intersection data
      plot_mean_results(subgoal_dist, out_dir, [subgoal_reachs[i]] if subgoal_reachs else None, 
                       label=f"Distance to Subgoal {i+1}", detected_flags=None, 
                       subgoal_reachs_gt=[subgoal_reachs_gt[i]] if subgoal_reachs_gt else None,
                       show_intersection=True)
    
    # save .json with detected and GT subgoal reaching timesteps for each trajectory
    save_subgoal_idxs(subgoal_reachs, traj_ids, os.path.join(out_dir, 'subgoal_idxs.json'), subgoal_reachs_gt=subgoal_reachs_gt)
    
    
  # plot trajectory-wise curves, and save per-trajectory reward metrics
  if not args.no_plot_trajs:
    plot_trajectory_samples(rewards, traj_ids, trajs_out_dir, None, label="Learned Reward", count=args.count)
  else:
    print("Skipping trajectory-wise plots for learned rewards (no_plot_trajs flag is set)")
  
  save_reward_metrics(rewards, os.path.join(out_dir, 'reward_metrics_per_traj.json'), avg=False, traj_ids=traj_ids)
  
  if subgoal_rewards is not None:
    # plot for traj samples, and save per-trajectory subgoal reward metrics
    if not args.no_plot_subgoal_trajs:
      plot_trajectory_samples(subgoal_rewards, traj_ids, trajs_out_dir, subgoal_reachs, 
                              label="Learned Subgoal Reward", count=args.count, 
                              subgoal_reachs_gt=subgoal_reachs_gt, show_intersection=False)
    else:
      print("Skipping trajectory-wise plots for subgoal rewards (no_plot_subgoal_trajs flag is set)")
    
    save_reward_metrics(subgoal_rewards, os.path.join(out_dir, 'subgoal_reward_metrics_per_traj.json'), avg=False, traj_ids=traj_ids)

    if args.plot_subgoal_dists:
      for i, subgoal_dist in enumerate(subgoal_dists):
        # For distance trajectory plots, plot both GT and detected with intersection
        plot_trajectory_samples(subgoal_dist, traj_ids, trajs_out_dir, [subgoal_reachs[i]] if subgoal_reachs else None, 
                               label=f"Distance to Subgoal {i+1}", count=args.count, 
                               subgoal_reachs_gt=[subgoal_reachs_gt[i]] if subgoal_reachs_gt else None,
                               show_intersection=True)


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
