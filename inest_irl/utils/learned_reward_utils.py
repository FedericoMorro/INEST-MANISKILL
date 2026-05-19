import os

import json
import matplotlib.pyplot as plt
import numpy as np
from torchkit import CheckpointManager
from tqdm.auto import tqdm

from inest_irl.maniskill3.stack_pyramid import MAX_SUBGOAL
from inest_irl.utils.utils import to_json_serializable, is_nan_or_none


# report-friendly plotting defaults (compact figure size with readable text)
FIGSIZE_TRAJ = (7.0, 3.6)
FIGSIZE_MEAN = (7.0, 3.6)
FIGSIZE_HIST = (7.0, 2.8)
FS_LABEL = 12
FS_TITLE = 13
FS_LEGEND = 10
FS_TRAJ_TITLE = 12


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
  
  def to_dict(self):
    return self.__dict__
  
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
  
  
  def to_file(self, out_dir):
    out_path = os.path.join(out_dir, 'learned_rewards.json')
    with open(out_path, 'w') as f:
      json.dump(to_json_serializable({traj_id: traj_lr.to_dict() for traj_id, traj_lr in self.traj_lrs.items()}), f, indent=2)
  
  
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
          
          
  def plot_trajectory_lengths(self, output_dir):
    print("Plotting trajectory lengths...")
    lengths = self.traj_lens()
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


def pad_trajectories(rewards):
  """Pad trajectories to same length for easier analysis"""
  lengths = np.array([len(r) for r in rewards])
  max_length = int(np.max(lengths))
  
  padded_rewards = np.full((len(rewards), max_length), np.nan)
  for i, reward in enumerate(rewards):
    padded_rewards[i, :len(reward)] = reward
  
  return padded_rewards, lengths


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
      if subgoal_reachs is not None and len(subgoal_reachs) > show_subgoals and not is_nan_or_none(subgoal_reachs[show_subgoals]):
        ax.axvline(subgoal_reachs[show_subgoals], color='green', linestyle='--', linewidth=1.7, alpha=0.9, label=f'Reached Subgoal {show_subgoals} (Detected)')
        if show_intersection:
          ax.plot(subgoal_reachs[show_subgoals], values[int(subgoal_reachs[show_subgoals])], 'o', color='green', markersize=4, zorder=5)
      if subgoal_reachs_gt is not None and len(subgoal_reachs_gt) > show_subgoals and not is_nan_or_none(subgoal_reachs_gt[show_subgoals]):
        ax.axvline(subgoal_reachs_gt[show_subgoals], color='purple', linestyle=':', linewidth=1.7, alpha=0.9, label=f'Reached Subgoal {show_subgoals} (GT)')
        if show_intersection:
          ax.plot(subgoal_reachs_gt[show_subgoals], values[int(subgoal_reachs_gt[show_subgoals])], 'o', color='purple', markersize=4, zorder=5)
    # show all subgoals 
    elif show_subgoals >= MAX_SUBGOAL:
      if subgoal_reachs is not None:
        for j, subgoal_step in enumerate(subgoal_reachs):
          if not is_nan_or_none(subgoal_step):
            ax.axvline(subgoal_step, color='green', linestyle='--', linewidth=1.7, alpha=0.9, label=f'Reached Subgoal {j} (Detected)' if j == 0 else None)
      if subgoal_reachs_gt is not None:
        for j, subgoal_step in enumerate(subgoal_reachs_gt):
          if not is_nan_or_none(subgoal_step):
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
