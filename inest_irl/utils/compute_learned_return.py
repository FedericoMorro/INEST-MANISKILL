"""Compute and visualize learned reward signal based on goal embeddings."""

"""
Example usage:

python inest_irl/utils/eval_return.py
    --experiment_path ../data/inest-maniskill/_experiments/pretrain/render-cam/


# for different trajs
python inest_irl/utils/eval_return.py
    --experiment_path ../data/inest-maniskill/_experiments/pretrain/render-cam/
    --diff_trajs_dataset ../data/inest-maniskill/different-trajs/
"""

import os
import typing
from pathlib import Path

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import torch
from torchkit import CheckpointManager
from tqdm.auto import tqdm

from inest_irl.maniskill3.stack_pyramid import MAX_SUBGOAL
from utils import load_config_from_dir
from xirl import common
from xirl.models import SelfSupervisedModel


ModelType = SelfSupervisedModel
DataLoaderType = typing.Dict[str, torch.utils.data.DataLoader]


C_VALUE = 0.25   # additional reward for reaching any subgoal
DISTANCE_THRESHOLDS = [3, 3, 3, 3]  # distance threshold for considering a subgoal reached (in embedding space)
PATIENCE_THRESHOLD = 0  # number of consecutive timesteps below distance threshold to consider subgoal reached

# report-friendly plotting defaults (compact figure size with readable text)
FIGSIZE_TRAJ = (7.0, 3.6)
FIGSIZE_MEAN = (7.0, 3.6)
FIGSIZE_HIST = (7.0, 2.8)
FS_LABEL = 12
FS_TITLE = 13
FS_LEGEND = 10
FS_TRAJ_TITLE = 12


def setup_from_pretrain(experiment_path, use_cpu, diff_dataset_path=None):
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
  
  # load data -> debug active if use_cpu, otherwise use GPU-optimized dataloader settings
  train_loader = common.get_downstream_dataloaders(config, debug=use_cpu)["train"]
  #! note batch_size=1 is enforced in the dataloader

  # if a different trajectory dataset path is provided, load it instead of the validation set for evaluation
  if diff_dataset_path is not None:
    print(f"Loading different trajectory dataset from: {diff_dataset_path}")
    config.data.root = diff_dataset_path
  else:
    print("No different trajectory dataset provided - using validation set for evaluation")
  valid_loader = common.get_downstream_dataloaders(config, debug=use_cpu)["valid"]

  # search for subgoal_frames.json to plot also subgoal rewards
  subgoal_frames_path = Path(config.data.root) / "subgoal_frames.json"
  if subgoal_frames_path.exists():
    with open(subgoal_frames_path, 'r') as f:
      subgoal_frames = json.load(f)
    print(f"Found subgoal frames file with {len(subgoal_frames)} trajectories - will compute and plot subgoal rewards")
  else:
    subgoal_frames = None
    print("No subgoal frames file found - will only compute and plot rewards to final goal")
  
  return model, train_loader, valid_loader, subgoal_frames, global_step, device


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
        subgoal_idxs = subgoal_frames[traj_id]

        # if empty list, add empty lists inside with the length of the number of subgoals
        if len(subgoal_embs_list) == 0:
          for _ in range(MAX_SUBGOAL):
            subgoal_embs_list.append([])

        # add subgoal embeddings to the corresponding subgoal index list
        for i, idx in enumerate(subgoal_idxs):
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
  
  return goal_emb, subgoal_embs, dist_scale


def compute_reward_signals(model, valid_loader, goal_emb, subgoal_embs, dist_scale, device):
  """Compute the negative distance to goal as reward signal for each trajectory"""
  rewards = []
  traj_ids = []
  subgoal_rewards = [] if subgoal_embs is not None else None
  subgoal_reachs = [ [] for _ in range(len(subgoal_embs)) ] if subgoal_embs is not None else None
  subgoal_dists = [ [] for _ in range(len(subgoal_embs)) ] if subgoal_embs is not None else None
  
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

  return rewards, traj_ids, subgoal_rewards, subgoal_dists, subgoal_reachs


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


def plot_trajectory_samples(rewards, traj_ids, output_dir, subgoal_reachs, label="Learned Reward", count=None):
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

    if subgoal_reachs is not None:
      for j, subgoal_steps in enumerate(subgoal_reachs):
        if idx < len(subgoal_steps):
          step = subgoal_steps[idx]
          if not np.isnan(step) and step < len(sample_reward):
            vline_label = "Reached Subgoal" if j == 0 else None
            ax.axvline(step, color='crimson', linestyle=':', linewidth=1.8, alpha=0.9, label=vline_label)
    
    ax.set_xlabel('Timestep', fontsize=FS_LABEL)
    ax.set_ylabel(label, fontsize=FS_LABEL)
    ax.set_title(f'Trajectory {traj_id} (Length: {len(sample_reward)})', fontsize=FS_TRAJ_TITLE, fontweight='bold')
    ax.grid(True)
    ax.legend(fontsize=FS_LEGEND)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{traj_id}_{plot_type}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_results(rewards, output_dir, subgoal_reachs, label="Learned Reward"):
  """Plot mean reward per timestep and return padded trajectories info."""
  os.makedirs(output_dir, exist_ok=True)
  
  # pad trajectories for analysis
  padded_rewards, lengths = pad_trajectories(rewards)
  
  # mean reward per timestep
  print(f"Plotting mean reward per timestep of {label.lower()}...")
  fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_MEAN)
  
  # calculate mean and std per timestep (ignoring NaNs)
  mean_reward = np.nanmean(padded_rewards, axis=0)
  std_reward = np.nanstd(padded_rewards, axis=0)
  timesteps = np.arange(len(mean_reward))
  
  ax.plot(timesteps, mean_reward, 'b-', linewidth=2, label='Mean Reward')
  ax.fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward, 
                    alpha=0.3, label='+-1 Std Dev')

  # aggregate each subgoal across trajectories and mark mean reaching timestep
  if subgoal_reachs is not None and len(subgoal_reachs) > 0:
    for subgoal_i, subgoal_steps in enumerate(subgoal_reachs):
      reached_steps = [step for step in subgoal_steps if not np.isnan(step)]
      if len(reached_steps) == 0:
        continue

      mean_step = float(np.mean(reached_steps))
      mean_label = f'Subgoal mean reach' if subgoal_i == 0 else None
      ax.axvline(mean_step, color='crimson', linestyle=':', linewidth=2, alpha=0.95, label=mean_label)

  ax.set_xlabel('Timestep', fontsize=FS_LABEL)
  ax.set_ylabel(label, fontsize=FS_LABEL)
  ax.set_title(f'Mean {label} per Timestep (N={len(rewards)} trajectories)', 
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
  model, train_loader, valid_loader, subgoal_frames, global_step, device = setup_from_pretrain(
    args.experiment_path, 
    args.use_cpu, 
    diff_dataset_path=args.diff_trajs_dataset
  )

  # check for cached results in output directory
  checkpoint_dir = os.path.join(args.experiment_path, "checkpoints")
  cache_path = os.path.join(checkpoint_dir, f"cached_embeddings_step_{global_step}.pkl")
  if not os.path.exists(cache_path):
    print("No cached embedddings found - computing from scratch...")

    # compute goal embedding
    goal_emb, subgoal_embs, dist_scale = compute_goal_embedding(model, train_loader, subgoal_frames, device)
    print(f"Goal embedding computed - shape: {goal_emb.shape}")
    if subgoal_embs is not None:
      print(f"Subgoals identified: {len(subgoal_embs)} - shape: {subgoal_embs[0].shape}")

    # save goal and subgoal embeddings to cache for future use
    with open(cache_path, 'wb') as f:
      pickle.dump((goal_emb, subgoal_embs, dist_scale), f)
    print(f"Saved computed embeddings and distance scale to cache at: {cache_path}")

  else:
    print(f"Found cached embeddings at {cache_path} - loading...")
    with open(cache_path, 'rb') as f:
      goal_emb, subgoal_embs, dist_scale = pickle.load(f)
    print(f"Goal embedding loaded - shape: {goal_emb.shape}")
    if subgoal_embs is not None:
      print(f"Subgoals identified: {len(subgoal_embs)} - shape: {subgoal_embs[0].shape}")
  
  # compute reward signals
  if not args.no_plot_trajs or not args.no_plot_subgoal_trajs:
    (rewards, traj_ids, subgoal_rewards, subgoal_dists, subgoal_reachs
    ) = compute_reward_signals(model, valid_loader, goal_emb, subgoal_embs, dist_scale, device)
    print(f"Computed {len(rewards)} reward signals")


  if not args.no_plot_trajs:
    # plot trajectory lengths once using base rewards, to understand distribution of trajectory lengths and mean-std plots better
    _, lengths = pad_trajectories(rewards)
    plot_trajectory_lengths(lengths, out_dir)

    # save one trajectory plot per eval sample (or up to count)
    plot_trajectory_samples(rewards, traj_ids, trajs_out_dir, None, label="Learned Reward", count=args.count)
  
    # plot results
    plot_results(rewards, out_dir, None, label="Learned Reward")

  if not args.no_plot_subgoal_trajs and subgoal_rewards is not None:
    plot_trajectory_samples(subgoal_rewards, traj_ids, trajs_out_dir, subgoal_reachs, label="Learned Subgoal Reward", count=args.count)
    plot_results(subgoal_rewards, out_dir, subgoal_reachs, label="Learned Subgoal Reward")

    if args.plot_subgoal_dists:
      for i, subgoal_dist in enumerate(subgoal_dists):
        plot_trajectory_samples(subgoal_dist, traj_ids, trajs_out_dir, subgoal_reachs, label=f"Distance to Subgoal {i+1}", count=args.count)
        plot_results(subgoal_dist, out_dir, subgoal_reachs, label=f"Distance to Subgoal {i+1}")

    # save .txt with per trajectory subgoal reaching timesteps
    traj_substep = []
    for subgoal_idx, subgoal_steps in enumerate(subgoal_reachs):
      for traj_idx, subgoal_step in enumerate(subgoal_steps):
        if len(traj_substep) < traj_idx + 1:
          traj_substep.append([])
        else:
          traj_substep[traj_idx].append(subgoal_step)

    with open(os.path.join(out_dir, 'sub_reach_times.txt'), 'w') as f:
      f.write("Trajectory ID: Subgoal i-th reaching timesteps (NaN if not reached)\n")
      for i, ts in enumerate(traj_substep):
        f.write(f"{i:>4.0f}: " + ", ".join([f"{t:3.0f}" for t in ts]) + "\n")

  print(f"All results saved to: {out_dir}")


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser(description="Evaluate learned reward signals based on goal embeddings")
  arg_parser.add_argument("--experiment_path", type=str, required=True,
                          help="Path to the pretraining experiment directory")
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
  args = arg_parser.parse_args()

  main(args)
