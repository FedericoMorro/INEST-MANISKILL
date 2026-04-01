"""Compute and visualize learned reward signal based on goal embeddings."""

import os
import typing
from pathlib import Path

import argparse
import numpy as np
import torch
from torchkit import CheckpointManager
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import sys

from utils import load_config_from_dir
from xirl import common
from xirl.models import SelfSupervisedModel


ModelType = SelfSupervisedModel
DataLoaderType = typing.Dict[str, torch.utils.data.DataLoader]


def _setup(experiment_path):
  """Load the latest embedder checkpoint and dataloaders"""

  config = load_config_from_dir(experiment_path)
  model = common.get_model(config)
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
  
  # load data
  train_loader = common.get_downstream_dataloaders(config, False)["train"]
  valid_loader = common.get_downstream_dataloaders(config, False)["valid"]
  
  return model, train_loader, valid_loader, device


def compute_goal_embedding(model, train_loader, device):
  """Compute the mean goal embedding from the last frames of trajectories"""
  init_embs, goal_embs = [], []

  for class_name, class_loader in train_loader.items():
    for batch in tqdm(iter(class_loader), leave=False, desc=f"Embedding {class_name}"):
      out = model.infer(batch["frames"].to(device))
      emb = out.numpy().embs  # shape: (seq_len, embedding_dim)
      
      init_embs.append(emb[0, :])   # first frame embedding
      goal_embs.append(emb[-1, :])  # last frame embedding
  
  goal_emb = np.mean(np.stack(goal_embs, axis=0), axis=0, keepdims=True)
  dist_to_goal = np.linalg.norm(np.stack(init_embs, axis=0) - goal_emb, axis=1).mean()
  dist_scale = 1.0 / (dist_to_goal + 1e-8)
  
  return goal_emb, dist_scale


def compute_reward_signals(model, valid_loader, goal_emb, dist_scale, device):
  """Compute the negative distance to goal as reward signal for each trajectory"""
  rewards = []
  
  for class_name, class_loader in valid_loader.items():
    for batch in tqdm(iter(class_loader), leave=False, desc=f"Computing rewards for {class_name}"):
      out = model.infer(batch["frames"].to(device))
      traj_emb = out.numpy().embs  # shape: (seq_len, embedding_dim)
      
      # compute euclidean distance from each frame to goal
      dist = np.linalg.norm(goal_emb - traj_emb, axis=1)**2  # shape: (seq_len,)
      rewards.append(- dist * dist_scale)
  
  return rewards


def pad_trajectories(rewards):
  """Pad trajectories to same length for easier analysis"""
  lengths = np.array([len(r) for r in rewards])
  max_length = int(np.max(lengths))
  
  padded_rewards = np.full((len(rewards), max_length), np.nan)
  for i, reward in enumerate(rewards):
    padded_rewards[i, :len(reward)] = reward
  
  return padded_rewards, lengths


def plot_results(rewards, output_dir, exp_name):
  """Plot mean reward per timestep and sample trajectory"""
  os.makedirs(output_dir, exist_ok=True)
  
  # 4 sample trajectories in 2x2 grid
  print(f"Plotting sample trajectories...")
  fig, axes = plt.subplots(2, 2, figsize=(14, 10))
  axes = axes.flatten()
  
  for i in range(4):
    idx = np.random.choice(len(rewards)) 
    sample_reward = rewards[idx]
    
    ax = axes[i]
    ax_twin = ax.twinx()
    
    ax.plot(range(len(sample_reward)), sample_reward, 'b-', 
                    linewidth=2, label='Reward', marker='o', markersize=3)
    
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Learned Reward', fontsize=10)
    ax.set_title(f'Sample {i+1} (Length: {len(sample_reward)})', fontsize=11, fontweight='bold')
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8)
  
  plt.suptitle('Sample Trajectories', fontsize=14, fontweight='bold', y=0.995)
  plt.tight_layout()
  output_path = os.path.join(output_dir, f'{exp_name}_sample-trajs_learn-reward.png')
  plt.savefig(output_path, dpi=300, bbox_inches='tight')
  plt.close()
  
  # pad trajectories for analysis
  padded_rewards, lengths = pad_trajectories(rewards)
  
  # mean reward per timestep with trajectory lengths bar chart
  print(f"Plotting mean reward per timestep and trajectory lengths...")
  fig, axs = plt.subplots(2, 1, figsize=(12, 10))
  
  # calculate mean and std per timestep (ignoring NaNs)
  mean_reward = np.nanmean(padded_rewards, axis=0)
  std_reward = np.nanstd(padded_rewards, axis=0)
  timesteps = np.arange(len(mean_reward))
  
  axs[0].plot(timesteps, mean_reward, 'b-', linewidth=2, label='Mean Reward')
  axs[0].fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward, 
                    alpha=0.3, label='+-1 Std Dev')
  axs[0].set_xlabel('Timestep', fontsize=12)
  axs[0].set_ylabel('Learned Reward (Negative Distance to Goal)', fontsize=12)
  axs[0].set_title(f'Mean Learned Reward per Timestep (N={len(rewards)} trajectories)', 
                fontsize=14, fontweight='bold')
  axs[0].grid(True, alpha=0.3)
  axs[0].legend(fontsize=10)
  
  # bar chart of trajectory lengths
  axs[1].hist(lengths, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
  axs[1].set_xlabel('Length (timesteps)', fontsize=11)
  axs[1].set_ylabel('Count', fontsize=11)
  axs[1].set_title('Trajectory Lengths', fontsize=12, fontweight='bold')
  axs[1].grid(True, alpha=0.3, axis='y')
  
  plt.tight_layout()
  output_path = os.path.join(output_dir, f'{exp_name}_mean-std_learn-rewards.png')
  plt.savefig(output_path, dpi=300, bbox_inches='tight')
  plt.close()
  
  return padded_rewards, lengths, mean_reward


def main(args):
  # set random seeds for reproducibility
  torch.manual_seed(22)
  torch.cuda.manual_seed_all(22)
  np.random.seed(22)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  
  # Setup model and data
  print(f"Loading model from: {args.experiment_path}")
  model, train_loader, valid_loader, device = _setup(args.experiment_path)
  
  # Compute goal embedding
  goal_emb, dist_scale = compute_goal_embedding(model, train_loader, device)
  print(f"Goal embedding computed - shape: {goal_emb.shape}")
  
  # compute reward signals
  rewards = compute_reward_signals(model, valid_loader, goal_emb, dist_scale, device)
  print(f"Computed {len(rewards)} reward signals")
  
  # plot results
  plot_results(rewards, args.output_dir, args.experiment_path.strip('/').split('/')[-1])
  print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser(description="Evaluate learned reward signals based on goal embeddings")
  arg_parser.add_argument("--experiment_path", type=str, required=True,
                          help="Path to the pretraining experiment directory")
  arg_parser.add_argument("--output_dir", type=str, default="/home/fmorro/INEST-MANISKILL/out/reward_plots",
                          help="Directory to save the generated plots")
  args = arg_parser.parse_args()

  main(args)
