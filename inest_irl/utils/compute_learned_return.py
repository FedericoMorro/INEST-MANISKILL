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
import json
import numpy as np
import pickle
import torch
from torchkit import CheckpointManager
from tqdm.auto import tqdm

from xirl import common
from xirl.models import SelfSupervisedModel

from inest_irl.maniskill3.stack_pyramid import MAX_SUBGOAL
from inest_irl.utils.learned_reward_utils import DatasetLearnedReward, TrajectoryLearnedReward
from inest_irl.utils.utils import load_config_from_dir


ModelType = SelfSupervisedModel
DataLoaderType = typing.Dict[str, torch.utils.data.DataLoader]


C_VALUE = 0.25   # additional reward for reaching any subgoal
DISTANCE_THRESHOLDS = [0.5, 0.5, 0.5, 0.5]  # distance threshold for considering a subgoal reached (in embedding space)
PATIENCE_THRESHOLD = 2  # number of consecutive timesteps below distance threshold to consider subgoal reached

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
  
  # save all results to a json file
  dataset_lr.to_file(out_dir)
  
  # save metrics to file
  dataset_lr.reward_metrics_to_file(out_dir)
  dataset_lr.subgoal_idxs_to_file(out_dir)

  # plot trajectory lengths once using base rewards, to understand distribution of trajectory lengths and mean-std plots better
  dataset_lr.plot_trajectory_lengths(out_dir)
  
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
