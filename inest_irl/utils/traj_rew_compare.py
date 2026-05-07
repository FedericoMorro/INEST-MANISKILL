"""
Compute trajectory reward similarity between positive and negative demonstrations.
Uses DTW alignment to normalize sequences before comparison.
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from torchkit import CheckpointManager
from tqdm import tqdm

# Add xirl to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xirl import common
import utils


def dtw_reward_similarity(pos_rewards: np.ndarray, neg_rewards: np.ndarray) -> float:
    """
    Compute similarity between positive and negative trajectory rewards using DTW alignment.
    
    Args:
        pos_rewards: Positive trajectory rewards, shape (T_pos,)
        neg_rewards: Negative trajectory rewards, shape (T_neg,)
    
    Returns:
        Similarity score in [-1, 1]: 1 = identical, 0 = uncorrelated, -1 = opposite.
        Computed as Pearson correlation of DTW-aligned sequences.
    """
    # Reshape to column vectors for DTW
    pos_vec = pos_rewards.reshape(-1, 1).astype(np.float64)
    neg_vec = neg_rewards.reshape(-1, 1).astype(np.float64)
    
    # Align trajectories using DTW
    _, path = fastdtw(pos_vec, neg_vec, dist=euclidean)
    
    # Map to reference (positive) timeline
    pos_indices = np.array([p[0] for p in path])
    neg_indices = np.array([p[1] for p in path])
    
    # Extract aligned values
    pos_aligned = pos_rewards[pos_indices]
    neg_aligned = neg_rewards[neg_indices]
    
    # Compute Pearson correlation as similarity
    if len(pos_aligned) < 2:
        return 0.0
    
    corr = np.corrcoef(pos_aligned, neg_aligned)[0, 1]
    return float(np.nan_to_num(corr, nan=0.0))


def _compute_trajectory_rewards(
    frames_tensor: torch.Tensor,
    goal_emb: np.ndarray,
    distance_scale: float,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """
    Compute learned reward for each frame of a trajectory.
    
    Args:
        frames_tensor: shape (1, T, C, H, W) - actual image frames
        goal_emb: shape (emb_dim,)
        distance_scale: scalar
        model: model with infer() method
        device: torch device
    
    Returns:
        rewards: shape (T,) - reward for each frame
    """
    goal = torch.tensor(goal_emb, dtype=torch.float32, device=device)
    if goal.ndim > 1:
        goal = goal.reshape(-1)
    
    frames_tensor = frames_tensor.to(device)
    
    with torch.no_grad():
        out = model.infer(frames_tensor)
    
    if hasattr(out, "embs"):
        embs = out.embs  # shape: (1, T, embedding_dim)
    elif hasattr(out, "embedding"):
        embs = out.embedding
    else:
        embs = out
    
    # Extract embeddings for all frames
    embs = embs.squeeze(0)  # (T, embedding_dim)
    if embs.ndim == 1:
        embs = embs.unsqueeze(0)
    
    # Ensure embeddings are on same device as goal
    embs = embs.to(device)
    
    # Compute distances
    dists = torch.norm(embs - goal.unsqueeze(0), dim=1).cpu().numpy()
    rewards = -dists * distance_scale
    
    return np.array(rewards, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute trajectory reward similarity between positive and negative demonstrations."
    )
    parser.add_argument(
        "--pos_dataset",
        type=str,
        required=True,
        help="Path to positive demonstrations dataset root (supports glob patterns).",
    )
    parser.add_argument(
        "--neg_dataset",
        type=str,
        required=True,
        help="Path to negative demonstrations dataset root (supports glob patterns).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model (config + checkpoint).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/traj_rew_comparison",
        help="Output directory for visualizations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Compute device.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Maximum number of demonstrations to load from each dataset.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> str:
    """Resolve glob patterns in path and return the first match or the original path."""
    # Handle escaped asterisks (e.g., from shell)
    path = path.replace(r"\*", "*")
    
    # Try glob expansion
    matches = glob.glob(path)
    if matches:
        resolved = sorted(matches)[0]
        print(f"  Resolved glob pattern '{path}' to '{resolved}'")
        return resolved
    
    # If no glob pattern or no matches, return as-is
    return path


def main():
    args = parse_args()
    
    # Resolve glob patterns in dataset paths
    print("Resolving dataset paths...")
    pos_dataset = resolve_path(args.pos_dataset)
    neg_dataset = resolve_path(args.neg_dataset)
    
    # Validate paths
    if not os.path.isdir(pos_dataset):
        print(f"ERROR: Positive dataset path does not exist: {pos_dataset}")
        return
    if not os.path.isdir(neg_dataset):
        print(f"ERROR: Negative dataset path does not exist: {neg_dataset}")
        return
    
    print(f"Positive dataset: {pos_dataset}")
    print(f"Negative dataset: {neg_dataset}")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load config and model
    print(f"Loading model from {args.model_path}")
    config = utils.load_config_from_dir(args.model_path)
    model = common.get_model(config)
    model.to(device).eval()
    
    checkpoint_dir = os.path.join(args.model_path, "checkpoints")
    ckpt_mgr = CheckpointManager(checkpoint_dir, model=model)
    ckpt_mgr.restore_or_initialize()
    print("Model loaded.")
    
    # Load positive dataset
    config.data.root = pos_dataset
    print(f"Loading positive dataset from {pos_dataset}")
    downstream_loaders_pos = common.get_downstream_dataloaders(config, debug=False)["valid"]
    print(f"  Found {len(downstream_loaders_pos)} classes in positive dataset")
    
    # Load negative dataset
    config.data.root = neg_dataset
    print(f"Loading negative dataset from {neg_dataset}")
    downstream_loaders_neg = common.get_downstream_dataloaders(config, debug=False)["valid"]
    print(f"  Found {len(downstream_loaders_neg)} classes in negative dataset")
    
    # Compute goal embedding from positive demonstrations (final frames)
    print("Computing goal embedding from positive demonstrations...")
    goal_embs = []
    dem_count = 0
    for class_name, class_loader in downstream_loaders_pos.items():
        print(f"  Processing class: {class_name}")
        for batch in tqdm(class_loader, desc=f"Processing {class_name}", leave=False, total=min(len(class_loader), args.count)):
            if dem_count >= args.count:
                break
            frames = batch["frames"]  # shape: (1, T, C, H, W)
            if isinstance(frames, np.ndarray):
                frames = torch.tensor(frames, dtype=torch.float32)
            
            # Get embedding of final frame
            final_frame = frames[:, -1:, :, :, :]  # (1, 1, C, H, W)
            
            with torch.no_grad():
                out = model.infer(final_frame.to(device))
            
            if hasattr(out, "embs"):
                emb = out.embs  # (1, 1, embedding_dim)
            elif hasattr(out, "embedding"):
                emb = out.embedding
            else:
                emb = out
            
            emb = emb.squeeze().detach().cpu().numpy()  # (embedding_dim,)
            if emb.ndim > 1:
                emb = emb.reshape(-1)
            goal_embs.append(emb)
            dem_count += 1
        if dem_count >= args.count:
            break
    
    goal_emb = np.mean(goal_embs, axis=0)
    print(f"Goal embedding shape: {goal_emb.shape}")
    print(f"Computed from {len(goal_embs)} trajectories")
    
    if len(goal_embs) == 0:
        print("ERROR: No goal embeddings computed. Check dataset loading.")
        return
    
    # Compute distance scale from initial frames
    print("Computing distance scale from initial frames...")
    init_embs = []
    dem_count = 0
    for class_name, class_loader in downstream_loaders_pos.items():
        print(f"  Processing class: {class_name}")
        for batch in tqdm(class_loader, desc=f"Scale {class_name}", leave=False, total=min(len(class_loader), args.count)):
            if dem_count >= args.count:
                break
            frames = batch["frames"]  # shape: (1, T, C, H, W)
            if isinstance(frames, np.ndarray):
                frames = torch.tensor(frames, dtype=torch.float32)
            
            init_frame = frames[:, 0:1, :, :, :]  # (1, 1, C, H, W)
            
            with torch.no_grad():
                out = model.infer(init_frame.to(device))
            
            if hasattr(out, "embs"):
                emb = out.embs
            elif hasattr(out, "embedding"):
                emb = out.embedding
            else:
                emb = out
            
            emb = emb.squeeze().detach().cpu().numpy()
            if emb.ndim > 1:
                emb = emb.reshape(-1)
            init_embs.append(emb)
            dem_count += 1
        if dem_count >= args.count:
            break
    
    init_embs = np.stack(init_embs, axis=0)
    dist_to_goal = np.linalg.norm(init_embs - goal_emb, axis=1).mean()
    distance_scale = 1.0 / (dist_to_goal + 1e-8)
    print(f"Distance scale: {distance_scale:.6f}")
    
    if len(init_embs) == 0:
        print("ERROR: No initial embeddings computed. Check dataset loading.")
        return
    
    # Compute rewards for positive trajectories
    print("Computing rewards for positive trajectories...")
    pos_rewards_list = []
    dem_count = 0
    for class_name, class_loader in downstream_loaders_pos.items():
        print(f"  Processing class: {class_name}")
        for batch in tqdm(class_loader, desc=f"Pos {class_name}", leave=False, total=min(len(class_loader), args.count)):
            if dem_count >= args.count:
                break
            frames = batch["frames"]  # shape: (1, T, C, H, W)
            if isinstance(frames, np.ndarray):
                frames = torch.tensor(frames, dtype=torch.float32)
            
            try:
                rewards = _compute_trajectory_rewards(
                    frames, goal_emb, distance_scale, model, device
                )
                pos_rewards_list.append(rewards)
                dem_count += 1
            except Exception as e:
                print(f"    Error computing rewards: {e}")
                continue
        if dem_count >= args.count:
            break
    print(f"  Computed rewards for {len(pos_rewards_list)} positive trajectories")
    
    # Compute rewards for negative trajectories
    print("Computing rewards for negative trajectories...")
    neg_rewards_list = []
    dem_count = 0
    for class_name, class_loader in downstream_loaders_neg.items():
        print(f"  Processing class: {class_name}")
        for batch in tqdm(class_loader, desc=f"Neg {class_name}", leave=False, total=min(len(class_loader), args.count)):
            if dem_count >= args.count:
                break
            frames = batch["frames"]  # shape: (1, T, C, H, W)
            if isinstance(frames, np.ndarray):
                frames = torch.tensor(frames, dtype=torch.float32)
            
            try:
                rewards = _compute_trajectory_rewards(
                    frames, goal_emb, distance_scale, model, device
                )
                neg_rewards_list.append(rewards)
                dem_count += 1
            except Exception as e:
                print(f"    Error computing rewards: {e}")
                continue
        if dem_count >= args.count:
            break
    print(f"  Computed rewards for {len(neg_rewards_list)} negative trajectories")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if we have data
    print(f"\nLoaded {len(pos_rewards_list)} positive trajectories")
    print(f"Loaded {len(neg_rewards_list)} negative trajectories")
    
    if len(pos_rewards_list) == 0 or len(neg_rewards_list) == 0:
        print("ERROR: One or both datasets are empty. Cannot compute similarities.")
        return
    
    # Compute similarities
    print("Computing similarity scores...")
    tbar = tqdm(total=len(pos_rewards_list) * len(neg_rewards_list), desc="Comparing trajectories")
    similarities = []
    for pos_rew in pos_rewards_list:
        for neg_rew in neg_rewards_list:
            sim = dtw_reward_similarity(pos_rew, neg_rew)
            similarities.append(sim)
            tbar.update(1)
    tbar.close()
    
    if len(similarities) == 0:
        print("ERROR: No similarities computed. Check reward computation.")
        return
    
    # Statistics
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)
    
    print(f"\nSimilarity Statistics:")
    print(f"  Mean: {avg_similarity:.4f}")
    print(f"  Std: {std_similarity:.4f}")
    print(f"  Min: {min_similarity:.4f}")
    print(f"  Max: {max_similarity:.4f}")
    print(f"  Num comparisons: {len(similarities)}")
    
    # Save statistics
    stats_path = os.path.join(args.output_dir, "similarity_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Mean similarity: {avg_similarity:.4f}\n")
        f.write(f"Std similarity: {std_similarity:.4f}\n")
        f.write(f"Min similarity: {min_similarity:.4f}\n")
        f.write(f"Max similarity: {max_similarity:.4f}\n")
        f.write(f"Num comparisons: {len(similarities)}\n")
    
    # Visualize sample pairs
    n_samples = min(3, len(pos_rewards_list), len(neg_rewards_list))
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for idx in range(n_samples):
        pos_rew = pos_rewards_list[idx]
        neg_rew = neg_rewards_list[idx]
        sim = dtw_reward_similarity(pos_rew, neg_rew)
        
        ax = axes[idx]
        ax.plot(pos_rew, "o-", label="Positive", alpha=0.7, linewidth=2)
        ax.plot(neg_rew, "s-", label="Negative", alpha=0.7, linewidth=2)
        ax.set_title(f"Sample {idx + 1}: Similarity = {sim:.4f}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    sample_path = os.path.join(args.output_dir, "sample_comparisons.png")
    plt.savefig(sample_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sample comparisons to {sample_path}")
    
    # Visualize average trajectories
    # Pad trajectories to same length for averaging
    max_len_pos = max(len(r) for r in pos_rewards_list) if pos_rewards_list else 0
    max_len_neg = max(len(r) for r in neg_rewards_list) if neg_rewards_list else 0
    
    if max_len_pos > 0 and max_len_neg > 0:
        pos_padded = [np.pad(r, (0, max_len_pos - len(r)), mode='edge') for r in pos_rewards_list]
        neg_padded = [np.pad(r, (0, max_len_neg - len(r)), mode='edge') for r in neg_rewards_list]
        
        avg_pos = np.mean(pos_padded, axis=0)
        avg_neg = np.mean(neg_padded, axis=0)
        avg_sim = dtw_reward_similarity(
            avg_pos[:max_len_pos], 
            avg_neg[:max_len_neg]
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        steps_pos = np.arange(len(avg_pos))
        steps_neg = np.arange(len(avg_neg))
        ax.plot(steps_pos, avg_pos, "o-", label="Positive (avg)", alpha=0.8, linewidth=2.5)
        ax.plot(steps_neg, avg_neg, "s-", label="Negative (avg)", alpha=0.8, linewidth=2.5)
        ax.set_title(f"Average Trajectory Rewards: Similarity = {avg_sim:.4f}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        avg_path = os.path.join(args.output_dir, "average_comparison.png")
        plt.savefig(avg_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved average comparison to {avg_path}")
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
