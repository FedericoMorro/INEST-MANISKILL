#!/usr/bin/env python3
"""
Test script to compare env.render() vs obs returned by setting obs_mode to rgb
using the local StackPyramid environment.

This script:
1. Creates env with state obs and calls env.render() while performing random actions
2. Creates env with rgb obs and performs the same random actions (same seed)
3. Compares the rendered frames with the rgb observations frame by frame
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gymnasium as gym
import torch

# Add parent directories to path to import local modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from maniskill3.stack_pyramid import StackPyramidEnv


def _convert_to_numpy(data):
    """Convert torch tensors (including CUDA tensors) to numpy arrays."""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return data


def _squeeze_and_normalize(frame):
    """Remove batch dimension if present and normalize to [0, 255] range."""
    if frame is None:
        return None
    
    # Remove batch dimension if present (e.g., (1, H, W, C) -> (H, W, C))
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    
    # Normalize to [0, 255] range
    if frame.dtype == np.float32 or frame.dtype == np.float64:
        # If values are in [0, 1], scale to [0, 255]
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    
    return frame


def test_render_vs_rgb_obs(num_steps: int = 10,
                           seed: int = 22,
                           save_comparison: bool = False,
                           output_dir: str = "./out/render_comparison",
                           robot_uids: str = "panda_wristcam",
                           env_reward_type: str = "sparse"):
    """
    Test that env.render() and obs with obs_mode='rgb' produce identical frames
    using the local StackPyramid environment.
    
    Args:
        num_steps: Number of random steps to take
        seed: Random seed for reproducibility
        save_comparison: Whether to save comparison images
        output_dir: Directory to save comparison images
        robot_uids: Robot UID for the environment
        env_reward_type: Reward type for the environment
    
    Returns:
        Dictionary with comparison results
    """
    
    print(f"Testing StackPyramid environment")
    print(f"Number of steps: {num_steps}")
    print(f"Seed: {seed}")
    print(f"Robot: {robot_uids}")
    print("-" * 80)
    
    # Create output directory if needed
    if save_comparison:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # Part 1: Environment with state obs, using env.render()
    # ============================================================================
    print("\n[Part 1] Creating StackPyramid environment with STATE obs and using env.render()...")
    
    env_render = StackPyramidEnv(
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        robot_uids=robot_uids,
        env_reward_type=env_reward_type,
    )
    
    # Reset and get initial state
    env_render.set_seed(seed)
    obs_render, info = env_render.reset()
    print(f"Initial observation shape (state): {obs_render['agent']['qpos'].shape if isinstance(obs_render, dict) else obs_render.shape}")
    
    # Get initial rendered frame
    try:
        frame_render_initial = env_render.render()
    except TypeError:
        # Fallback for older Gym versions
        frame_render_initial = env_render.render(mode="rgb_array")
    
    # Convert CUDA tensors to numpy
    frame_render_initial = _convert_to_numpy(frame_render_initial)
    # Squeeze batch dimension and normalize
    frame_render_initial = _squeeze_and_normalize(frame_render_initial)
    
    if frame_render_initial is None:
        print("Error: env.render() returned None. Make sure render_mode='rgb_array' is set.")
        return None
    
    print(f"Initial rendered frame shape: {frame_render_initial.shape}, dtype: {frame_render_initial.dtype}")
    
    # Store all rendered frames and actions
    rendered_frames = [frame_render_initial]
    random_actions = []
    
    # Perform random actions and collect rendered frames
    for step in range(num_steps):
        action = env_render.action_space.sample()
        random_actions.append(action)
        
        obs_state, reward, terminated, truncated, info = env_render.step(action)
        
        # Get rendered frame
        try:
            frame = env_render.render()
        except TypeError:
            frame = env_render.render(mode="rgb_array")
        
        # Convert CUDA tensors to numpy
        frame = _convert_to_numpy(frame)
        # Squeeze batch dimension and normalize
        frame = _squeeze_and_normalize(frame)
        
        if frame is not None:
            rendered_frames.append(frame)
        
        if terminated or truncated:
            obs_state, info = env_render.reset()
        
        print(f"Step {step+1}/{num_steps}: action shape={action.shape}, frame shape={frame.shape if frame is not None else 'None'}")
    
    rendered_frames = np.array(rendered_frames)
    print(f"\nTotal rendered frames collected: {len(rendered_frames)}")
    print(f"Rendered frames shape: {rendered_frames.shape}")
    
    env_render.close()
    
    # ============================================================================
    # Part 2: Environment with RGB obs, performing same actions
    # ============================================================================
    print("\n" + "=" * 80)
    print("[Part 2] Creating StackPyramid environment with RGB obs...")
    
    env_rgb = StackPyramidEnv(
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        robot_uids=robot_uids,
        env_reward_type=env_reward_type,
    )
    
    # Reset with same seed
    env_rgb.set_seed(seed)
    obs_rgb, info = env_rgb.reset()
    
    # Extract RGB observation from obs dict if necessary
    if isinstance(obs_rgb, dict):
        print(f"Observation is a dict with keys: {obs_rgb.keys()}")
        # Look for RGB in sensor_data
        if 'sensor_data' in obs_rgb:
            sensor_data = obs_rgb['sensor_data']
            if isinstance(sensor_data, dict):
                # Try different camera keys
                if 'base_camera' in sensor_data:
                    camera_data = sensor_data['base_camera']
                    if isinstance(camera_data, dict):
                        rgb_obs = camera_data.get('rgb', camera_data.get('image', None))
                    else:
                        rgb_obs = camera_data
                elif 'hand_camera' in sensor_data:
                    camera_data = sensor_data['hand_camera']
                    if isinstance(camera_data, dict):
                        rgb_obs = camera_data.get('rgb', camera_data.get('image', None))
                    else:
                        rgb_obs = camera_data
                else:
                    rgb_obs = None
            else:
                rgb_obs = sensor_data
        # Fall back to other possibilities
        elif 'image' in obs_rgb:
            rgb_obs = obs_rgb['image']
        elif 'rgb' in obs_rgb:
            rgb_obs = obs_rgb['rgb']
        else:
            # Print the structure to debug
            print(f"Available keys in obs dict: {list(obs_rgb.keys())}")
            for key, val in obs_rgb.items():
                if isinstance(val, dict):
                    print(f"  {key}: dict with keys {list(val.keys())}")
                elif isinstance(val, (torch.Tensor, np.ndarray)):
                    print(f"  {key}: shape {val.shape}")
                else:
                    print(f"  {key}: {type(val)}")
            rgb_obs = None
    else:
        rgb_obs = obs_rgb
    
    # Convert CUDA tensors to numpy
    rgb_obs = _convert_to_numpy(rgb_obs)
    # Squeeze batch dimension and normalize
    rgb_obs = _squeeze_and_normalize(rgb_obs)
    
    print(f"Initial observation shape (rgb): {rgb_obs.shape if rgb_obs is not None else 'None'}, dtype: {rgb_obs.dtype if rgb_obs is not None else 'None'}")
    if rgb_obs is not None:
        print(f"Observation value range: [{rgb_obs.min()}, {rgb_obs.max()}]")
    
    # Store all rgb observations
    rgb_observations = [rgb_obs]
    
    # Perform the same random actions and collect rgb observations
    for step, action in enumerate(random_actions):
        obs_rgb, reward, terminated, truncated, info = env_rgb.step(action)
        
        # Extract RGB observation from obs dict if necessary
        if isinstance(obs_rgb, dict):
            if 'sensor_data' in obs_rgb:
                sensor_data = obs_rgb['sensor_data']
                if isinstance(sensor_data, dict):
                    if 'base_camera' in sensor_data:
                        camera_data = sensor_data['base_camera']
                        if isinstance(camera_data, dict):
                            rgb_obs = camera_data.get('rgb', camera_data.get('image', None))
                        else:
                            rgb_obs = camera_data
                    elif 'hand_camera' in sensor_data:
                        camera_data = sensor_data['hand_camera']
                        if isinstance(camera_data, dict):
                            rgb_obs = camera_data.get('rgb', camera_data.get('image', None))
                        else:
                            rgb_obs = camera_data
                    else:
                        rgb_obs = None
                else:
                    rgb_obs = sensor_data
            elif 'image' in obs_rgb:
                rgb_obs = obs_rgb['image']
            elif 'rgb' in obs_rgb:
                rgb_obs = obs_rgb['rgb']
            else:
                rgb_obs = None
        else:
            rgb_obs = obs_rgb
        
        # Convert CUDA tensors to numpy
        rgb_obs = _convert_to_numpy(rgb_obs)
        # Squeeze batch dimension and normalize
        rgb_obs = _squeeze_and_normalize(rgb_obs)
        
        rgb_observations.append(rgb_obs)
        
        if terminated or truncated:
            obs_rgb, info = env_rgb.reset()
        
        if rgb_obs is not None:
            print(f"Step {step+1}/{num_steps}: obs shape={rgb_obs.shape}, "
                  f"value range=[{rgb_obs.min()}, {rgb_obs.max()}]")
        else:
            print(f"Step {step+1}/{num_steps}: obs is None")
    
    rgb_observations = np.array(rgb_observations)
    print(f"\nTotal RGB observations collected: {len(rgb_observations)}")
    print(f"RGB observations shape: {rgb_observations.shape}")
    
    env_rgb.close()
    
    # ============================================================================
    # Part 3: Compare frames
    # ============================================================================
    print("\n" + "=" * 80)
    print("[Part 3] Comparing rendered frames with RGB observations...")
    
    # Verify shapes match
    if rendered_frames.shape != rgb_observations.shape:
        print(f"⚠️  WARNING: Shape mismatch!")
        print(f"   Rendered frames shape: {rendered_frames.shape}")
        print(f"   RGB observations shape: {rgb_observations.shape}")
    else:
        print(f"✓ Shapes match: {rendered_frames.shape}")
    
    # Compare frame by frame
    comparison_results = {
        "num_steps": num_steps,
        "total_frames": len(rendered_frames),
        "frame_comparisons": []
    }
    
    max_diff = 0
    mean_diff = 0
    num_frames_identical = 0
    num_frames_valid = 0
    
    for frame_idx in range(min(len(rendered_frames), len(rgb_observations))):
        frame_render = rendered_frames[frame_idx]
        frame_rgb = rgb_observations[frame_idx]
        
        # Skip if either frame is None
        if frame_render is None or frame_rgb is None:
            print(f"Frame {frame_idx:2d}: ✗ SKIPPED (None value)")
            continue
        
        num_frames_valid += 1
        
        # Handle potential dtype differences
        if frame_render.dtype != frame_rgb.dtype:
            print(f"Frame {frame_idx}: dtype mismatch: {frame_render.dtype} vs {frame_rgb.dtype}")
            # Convert both to float for comparison
            frame_render_f = frame_render.astype(np.float32)
            frame_rgb_f = frame_rgb.astype(np.float32)
        else:
            frame_render_f = frame_render.astype(np.float32)
            frame_rgb_f = frame_rgb.astype(np.float32)
        
        # Calculate differences
        diff = np.abs(frame_render_f - frame_rgb_f)
        max_frame_diff = np.max(diff)
        mean_frame_diff = np.mean(diff)
        is_identical = np.allclose(frame_render_f, frame_rgb_f)
        
        max_diff = max(max_diff, max_frame_diff)
        mean_diff += mean_frame_diff
        
        if is_identical:
            num_frames_identical += 1
            status = "✓ IDENTICAL"
        else:
            status = f"✗ DIFFERENT (max diff={max_frame_diff:.4f}, mean diff={mean_frame_diff:.4f})"
        
        frame_result = {
            "frame_idx": frame_idx,
            "is_identical": is_identical,
            "max_diff": float(max_frame_diff),
            "mean_diff": float(mean_frame_diff),
        }
        comparison_results["frame_comparisons"].append(frame_result)
        
        print(f"Frame {frame_idx:2d}: {status}")
    
    if num_frames_valid > 0:
        mean_diff /= num_frames_valid
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Total frames collected: {len(rendered_frames)}")
    print(f"Valid frames compared: {num_frames_valid}")
    print(f"Identical frames: {num_frames_identical}/{num_frames_valid} "
          f"({100*num_frames_identical/num_frames_valid:.1f}%)" if num_frames_valid > 0 else "N/A")
    print(f"Maximum difference across all frames: {max_diff:.6f}")
    print(f"Mean difference across all frames: {mean_diff:.6f}")
    
    comparison_results.update({
        "num_frames_valid": num_frames_valid,
        "num_frames_identical": num_frames_identical,
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
    })
    
    # ============================================================================
    # Part 4: Save comparison visualizations (optional)
    # ============================================================================
    if save_comparison:
        print("\n" + "=" * 80)
        print("[Part 4] Saving comparison visualizations...")
        
        saved_count = 0
        for frame_idx in range(min(len(rendered_frames), len(rgb_observations))):  # Save first 5 frames
            if saved_count >= 5:
                break
            
            frame_render = rendered_frames[frame_idx]
            frame_rgb = rgb_observations[frame_idx]
            
            # Skip if either is None
            if frame_render is None or frame_rgb is None:
                continue
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Render frame
            axes[0].imshow(frame_render.astype(np.uint8) if frame_render.dtype != np.uint8 else frame_render)
            axes[0].set_title(f"env.render()\nFrame {frame_idx}")
            axes[0].axis('off')
            
            # RGB observation
            axes[1].imshow(frame_rgb.astype(np.uint8) if frame_rgb.dtype != np.uint8 else frame_rgb)
            axes[1].set_title(f"obs_mode='rgb'\nFrame {frame_idx}")
            axes[1].axis('off')
            
            # Difference
            diff = np.abs(frame_render.astype(np.float32) - frame_rgb.astype(np.float32))
            im = axes[2].imshow(diff, cmap='hot')
            axes[2].set_title(f"Absolute Difference\nFrame {frame_idx}")
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2])
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"frame_comparison_{frame_idx:02d}.png")
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close()
            saved_count += 1
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(
        description="Test StackPyramid env.render() vs obs_mode='rgb' frame comparison"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of random steps to take"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save comparison visualizations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./render_comparison",
        help="Output directory for saved comparisons"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="panda_wristcam",
        help="Robot UID (panda_wristcam, panda, fetch)"
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="sparse",
        help="Reward type (sparse, dense, normalized_dense)"
    )
    
    args = parser.parse_args()
    
    results = test_render_vs_rgb_obs(
        num_steps=args.steps,
        seed=args.seed,
        save_comparison=args.save,
        output_dir=args.output_dir,
        robot_uids=args.robot,
        env_reward_type=args.reward_type
    )
    
    if results is not None:
        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Test failed!")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
