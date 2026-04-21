import argparse
import h5py
import imageio
import inspect
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm
from types import MethodType, SimpleNamespace
import yaml

from stable_baselines3 import SAC

from inest_irl.utils import utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


HORIZON = 100
DETERMINISTIC = True

# Report-friendly plotting defaults
FIGSIZE_TRAJ = (7.0, 3.6)
FIGSIZE_SUMMARY = (12.0, 4.5)
FS_LABEL = 12
FS_TITLE = 13
FS_LEGEND = 10


def safe_render(env, mode="rgb_array", **kwargs):
    """Safely call env.render, handling ManiSkill/Gym differences and ensuring
    a single (H, W, 3) RGB NumPy frame is always returned (even if batched)."""
    
    frame = None
    try:
        # Try Gym-style render first
        frame = env.render(mode=mode, **kwargs)
    except TypeError as e:
        if "positional argument" in str(e) or "unexpected keyword argument" in str(e):
            # Fallback to ManiSkill-style render()
            try:
                frame = env.render()
            except Exception:
                raise e
        else:
            raise e

    # ---- Normalize ManiSkill / Gym render output ----
    if frame is None:
        return None

    # Handle dicts or lists (e.g., ManiSkill multiple cameras)
    if isinstance(frame, dict):
        # Pick the first camera image
        frame = next(iter(frame.values()))
    elif isinstance(frame, (list, tuple)):
        frame = frame[0]

    # Convert torch.Tensor → numpy
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()

    # Handle batched ManiSkill outputs (num_envs, H, W, 3)
    if isinstance(frame, np.ndarray):
        if frame.ndim == 4 and frame.shape[-1] == 3:
            frame = frame[0]
        elif frame.ndim == 5 and frame.shape[-1] == 3:
            frame = frame[0, 0]

    # Convert float images to uint8 if needed
    if isinstance(frame, np.ndarray) and frame.dtype != np.uint8:
        fmin, fmax = frame.min(), frame.max()
        if fmin >= 0.0 and fmax <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        elif fmin >= -1.0 and fmax <= 1.0:
            frame = ((frame + 1.0) / 2.0 * 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Final sanity check
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[-1] != 3:
        logging.debug(f"Warning: unexpected frame shape {getattr(frame, 'shape', None)}")
        return None

    return frame


def patch_env_render_compatibility(env):
    """Automatically patch any wrapper in the env chain that has render signature mismatch."""
    
    def make_compatible_render(original_render):
        """Create a compatible render method from an incompatible one."""
        def compatible_render(self, mode="rgb_array", **kwargs):
            try:
                # First try to call original with mode
                sig = inspect.signature(original_render)
                if "mode" in sig.parameters or len(sig.parameters) > 1:
                    return original_render(mode=mode, **kwargs)
                else:
                    # Original only accepts self, call without args
                    return original_render()
            except TypeError:
                # Fallback to no-args call
                return original_render()
        return compatible_render
    
    # Walk the wrapper chain and patch incompatible render methods
    current_env = env
    patched_count = 0
    
    while current_env is not None:
        render_method = getattr(current_env, 'render', None)
        if render_method is not None:
            try:
                sig = inspect.signature(render_method)
                params = list(sig.parameters.keys())
                
                # Check if this render method only accepts 'self'
                if len(params) <= 1 and "mode" not in sig.parameters:
                    # Patch this wrapper's render method
                    compatible_method = make_compatible_render(render_method)
                    current_env.render = MethodType(compatible_method, current_env)
                    logging.info(f"Patched render method on {type(current_env).__name__}")
                    patched_count += 1
            except (ValueError, TypeError):
                # Can't inspect signature, skip
                pass
        
        # Move to next wrapper in chain
        if hasattr(current_env, 'env'):
            current_env = current_env.env
        elif hasattr(current_env, 'unwrapped') and current_env.unwrapped != current_env:
            current_env = current_env.unwrapped
        else:
            break
    
    logging.info(f"Patched {patched_count} wrapper(s) for render compatibility")
    return env

def _save_video(video_path, frames, fps=10):
    """Save frames to video file using imageio."""
    if not frames:
        logging.warning(f"No frames to save for video")
        return
    
    try:
        #logging.info(f"Saving video: {video_path}")
        with imageio.get_writer(video_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        #logging.info(f"Saved video: {video_path} ({len(frames)} frames)")
    except Exception as e:
        logging.error(f"Error saving video {video_path}: {e}")
        import traceback
        traceback.print_exc()

def _save_reward_plot(plot_path, step_rewards, subgoal_idxs=None):
    """Save reward curve plot as PNG with per-step and cumulative plots."""
    try:
        plt.figure(figsize=FIGSIZE_TRAJ)
        
        # Rewards with average line
        plt.plot(step_rewards, linewidth=2, color='steelblue', label='Reward')
        plt.axhline(np.mean(step_rewards), color='green', linestyle=':', linewidth=2, alpha=0.7,
                    label=f'Avg: {np.mean(step_rewards):.2f}')
        if subgoal_idxs:
            for idx in subgoal_idxs:
                plt.axvline(idx, color='red', linestyle='--', alpha=0.7, label='Subgoal(s)' if idx == subgoal_idxs[0] else "")
        plt.xlabel('Step', fontsize=FS_LABEL)
        plt.ylabel('Reward', fontsize=FS_LABEL)
        plt.title('Reward', fontsize=FS_TITLE, fontweight='bold')
        plt.legend(fontsize=FS_LEGEND)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        #logging.info(f"Saved reward plot: {plot_path}")
    except Exception as e:
        logging.error(f"Error saving reward plot {plot_path}: {e}")

def _save_trajectory_h5(h5_path, json_path, env, episodes_data, save_rgb=False):
    """Save evaluation trajectories to H5 and metadata to JSON with optional RGB observations, rewards, and subgoals."""
    try:
        # Save metadata JSON
        json_data = {
            "env_info": {
                "env_id": env.unwrapped.spec.id,
                "env_kwargs": {
                    "sim_backend": getattr(env.unwrapped.backend, 'sim_backend', 'physx_cpu'),
                    "obs_mode": env.unwrapped.obs_mode,
                    "control_mode": env.unwrapped.control_mode,
                },
                "max_episode_steps": env.spec.max_episode_steps if env.spec else 100,
            },
            "episodes": []
        }
        
        # Save trajectories to H5
        with h5py.File(h5_path, 'w') as f:
            for episode_id, data in enumerate(episodes_data):
                actions = data.get('actions', [])
                observations = data.get('observations', []) if save_rgb else []
                rewards = data.get('rewards', [])
                subgoal_idxs = data.get('subgoal_idxs', [])
                
                if len(actions) > 0:
                    actions_array = np.array(actions, dtype=np.float32)
                    f.create_dataset(f'traj_{episode_id}/actions', data=actions_array, compression='gzip')
                
                if save_rgb and len(observations) > 0:
                    obs_array = np.array(observations, dtype=np.uint8)
                    f.create_dataset(f'traj_{episode_id}/obs/sensor_data/hand_camera/rgb', data=obs_array, compression='gzip')
                
                if len(rewards) > 0:
                    rewards_array = np.array(rewards, dtype=np.float32)
                    f.create_dataset(f'traj_{episode_id}/rewards', data=rewards_array, compression='gzip')
                
                if len(subgoal_idxs) > 0:
                    subgoal_array = np.array(subgoal_idxs, dtype=np.int32)
                    f.create_dataset(f'traj_{episode_id}/subgoal_idxs', data=subgoal_array, compression='gzip')
                
                json_data["episodes"].append({
                    "episode_id": episode_id,
                    "episode_seed": data.get('seed', 0),
                    "reset_kwargs": {"seed": data.get('seed', 0)},
                    "control_mode": env.unwrapped.control_mode,
                    "elapsed_steps": data.get('elapsed_steps', 0),
                })
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logging.info(f"Saved trajectory H5 to {h5_path} and metadata to {json_path}")
    except Exception as e:
        logging.error(f"Error saving trajectory H5: {e}")
        import traceback
        traceback.print_exc()

def _run_episode(model, env, video_path=None, save_video=True, save_rgb=False):
    """Run single evaluation episode, optionally collecting RGB observations, actions, rewards, and subgoal indices."""
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
    
    done = False
    step_count = 0
    episode_reward = 0.0
    step_rewards = []
    frames = []
    subgoal_idxs = []
    actions = []
    observations = []
    rewards = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=DETERMINISTIC)
        actions.append(action)
        step_result = env.step(action)
        
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, terminated, info = step_result
            truncated = False
        
        reward = float(reward)
        episode_reward += reward
        step_rewards.append(reward)
        rewards.append(reward)
        step_count += 1
        done = terminated or truncated

        curr_subgoal = info.get("subgoal", None)
        if curr_subgoal is not None and curr_subgoal > len(subgoal_idxs):
            subgoal_idxs.append(step_count)
        
        # Render frame for both video and optional observation storage
        if save_video or save_rgb:
            try:
                frame = safe_render(env, mode="rgb_array")
                if frame is not None:
                    if save_video:
                        frames.append(frame)
                    if save_rgb:
                        observations.append(frame)
            except Exception as e:
                logging.debug(f"Error getting frame: {e}")
    
    # Save video if frames were collected
    if video_path and frames:
        _save_video(video_path, frames)
    
    return episode_reward, step_count, len(frames), step_rewards, subgoal_idxs, actions, observations, rewards


def _create_plots(results, output_dir):
    """Generate evaluation plots."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_SUMMARY)

    # Plot 1: Reward distribution
    rewards = results["individual_rewards"]
    axes[0].bar(range(len(rewards)), rewards, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    axes[0].axhline(results["mean_reward"], color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {results['mean_reward']:.2f}")
    axes[0].fill_between(
        range(len(rewards)),
        results["mean_reward"] - results["std_reward"],
        results["mean_reward"] + results["std_reward"],
        alpha=0.2,
        color='red',
        label='±1 Std Dev'
    )
    axes[0].set_xlabel("Episode", fontsize=FS_LABEL)
    axes[0].set_ylabel("Reward", fontsize=FS_LABEL)
    axes[0].set_title("Episode Rewards", fontsize=FS_TITLE, fontweight='bold')
    axes[0].legend(fontsize=FS_LEGEND)
    axes[0].grid(alpha=0.3)

    # Plot 2: Episode length distribution
    lengths = results["episode_lengths"]
    axes[1].bar(range(len(lengths)), lengths, alpha=0.7, color='coral', edgecolor='black', linewidth=0.5)
    axes[1].axhline(results["mean_length"], color='darkred', linestyle='--', linewidth=2,
                    label=f"Mean: {results['mean_length']:.1f}")
    axes[1].set_xlabel("Episode", fontsize=FS_LABEL)
    axes[1].set_ylabel("Episode Length (steps)", fontsize=FS_LABEL)
    axes[1].set_title("Episode Lengths", fontsize=FS_TITLE, fontweight='bold')
    axes[1].legend(fontsize=FS_LEGEND)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Plots saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL policy")
    parser.add_argument("model_path", type=str, help="Path to the trained model checkpoint")
    parser.add_argument("--seed", type=int, default=2222, help="Random seed for evaluation")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--save_viz", action="store_true", default=True,
                        help="Whether to save visualization (videos and plots)")
    parser.add_argument("--no_save_video", action="store_false", dest="save_video",
                        help="Disable video recording")
    parser.add_argument("--save_rgb", action="store_true", default=False,
                        help="Save RGB observations to H5 file for dataset creation")
    parser.add_argument("--no_progress_bar", action="store_true", default=False,
                        help="Disable progress bar")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (e.g., 'cpu', 'cuda:0')")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate checkpoint path
    if not args.model_path:
        raise ValueError("model_path is required")
    
    checkpoint_dir = args.model_path
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        logging.warning("No GPU found, using CPU")
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Load configuration from YAML in experiment root folder (e.g., root/checkpoints/model.zip -> root/config.yaml)
    config_root = os.path.dirname(os.path.dirname(checkpoint_dir))
    
    config_path = os.path.join(config_root, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config YAML not found at {config_path}")
    
    logging.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    def _dict_to_namespace(data):
        """Recursively convert nested dicts to SimpleNamespace objects."""
        if isinstance(data, dict):
            return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in data.items()})
        elif isinstance(data, list):
            return [_dict_to_namespace(item) for item in data]
        return data
    
    # Convert dict to namespace with nested support
    config = _dict_to_namespace(config_dict)

    # Create evaluation environment
    logging.info(f"Creating environment: {config.env_name}")
    eval_env = utils.make_env(
        config.env_name,
        seed=args.seed,
        reward_type=config.reward_wrapper.type,
        action_repeat=config.action_repeat,
        frame_stack=config.frame_stack,
    )
    
    # Patch render compatibility in the wrapper chain
    eval_env = patch_env_render_compatibility(eval_env)

    # Load model
    logging.info(f"Loading checkpoint from {checkpoint_dir}...")
    model = SAC.load(checkpoint_dir, device=device)

    # Create output directories: parent/eval_results/{model_basename}/
    model_basename = os.path.basename(checkpoint_dir).split('.')[0]
    parent_dir = os.path.dirname(os.path.dirname(checkpoint_dir))
    results_dir = os.path.join(parent_dir, "eval_results", model_basename)
    traj_dir = os.path.join(results_dir, "trajs") if args.save_viz else None
    if traj_dir:
        os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Run evaluation episodes manually
    logging.info(f"Running {args.num_episodes} evaluation episodes...")
    rewards = []
    lengths = []
    frame_counts = []
    subgoal_idxs_all = []
    episodes_data = []
        
    for episode_num in tqdm(range(args.num_episodes), desc="Evaluating", unit="episode", disable=args.no_progress_bar):
        video_path = None
        if args.save_viz and traj_dir:
            video_path = os.path.join(traj_dir, f"{episode_num}.mp4")
            
        eval_env.unwrapped.set_seed(args.seed + episode_num)
        
        episode_reward, episode_length, frame_count, step_rewards, subgoal_idxs, actions, observations, ep_rewards = _run_episode(
            model, eval_env, video_path, save_video=args.save_viz, save_rgb=args.save_rgb
        )
        rewards.append(episode_reward)
        lengths.append(episode_length)
        frame_counts.append(frame_count)
        subgoal_idxs_all.append(subgoal_idxs)
        
        # Store trajectory data for H5 saving
        episodes_data.append({
            'seed': args.seed + episode_num,
            'elapsed_steps': episode_length,
            'actions': actions,
            'observations': observations,
            'rewards': ep_rewards,
            'subgoal_idxs': subgoal_idxs,
        })
        
        # Save reward curve plot
        if args.save_viz and traj_dir:
            plot_path = os.path.join(traj_dir, f"{episode_num}.png")
            _save_reward_plot(plot_path, step_rewards, subgoal_idxs)
        
        logging.info(f"Episode {episode_num}: reward={episode_reward:.4f}, length={episode_length}, frames={frame_count}, subgoal={len(subgoal_idxs)}")

    # Compute statistics
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_length = float(np.mean(lengths))
    std_length = float(np.std(lengths))
    
    # Convert subgoal indices to dict of reach rates
    max_subgoal = eval_env.unwrapped.max_subgoal
    episode_subgoals_dict = {i: 0 for i in range(max_subgoal + 1)}
    for subgoal_idxs in subgoal_idxs_all:
        # interpret first as failure rate, then cumulative reach rates for each subgoal
        subgoal_reached = len(subgoal_idxs)
        if subgoal_reached == 0:
            episode_subgoals_dict[0] += 1
        else:
            for idx in range(1, subgoal_reached + 1):
                episode_subgoals_dict[idx] += 1
    episode_subgoals_dict = {k: v / args.num_episodes for k, v in episode_subgoals_dict.items()}
    
    success_rate = episode_subgoals_dict.get(max_subgoal, 0.0)
    avg_subgoal_reached = np.mean([len(idxs) for idxs in subgoal_idxs_all])

    # Log results
    logging.info("=" * 50)
    logging.info("Evaluation Results")
    logging.info("=" * 50)
    logging.info(f"Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
    logging.info(f"Min-Max Reward: {np.min(rewards):.4f} - {np.max(rewards):.4f}")
    logging.info(f"Success Rate: {success_rate:.4f}")
    logging.info(f"Average Subgoals Reached: {avg_subgoal_reached:.2f} / {max_subgoal}")
    if traj_dir:
        logging.info(f"Trajectories saved to: {traj_dir}")
    logging.info("=" * 50)

    # Save results to JSON
    results = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "std_length": std_length,
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "success_rate": success_rate,
        "average_subgoals_reached": avg_subgoal_reached,
        "subgoal_reach_rates": episode_subgoals_dict,
        "individual_rewards": rewards,
        "episode_lengths": lengths,
        "frame_counts": frame_counts,
        "subgoal_idxs": subgoal_idxs_all
    }
    
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_path}")

    # Save trajectories to H5 for replay compatibility
    h5_path = os.path.join(results_dir, "trajectories.h5")
    json_traj_path = os.path.join(results_dir, "trajectories.json")
    _save_trajectory_h5(h5_path, json_traj_path, eval_env, episodes_data, save_rgb=args.save_rgb)

    # Create plots
    _create_plots(results, results_dir)

    eval_env.close()
    logging.info("Evaluation complete!")



if __name__ == "__main__":
    main()