from absl import app, flags, logging
import json
import matplotlib.pyplot as plt
import mani_skill.envs
import numpy as np
import os
from pathlib import Path
import torch
import gymnasium as gym
from types import MethodType
import inspect
import imageio

from stable_baselines3 import SAC

from inest_irl.utils import utils

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", None, "Path to checkpoint directory (required).")
flags.DEFINE_string("config_path", "/home/fmorro/INEST-MANISKILL/scripts/configs/sb3_sac.py", "Path to the configuration file.")
flags.DEFINE_integer("num_episodes", 10, "Number of evaluation episodes.")
flags.DEFINE_boolean("save_video", True, "Whether to save video recordings.")
flags.DEFINE_string("device", "cpu", "Device to use (e.g., 'cpu', 'cuda:0').")

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
        logging.info(f"Saving video: {video_path}")
        with imageio.get_writer(video_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        logging.info(f"Saved video: {video_path} ({len(frames)} frames)")
    except Exception as e:
        logging.error(f"Error saving video {video_path}: {e}")
        import traceback
        traceback.print_exc()

def _save_reward_plot(plot_path, step_rewards):
    """Save reward curve plot as PNG with per-step and cumulative plots."""
    try:
        plt.figure(figsize=FIGSIZE_TRAJ)
        
        # Rewards with average line
        plt.plot(step_rewards, linewidth=2, color='steelblue', label='Reward')
        plt.axhline(np.mean(step_rewards), color='red', linestyle=':', linewidth=2, 
                    label=f'Avg: {np.mean(step_rewards):.2f}')
        plt.xlabel('Step', fontsize=FS_LABEL)
        plt.ylabel('Reward', fontsize=FS_LABEL)
        plt.title('Reward', fontsize=FS_TITLE, fontweight='bold')
        plt.legend(fontsize=FS_LEGEND)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved reward plot: {plot_path}")
    except Exception as e:
        logging.error(f"Error saving reward plot {plot_path}: {e}")

def _run_episode(model, env, video_path=None):
    """Run single evaluation episode manually, optionally recording video."""
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
    
    while step_count < 250:
        action, _ = model.predict(obs, deterministic=True)
        step_result = env.step(action)
        
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, terminated, truncated = step_result
        
        reward = float(reward)
        episode_reward += reward
        step_rewards.append(reward)
        step_count += 1
        done = terminated or truncated
        
        # Render frame if video saving enabled
        if FLAGS.save_video:
            try:
                frame = safe_render(env, mode="rgb_array")
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                logging.debug(f"Error getting frame: {e}")
    
    # Save video if frames were collected
    if video_path and frames:
        _save_video(video_path, frames)
    
    return episode_reward, step_count, len(frames), step_rewards


def main(_):
    # Validate checkpoint path
    if not FLAGS.checkpoint_dir:
        raise ValueError("--checkpoint_dir is required")
    if not os.path.exists(FLAGS.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found: {FLAGS.checkpoint_dir}")

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(FLAGS.device)
    else:
        logging.warning("No GPU found, using CPU")
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Load configuration
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", FLAGS.config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.get_config() if hasattr(config_module, 'get_config') else config_module.config

    # Create evaluation environment
    logging.info(f"Creating environment: {config.env_name}")
    eval_env = utils.make_env(
        config.env_name,
        seed=42,
        action_repeat=config.action_repeat,
        frame_stack=config.frame_stack,
    )
    
    # Patch render compatibility in the wrapper chain
    eval_env = patch_env_render_compatibility(eval_env)

    # Load model
    logging.info(f"Loading checkpoint from {FLAGS.checkpoint_dir}...")
    model = SAC.load(FLAGS.checkpoint_dir, device=device)

    # Create output directories
    checkpoint_dir = os.path.dirname(FLAGS.checkpoint_dir)
    results_dir = os.path.join(checkpoint_dir, "eval_results")
    traj_dir = os.path.join(results_dir, "trajs") if FLAGS.save_video else None
    if traj_dir:
        os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Run evaluation episodes manually
    logging.info(f"Running {FLAGS.num_episodes} evaluation episodes...")
    rewards = []
    lengths = []
    frame_counts = []

    for episode_num in range(FLAGS.num_episodes):
        video_path = None
        if FLAGS.save_video and traj_dir:
            video_path = os.path.join(traj_dir, f"{episode_num}.mp4")
        
        episode_reward, episode_length, frame_count, step_rewards = _run_episode(model, eval_env, video_path)
        rewards.append(episode_reward)
        lengths.append(episode_length)
        frame_counts.append(frame_count)
        
        # Save reward curve plot
        if FLAGS.save_video and traj_dir:
            plot_path = os.path.join(traj_dir, f"{episode_num}.png")
            _save_reward_plot(plot_path, step_rewards)
        
        logging.info(f"Episode {episode_num}: reward={episode_reward:.4f}, length={episode_length}, frames={frame_count}")

    # Compute statistics
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_length = float(np.mean(lengths))
    std_length = float(np.std(lengths))

    # Log results
    logging.info("=" * 50)
    logging.info("Evaluation Results")
    logging.info("=" * 50)
    logging.info(f"Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
    logging.info(f"Mean Episode Length: {mean_length:.2f} ± {std_length:.2f}")
    logging.info(f"Min Reward: {np.min(rewards):.4f}")
    logging.info(f"Max Reward: {np.max(rewards):.4f}")
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
        "individual_rewards": rewards,
        "episode_lengths": lengths,
        "frame_counts": frame_counts,
    }
    
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_path}")

    # Create plots
    _create_plots(results, results_dir)

    eval_env.close()
    logging.info("Evaluation complete!")


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


if __name__ == "__main__":
    app.run(main)