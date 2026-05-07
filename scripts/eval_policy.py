import argparse
import h5py
import imageio
import json
import logging
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
from PIL import Image
import torch
from tqdm import tqdm
from types import SimpleNamespace
import yaml

from stable_baselines3 import SAC

from inest_irl.maniskill3.stack_pyramid import MAX_SUBGOAL
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


class EvalResults:
    """Structured results from evaluation episodes."""
    def __init__(self):
        self.rewards = []
        self.lengths = []
        self.frame_counts = []
        self.subgoal_idxs_all = []
        self.env_rewards = []
        self.detected_subgoal_idxs_all = []
        
    def init_episodes_data(self, seed):
        self.seed = seed
        self.episodes_data = []
        
    def add_episode(self, rewards, length, frame_count, subgoal_idxs, env_rewards, detected_subgoal_idxs):
        self.rewards.append(rewards)
        self.lengths.append(length)
        self.frame_counts.append(frame_count)
        self.subgoal_idxs_all.append(subgoal_idxs)
        self.env_rewards.append(env_rewards)
        self.detected_subgoal_idxs_all.append(detected_subgoal_idxs)
        
    def add_episode_data(self, episode_num, length, actions, observations, rewards, subgoal_idxs):
        self.episodes_data.append({
            'seed': self.seed + episode_num,
            'elapsed_steps': length,
            'actions': actions,
            'observations': observations,
            'rewards': rewards,
            'subgoal_idxs': subgoal_idxs,
        })
        
    def get_last_episode(self):
        return {
            "rewards": self.rewards[-1] if self.rewards else None,
            "length": self.lengths[-1] if self.lengths else None,
            "frame_count": self.frame_counts[-1] if self.frame_counts else None,
            "subgoal_idxs": self.subgoal_idxs_all[-1] if self.subgoal_idxs_all else None,
            "env_rewards": self.env_rewards[-1] if self.env_rewards else None,
            "detected_subgoal_idxs": self.detected_subgoal_idxs_all[-1] if self.detected_subgoal_idxs_all else None,
        }
        
    def to_dict(self):
        return {
            "rewards": self.rewards,
            "lengths": self.lengths,
            "frame_counts": self.frame_counts,
            "subgoal_idxs_all": self.subgoal_idxs_all,
            "env_rewards": self.env_rewards,
            "detected_subgoal_idxs_all": self.detected_subgoal_idxs_all,
            "episodes_data": self.episodes_data if hasattr(self, 'episodes_data') else [],
        }
        
    def merge_results(self, other):
        self.rewards.extend(other.rewards)
        self.lengths.extend(other.lengths)
        self.frame_counts.extend(other.frame_counts)
        self.subgoal_idxs_all.extend(other.subgoal_idxs_all)
        self.env_rewards.extend(other.env_rewards)
        self.detected_subgoal_idxs_all.extend(other.detected_subgoal_idxs_all)
        if hasattr(self, 'episodes_data') and hasattr(other, 'episodes_data'):
            self.episodes_data.extend(other.episodes_data)


def _save_video(video_path, frames, fps=10):
    """Save frames to video file using imageio."""
    if not frames:
        logging.warning(f"No frames to save for video")
        return
    
    def _upscale_frame(frame, target_size=(512, 512)):
        img = Image.fromarray(frame)
        img = img.resize(target_size, resample=Image.BILINEAR)
        return np.array(img)
    
    try:
        #logging.info(f"Saving video: {video_path}")
        with imageio.get_writer(video_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(_upscale_frame(frame))
        #logging.info(f"Saved video: {video_path} ({len(frames)} frames)")
    except Exception as e:
        logging.error(f"Error saving video {video_path}: {e}")
        import traceback
        traceback.print_exc()
        
        
def generate_reward_plot(ax, rewards, subgoal_idxs=None, env_rewards=None, detected_subgoal_idxs=None, title="Reward", fontsizes={}):
    """Generate reward curve plot on given axis with optional subgoal and environment reward annotations."""
    
    ax.plot(rewards, linewidth=2, color='steelblue', label='Reward')
    ax.axhline(np.mean(rewards), color='blue', linestyle='-.', linewidth=2, alpha=0.5,
                label=f'Avg: {np.mean(rewards):.2f}')
    if subgoal_idxs:
        for idx in subgoal_idxs:
            ax.axvline(idx, color='purple', linestyle=':', alpha=0.7, label='GT Subgoal(s)' if idx == subgoal_idxs[0] else "")
    if detected_subgoal_idxs:
        for idx in detected_subgoal_idxs:
            ax.axvline(idx, color='green', linestyle='--', alpha=0.7, label='Detected Subgoal(s)' if idx == detected_subgoal_idxs[0] else "")

    ax2 = None
    if env_rewards is not None:
        ax2 = ax.twinx()
        ax2.plot(env_rewards, linewidth=2, color='orange', label='Env Reward')
        ax2.set_ylabel('Env Reward', fontsize=fontsizes.get('label', None))
        ax2.tick_params(axis='y', labelcolor='orange')
        
    ax.set_xlabel('Step', fontsize=fontsizes.get('label', None))
    ax.set_ylabel('Reward', fontsize=fontsizes.get('label', None))
    ax.set_title(title, fontsize=fontsizes.get('title', None), fontweight='bold')
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
        
    ax.legend(handles, labels, fontsize=fontsizes.get('legend', None))

def _save_reward_plot(plot_path, step_rewards, subgoal_idxs=None, env_rewards=None, detected_subgoal_idxs=None):
    """Save reward curve plot as PNG with per-step and cumulative plots."""
    try:
        plt.figure(figsize=FIGSIZE_TRAJ)
        ax1 = plt.gca()
        fontsizes = {'label': FS_LABEL, 'title': FS_TITLE, 'legend': FS_LEGEND}
        
        generate_reward_plot(ax1, step_rewards, subgoal_idxs, env_rewards, detected_subgoal_idxs, "Reward", fontsizes)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        #logging.info(f"Saved reward plot: {plot_path}")
    except Exception as e:
        logging.error(f"Error saving reward plot {plot_path}: {e}")

def _save_trajectory_h5(h5_path, json_path, env_config, episodes_data, save_rgb=False):
    """Save evaluation trajectories to H5 and metadata to JSON with optional RGB observations, rewards, and subgoals."""
    env = utils.make_env(
        env_config.env_name,
        seed=env_config.seed,
        reward_type=env_config.reward_wrapper.type,
        action_repeat=env_config.action_repeat,
        frame_stack=env_config.frame_stack,
        learned_reward_data=env_config.learned_reward_data,
    )
    
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
                actions = data['actions']
                observations = data['observations']
                rewards = data['rewards']
                subgoal_idxs = data.get('subgoal_idxs', [])
                
                actions_array = np.array(actions, dtype=np.float32)
                f.create_dataset(f'traj_{episode_id}/actions', data=actions_array, compression='gzip')
                
                if save_rgb:
                    obs_array = np.array(observations, dtype=np.uint8)
                    f.create_dataset(f'traj_{episode_id}/obs/sensor_data/base_camera/rgb', data=obs_array, compression='gzip')
                
                rewards_array = np.array(rewards, dtype=np.float32)
                f.create_dataset(f'traj_{episode_id}/rewards', data=rewards_array, compression='gzip')
                
                # convert subgoal idxs to per-step subgoal data and save as extra obs for dataset creation handling
                subgoal_data = []
                for i in range(len(actions)):
                    subgoal_data.append([len([idx for idx in subgoal_idxs if idx <= i])])
                subgoal_array = np.array(subgoal_data, dtype=np.int32)
                f.create_dataset(f'traj_{episode_id}/obs/extra/subgoal', data=subgoal_array, compression='gzip')
                
                json_data["episodes"].append({
                    "episode_id": episode_id,
                    "episode_seed": data.get('seed', 0),
                    "reset_kwargs": {"seed": data.get('seed', 0)},
                    "control_mode": env.unwrapped.control_mode,
                    "elapsed_steps": data.get('elapsed_steps', 0),
                })
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        env.close()
        
        logging.info(f"Saved trajectory H5 to {h5_path} and metadata to {json_path}")
    except Exception as e:
        env.close()
        logging.error(f"Error saving trajectory H5: {e}")
        import traceback
        traceback.print_exc()

def _run_episode(model, env, eval_results, episode_num, video_path=None, save_video=True, save_rgb=False):
    """Run single evaluation episode, optionally collecting RGB observations, actions, rewards, and subgoal indices."""
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
    
    done = False
    step_count = 0
    ep_cum_reward = 0.0
    frames = []
    actions = []
    observations = []
    rewards = []
    subgoal_idxs = None
    env_rewards = None
    detected_subgoal_idxs = None
    
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
        ep_cum_reward += reward
        rewards.append(reward)
        step_count += 1
        done = terminated or truncated

        curr_subgoal = info.get("subgoal", None)
        if curr_subgoal is not None and subgoal_idxs is None:
            subgoal_idxs = []
        if curr_subgoal is not None and curr_subgoal > len(subgoal_idxs):
            subgoal_idxs.append(step_count)
            
        env_reward = info.get("env_reward", None)
        if env_reward is not None and env_rewards is None:
            env_rewards = []
        if env_reward is not None:
            env_rewards.append(float(env_reward))
            
        detected_subgoal = info.get("detected_subgoal", None)
        if detected_subgoal is not None and detected_subgoal_idxs is None:
            detected_subgoal_idxs = []
        if detected_subgoal is not None and detected_subgoal > len(detected_subgoal_idxs):
            detected_subgoal_idxs.append(step_count)
        
        # Render frame for both video and optional observation storage
        if save_video or save_rgb:
            try:
                frame = utils.safe_render(env, mode="rgb_array")
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
        
    # add episode results to eval_results
    eval_results.add_episode(
        rewards=rewards,
        length=step_count,
        frame_count=len(frames),
        subgoal_idxs=subgoal_idxs,
        env_rewards=env_rewards,
        detected_subgoal_idxs=detected_subgoal_idxs,
    )
    
    eval_results.add_episode_data(
        episode_num=episode_num,
        length=step_count,
        actions=actions,
        observations=observations,
        rewards=rewards,
        subgoal_idxs=subgoal_idxs,
    )
    
    return eval_results.get_last_episode()
        

def _single_process_evaluation(pipe, checkpoint_dir, eval_env_config, args, ep_num_bounds, device, traj_dir=None):
    """Run evaluation episodes sequentially in a single process."""
    # load model
    logging.info(f"Loading checkpoint from {checkpoint_dir}...")
    model = SAC.load(checkpoint_dir, device=device)
    
    # create EvalResults instance to store results from this process
    eval_results = EvalResults()
    eval_results.init_episodes_data(args.seed)
    
    # create evaluation environment
    logging.info(f"Creating environment: {eval_env_config.env_name}")
    eval_env = utils.make_env(
        eval_env_config.env_name,
        seed=eval_env_config.seed,
        reward_type=eval_env_config.reward_wrapper.type,
        action_repeat=eval_env_config.action_repeat,
        frame_stack=eval_env_config.frame_stack,
        learned_reward_data=eval_env_config.learned_reward_data,
    )
    
    # wait for others to init before starting evaluation loop
    pipe.send("ready")
    pipe.recv()  # wait for signal to start evaluation
    
    start_ep_num, end_ep_num = ep_num_bounds
        
    for episode_num in tqdm(range(start_ep_num, end_ep_num), desc="Evaluating", unit="episode", disable=args.no_progress_bar):
        video_path = None
        if args.save_viz and traj_dir:
            video_path = os.path.join(traj_dir, f"{episode_num}.mp4")
            
        eval_env.unwrapped.set_seed(args.seed + episode_num)    #! crucial to have (start, end) to have different seeds for different processes
         
        ep_data = _run_episode(
            model, eval_env, eval_results, episode_num, video_path, save_video=args.save_viz, save_rgb=args.save_rgb
        )
        
        # save reward curve plot
        if args.save_viz and traj_dir:
            plot_path = os.path.join(traj_dir, f"{episode_num}.png")
            _save_reward_plot(plot_path, ep_data["rewards"], ep_data["subgoal_idxs"], ep_data["env_rewards"], ep_data["detected_subgoal_idxs"])
        
        add_info = f", env_reward={np.sum(ep_data['env_rewards']):.2f}" if ep_data['env_rewards'] is not None else ""
        add_info += f", subgoal={len(ep_data['subgoal_idxs'])}" if ep_data['subgoal_idxs'] is not None else ""
        add_info += f", detected_subgoal={len(ep_data['detected_subgoal_idxs'])}" if ep_data['detected_subgoal_idxs'] is not None else ""
        logging.info(f"Episode {episode_num}: reward={np.sum(ep_data['rewards']):.2f}, length={ep_data['length']}, frames={ep_data['frame_count']}{add_info}")
        
    print(eval_results.to_dict())
        
    eval_env.close()
    pipe.send(eval_results)


def _create_plots(results, output_dir):
    """Generate evaluation plots."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_SUMMARY)

    # Plot 1: Reward distribution
    rewards = results["cumulative_rewards"]
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
    parser.add_argument("--num_processes", type=int, default=1,
                        help="Number of parallel processes for evaluation")
    parser.add_argument("--save_viz", action="store_true", default=True,
                        help="Whether to save visualization (videos and plots)")
    parser.add_argument("--no_save_video", action="store_false", dest="save_video",
                        help="Disable video recording")
    parser.add_argument("--save_rgb", action="store_true", default=False,
                        help="Save RGB observations to H5 file for dataset creation")
    parser.add_argument("--no_progress_bar", action="store_true", default=False,
                        help="Disable progress bar")
    parser.add_argument("--learned_reward_model_path", type=str, default=None,
                        help="Path to the learned reward model checkpoint (if using a learned reward wrapper) - overrides config path if specified")
    parser.add_argument("--learned_reward_data_dir", type=str, default=None,
                        help="Data directory to use for computing learned reward embeddings (overrides config and checkpoint paths if specified)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (e.g., 'cpu', 'cuda:0'), NOTE: GPU does not support rendering, so videos saving is compromised when using GPU")
    
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
    
    # load learned reward model if specified and needed for reward wrapper
    if config.reward_wrapper.type not in ["sparse", "env", "env_state-intrinsic"]:
        logging.info(f"Getting learned reward model for reward wrapper type {config.reward_wrapper.type}...")
        if args.learned_reward_model_path is not None:
            logging.info(f"Loading learned reward model from {args.learned_reward_model_path}...")
            learned_reward_data = utils.load_learned_reward_data(args.learned_reward_model_path, device=device, data_dir=args.learned_reward_data_dir)
        else:
            logging.info(f"Getting learned reward model path from config {config.reward_wrapper.pretrained_path}")
            learned_reward_data = utils.load_learned_reward_data(config.reward_wrapper.pretrained_path, device=device, data_dir=args.learned_reward_data_dir)
    else:
        learned_reward_data = None
    
    # Patch render compatibility in the wrapper chain
    #eval_env = utils.patch_env_render_compatibility(eval_env)
    
    # populate eval_env_config with necessary info for creating envs in each process
    eval_env_config = SimpleNamespace(
        env_name=config.env_name,
        seed=args.seed,
        reward_wrapper=config.reward_wrapper,
        action_repeat=config.action_repeat,
        frame_stack=config.frame_stack,
        learned_reward_data=learned_reward_data,
    )

    # Create output directories: parent/out_eval-policy-py/{model_basename}/
    model_basename = os.path.basename(checkpoint_dir).split('.')[0]
    parent_dir = os.path.dirname(os.path.dirname(checkpoint_dir))
    results_dir = os.path.join(parent_dir, "out_eval-policy-py", model_basename)
    traj_dir = os.path.join(results_dir, "trajs") if args.save_viz else None
    if traj_dir:
        os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Run evaluation episodes manually
    logging.info(f"Running {args.num_episodes} evaluation episodes...")
    procs = []
    pipes = []
    mp.set_start_method('spawn', force=True)
    for proc_id in range(args.num_processes):
        logging.info(f"Starting evaluation on process {proc_id + 1}/{args.num_processes}...")
        ep_num_bounds = (proc_id * (args.num_episodes // args.num_processes), (proc_id + 1) * (args.num_episodes // args.num_processes))
        parent_pipe, child_pipe = mp.Pipe()
        proc = mp.Process(target=_single_process_evaluation, args=(child_pipe, checkpoint_dir, eval_env_config, args, ep_num_bounds, device, traj_dir))
        proc.start()
        procs.append(proc)
        pipes.append(parent_pipe)
        
    [pipe.recv() for pipe in pipes]  # wait for all processes to finish init
    [pipe.send("start") for pipe in pipes]  # signal all processes to continue
    
    # collect results from all processes
    eval_results = [pipe.recv() for pipe in pipes]
    [proc.join() for proc in procs]  # wait for all processes to finish
    
    # merge results from all processes 
    eval_result = eval_results[0]
    if len(eval_results) > 1:
        for res in eval_results[1:]:
            eval_result.merge_results(res)
        
    # extract results for statistics and plotting
    rewards = eval_result.rewards
    lengths = eval_result.lengths
    frame_counts = eval_result.frame_counts
    subgoal_idxs_all = eval_result.subgoal_idxs_all
    env_rewards = eval_result.env_rewards
    detected_subgoal_idxs_all = eval_result.detected_subgoal_idxs_all
    episodes_data = eval_result.episodes_data if hasattr(eval_result, 'episodes_data') else []

    # Compute statistics
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_length = float(np.mean(lengths))
    std_length = float(np.std(lengths))
    mean_env_reward = float(np.mean(env_rewards)) if env_rewards else 0.0
    std_env_reward = float(np.std(env_rewards)) if env_rewards else 0.0
    
    # Convert subgoal indices to dict of reach rates
    episode_subgoals_dict = {}
    if subgoal_idxs_all:
        max_subgoal = MAX_SUBGOAL
        episode_subgoals_dict = {i: 0 for i in range(max_subgoal + 1)}
        for subgoal_idxs in subgoal_idxs_all:
            # interpret first as failure rate, then cumulative reach rates for each subgoal
            subgoal_reached = len(subgoal_idxs) if subgoal_idxs is not None else 0
            if subgoal_reached == 0:
                episode_subgoals_dict[0] += 1
            else:
                for idx in range(1, subgoal_reached + 1):
                    episode_subgoals_dict[idx] += 1
        episode_subgoals_dict = {k: v / args.num_episodes for k, v in episode_subgoals_dict.items()}
    
    detected_subgoals_dict = {}
    if detected_subgoal_idxs_all:
        detected_subgoals_dict = {i: 0 for i in range(max_subgoal + 1)}
        for subgoal_idxs in detected_subgoal_idxs_all:
            subgoal_reached = len(subgoal_idxs) if subgoal_idxs is not None else 0
            if subgoal_reached == 0:
                detected_subgoals_dict[0] += 1
            else:
                for idx in range(1, subgoal_reached + 1):
                    detected_subgoals_dict[idx] += 1
        detected_subgoals_dict = {k: v / args.num_episodes for k, v in detected_subgoals_dict.items()}
    
    success_rate = episode_subgoals_dict.get(max_subgoal, 0.0)
    avg_subgoal_reached = np.mean([len(idxs) if idxs is not None else 0 for idxs in subgoal_idxs_all])

    # Log results
    logging.info("=" * 50)
    logging.info("Evaluation Results")
    logging.info("=" * 50)
    logging.info(f"Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
    logging.info(f"Min-Max Reward: {np.min(rewards):.4f} - {np.max(rewards):.4f}")
    logging.info(f"Success Rate: {success_rate:.4f}")
    
    if episode_subgoals_dict:
        logging.info(f"Average Subgoals Reached: {avg_subgoal_reached:.2f} / {max_subgoal}")
        logging.info("Subgoal Reach Rates:")
        for subgoal_idx in range(max_subgoal + 1):
            reach_rate = episode_subgoals_dict.get(subgoal_idx, 0.0)
            logging.info(f"  Subgoal {subgoal_idx}: {reach_rate:.4f}")
        
    if env_rewards:
        logging.info(f"Mean Environment Reward: {mean_env_reward:.4f} ± {std_env_reward:.4f}")
        logging.info(f"Min-Max Environment Reward: {np.min(env_rewards):.4f} - {np.max(env_rewards):.4f}")
        
    if detected_subgoals_dict:
        avg_detected_subgoal_reached = np.mean([len(idxs) if idxs is not None else 0 for idxs in detected_subgoal_idxs_all])
        logging.info(f"Average Detected Subgoals Reached: {avg_detected_subgoal_reached:.2f} / {max_subgoal}")
        logging.info("Detected Subgoal Reach Rates:")
        for subgoal_idx in range(max_subgoal + 1):
            reach_rate = detected_subgoals_dict.get(subgoal_idx, 0.0)
            logging.info(f"  Subgoal {subgoal_idx}: {reach_rate:.4f}")
            
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
        "cumulative_rewards": np.sum(rewards, axis=1).tolist(),
        "individual_rewards": rewards,
        "episode_lengths": lengths,
        "frame_counts": frame_counts,
        "subgoal_idxs": subgoal_idxs_all,
        "env_rewards": env_rewards,
        "detected_subgoal_idxs": detected_subgoals_dict,
    }
    
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_path}")

    # Save trajectories to H5 for replay compatibility
    h5_path = os.path.join(results_dir, "trajectories.h5")
    json_traj_path = os.path.join(results_dir, "trajectories.json")
    _save_trajectory_h5(h5_path, json_traj_path, eval_env_config, episodes_data, save_rgb=args.save_rgb)

    # Create plots
    _create_plots(results, results_dir)

    logging.info("Evaluation complete!")



if __name__ == "__main__":
    main()