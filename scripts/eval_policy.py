"""
Example usage:

python scripts/eval_policy.py \
    ../data/inest-maniskill/_experiments/lr-sb3/min_fr40_d0.95/22/checkpoints/best_model.zip \
    --num_episodes 10 --save_rgb --learned_reward_model_path \
    ../data/inest-maniskill/_experiments/pretrain/min_mc_b8_fr40/ \
    --learned_reward_data_dir ../data/inest-maniskill/datasets/dataset-min-rand/
"""

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
from inest_irl.utils.learned_reward_utils import (
    TrajectoryLearnedReward, DatasetLearnedReward, is_nan_or_none, save_reward_metrics
)
from inest_irl.viz.reward_plot import generate_reward_plot

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


class EvalEpisodeData:
    """Manages per-episode trajectory data (actions, observations, etc)."""
    def __init__(self):
        self.episodes_data = {}
    
    def add_episode(self, traj_id, seed, length, actions, observations, rewards, subgoal_idxs):
        """Add episode data for a trajectory."""
        self.episodes_data[traj_id] = {
            'seed': seed,
            'elapsed_steps': length,
            'actions': actions,
            'observations': observations,
            'rewards': rewards,
            'subgoal_idxs': subgoal_idxs,
        }
    
    def get(self, traj_id):
        """Get episode data for a trajectory."""
        return self.episodes_data.get(traj_id)
    
    def values(self):
        """Get all episode data."""
        return list(self.episodes_data.values())
    
    def merge(self, other, traj_id_offset=0):
        """Merge another EvalEpisodeData with trajectory ID shifting."""
        for traj_id, data in other.episodes_data.items():
            new_traj_id = traj_id + traj_id_offset
            self.episodes_data[new_traj_id] = data


class EvalTrajectoryLearnedReward(TrajectoryLearnedReward):
    """Extends TrajectoryLearnedReward with environment rewards."""
    def __init__(self, rewards, env_rewards=None, subgoal_rewards=None, 
                 subgoal_dists=None, subgoal_reachs=None, subgoal_reachs_gt=None):
        super().__init__(rewards, subgoal_rewards, subgoal_dists, 
                        subgoal_reachs, subgoal_reachs_gt)
        self.env_rewards = env_rewards


class EvalDatasetLearnedReward(DatasetLearnedReward):
    """Extends DatasetLearnedReward for evaluation results."""
    def __init__(self):
        super().__init__()
        self.episode_data = EvalEpisodeData()
    
    def add_eval_traj(self, traj_id, eval_traj_lr, episode_data):
        """Add evaluated trajectory with episode data."""
        self.add_traj(traj_id, eval_traj_lr)
        if episode_data:
            self.episode_data.add_episode(traj_id=traj_id, **episode_data)
    
    def merge_dataset(self, other, traj_id_offset=None):
        """Merge another EvalDatasetLearnedReward with optional trajectory ID shifting."""
        if traj_id_offset is None:
            # Auto-compute offset as max current trajectory ID + 1
            traj_id_offset = max(self.traj_lrs.keys()) + 1 if self.traj_lrs else 0
        
        for traj_id, traj_lr in other.traj_lrs.items():
            new_traj_id = traj_id + traj_id_offset
            self.traj_lrs[new_traj_id] = traj_lr
        
        self.episode_data.merge(other.episode_data, traj_id_offset)
    
    def subgoal_reach_rates(self):
        """Convert subgoal reaching steps to reach rate statistics."""
        def _count_subgoal_reaches(traj_subgoal_reachs_attr):
            cum_count = [0 for _ in range(MAX_SUBGOAL)]
            for traj_lr in self.traj_lrs.values():
                subgoal_reachs = getattr(traj_lr, traj_subgoal_reachs_attr)
                if subgoal_reachs is not None:
                    for i, reach_step in enumerate(subgoal_reachs):
                        if not is_nan_or_none(reach_step) and i < MAX_SUBGOAL:
                            cum_count[i] += 1
            return cum_count
        
        # Compute reach rates for GT and detected subgoals
        gt_reach_counts = _count_subgoal_reaches('subgoal_reachs_gt')
        detected_reach_counts = _count_subgoal_reaches('subgoal_reachs')
        
        num_trajs = len(self.traj_lrs)
        gt_rates = {i: count / num_trajs for i, count in enumerate(gt_reach_counts)}
        detected_rates = {i: count / num_trajs for i, count in enumerate(detected_reach_counts)}
        
        return {'gt': gt_rates, 'detected': detected_rates}
    
    def subgoal_reach_rates_to_file(self, out_dir):
        """Save subgoal reach rates to JSON file."""
        reach_rates = self.subgoal_reach_rates()
        with open(os.path.join(out_dir, 'subgoal_reach_rates.json'), 'w') as f:
            json.dump(reach_rates, f, indent=2)
    
    def reward_metrics_to_file(self, out_dir):
        """Override: Save reward metrics including per-trajectory and aggregate eval metrics."""
        # call parent behavior for per-trajectory metrics
        traj_rews = {traj_id: traj_lr.rewards for traj_id, traj_lr in self.traj_lrs.items()} 
        save_reward_metrics(traj_rews, os.path.join(out_dir, 'reward_metrics.json'), avg=True)
        save_reward_metrics(traj_rews, os.path.join(out_dir, 'reward_metrics_per_traj.json'), avg=False)
        
        if any(traj_lr.env_rewards is not None for traj_lr in self.traj_lrs.values()):
            traj_env_rews = {traj_id: traj_lr.env_rewards for traj_id, traj_lr in self.traj_lrs.items() if traj_lr.env_rewards is not None}
            save_reward_metrics(traj_env_rews, os.path.join(out_dir, 'env_reward_metrics.json'), avg=True)
            save_reward_metrics(traj_env_rews, os.path.join(out_dir, 'env_reward_metrics_per_traj.json'), avg=False)


def _extract_rgb_from_info(info):
    """Extract RGB frame(s) from info dict. Handles both single and multi-camera cases.
    
    Args:
        info: info dict that may contain camera data
        camera_names: optional list of camera names to extract in order
        
    Returns:
        Dict mapping camera name to frame (H,W,3), or empty dict if no frames found
    """
    if not isinstance(info, dict):
        return {}
    
    frames_dict = {}
    
    # Try to extract sensor_data from info (ManiSkill format)
    sensor_data = info.get("sensor_data", {})
    if isinstance(sensor_data, dict):
        # extract all RGB data available
        for cam_name, cam_data in sensor_data.items():
            if isinstance(cam_data, dict) and "rgb" in cam_data:
                frames_dict[cam_name] = cam_data["rgb"][0,...]  # remove batch dimension
    
    return frames_dict


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
        obs_mode=env_config.obs_mode,
        action_repeat=env_config.action_repeat,
        frame_stack=env_config.frame_stack,
        env_randomization=env_config.env_randomization,
        render_camera=env_config.render_camera,
        reward_scaling=env_config.reward_scaling,
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

def _run_episode(model, env, episode_num, seed, video_path=None, save_video=True, save_rgb=False):
    """Run single evaluation episode, optionally collecting RGB observations, actions, rewards, and subgoal indices.
    
    Returns:
        tuple: (eval_traj_lr, episode_data) where eval_traj_lr is EvalTrajectoryLearnedReward
               and episode_data is a dict with seed, length, actions, observations, rewards, subgoal_idxs
    """
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
    
    done = False
    step_count = 0
    frames = {}  # dict mapping camera_name -> list of frames
    actions = []
    observations = []
    rewards = []
    subgoal_reachs_gt = [np.nan for _ in range(MAX_SUBGOAL)]  # Initialize to np.nan
    env_rewards = None
    subgoal_reachs = [np.nan for _ in range(MAX_SUBGOAL)]  # Initialize to np.nan
    
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
        rewards.append(reward)
        step_count += 1
        done = terminated or truncated

        # Track ground truth subgoal reaching steps
        curr_subgoal = info.get("subgoal", None)
        if curr_subgoal is not None and curr_subgoal > 0 and curr_subgoal <= MAX_SUBGOAL:
            if np.isnan(subgoal_reachs_gt[curr_subgoal - 1]):  # Only set once
                subgoal_reachs_gt[curr_subgoal - 1] = step_count
            
        # Track environment rewards
        env_reward = info.get("env_reward", None)
        if env_reward is not None:
            if env_rewards is None:
                env_rewards = []
            env_rewards.append(float(env_reward))
            
        # Track detected subgoal reaching steps
        detected_subgoal = info.get("detected_subgoal", None)
        if detected_subgoal is not None and detected_subgoal > 0 and detected_subgoal <= MAX_SUBGOAL:
            if np.isnan(subgoal_reachs[detected_subgoal - 1]):  # Only set once
                subgoal_reachs[detected_subgoal - 1] = step_count
        
        # Extract RGB frames from info for both video and optional observation storage
        if save_video or save_rgb:
            rgb_frames_dict = _extract_rgb_from_info(info)
            if rgb_frames_dict:
                # Initialize frame lists for each camera on first frame
                for cam_name, frame in rgb_frames_dict.items():
                    if cam_name not in frames:
                        frames[cam_name] = []
                    if save_video:
                        frames[cam_name].append(frame)
                    # For RGB storage, use first camera only
                    if save_rgb and cam_name == list(rgb_frames_dict.keys())[0]:
                        observations.append(frame)
    
    # Save video for each camera if frames were collected
    if video_path and frames:
        for cam_name, cam_frames in frames.items():
            if cam_frames:
                # Insert camera name in video path (e.g., "episode_0.mp4" -> "episode_0_base_camera.mp4")
                path_parts = os.path.splitext(video_path)
                cam_video_path = f"{path_parts[0]}_{cam_name}{path_parts[1]}"
                _save_video(cam_video_path, cam_frames)
    
    # Convert np.nan sentinels in subgoal reaching steps to np.nan (they already are)
    subgoal_reachs_gt = np.array(subgoal_reachs_gt, dtype=np.float32)
    subgoal_reachs = np.array(subgoal_reachs, dtype=np.float32)
    
    # Create EvalTrajectoryLearnedReward object
    eval_traj_lr = EvalTrajectoryLearnedReward(
        rewards=np.array(rewards, dtype=np.float32),
        env_rewards=np.array(env_rewards, dtype=np.float32) if env_rewards else None,
        subgoal_reachs=subgoal_reachs,
        subgoal_reachs_gt=subgoal_reachs_gt,
    )
    
    # Prepare episode data
    episode_data = {
        'seed': seed,
        'length': step_count,
        'actions': actions,
        'observations': observations,
        'rewards': rewards,
        'subgoal_idxs': subgoal_reachs_gt,
    }
    
    return eval_traj_lr, episode_data
        

def _single_process_evaluation(pipe, checkpoint_dir, eval_env_config, args, ep_num_bounds, device, traj_dir=None):
    """Run evaluation episodes sequentially in a single process."""
    # load model
    logging.info(f"Loading checkpoint from {checkpoint_dir}...")
    model = SAC.load(checkpoint_dir, device=device)
    
    # create EvalDatasetLearnedReward instance to store results from this process
    eval_dataset = EvalDatasetLearnedReward()
    
    # create evaluation environment
    logging.info(f"Creating environment: {eval_env_config.env_name}")
    eval_env = utils.make_env(
        eval_env_config.env_name,
        seed=eval_env_config.seed,
        reward_type=eval_env_config.reward_wrapper.type,
        obs_mode=eval_env_config.obs_mode,
        action_repeat=eval_env_config.action_repeat,
        frame_stack=eval_env_config.frame_stack,
        env_randomization=eval_env_config.env_randomization,
        render_camera=eval_env_config.render_camera,
        reward_scaling=eval_env_config.reward_scaling,
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
         
        eval_traj_lr, episode_data = _run_episode(
            model, eval_env, episode_num, seed=args.seed + episode_num, 
            video_path=video_path, 
            save_video=args.save_viz, save_rgb=args.save_rgb,
        )
        
        # Add to dataset
        eval_dataset.add_eval_traj(episode_num, eval_traj_lr, episode_data)
        
        # save reward curve plot
        if args.save_viz and traj_dir:
            plot_path = os.path.join(traj_dir, f"{episode_num}.png")
            _save_reward_plot(plot_path, eval_traj_lr.rewards, eval_traj_lr.subgoal_reachs_gt, 
                            eval_traj_lr.env_rewards, eval_traj_lr.subgoal_reachs)
        
        # Log episode info
        num_gt_subgoals = int(np.sum(~np.isnan(eval_traj_lr.subgoal_reachs_gt)))
        num_detected_subgoals = int(np.sum(~np.isnan(eval_traj_lr.subgoal_reachs)))
        add_info = f", env_reward={np.sum(eval_traj_lr.env_rewards):.2f}" if eval_traj_lr.env_rewards is not None else ""
        add_info += f", subgoal_gt={num_gt_subgoals}"
        add_info += f", subgoal_det={num_detected_subgoals}"
        logging.info(f"Episode {episode_num}: reward={np.sum(eval_traj_lr.rewards):.2f}, length={len(eval_traj_lr.rewards)}{add_info}")
        
    eval_env.close()
    pipe.send(eval_dataset)


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
        obs_mode="state+rgb",  # Force RGB observation mode for video frame extraction
        env_randomization=config.env_randomization,
        render_camera=config.render_camera,
        reward_scaling=config.reward_scaling,
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
    eval_datasets = [pipe.recv() for pipe in pipes]
    [proc.join() for proc in procs]  # wait for all processes to finish
    
    # merge results from all processes 
    eval_dataset = eval_datasets[0]
    if len(eval_datasets) > 1:
        for dataset in eval_datasets[1:]:
            eval_dataset.merge_dataset(dataset)
        
    # extract results for statistics and plotting
    rewards_list = [traj_lr.rewards for traj_lr in eval_dataset.traj_lrs.values()]
    env_rewards_list = [traj_lr.env_rewards for traj_lr in eval_dataset.traj_lrs.values() if traj_lr.env_rewards is not None]
    
    # Compute statistics
    cumulative_rewards = [np.sum(r) for r in rewards_list]
    mean_reward = float(np.mean(cumulative_rewards))
    std_reward = float(np.std(cumulative_rewards))
    
    episode_lengths = [len(r) for r in rewards_list]
    mean_length = float(np.mean(episode_lengths))
    std_length = float(np.std(episode_lengths))
    
    mean_env_reward = float(np.mean([np.sum(r) for r in env_rewards_list])) if env_rewards_list else 0.0
    std_env_reward = float(np.std([np.sum(r) for r in env_rewards_list])) if env_rewards_list else 0.0
    
    # Get subgoal reach rates using the dataset method
    subgoal_rates = eval_dataset.subgoal_reach_rates()
    gt_reach_rates = subgoal_rates['gt']
    detected_reach_rates = subgoal_rates['detected']
    
    # Compute success rate (reaching all subgoals)
    success_rate = gt_reach_rates.get(MAX_SUBGOAL - 1, 0.0)
    
    # Compute average subgoals reached
    avg_subgoal_reached = np.mean([
        int(np.sum(~np.isnan(traj_lr.subgoal_reachs_gt))) 
        for traj_lr in eval_dataset.traj_lrs.values()
    ])

    # Log results
    logging.info("=" * 50)
    logging.info("Evaluation Results")
    logging.info("=" * 50)
    logging.info(f"Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
    logging.info(f"Min-Max Reward: {np.min(cumulative_rewards):.4f} - {np.max(cumulative_rewards):.4f}")
    logging.info(f"Success Rate: {success_rate:.4f}")
    
    logging.info(f"Average Subgoals Reached: {avg_subgoal_reached:.2f} / {MAX_SUBGOAL}")
    logging.info("GT Subgoal Reach Rates:")
    for subgoal_idx in range(MAX_SUBGOAL):
        reach_rate = gt_reach_rates.get(subgoal_idx, 0.0)
        logging.info(f"  Subgoal {subgoal_idx}: {reach_rate:.4f}")
        
    if env_rewards_list:
        logging.info(f"Mean Environment Reward: {mean_env_reward:.4f} ± {std_env_reward:.4f}")
        logging.info(f"Min-Max Environment Reward: {np.min([np.sum(r) for r in env_rewards_list]):.4f} - {np.max([np.sum(r) for r in env_rewards_list]):.4f}")
        
    logging.info("Detected Subgoal Reach Rates:")
    for subgoal_idx in range(MAX_SUBGOAL):
        reach_rate = detected_reach_rates.get(subgoal_idx, 0.0)
        logging.info(f"  Subgoal {subgoal_idx}: {reach_rate:.4f}")
            
    if traj_dir:
        logging.info(f"Trajectories saved to: {traj_dir}")
    logging.info("=" * 50)
    
    # Save metrics including results to JSON file
    logging.info(f"Metrics and results saved to {results_dir}")
    eval_dataset.reward_metrics_to_file(results_dir)
    eval_dataset.to_file(results_dir)
    eval_dataset.subgoal_idxs_to_file(results_dir)
    eval_dataset.subgoal_reach_rates_to_file(results_dir)

    # Save trajectories to H5 for replay compatibility
    h5_path = os.path.join(results_dir, "trajectories.h5")
    json_traj_path = os.path.join(results_dir, "trajectories.json")
    episodes_data = eval_dataset.episode_data.values()
    _save_trajectory_h5(h5_path, json_traj_path, eval_env_config, episodes_data, save_rgb=args.save_rgb)

    # Create plots using learned reward utilities
    eval_dataset.plot_results(results_dir, mean=True, count=None, 
                             rewards_flag=True, subgoal_rewards_flag=False, distances_flag=False)
    eval_dataset.plot_trajectory_lengths(results_dir)

    logging.info("Evaluation complete!")



if __name__ == "__main__":
    main()