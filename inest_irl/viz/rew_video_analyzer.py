"""
Example usage:

# experiments
python inest_irl/viz/rew_video_analyzer.py \
    --h5 ../data/inest-maniskill/_experiments/lr-sb3/min_fr40_d0.95/22/out_eval-policy-py/best_model/trajectories.h5 \
    --rewards ../data/inest-maniskill/_experiments/lr-sb3/min_fr40_d0.95/22/out_eval-policy-py/best_model/learned_rewards.json \
    --video_name min_fr40_d0.95 \
    --only_video \
    --count 2
    
# dataset
python inest_irl/viz/rew_video_analyzer.py \
    --h5 ../data/maniskill/StackPyramid-v1_data-min-rand/trajectory.rgb+state_dict.pd_joint_pos.physx_cpu.h5 \
    --rewards out/reward_plots/min_mc_b8_fr40/learned_rewards.json \
    --video_name min_mc_b8_fr40_data \
    --only_video \
    --count 2
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple, Optional, Any

if '--only_video' in sys.argv:
    import matplotlib
    matplotlib.use('Agg', force=True)

import h5py
import yaml
from inest_irl.dataset_utils.h5_to_dataset import _access_nested_group
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FFMpegWriter
import tqdm

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

FPS_DEFAULT = 20
FPS_VALUES = [0.5, 1, 2, 5, 10, 15, 20, 30, 50]


def find_image_keys_in_demo(demo: h5py.Group, obs_keys: List[str], max_cameras: int = 2) -> List[Tuple[str, str]]:
    """
    Find up to max_cameras image keys in a demonstration using a list of nested keys.

    Args:
        demo: h5py group representing a single demo
        obs_keys: list of nested keys (e.g. 'obs/sensor_data/base_camera/rgb')
        max_cameras: Maximum number of cameras to find

    Returns:
        List of (key, name) tuples where key is the nested key and name is display name
    """
    found_keys = []
    for key in obs_keys:
        if len(found_keys) >= max_cameras:
            break
        try:
            # Attempt to access nested group; _access_nested_group will raise if missing
            _access_nested_group(demo, key)
            key_parts = key.split('/')
            display_name = key_parts[-2] if len(key_parts) >= 2 else key_parts[-1]
            found_keys.append((key, display_name))
        except Exception:
            continue
    return found_keys


class TrajectoryLoader:
    """Loads trajectories from HDF5 and rewards from JSON."""

    def __init__(self, hdf5_path: str, rewards_json_path: str, config_path: Optional[str] = None):
        """Initialize loader with HDF5 and rewards JSON paths.

        Args:
            config_path: optional path to YAML config that provides `obs_keys` list
        """
        self.hdf5_path = hdf5_path
        self.rewards_json_path = rewards_json_path
        self.trajectories = {}
        self.rewards_data = {}
        # Load obs keys from config or use defaults
        self.obs_keys = []
        if config_path is None:
            # default config in dataset_utils
            default_cfg = os.path.join(os.path.dirname(__file__), '..', 'dataset_utils', 'configs_h5_to_dataset', 'maniskill_demos_merged.yaml')
            config_path = default_cfg
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as cf:
                    cfg = yaml.safe_load(cf)
                    if isinstance(cfg, dict) and 'obs_keys' in cfg:
                        self.obs_keys = cfg['obs_keys']
        except Exception:
            self.obs_keys = []
        # Fallback to a few common keys if config missing
        if not self.obs_keys:
            self.obs_keys = [
                'obs/sensor_data/base_camera/rgb',
                'obs/sensor_data/hand_camera/rgb',
            ]
        
    def load(self, max_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Load trajectories and rewards.
        
        Returns:
            Dictionary with trajectory data and rewards
        """
        print(f"Loading rewards from {self.rewards_json_path}...")
        self._load_rewards_json()

        print(f"Loading trajectories from {self.hdf5_path}...")
        self._load_h5(max_count=max_count)
        
        return self.trajectories

    def _load_episode_ids_from_sidecar_json(self) -> List[str]:
        """Load ordered episode ids from a sibling JSON with the same base name as the H5."""
        sidecar_json_path = os.path.splitext(self.hdf5_path)[0] + '.json'
        if not os.path.exists(sidecar_json_path):
            return []

        try:
            with open(sidecar_json_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read sidecar JSON '{sidecar_json_path}': {e}")
            return []

        episodes = metadata.get('episodes', []) if isinstance(metadata, dict) else []
        if not isinstance(episodes, list):
            return []

        episode_ids = []
        for ep in episodes:
            if isinstance(ep, dict) and 'episode_id' in ep:
                episode_ids.append(str(ep['episode_id']))
        return episode_ids
    
    def _load_h5(self, max_count: Optional[int] = None):
        """Load trajectory data from HDF5 file."""
        episode_ids = self._load_episode_ids_from_sidecar_json()
        reward_keys = set(self.rewards_data.keys())

        with h5py.File(self.hdf5_path, 'r') as f:
            # Check for 'data' or 'trajectories' group
            if 'data' in f:
                root = f['data']
            elif 'trajectories' in f:
                root = f['trajectories']
            else:
                root = f
            
            traj_ids = list(root.keys())
            print(f"Found {len(traj_ids)} trajectories")
            if episode_ids:
                print(f"Loaded {len(episode_ids)} episode ids from sidecar JSON")
            if reward_keys:
                print(f"Reward JSON contains {len(reward_keys)} trajectory ids")
            
            for traj_idx, traj_id in enumerate(traj_ids):
                if max_count is not None and len(self.trajectories) >= max_count:
                    break

                demo = root[traj_id]

                # Prefer episode ids from sidecar JSON; fallback to H5 key.
                mapped_traj_id = episode_ids[traj_idx] if traj_idx < len(episode_ids) else str(traj_id)

                if reward_keys and mapped_traj_id not in reward_keys:
                    continue

                # Find image keys (up to 2 cameras) using config-driven obs_keys
                image_keys = find_image_keys_in_demo(demo, self.obs_keys, max_cameras=2)

                if not image_keys:
                    print(f"Warning: No images found in trajectory {traj_id}, skipping")
                    continue

                # Load images for each camera using nested access
                images_list = []
                for nested_key, name in image_keys:
                    try:
                        images = _access_nested_group(demo, nested_key)
                        images = np.array(images)
                        images_list.append({
                            'images': images,
                            'name': name,
                            'key': nested_key
                        })
                    except Exception as e:
                        print(f"Warning: Could not load {nested_key} from {traj_id}: {e}")

                if not images_list:
                    print(f"Warning: Could not load any images from {traj_id}, skipping")
                    continue

                # Load actions if available
                actions = None
                try:
                    actions = _access_nested_group(demo, 'actions') if 'actions' in demo else None
                    if actions is not None:
                        actions = np.array(actions)
                except Exception:
                    actions = None

                # Get trajectory length from first camera
                trajectory_length = len(images_list[0]['images'])

                self.trajectories[traj_id] = {
                    'images_list': images_list,
                    'actions': actions,
                    'length': trajectory_length,
                    'traj_id': mapped_traj_id,
                    'h5_traj_id': str(traj_id),
                }
    
    def _load_rewards_json(self):
        """Load reward data from JSON file."""
        try:
            with open(self.rewards_json_path, 'r') as f:
                rewards_json = json.load(f)

            if not isinstance(rewards_json, dict):
                print("Warning: Expected JSON to be a dict keyed by trajectory id")
                self.rewards_data = {}
            else:
                # Exact expected structure:
                # {"0": {"rewards": [...], "subgoal_reachs": [...], "subgoal_reachs_gt": [...], "env_rewards": [...]}, ...}
                self.rewards_data = {str(key): value for key, value in rewards_json.items() if isinstance(value, dict)}

            print(f"Loaded rewards for {len(self.rewards_data)} trajectories")
            
        except Exception as e:
            print(f"Error loading rewards JSON: {e}")
            self.rewards_data = {}
    
    def get_trajectory_with_rewards(self, traj_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trajectory data merged with rewards.
        
        Args:
            traj_id: Trajectory ID
            
        Returns:
            Dict with trajectory and reward data, or None if not found
        """
        if traj_id not in self.trajectories:
            return None
        
        traj_data = self.trajectories[traj_id].copy()
        
        # Add rewards if available
        reward_info = None

        def _add_candidate_variants(key: str, out: List[str]):
            if not key:
                return
            if key not in out:
                out.append(key)
            if key.startswith('traj_'):
                stripped = key.replace('traj_', '', 1)
                if stripped not in out:
                    out.append(stripped)
            elif key.isdigit():
                prefixed = f'traj_{key}'
                if prefixed not in out:
                    out.append(prefixed)

        candidate_keys: List[str] = []
        _add_candidate_variants(str(traj_data.get('traj_id', '')), candidate_keys)
        _add_candidate_variants(str(traj_data.get('h5_traj_id', '')), candidate_keys)
        _add_candidate_variants(str(traj_id), candidate_keys)

        for candidate_key in candidate_keys:
            if candidate_key in self.rewards_data:
                reward_info = self.rewards_data[candidate_key]
                break

        if reward_info is not None:
            rewards = reward_info.get('rewards', [])
            env_rewards = reward_info.get('env_rewards', None)
            subgoal_idxs = reward_info.get('subgoal_reachs', [])
            gt_subgoal_idxs = reward_info.get('subgoal_reachs_gt', [])

            traj_data['rewards'] = np.array(rewards, dtype=float)
            traj_data['env_rewards'] = np.array(env_rewards, dtype=float) if env_rewards is not None and len(env_rewards) > 0 else None
            traj_data['subgoal_idxs'] = list(subgoal_idxs) if subgoal_idxs is not None else []
            traj_data['gt_subgoal_idxs'] = list(gt_subgoal_idxs) if gt_subgoal_idxs is not None else []
        else:
            # Create dummy rewards if not found
            traj_data['rewards'] = np.zeros(traj_data['length'])
            traj_data['env_rewards'] = None
            traj_data['subgoal_idxs'] = []
            traj_data['gt_subgoal_idxs'] = []
        
        return traj_data


class RewardPlotter:
    """Handles reward plotting with current timestep highlighting."""
    
    def __init__(self, ax):
        """Initialize plotter with matplotlib axis."""
        self.ax = ax
        self.ax2 = None
        self.rewards = None
        self.current_line = None
        self.reward_line = None
        
    def setup_plot(self, rewards: np.ndarray, traj_id: str, 
                   env_rewards: Optional[np.ndarray] = None,
                   subgoal_idxs: Optional[List] = None,
                   gt_subgoal_idxs: Optional[List] = None):
        """Setup the reward plot for a trajectory."""
        # If a previous trajectory created a secondary y-axis, remove it first.
        if self.ax2 is not None:
            self.ax2.remove()
            self.ax2 = None

        self.rewards = rewards
        self.ax.clear()
        
        # Plot reward curve
        timesteps = np.arange(len(rewards))
        self.reward_line, = self.ax.plot(timesteps, rewards, 'b-', linewidth=2, alpha=0.7, label='Reward')
        
        # Add mean line
        mean_rew = np.mean(rewards)
        self.ax.axhline(mean_rew, color='blue', linestyle='-.', linewidth=2, alpha=0.5,
                       label=f'Avg: {mean_rew:.2f}')
        
        # Mark subgoals
        if gt_subgoal_idxs:
            valid_idxs = [idx for idx in gt_subgoal_idxs if not np.isnan(idx)]
            for idx in valid_idxs:
                self.ax.axvline(int(idx), color='purple', linestyle=':', alpha=0.7,
                              label='GT Subgoal(s)' if idx == valid_idxs[0] else "")
        
        if subgoal_idxs:
            valid_idxs = [idx for idx in subgoal_idxs if not np.isnan(idx)]
            for idx in valid_idxs:
                self.ax.axvline(int(idx), color='green', linestyle='--', alpha=0.7,
                              label='Detected Subgoal(s)' if idx == valid_idxs[0] else "")
        
        # Setup current timestep as vertical line
        self.current_line = self.ax.axvline(x=0, color='red', linestyle='-', alpha=0.8, linewidth=2, zorder=4)
        
        # Setup secondary axis for env rewards
        if env_rewards is not None and len(env_rewards) > 0:
            self.ax2 = self.ax.twinx()
            self.ax2.plot(timesteps, env_rewards, 'orange', linewidth=2, alpha=0.7, label='Env Reward')
            self.ax2.set_ylabel('Env Reward', color='orange', labelpad=14)
            self.ax2.tick_params(axis='y', labelcolor='orange', pad=4)
            self.ax2.spines['right'].set_position(('outward', 10))
        
        # Formatting
        self.ax.set_xlabel('Timestep')
        self.ax.set_ylabel('Reward', color='b')
        self.ax.tick_params(axis='y', labelcolor='b')
        self.ax.set_title(f'{traj_id}\nTotal Reward: {np.sum(rewards):.3f} | '
                         f'Avg Reward: {mean_rew:.3f}')
        self.ax.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        reward_min, reward_max = np.min(rewards), np.max(rewards)
        margin = (reward_max - reward_min) * 0.1 if reward_max != reward_min else 0.1
        self.ax.set_ylim(reward_min - margin, reward_max + margin)
        self.ax.set_xlim(-1, len(rewards))
        
        # Legend
        handles, labels = self.ax.get_legend_handles_labels()
        if self.ax2 is not None:
            handles2, labels2 = self.ax2.get_legend_handles_labels()
            handles += handles2
            labels += labels2
        
        if handles:
            self.ax.legend(handles, labels, loc='upper right', fontsize=8)
    
    def update_timestep(self, timestep: int):
        """Update the current timestep indicator."""
        if self.current_line is not None and self.rewards is not None:
            if 0 <= timestep < len(self.rewards):
                self.current_line.set_xdata([timestep, timestep])


class VideoVisualizer:
    """Main visualization class for reward videos."""
    
    def __init__(self, figsize=(16, 6), title=None):
        """Initialize the visualizer."""
        self.fig = plt.figure(figsize=figsize)
        self.title = title if title is not None else 'Trajectory Reward Visualizer'
        self.fig.suptitle(self.title, fontsize=14)
        
        # We'll set up axes dynamically based on number of cameras
        self.axes_images = []
        self.axes_reward = None
        self.reward_plotter = None
        
        # Setup reward plotter
        self.reward_plotter = RewardPlotter(None)  # Will set ax later
        
        # Video playback state
        self.traj_data_list = []
        self.traj_ids = []
        self.current_traj_idx = 0
        self.current_timestep = 0
        self.is_playing = False
        self.fps = FPS_DEFAULT
        self.last_frame_time = 0
        self.should_exit = False
        self.num_cameras = 0
        
        # Video saving
        self.video_writer = None
        self.save_video_path = None
        
        # Setup controls
        self._setup_controls()
        self._setup_key_bindings()
        self._setup_close_handler()
    
    def _setup_axes(self, num_cameras: int):
        """Setup matplotlib axes based on number of cameras."""
        self.num_cameras = num_cameras

        # Remove only the dynamic content axes, keeping the control buttons alive.
        for ax in list(self.axes_images):
            if ax in self.fig.axes:
                self.fig.delaxes(ax)
        if self.axes_reward is not None and self.axes_reward in self.fig.axes:
            self.fig.delaxes(self.axes_reward)
        
        self.axes_images = []

        # Reserve space at the bottom so buttons remain visible and give the right axis room.
        self.fig.subplots_adjust(left=0.05, right=0.90, top=0.93, bottom=0.16, wspace=0.22, hspace=0.25)

        if num_cameras <= 1:
            # One image on the left, reward on the right.
            gs = self.fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.18)
            self.axes_images.append(self.fig.add_subplot(gs[0, 0]))
            self.axes_reward = self.fig.add_subplot(gs[0, 1])
        else:
            # Two cameras stacked on the left, reward plot on the right.
            gs = self.fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1, 1], wspace=0.18, hspace=0.18)
            self.axes_images = [self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[1, 0])]
            self.axes_reward = self.fig.add_subplot(gs[:, 1])
        
        # Update reward plotter axis
        self.reward_plotter.ax = self.axes_reward
    
    def _setup_controls(self):
        """Setup interactive controls."""
        button_height = 0.04
        button_width = 0.08
        gap = 0.01
        button_y = 0.01
        start_x = 0.04
        
        # -1 button
        self.ax_minus_one = plt.axes([start_x, button_y, button_width, button_height])
        self.btn_minus_one = Button(self.ax_minus_one, '-1')
        self.btn_minus_one.on_clicked(self._step_backward)
        
        # Play/Pause button
        play_x = start_x + (button_width + gap)
        self.ax_play = plt.axes([play_x, button_y, button_width, button_height])
        self.btn_play = Button(self.ax_play, 'Play')
        self.btn_play.on_clicked(self._toggle_play)
        
        # +1 button
        plus_x = play_x + (button_width + gap)
        self.ax_plus_one = plt.axes([plus_x, button_y, button_width, button_height])
        self.btn_plus_one = Button(self.ax_plus_one, '+1')
        self.btn_plus_one.on_clicked(self._step_forward)
        
        # Previous traj button
        prev_x = plus_x + (button_width + gap)
        self.ax_prev_traj = plt.axes([prev_x, button_y, button_width, button_height])
        self.btn_prev_traj = Button(self.ax_prev_traj, 'Prev Traj')
        self.btn_prev_traj.on_clicked(self._prev_traj)
        
        # Next traj button
        next_x = prev_x + (button_width + gap)
        self.ax_next_traj = plt.axes([next_x, button_y, button_width, button_height])
        self.btn_next_traj = Button(self.ax_next_traj, 'Next Traj')
        self.btn_next_traj.on_clicked(self._next_traj)
        
        # Slower button
        slower_x = next_x + (button_width + gap)
        self.ax_slower = plt.axes([slower_x, button_y, button_width, button_height])
        self.btn_slower = Button(self.ax_slower, 'Slower')
        self.btn_slower.on_clicked(self._slower)
        
        # Faster button
        faster_x = slower_x + (button_width + gap)
        self.ax_faster = plt.axes([faster_x, button_y, button_width, button_height])
        self.btn_faster = Button(self.ax_faster, 'Faster')
        self.btn_faster.on_clicked(self._faster)
        
        # Reset button
        reset_x = faster_x + (button_width + gap)
        self.ax_reset = plt.axes([reset_x, button_y, button_width, button_height])
        self.btn_reset = Button(self.ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._reset_timestep)
    
    def _setup_key_bindings(self):
        """Setup keyboard controls."""
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
    
    def _setup_close_handler(self):
        """Setup window close event handler."""
        self.fig.canvas.mpl_connect('close_event', self._on_close)
    
    def _on_close(self, event):
        """Handle window close event."""
        self.should_exit = True
        if self.video_writer is not None:
            self.video_writer.finish()
        plt.close('all')
    
    def _on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == ' ':
            self._toggle_play(None)
        elif event.key == 'left':
            self._step_backward()
        elif event.key == 'right':
            self._step_forward()
        elif event.key == 'up':
            self._next_traj(None)
        elif event.key == 'down':
            self._prev_traj(None)
        elif event.key == 'r':
            self._reset_timestep(None)
        elif event.key == '+' or event.key == '=':
            self._faster(None)
        elif event.key == '-':
            self._slower(None)
        elif event.key == 'q' or event.key == 'escape':
            self._on_close(None)
    
    def load_data(self, hdf5_path: str, rewards_json_path: str, max_count: Optional[int] = None) -> bool:
        """Load trajectory and reward data."""
        loader = TrajectoryLoader(hdf5_path, rewards_json_path)
        trajectories = loader.load(max_count=max_count)
        
        if not trajectories:
            print("No valid trajectories found!")
            return False
        
        # Merge with rewards
        self.traj_data_list = []
        self.traj_ids = []
        
        # Preserve original H5 trajectory ids and insertion order from loader.
        for traj_id in trajectories.keys():
            traj_with_rewards = loader.get_trajectory_with_rewards(traj_id)
            if traj_with_rewards is not None:
                self.traj_data_list.append(traj_with_rewards)
                self.traj_ids.append(traj_id)
        
        print(f"Loaded {len(self.traj_data_list)} trajectories with reward data")
        
        if not self.traj_data_list:
            print("No trajectories with rewards found!")
            return False
        
        # Setup axes based on number of cameras in first traj
        num_cameras = len(self.traj_data_list[0]['images_list'])
        self._setup_axes(num_cameras)
        
        self.current_traj_idx = 0
        self.current_timestep = 0
        self._update_display()
        return True
    
    def set_trajectory(self, traj_idx: int):
        """Set the current trajectory index."""
        if 0 <= traj_idx < len(self.traj_data_list):
            self.current_traj_idx = traj_idx
            self.current_timestep = 0
            self._update_display()
    
    def _update_display(self):
        """Update display with current trajectory and timestep."""
        if not self.traj_data_list:
            return
        
        traj = self.traj_data_list[self.current_traj_idx]
        
        # Update image displays
        for cam_idx, camera_data in enumerate(traj['images_list']):
            if cam_idx < len(self.axes_images):
                ax = self.axes_images[cam_idx]
                
                if self.current_timestep < len(camera_data['images']):
                    image = camera_data['images'][self.current_timestep]
                    
                    # Handle different image formats
                    if image.dtype == np.uint8:
                        display_image = image
                    else:
                        # Normalize to [0, 255]
                        img_min, img_max = image.min(), image.max()
                        if img_max > img_min:
                            display_image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                        else:
                            display_image = image.astype(np.uint8)
                    
                    ax.clear()
                    ax.imshow(display_image)
                    ax.set_title(f'{camera_data["name"]} - '
                                f'Step {self.current_timestep + 1}/{traj["length"]}')
                    ax.axis('off')
        
        # Update reward plot
        self.reward_plotter.setup_plot(
            traj['rewards'],
            traj['traj_id'],
            env_rewards=traj.get('env_rewards'),
            subgoal_idxs=traj.get('subgoal_idxs'),
            gt_subgoal_idxs=traj.get('gt_subgoal_idxs')
        )
        self.reward_plotter.update_timestep(self.current_timestep)
        
        # Update window title
        self.fig.suptitle(f'{self.title} - Traj {self.current_traj_idx + 1}/{len(self.traj_data_list)} '
                         f'| FPS: {self.fps} | {"Playing" if self.is_playing else "Paused"}',
                         fontsize=14)
        
        plt.draw()
        
        # Save frame if video writer is active
        if self.video_writer is not None:
            self.video_writer.grab_frame()

    def export_all_demos(self, output_path: str, fps: int = 20, dpi: int = 100):
        """Export all demos to a video file without opening the interactive window."""
        self.start_video_export(output_path, fps=fps, dpi=dpi)
        if self.video_writer is None:
            raise RuntimeError(f'Could not initialize video export for {output_path}')

        total_frames = sum(demo['length'] for demo in self.traj_data_list)
        frame_progress = tqdm.tqdm(total=total_frames, desc='Saving video frames', unit='frame')

        try:
            for traj_idx, traj in enumerate(self.traj_data_list):
                self.current_traj_idx = traj_idx
                for timestep in range(traj['length']):
                    self.current_timestep = timestep
                    self._update_display()
                    frame_progress.update(1)
        finally:
            frame_progress.close()
            if self.video_writer is not None:
                self.video_writer.finish()
                self.video_writer = None
            plt.close(self.fig)
    
    def _toggle_play(self, event):
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Pause' if self.is_playing else 'Play')
    
    def _step_forward(self, event=None):
        """Step forward one timestep."""
        if self.traj_data_list and self.current_timestep < self.traj_data_list[self.current_traj_idx]['length'] - 1:
            self.current_timestep += 1
            self._update_display()
    
    def _step_backward(self, event=None):
        """Step backward one timestep."""
        if self.current_timestep > 0:
            self.current_timestep -= 1
            self._update_display()
    
    def _next_traj(self, event):
        """Switch to next trajectory."""
        if self.current_traj_idx < len(self.traj_data_list) - 1:
            self.current_traj_idx += 1
            self.current_timestep = 0
            self._update_display()
    
    def _prev_traj(self, event):
        """Switch to previous trajectory."""
        if self.current_traj_idx > 0:
            self.current_traj_idx -= 1
            self.current_timestep = 0
            self._update_display()
    
    def _faster(self, event):
        """Increase playback speed."""
        if self.fps in FPS_VALUES:
            curr_idx = FPS_VALUES.index(self.fps)
            self.fps = FPS_VALUES[min(curr_idx + 1, len(FPS_VALUES) - 1)]
        self._update_display()
    
    def _slower(self, event):
        """Decrease playback speed."""
        if self.fps in FPS_VALUES:
            curr_idx = FPS_VALUES.index(self.fps)
            self.fps = FPS_VALUES[max(curr_idx - 1, 0)]
        self._update_display()
    
    def _reset_timestep(self, event):
        """Reset to first timestep."""
        self.current_timestep = 0
        self._update_display()
    
    def start_video_export(self, output_path: str, fps: int = 30, dpi: int = 100):
        """Start exporting visualization to video file.
        
        Args:
            output_path: Path to save video file (should end in .mp4)
            fps: Frames per second for video
            dpi: DPI for figure rendering
        """
        if not CV2_AVAILABLE and output_path.endswith('.mp4'):
            print("Warning: OpenCV not available, will use matplotlib backend (slower)")
        
        self.save_video_path = output_path
        print(f"Will save video to: {output_path}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Setup FFMpeg writer
        try:
            # Get current figure size and DPI
            fig_width_inches = self.fig.get_figwidth()
            fig_height_inches = self.fig.get_figheight()
            figsize_pixels = (int(fig_width_inches * dpi), int(fig_height_inches * dpi))
            
            self.video_writer = FFMpegWriter(fps=fps, metadata={'artist': 'rew_video_analyzer'})
            self.video_writer.setup(self.fig, output_path, dpi=dpi)
            print(f"Video export setup: {figsize_pixels} @ {fps} FPS")
        except Exception as e:
            print(f"Error setting up video export: {e}")
            self.video_writer = None
    
    def run(self, auto_play: bool = False, only_one: bool = False):
        """Run the visualization loop.
        
        Args:
            auto_play: Start playing automatically
            only_one: Stop after playing first trajectory
        """
        if auto_play:
            self.is_playing = True
            self.btn_play.label.set_text('Pause')
        
        print("\n=== Controls ===")
        print("  Spacebar: Play/Pause")
        print("  Left/Right arrows: Step backward/forward")
        print("  Up/Down arrows: Next/Previous trajectory")
        print("  +/-: Faster/Slower playback")
        print("  r: Reset to first timestep")
        print("  q/Escape: Exit")
        print("  Close window to exit")
        
        try:
            while not self.should_exit:
                if self.is_playing and self.traj_data_list:
                    current_time = time.time()
                    if current_time - self.last_frame_time >= 1.0 / self.fps:
                        self._step_forward()
                        self.last_frame_time = current_time
                        
                        # Auto-advance to next trajectory when current one ends
                        traj = self.traj_data_list[self.current_traj_idx]
                        if self.current_timestep >= traj['length'] - 1:
                            if only_one:
                                self.should_exit = True
                                continue
                            
                            if self.current_traj_idx < len(self.traj_data_list) - 1:
                                self._next_traj(None)
                            else:
                                self.is_playing = False
                                self.btn_play.label.set_text('Play')
                                print("Reached end of all trajectories")
                
                # Check if figure is still open
                if not plt.fignum_exists(self.fig.number):
                    self.should_exit = True
                    break
                
                plt.pause(0.01)
                
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.video_writer is not None:
                self.video_writer.finish()
            plt.close('all')


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Visualize .h5 trajectory videos with rewards from JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--h5', required=True, help='Path to .h5 trajectory file')
    parser.add_argument('--rewards', required=True, help='Path to JSON rewards file')
    parser.add_argument('--traj_idx', type=int, default=0, help='Initial trajectory index (default: 0)')
    parser.add_argument('--fps', type=float, default=20.0, help='Playback FPS (default: 20.0)')
    parser.add_argument('--auto_play', action='store_true', help='Start playing automatically')
    parser.add_argument('--only_one', action='store_true', help='Play only first trajectory')
    parser.add_argument('--count', type=int, default=None, help='Maximum number of trajectories to load/process')
    parser.add_argument('--figsize', nargs=2, type=float, default=[18, 8], help='Figure size [width height]')
    parser.add_argument('--title', type=str, help='Visualization title')
    parser.add_argument('--save_video', action='store_true', help='Save visualization as video file')
    parser.add_argument('--video_dir', type=str, default='out/viz_rew', help='Save visualization as video (e.g., output.mp4)')
    parser.add_argument('--video_name', type=str, default='out', help='Name of the output video file')
    parser.add_argument('--only_video', action='store_true', help='Export a video for all demos and exit')
    parser.add_argument('--video_fps', type=int, default=20, help='Video export FPS (default: 30)')
    parser.add_argument('--video_dpi', type=int, default=100, help='Video export DPI (default: 100)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.h5):
        print(f"Error: H5 file '{args.h5}' not found!")
        sys.exit(1)
    
    if not os.path.exists(args.rewards):
        print(f"Error: Rewards JSON file '{args.rewards}' not found!")
        sys.exit(1)
    
    # Create visualizer
    visualizer = VideoVisualizer(figsize=tuple(args.figsize), title=args.title)
    visualizer.fps = args.fps

    if args.only_video:
        args.auto_play = True
        args.only_one = False
        
    out_video_path = os.path.join(args.video_dir, args.video_name + '.mp4')
    
    # Setup video export if requested in interactive mode.
    if args.save_video and not args.only_video:
        visualizer.start_video_export(out_video_path, fps=args.video_fps, dpi=args.video_dpi)
    
    # Load data
    if not visualizer.load_data(args.h5, args.rewards, max_count=args.count):
        sys.exit(1)
    
    # Set initial trajectory
    visualizer.set_trajectory(args.traj_idx)

    if args.only_video:
        visualizer.export_all_demos(out_video_path, fps=args.video_fps, dpi=args.video_dpi)
    else:
        # Run visualization
        visualizer.run(auto_play=args.auto_play, only_one=args.only_one)


if __name__ == '__main__':
    main()
