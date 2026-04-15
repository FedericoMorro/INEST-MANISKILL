# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environment wrappers."""

import abc
import collections
import os
import time
import typing

import cv2
import gym
import imageio
import numpy as np
import torch
from xirl.models import SelfSupervisedModel
import wandb
import pdb
import glob
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
import torchvision.transforms as T
#from vip import load_vip
from PIL import Image  
from scipy.optimize import linear_sum_assignment



from sac.state_entropy_tracker import StateEntropyTracker, IntrinsicRewardWrapper


class DictToArrayWrapper(gym.ObservationWrapper):
    """Convert dictionary observations to numpy arrays for SAC compatibility."""
    
    def __init__(self, env):
        super().__init__(env)
        
        # Test the environment to see what observations look like
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            test_obs, _ = reset_result  # Gymnasium format: (obs, info)
        else:
            test_obs = reset_result  # Gym format: obs
        
        if isinstance(test_obs, dict):
            # Observations are dictionaries, determine keys and shapes
            self._keys = sorted(test_obs.keys())
            self._shapes = [test_obs[key].shape for key in self._keys]
            self._dtypes = [test_obs[key].dtype for key in self._keys]
            
            # Calculate total flattened size
            total_size = sum(np.prod(shape) for shape in self._shapes)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32
            )
            print(f"[DictToArrayWrapper] Converting dict obs with keys {self._keys} to array of size {total_size}")
        elif hasattr(env.observation_space, 'spaces'):
            # This is a Dict observation space
            self._keys = sorted(env.observation_space.spaces.keys())
            self._shapes = [env.observation_space.spaces[key].shape for key in self._keys]
            self._dtypes = [env.observation_space.spaces[key].dtype for key in self._keys]
            
            # Calculate total flattened size
            total_size = sum(np.prod(shape) for shape in self._shapes)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32
            )
            print(f"[DictToArrayWrapper] Converting dict obs with keys {self._keys} to array of size {total_size}")
        else:
            # Already a Box space, no conversion needed
            self.observation_space = env.observation_space
            self._keys = None
            print(f"[DictToArrayWrapper] No conversion needed - observations are already arrays")
    
    def observation(self, obs):
        if self._keys is None:
            return obs
        
        # Convert dictionary to flattened array
        arrays = []
        for key in self._keys:
            if key in obs:
                arrays.append(obs[key].flatten())
            else:
                # Fill with zeros if key is missing
                if hasattr(self.env.observation_space, 'spaces') and key in self.env.observation_space.spaces:
                    shape = self.env.observation_space.spaces[key].shape
                else:
                    # Fallback: use the shape from the test observation
                    shape = self._shapes[self._keys.index(key)]
                arrays.append(np.zeros(np.prod(shape), dtype=np.float32))
        
        return np.concatenate(arrays).astype(np.float32)


class XMagicalPretrainedIntrinsicWrapper(gym.Wrapper):
    """Wrapper that adds pretrained intrinsic rewards to X-Magical environments."""
    
    def __init__(
        self,
        env,
        pretrained_model,
        device,
        intrinsic_weight: float = 0.1,
        extrinsic_weight: float = 1.0,
        re3_cfg: dict = None,
        use_images: bool = False,
        use_bboxes: bool = True
    ):
        """Initialize the pretrained intrinsic reward wrapper for X-Magical.
        
        Args:
            env: X-Magical environment to wrap
            pretrained_model: Pretrained model for computing embeddings
            device: Device to use for computations
            intrinsic_weight: Weight for intrinsic reward component
            extrinsic_weight: Weight for extrinsic reward component
            re3_cfg: Configuration for pretrained method
            use_images: Whether to use image observations
            use_bboxes: Whether to use bbox extraction from state (for X-Magical)
        """
        super().__init__(env)
        
        # Initialize state entropy tracker with pretrained model
        self.state_entropy_tracker = StateEntropyTracker(
            state_dim=env.observation_space.shape[0] if hasattr(env, 'observation_space') else 50,
            method="pretrained",
            intrinsic_weight=intrinsic_weight,
            device=device,
            re3_cfg=re3_cfg,
            pretrained_model=pretrained_model
        )
        
        # Set image environment if using images
        if use_images:
            self.state_entropy_tracker.set_image_env(env)
        
        # Set state-to-bboxes function if using bboxes
        if use_bboxes:
            # Create a state-to-bboxes function similar to DistanceToGoalBboxReward
            def state_to_bboxes_func(obs):
                return self.state_entropy_tracker._create_xmagical_state_to_bboxes(obs)
            self.state_entropy_tracker.set_state_to_bboxes_func(state_to_bboxes_func)
        
        self.intrinsic_weight = intrinsic_weight
        self.extrinsic_weight = extrinsic_weight
        
        # Statistics
        self.episode_intrinsic_rewards = []
        self.episode_extrinsic_rewards = []
        
    def reset(self, **kwargs):
        """Reset the environment and clear episode statistics."""
        obs = self.env.reset(**kwargs)
        self.episode_intrinsic_rewards = []
        self.episode_extrinsic_rewards = []
        return obs
    
    def step(self, action):
        """Take a step in the environment and compute combined reward."""
        obs, extrinsic_reward, done, info = self.env.step(action)
        
        # Compute intrinsic reward using pretrained encoder
        intrinsic_reward = self.state_entropy_tracker.add_state(obs)
        
        # Combine rewards
        total_reward = (self.extrinsic_weight * extrinsic_reward + 
                       self.intrinsic_weight * intrinsic_reward)
        
        # Store statistics
        self.episode_intrinsic_rewards.append(intrinsic_reward)
        self.episode_extrinsic_rewards.append(extrinsic_reward)
        
        # Add statistics to info
        info["intrinsic_reward"] = intrinsic_reward
        info["extrinsic_reward"] = extrinsic_reward
        info["total_reward"] = total_reward
        
        if done:
            info["episode_intrinsic_reward"] = sum(self.episode_intrinsic_rewards)
            info["episode_extrinsic_reward"] = sum(self.episode_extrinsic_rewards)
            info["episode_total_reward"] = sum(self.episode_intrinsic_rewards) + sum(self.episode_extrinsic_rewards)
        
        return obs, total_reward, done, info
    
    def render(self, mode="rgb_array", **kwargs):
        """Forward render call to the underlying environment."""
        if hasattr(self.env, 'render'):
            try:
                return self.env.render(mode=mode, **kwargs)
            except TypeError:
                # If the underlying env doesn't accept mode parameter, try without it
                return self.env.render(**kwargs)
        else:
            # Return a dummy image if render is not available
            print("❌ Render is not available for this environment")
            return np.zeros((64, 64, 3), dtype=np.uint8)


def create_xmagical_env_with_pretrained_intrinsic(
    env_name: str,
    pretrained_model,
    device,
    intrinsic_weight: float = 0.1,
    extrinsic_weight: float = 1.0,
    pretrained_cfg: dict = None,
    use_images: bool = False,
    use_bboxes: bool = True,
    use_dense_reward: bool = False,
    config: dict = None,
    add_episode_monitor: bool = True,
    action_repeat: int = 1,
    frame_stack: int = 1,
    save_dir: str = None,
    wandb_video_freq: int = 0,
    seed: int = 0
):
    """Create X-Magical environment with pretrained intrinsic rewards.
    
    Args:
        env_name: Name of the X-Magical environment
        pretrained_model: Pretrained model for computing embeddings
        device: Device to use for computations
        intrinsic_weight: Weight for intrinsic reward component
        extrinsic_weight: Weight for extrinsic reward component
        pretrained_cfg: Configuration for pretrained method
        use_images: Whether to use image observations
        use_bboxes: Whether to use bbox extraction from state (for X-Magical)
        use_dense_reward: Whether to use dense rewards
        config: Environment configuration
        add_episode_monitor: Whether to add episode monitoring
        action_repeat: Number of times to repeat each action
        frame_stack: Number of frames to stack
        save_dir: Directory to save videos
        wandb_video_freq: Frequency for wandb video logging
        seed: Random seed
        
    Returns:
        Wrapped X-Magical environment with pretrained intrinsic rewards
    """
    import xmagical
    from gym.wrappers import RescaleAction
    
    # Register X-Magical environments
    xmagical.register_envs()
    
    if env_name not in xmagical.ALL_REGISTERED_ENVS:
        raise ValueError(f"{env_name} is not a valid X-Magical environment name.")
    
    # Create base environment
    env = gym.make(env_name, use_dense_reward=use_dense_reward, config=config)
    
    # Add standard wrappers
    if add_episode_monitor:
        env = EpisodeMonitor(env)
    if action_repeat > 1:
        env = ActionRepeat(env, action_repeat)
    env = RescaleAction(env, -1.0, 1.0)
    if save_dir is not None:
        env = VideoRecorder(env, save_dir=save_dir, wandb_video_freq=wandb_video_freq)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    
    # Add pretrained intrinsic reward wrapper
    env = XMagicalPretrainedIntrinsicWrapper(
        env=env,
        pretrained_model=pretrained_model,
        device=device,
        intrinsic_weight=intrinsic_weight,
        extrinsic_weight=extrinsic_weight,
        re3_cfg=pretrained_cfg,
        use_images=use_images,
        use_bboxes=use_bboxes
    )
    
    # Set seed
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    return env


TimeStep = typing.Tuple[np.ndarray, float, bool, dict]
ModelType = SelfSupervisedModel
TensorType = torch.Tensor
DistanceFuncType = typing.Callable[[float], float]
InfoMetric = typing.Mapping[str, typing.Mapping[str, typing.Any]]


class FrameStack(gym.Wrapper):
  """Stack the last k frames of the env into a flat array.

  This is useful for allowing the RL policy to infer temporal information.

  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env, k):
    """Constructor.

    Args:
      env: A gym env.
      k: The number of frames to stack.
    """
    super().__init__(env)

    assert isinstance(k, int), "k must be an integer."

    self._k = k
    self._frames = collections.deque([], maxlen=k)

    shp = env.observation_space.shape
    self.observation_space = gym.spaces.Box(
        low=env.observation_space.low.min(),
        high=env.observation_space.high.max(),
        shape=((shp[0] * k,) + shp[1:]),
        dtype=env.observation_space.dtype,
    )

  def reset(self):
    obs = self.env.reset()
    for _ in range(self._k):
      self._frames.append(obs)
    return self._get_obs()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self._frames.append(obs)
    return self._get_obs(), reward, done, info

  def _get_obs(self):
    assert len(self._frames) == self._k
    return np.concatenate(list(self._frames), axis=0)


class ActionRepeat(gym.Wrapper):
  """Repeat the agent's action N times in the environment.

  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env, repeat):
    """Constructor.

    Args:
      env: A gym env.
      repeat: The number of times to repeat the action per single underlying env
        step.
    """
    super().__init__(env)

    assert repeat > 1, "repeat should be greater than 1."
    self._repeat = repeat

  def step(self, action):
    total_reward = 0.0
    for _ in range(self._repeat):
      obs, rew, done, info = self.env.step(action)
      total_reward += rew
      if done:
        break
    return obs, total_reward, done, info


class RewardScale(gym.Wrapper):
  """Scale the environment reward."""

  def __init__(self, env, scale):
    """Constructor.

    Args:
      env: A gym env.
      scale: How much to scale the reward by.
    """
    super().__init__(env)

    self._scale = scale

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    reward *= self._scale
    return obs, reward, done, info


class EpisodeMonitor(gym.ActionWrapper):
  """A class that computes episode metrics.

  At minimum, episode return, length and duration are computed. Additional
  metrics that are logged in the environment's info dict can be monitored by
  specifying them via `info_metrics`.

  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env):
    super().__init__(env)

    self._reset_stats()
    self.total_timesteps: int = 0

  def _reset_stats(self):
    self.reward_sum: float = 0.0
    self.episode_length: int = 0
    self.start_time = time.time()

  def step(self, action):
    result = self.env.step(action)
    if len(result) == 5:
        next_observation, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        next_observation, reward, done, info = result

    self.reward_sum += reward
    self.episode_length += 1
    self.total_timesteps += 1
    info["total"] = {"timesteps": self.total_timesteps}

    if done:
      info["episode"] = dict()
      info["episode"]["return"] = self.reward_sum
      info["episode"]["length"] = self.episode_length
      info["episode"]["duration"] = time.time() - self.start_time

    return next_observation, reward, done, info

  def reset(self):
    self._reset_stats()
    return self.env.reset()


class VideoRecorder(gym.Wrapper):
  """Wrapper for rendering and saving rollouts to disk.

  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(
      self,
      env,
      save_dir,
      resolution = (128, 128),
      fps = 30,
      wandb_video_freq=0
  ):
    super().__init__(env)

    self.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    self.height, self.width = resolution
    self.fps = fps
    self.enabled = True
    self.current_episode = 0
    self.frames = []
    self.wandb_video_freq = wandb_video_freq

  def step(self, action):
    frame = self.env.render(mode="rgb_array")
    if frame.shape[:2] != (self.height, self.width):
      frame = cv2.resize(
          frame,
          dsize=(self.width, self.height),
          interpolation=cv2.INTER_CUBIC,
      )
    self.frames.append(frame)
    observation, reward, done, info = self.env.step(action)
    if done:
      if self.wandb_video_freq!= 0:
        # remove old videos if already on wandb
        files = [f for f in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, f))]
        # for f in files:
        #     os.remove(f)
        if (self.current_episode+1) % self.wandb_video_freq  == 0:
          frames = np.array([frame.transpose(2, 0, 1)  for frame in self.frames])
          wandb.log({'eval/eval_video%s'%self.env: wandb.Video(frames, fps=self.fps, format="mp4") })
      filename = os.path.join(self.save_dir, f"{self.current_episode}.mp4")
      imageio.mimsave(filename, self.frames, fps=self.fps)
      self.frames = []
      self.current_episode += 1
    return observation, reward, done, info


# ========================================= #
# Learned reward wrappers.
# ========================================= #

# Note: While the below classes provide a nice wrapper API, they are not
# efficient for training RL policies as rewards are computed individually at
# every `env.step()` and so cannot take advantage of batching on the GPU.
# For actually training policies, it is better to use the learned replay buffer
# implementations in `sac.replay_buffer.py`. These store transitions in a
# staging buffer which is forwarded as a batch through the GPU.


class LearnedVisualReward(abc.ABC, gym.Wrapper):
  """Base wrapper class that replaces the env reward with a learned one.

  Subclasses should implement the `_get_reward_from_image` method.
  """

  def __init__(
      self,
      env,
      model,
      device,
      res_hw = None,
  ):
    """Constructor.

    Args:
      env: A gym env.
      model: A model that ingests RGB frames and returns embeddings. Should be a
        subclass of `xirl.models.SelfSupervisedModel`.
      device: Compute device.
      res_hw: Optional (H, W) to resize the environment image before feeding it
        to the model.
    """
    super().__init__(env)

    self._device = device
    self._model = model.to(device).eval()
    self._res_hw = res_hw

  def _to_tensor(self, x):
    x = torch.from_numpy(x).permute(2, 0, 1).float()[None, None, Ellipsis]
    # TODO(kevin): Make this more generic for other preprocessing.
    x = x / 255.0
    x = x.to(self._device)
    return x

  def _render_obs(self):
    """Render the pixels at the desired resolution."""
    # TODO(kevin): Make sure this works for mujoco envs.
    pixels = self.env.render(mode="rgb_array")
    if self._res_hw is not None:
      h, w = self._res_hw
      pixels = cv2.resize(pixels, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    return pixels

  @abc.abstractmethod
  def _get_reward_from_image(self, image):
    """Forward the pixels through the model and compute the reward."""

  def step(self, action):
    obs, env_reward, done, info = self.env.step(action)
    # We'll keep the original env reward in the info dict in case the user would
    # like to use it in conjunction with the learned reward.
    info["env_reward"] = env_reward
    pixels = self._render_obs()
    learned_reward = self._get_reward_from_image(pixels)
    return obs, learned_reward, done, info


class DistanceToGoalLearnedVisualReward(LearnedVisualReward):
  """Replace the environment reward with distances in embedding space."""

  def __init__(
      self,
      goal_emb,
      distance_scale = 1.0,
      **base_kwargs,
  ):
    """Constructor.

    Args:
      goal_emb: The goal embedding.
      distance_scale: Scales the distance from the current state embedding to
        that of the goal state. Set to `1.0` by default.
      **base_kwargs: Base keyword arguments.
    """
    super().__init__(**base_kwargs)

    self._goal_emb = np.atleast_2d(goal_emb)
    self._distance_scale = distance_scale

  def _get_reward_from_image(self, image):
    """Forward the pixels through the model and compute the reward."""
    image_tensor = self._to_tensor(image)
    emb = self._model.infer(image_tensor).numpy().embs
    dist = -1.0 * np.linalg.norm(emb - self._goal_emb)
    dist *= self._distance_scale
    return dist


class GoalClassifierLearnedVisualReward(LearnedVisualReward):
  """Replace the environment reward with the output of a goal classifier."""

  def _get_reward_from_image(self, image):
    """Forward the pixels through the model and compute the reward."""
    image_tensor = self._to_tensor(image)
    prob = torch.sigmoid(self._model.infer(image_tensor).embs)
    return prob.item()

class DistanceToGoalBboxReward(DistanceToGoalLearnedVisualReward):
  """Replace the environment reward with distances in embedding space."""

  def __init__(
      self,
      obj_det,
      **base_kwargs,
  ):
    """Constructor.

    Args:
      goal_emb: The goal embedding.
      distance_scale: Scales the distance from the current state embedding to
        that of the goal state. Set to `1.0` by default.
      **base_kwargs: Base keyword arguments.
    """
    super().__init__(**base_kwargs)
    self.gripper_size = 60   
    self.target_1_size = 25
    self.target_2_size = 20
    self.goal_h = 0.65* 384/ 2 
    self.goal_w = 0.55* 384/ 2 
    self.goal_x = 0.1
    self.goal_y = 0.7
    self.dist_1_size = 25
    self.dist_2_size = 25
    self.dist_3_size = 25

    # Position adjustment offsets for fine-tuning
    self.gripper_offset = np.array([0, 0])     # Move gripper down
    self.target_1_offset = np.array([0, 0])  # Move target_1 up-left
    self.target_2_offset = np.array([0, 0])    # Move target_2 down-right
    self.goal_offset = np.array([0, 0]) 

    self.obj_det = obj_det

    self.agent_in_bboxes = True
    self.distr_in_bboxes = True
    self.bboxes_from_state = True
    self.num_obj = sum([self.agent_in_bboxes]+[self.distr_in_bboxes]*3)+1+2
    self.class_id = self._model.class_id

  def convert_coordinates(self, pos):
      pos[0] = (pos[0] + 1.1) / 2.2 * 384
      pos[1] = 384 - ((pos[1] + 1.1) / 2.2) * 384
      return pos


  def state_to_bboxes(self, obs):
      state = np.copy(obs)
      bboxes = np.zeros((1, 4))
      #adapt to the env!!!
      gripper_pos = state[:2] 
      target_1_pos = state[2:4] # STAR
      target_2_pos = state[4:6] # RECT
      goal_pos = np.array([self.goal_x, self.goal_y])
      dist_1_pos = state[6:8] # PENT (blue)
      dist_2_pos = state[8:10] # CIRC (yellow)
      dist_3_pos = state[10:12] # PENT (yellow)

      gripper_pos = self.convert_coordinates(gripper_pos) + self.gripper_offset
      target_1_pos = self.convert_coordinates(target_1_pos) + self.target_1_offset # STAR
      target_2_pos = self.convert_coordinates(target_2_pos) + self.target_2_offset # RECT
      goal_pos = self.convert_coordinates(goal_pos) + self.goal_offset
      dist_1_pos = self.convert_coordinates(dist_1_pos)
      dist_2_pos = self.convert_coordinates(dist_2_pos)
      dist_3_pos = self.convert_coordinates(dist_3_pos)
        
      #adapt to the env!!!
      bboxes[0] = goal_pos[0], goal_pos[1], goal_pos[0] + self.goal_w, goal_pos[1] + self.goal_h
      if self.agent_in_bboxes:
        bboxes = np.append(bboxes, [[gripper_pos[0] - self.gripper_size, gripper_pos[1] - self.gripper_size, gripper_pos[0] + self.gripper_size, gripper_pos[1] + self.gripper_size]], axis=0)
        
      bboxes = np.append(bboxes, [[target_1_pos[0] - self.target_1_size, target_1_pos[1] - self.target_1_size, target_1_pos[0] + self.target_1_size, target_1_pos[1] + self.target_1_size]], axis=0)
      bboxes = np.append(bboxes, [[target_2_pos[0] - self.target_2_size, target_2_pos[1] - self.target_2_size, target_2_pos[0] + self.target_2_size, target_2_pos[1] + self.target_2_size]], axis=0)

      if self.distr_in_bboxes:
        bboxes = np.append(bboxes, [[dist_1_pos[0] - self.dist_1_size, dist_1_pos[1] - self.dist_1_size, dist_1_pos[0] + self.dist_1_size, dist_1_pos[1] + self.dist_1_size]], axis=0)
        bboxes = np.append(bboxes, [[dist_2_pos[0] - self.dist_2_size, dist_2_pos[1] - self.dist_2_size, dist_2_pos[0] + self.dist_2_size, dist_2_pos[1] + self.dist_2_size]], axis=0)
        bboxes = np.append(bboxes, [[dist_3_pos[0] - self.dist_3_size, dist_3_pos[1] - self.dist_3_size, dist_3_pos[0] + self.dist_3_size, dist_3_pos[1] + self.dist_3_size]], axis=0)
      
      bboxes =  np.reshape(bboxes, (1, len(bboxes), 4)) 
      return bboxes
  
  def retrieve_bboxes_from_state(self, state, id):
    bboxes = self.state_to_bboxes(state)
    if self.agent_in_bboxes:
          considered_objects =  [1, 2, 3, 4, 5, 6, 7] 
    else:
      considered_objects =  [1, 3, 4, 5, 6, 7] 
    # Find index of the first occurrence of `id`
    index = np.where(np.array(considered_objects) ==id)[0]

    if index.size <= 0:
        print("ID not found!")

    return bboxes[0][index[0]]

  # Function to get the class name from the label
  def get_class_name(self, label, names):
      return names[int(label)]

  # Function to reduce bbox size
  def reduce_bbox(self, bbox, reduction_percent, img_shape):
      x1, y1, x2, y2 = bbox
      width = x2 - x1
      height = y2 - y1

      # Calculate the reduction
      reduction_width = width * reduction_percent / 100
      reduction_height = height * reduction_percent / 100

      # Apply the reduction
      new_x1 = max(0, x1 + reduction_width / 2)  # Ensure not going out of bounds
      new_y1 = max(0, y1 + reduction_height / 2)
      new_x2 = min(img_shape[1], x2 - reduction_width / 2)  # img_shape[1] = image width
      new_y2 = min(img_shape[0], y2 - reduction_height / 2)  # img_shape[0] = image height

      return [new_x1, new_y1, new_x2, new_y2]

  def reorder_bboxes(self, bboxes, labels, img):
      names = {
            0: 'goal', 
            1: 'circle', 
            2: 'penthagon',  # Default pentagon
            3: 'rect', 
            4: 'robot', 
            5: 'star',
        }

      ordered_names = {1: 'goal', 2: 'robot', 3: 'star', 4: 'rect', 5: 'penthagon1', 6: 'circle', 7: 'penthagon2'}
      # Define the desired order of classes
      order = {
          'goal': 1,
          'robot': 2,
          'star': 3,
          'rect': 4,
          'penthagon1': 5,
          'circle': 6,
          'penthagon2': 7,
      }
      # Initialize arrays for ordered bboxes and labels
      ordered_bboxes = []
      ordered_labels = []

      if len(bboxes) != len(ordered_names.keys()):
        print(f"DANGER! Detection fails (detected {len(bboxes)} objects).")

      # Process each bounding box and label
      for bbox, label in zip(bboxes, labels):
          class_name = self.get_class_name(label, names)

          # Reduce bbox size for specific classes
          if class_name in ['penthagon', 'circle']:
              bbox = self.reduce_bbox(bbox, reduction_percent=15, img_shape=img.shape)
          elif class_name == 'star':
              bbox = self.reduce_bbox(bbox, reduction_percent=20, img_shape=img.shape)
          
          if class_name == 'penthagon':
              # Determine the specific pentagon type by color
              center_y = int((bbox[3] + bbox[1]) / 2)
              center_x = int((bbox[2] + bbox[0]) / 2)
              color = img[center_y, center_x]
              
              if np.array_equal(color, [254, 213, 123]):  # penthagon2 color
                  class_name = 'penthagon2'
              elif np.array_equal(color, [135, 185, 211]):  # penthagon1 color
                  class_name = 'penthagon1'
              else:
                  print("Detected wrongly a penthagon, skipping this bbox.")
                  continue  # Skip this bounding box and move to the next one
          
          # Append to ordered list
          ordered_bboxes.append((bbox, order[class_name]))

      # Sort the bounding boxes based on the desired order
      ordered_bboxes.sort(key=lambda x: x[1])

      # Separate the bboxes and labels
      sorted_bboxes = [item[0] for item in ordered_bboxes]
      sorted_labels = [item[1] for item in ordered_bboxes]

      return np.array(sorted_bboxes), np.array(sorted_labels)

  def image_to_bboxes(self, image, state):
        final_bboxes = []
        ordered_names = {1: 'goal', 2: 'robot', 3: 'star', 4: 'rect', 5: 'penthagon1', 6: 'circle', 7: 'penthagon2'}

        if self.agent_in_bboxes:
          considered_objects =  [1, 2, 3, 4, 5, 6, 7] 
        else:
          considered_objects =  [1, 3, 4, 5, 6, 7] 

        # Perform object detection on the frame
        results = self.obj_det.predict(image, iou=0.4, conf=0.20, verbose=False)
        # Extract bounding boxes and labels
        bboxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        labels = results[0].boxes.cls.cpu().numpy()  # Class labels
        bboxes, labels = self.reorder_bboxes(bboxes, labels, image)
        # Create a dictionary to group bboxes by label
        bbox_dict = {i: [] for i in range(len(ordered_names))}  # List for each label

        # Fill bbox_dict with actual bboxes
        for bbox, label in zip(bboxes, labels):
            label_int = int(label)
            centre_y = (bbox[3] + bbox[1]) / 2
            centre_x = (bbox[2] + bbox[0]) / 2
            if label_int == 3 or label_int == 5:
              if np.array_equal(image[round(centre_y),round(centre_x)],np.array([195, 208, 130])):
                bbox_dict[label_int-1].append(bbox[:4])  # Append bbox coordinates
            else:
              bbox_dict[label_int-1].append(bbox[:4])  # Append bbox coordinates
        for label_int in sorted(bbox_dict.keys()):
          if label_int in considered_objects:
              if bbox_dict[label_int]:  # If there are detected boxes for the label
                  for bbox in bbox_dict[label_int]:
                      if len(final_bboxes)<len(considered_objects):
                        final_bboxes.append(bbox)
              else:  # No detection for this label
                  print(f"DANGER! No detection for label '{ordered_names[label_int]}'")
                  if len(final_bboxes)<len(considered_objects):
                    sim_bbox = self.retrieve_bboxes_from_state(state, label_int)
                    final_bboxes.append(sim_bbox)  # Placeholder for missing bbox
        
        return np.array([final_bboxes])
    
  def append_distances(self, bboxes):
      
      bboxes = bboxes[0]
      
      # Calculate the centers of each bounding box
      centres_y = (bboxes[:, 3] + bboxes[:, 1]) / 2
      centres_x = (bboxes[:, 2] + bboxes[:, 0]) / 2 
      centres = np.column_stack([centres_x, centres_y])

      # Calculate pairwise distances between centers
      distances_matrix = pairwise_distances(centres, metric='euclidean', n_jobs=-1)

      num_bboxes = len(bboxes)  # Total number of bounding boxes
      distances = []

      # Collect pairwise distances dynamically
      for i in range(num_bboxes):
          for j in range(num_bboxes):
              if i != j:  # Exclude self-distances
                  distances.append(distances_matrix[i, j])

      # Reshape distances array to match each bounding box having distances to all others
      # Each bbox has `num_bboxes - 1` distances
      distances = np.reshape(distances, (num_bboxes, num_bboxes - 1))

      # Combine bboxes with the pairwise distances
      final_features = np.column_stack([bboxes, distances])

      # Reshape to the required format
      # Shape: (1, num_bboxes, 4 + (num_bboxes - 1)) -> (1, num_bboxes, 4 coordinates (+1 for class) + distances)
      return np.reshape(final_features, (1, num_bboxes, len(bboxes[0]) + (num_bboxes - 1)))

    
  
  def _get_reward_from_image(self, image, state=None):
    
      """Forward the pixels through the model and compute the reward."""
      # Generate the bounding boxes from the state
      bboxes_curr = self.state_to_bboxes(state)
      bboxes_normalized = np.copy(bboxes_curr)
      bboxes_normalized[0, :, [0, 2]] /= 384  # Normalize x-coordinates (x1 and x2)
      bboxes_normalized[0, :, [1, 3]] /= 384  # Normalize y-coordinates (y1 and y2)

      if not self.bboxes_from_state:
        bboxes_curr_from_image = self.image_to_bboxes(image, state)
        bboxes_from_image_normalized = np.copy(bboxes_curr_from_image)
        bboxes_from_image_normalized /= image.shape[0] # Normalize xy-coordinates (assumiing the image is squared)

      # VISUALIZE HERE THE BOUNDING BOX COMPUTED      
      # img = self.vis_bboxes(bboxes_curr[0], image, state)
      # img_objdet = self.vis_bboxes_from_objdet(bboxes_curr_from_image[0], image)

      # file_list = os.listdir("./tests/bboxes")
      # png_files = [file for file in file_list if file.endswith('.png')]

      # if not png_files:
      #     next_number = '1'  # If no PNG files exist, start with 1.png
      # else:
      #   # Extract the numbers from the filenames and sort them
      #   numbers = [int(file.split('_')[1].split('.')[0]) for file in png_files]
      #   numbers.sort()

      #   next_number = numbers[-1] + 1  # Get the next number
      
      # cv2.imwrite(os.path.join("./tests/bboxes", f"sim_{next_number}.png"), img)
      # cv2.imwrite(os.path.join("./tests/bboxes", f"objdet_{next_number}.png"), img_objdet)

      #print(bboxes_curr)
      # import pdb
      # pdb.set_trace()

      # Append distances to the normalized bounding boxes
      if self.bboxes_from_state:
        if self.class_id:
          bboxes_normalized = np.append(bboxes_normalized, np.array([[[i] for i in range(self.num_obj)]]), axis=2)
        bboxes_with_distances = self.append_distances(bboxes_normalized)
      else: 
        if self.class_id:
          bboxes_from_image_normalized = np.append(bboxes_from_image_normalized, np.array([[[i] for i in range(self.num_obj)]]), axis=2)
        bboxes_with_distances = self.append_distances(bboxes_from_image_normalized)

      # Convert to a tensor and send to the device for inference
      bboxes_tensor = torch.from_numpy(bboxes_with_distances).float()[None, None, Ellipsis]
      bboxes_tensor = bboxes_tensor.to(self._device)
      
      # Model inference
      emb = self._model.infer(bboxes_tensor).numpy().embs
      
      # Calculate the distance to the goal embedding
      dist = np.linalg.norm(emb - self._goal_emb)
      dist = -1.0 * dist * self._distance_scale
      
      return dist
  
  def vis_bboxes(self, bboxes, img, state):

    # if img:
    #   img = np.ones((384, 384, 3)) * 255
    img = cv2.resize(
          img,
          dsize=(384, 384),
          interpolation=cv2.INTER_CUBIC,
      )
    colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255), 6: (122, 255, 255)}

    for i in range(bboxes.shape[0]):
      x1, y1, x2, y2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]

      img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors[i], 2)

    return img
  
  def vis_bboxes_from_objdet(self, bboxes, img):
    img_bboxes = np.copy(img)
    for i in range(bboxes.shape[0]):
      x1, y1, x2, y2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]

      img_bboxes = cv2.rectangle(img_bboxes, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    return img_bboxes

  def step(self, action):
    obs, env_reward, done, info = self.env.step(action)
    # We'll keep the original env reward in the info dict in case the user would
    # like to use it in conjunction with the learned reward.
    info["env_reward"] = env_reward
    pixels = self._render_obs()
    state = np.copy(obs)
    learned_reward = self._get_reward_from_image(image=pixels, state=state) #+ env_reward

    # obs = cv2.resize(np.transpose(np.array(obs), (1, 2, 0)), (self.img_size, self.img_size))
    # obs = np.transpose(obs, (2, 0, 1))
    # info['env_reward'] = env_reward
    # info['state_full'] = state
    # state = np.concatenate([state[:3], np.array([state[-1]])]).astype('float64') #3d state
      
    
    return obs, learned_reward, done, info

class DistanceToGoalBboxReward_sweepToTop(DistanceToGoalLearnedVisualReward):
    def __init__(self, obj_det, **base_kwargs):
        super().__init__(**base_kwargs)
        self.gripper_size = 60 #CHECK THIS
        self.cube_size = 25 #CHECK THIS

        self.obj_det = obj_det
        self.agent_in_bboxes = True
        self.bboxes_from_state = False
        self.num_obj = 5  # goal, robot, 3 cubes
        self.class_id = self._model.class_id

        self.cube_id_to_slot = {}  # persistent mapping per rollout
        self.prev_cube_centers = None
        self.cube_slot_bboxes = None
        self.max_movement = 40.0  # pixels – used for distance thresholding

        self.debug = False
        self.idx = 0

    def convert_coordinates(self, pos):
        pos[0] = (pos[0] + 1.1) / 2.2 * 384
        pos[1] = 384 - ((pos[1] + 1.1) / 2.2) * 384
        return pos

    def state_to_bboxes(self, obs):
        state = np.copy(obs)
        bboxes = []

        gripper_pos = self.convert_coordinates(state[:2])
        cube1_pos = self.convert_coordinates(state[2:4])
        cube2_pos = self.convert_coordinates(state[4:6])
        cube3_pos = self.convert_coordinates(state[6:8])

        bboxes.append([0.8244, 0, 383.91, 61.664]) # goal position
        bboxes.append([
            gripper_pos[0] - self.gripper_size, gripper_pos[1] - self.gripper_size,
            gripper_pos[0] + self.gripper_size, gripper_pos[1] + self.gripper_size
        ])
        for pos in [cube1_pos, cube2_pos, cube3_pos]:
            bboxes.append([
                pos[0] - self.cube_size, pos[1] - self.cube_size,
                pos[0] + self.cube_size, pos[1] + self.cube_size
            ])
        return np.reshape(np.array([bboxes]), (1, 5, 4))

    def compute_center(self, bbox):
      return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    def reorder_bboxes(self, bboxes, labels, ids, frame_path=None, is_first_frame=False):
      object_slots = {1: None, 2: None, 3: None, 4: None, 5: None}
      label_map = {0: 'goal', 1: 'robot', 2: 'cube', 3: 'cube', 4: 'cube'}
      cube_bboxes = []

      for bbox, label in zip(bboxes, labels):
          class_name = label_map[int(label)]
          if class_name == 'goal':
              object_slots[1] = bbox
          elif class_name == 'robot':
              object_slots[2] = bbox
          elif class_name == 'cube':
              cube_bboxes.append(bbox)

      # FIRST FRAME: assign cubes by x-order left→right
      if is_first_frame or self.prev_cube_centers is None:
          self.prev_cube_centers = {}
          sorted_cubes = sorted(cube_bboxes[:3], key=lambda b: (b[0] + b[2]) / 2)  # center x
          for i, slot in enumerate([3, 4, 5]):
              if i < len(sorted_cubes):
                  object_slots[slot] = sorted_cubes[i]
                  self.prev_cube_centers[slot] = self.compute_center(sorted_cubes[i])
              else:
                  object_slots[slot] = [0.0, 0.0, 0.0, 0.0]
                  self.prev_cube_centers[slot] = None
          return np.array([object_slots[i] for i in range(1, 6)]), np.array(list(range(1, 6)))

      # SUBSEQUENT FRAMES
      matched = {3: None, 4: None, 5: None}
      assigned = set()
      curr_centers = [self.compute_center(b) for b in cube_bboxes]

      max_movement = 40.0

      for slot in [3, 4, 5]:
          prev_center = self.prev_cube_centers.get(slot)
          if prev_center is None:
              continue

          best_idx = None
          best_dist = float('inf')

          for i, center in enumerate(curr_centers):
              if i in assigned:
                  continue
              dist = np.linalg.norm(center - prev_center)
              if dist < best_dist:
                  best_dist = dist
                  best_idx = i

          if best_idx is not None and best_dist < max_movement:
              matched[slot] = cube_bboxes[best_idx]
              self.prev_cube_centers[slot] = curr_centers[best_idx]
              assigned.add(best_idx)
          else:
              # Mark slot as missing but keep prev center for future recovery
              matched[slot] = [0.0, 0.0, 0.0, 0.0]

      # Assign any unmatched cube to empty slot
      for i, bbox in enumerate(cube_bboxes):
          if i in assigned:
              continue
          for slot in [3, 4, 5]:
              if matched[slot] is not None and np.allclose(matched[slot], [0.0, 0.0, 0.0, 0.0]):
                  matched[slot] = bbox
                  self.prev_cube_centers[slot] = self.compute_center(bbox)
                  break

      # Finalize output
      for slot in [3, 4, 5]:
          object_slots[slot] = matched[slot] if matched[slot] is not None else [0.0, 0.0, 0.0, 0.0]

      final_bboxes = [object_slots[i] for i in range(1, 6)]
      final_labels = list(range(1, 6))

      return np.array(final_bboxes), np.array(final_labels)


    def image_to_bboxes(self, image, state):
        results = self.obj_det.predict(
                source=image,
                conf=0.25,
                imgsz=384,
                device=self._device,
                tracker="bytetrack.yaml",
                verbose=False,
                #persist=True
            )
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        ids = list(range(len(bboxes)))  # dummy IDs per frame

        is_first_frame = self.prev_cube_centers is None
        reordered_bboxes, _ = self.reorder_bboxes(bboxes, labels, ids, is_first_frame=is_first_frame)

        # Use sim fallback for missing detections
        sim_bboxes = self.state_to_bboxes(state)[0]
        final_bboxes = []
        for i, bbox in enumerate(reordered_bboxes):
            if np.allclose(bbox, [0, 0, 0, 0]):
                final_bboxes.append(sim_bboxes[i])
            else:
                final_bboxes.append(bbox)

        return np.array([final_bboxes])

    def append_distances(self, bboxes):
        bboxes = bboxes[0]
        centres_y = (bboxes[:, 3] + bboxes[:, 1]) / 2
        centres_x = (bboxes[:, 2] + bboxes[:, 0]) / 2
        centres = np.column_stack([centres_x, centres_y])
        distances_matrix = pairwise_distances(centres, metric='euclidean', n_jobs=-1)

        num_bboxes = len(bboxes)
        distances = []
        for i in range(num_bboxes):
            for j in range(num_bboxes):
                if i != j:
                    distances.append(distances_matrix[i, j])
        distances = np.reshape(distances, (num_bboxes, num_bboxes - 1))
        final_features = np.column_stack([bboxes, distances])
        return np.reshape(final_features, (1, num_bboxes, len(bboxes[0]) + (num_bboxes - 1)))
    
    def vis_bboxes_debug(self, image, bboxes, title="bbox_debug", idx=None):
      # Copy image to avoid in-place editing
      vis_img = np.copy(image)
      colors = {
          1: (0, 255, 0),   # Goal – green
          2: (255, 0, 0),   # Robot – red
          3: (0, 0, 255),   # Cube1 – blue
          4: (255, 255, 0), # Cube2 – yellow
          5: (255, 0, 255), # Cube3 – magenta
      }

      labels = ['goal', 'robot', 'cube1', 'cube2', 'cube3']

      for i in range(bboxes.shape[0]):
          x1, y1, x2, y2 = bboxes[i]
          cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), colors[i+1], 2)
          cv2.putText(vis_img, f"{labels[i]}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i+1], 1)

      # Show image with matplotlib (so it works in notebooks or headless envs)
      # plt.figure(figsize=(5, 5))
      # plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
      # plt.title(title)
      # plt.axis('off')
      # plt.show()
      image = Image.fromarray(vis_img)
      image.save(f"tests/test/{idx}.png")

    def _get_reward_from_image(self, image, state=None, idx=None):
        if self.bboxes_from_state:
            bboxes = self.state_to_bboxes(state)
        else:
            bboxes = self.image_to_bboxes(image, state)
        if self.debug:
          self.vis_bboxes_debug(image, bboxes[0], title="Visual check: BBoxes", idx=idx)
        norm_bboxes = np.copy(bboxes)
        norm_bboxes[0, :, [0, 2]] /= image.shape[1]
        norm_bboxes[0, :, [1, 3]] /= image.shape[0]

        if self.class_id:
            class_ids = np.array([[[i] for i in range(self.num_obj)]])
            norm_bboxes = np.append(norm_bboxes, class_ids, axis=2)

        bboxes_with_distances = self.append_distances(norm_bboxes)
        bboxes_tensor = torch.from_numpy(bboxes_with_distances).float()[None, None, Ellipsis].to(self._device)

        emb = self._model.infer(bboxes_tensor).numpy().embs
        dist = -1.0 * np.linalg.norm(emb - self._goal_emb) * self._distance_scale
        return dist

    def step(self, action):
      obs, env_reward, done, info = self.env.step(action)
      info["env_reward"] = env_reward
      pixels = self._render_obs()

      # 🔁 Only reset if this is the first frame of a new episode
      if self.idx == 0:
          self.cube_id_to_slot.clear()
          self.prev_cube_centers = None
          self.cube_slot_bboxes = None

      learned_reward = self._get_reward_from_image(image=pixels, state=np.copy(obs), idx=self.idx)
      self.idx += 1
      return obs, learned_reward, done, info

    def reset(self):
      obs = self.env.reset()  # ← reset the inner environment
      self.idx = 0
      self.prev_cube_centers = None
      self.cube_slot_bboxes = None
      self.cube_id_to_slot.clear()
      return obs


class DistanceToGoalLearnedVipReward(DistanceToGoalLearnedVisualReward):
  """Replace the environment reward with distances in embedding space."""

  def __init__(
      self,
      **base_kwargs,
  ):
    """Constructor.

    Args:
      **base_kwargs: Base keyword arguments.
    """
    self.model = load_vip()
    self.transform = T.Compose([T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor()])
    base_kwargs['model'] = self.model
    super().__init__(**base_kwargs)


  def _get_reward_from_image(self, image):
    """Forward the pixels through the model and compute the reward."""
    #image_tensor = self._to_tensor(image)
    #emb = self._model.infer(image_tensor).numpy().embs
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    img_cur = self.transform(Image.fromarray(image.astype(np.uint8)))
    img_cur = img_cur * 255 ## vip expects image input to be [0-255]
    img_cur.to(self._device) 
    with torch.no_grad():
      embeddings = self.model(torch.stack([img_cur]).cuda())
      embeddings = embeddings.cpu().numpy()
    
    dist = -1.0 * np.linalg.norm(embeddings - self._goal_emb)
    dist *= self._distance_scale
    return dist
