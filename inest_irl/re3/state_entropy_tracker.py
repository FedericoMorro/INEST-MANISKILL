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

"""State entropy tracker for intrinsic motivation.

This module provides functionality to track state entropy and compute intrinsic
rewards based on state novelty to encourage exploration.
"""

import numpy as np
import torch
import gym
import time
from collections import defaultdict, deque
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import hashlib
import pickle
import os
from typing import Dict, List, Tuple, Optional, Union

ENABLE_TIMING = False


def safe_rgb_frame(env):
    """Return an RGB frame (H,W,3) uint8 from env across Gym/Gymnasium versions.
    Order of attempts:
      A) Gymnasium / Gym >=0.26: env.render() with render_mode='rgb_array'
      B) Classic Gym <=0.21: env.render(mode='rgb_array')
      C) Classic Gym viewer buffer: env.viewer.get_array() after render()
      D) pygame display surface (if present)
    Raises RuntimeError if no RGB can be obtained.
    """
    # A) Gymnasium / Gym >=0.26: render_mode must be set at make()-time.
    # If it's set, env.render() should return a frame or None (some envs require a step first).
    try:
        # common attributes in Gymnasium
        if getattr(env, "render_mode", None) == "rgb_array" or \
           ("rgb_array" in getattr(getattr(env, "metadata", None), "get", lambda k, d=None: [])("render_modes", [])):
            frame = env.render()
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                return frame
            elif hasattr(frame, 'numpy') and hasattr(frame, 'ndim') and frame.ndim == 4:  # PyTorch tensor [B, H, W, C]
                # Convert PyTorch tensor to numpy array
                frame_np = frame.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to CPU numpy
                if frame_np.ndim == 3:  # Should be [H, W, C] now
                    return frame_np
    except Exception:
        pass

    # B) Classic Gym API (<=0.21)
    try:
        frame = env.render(mode="rgb_array")
        if isinstance(frame, np.ndarray) and frame.ndim == 3:
            return frame
        elif hasattr(frame, 'numpy') and hasattr(frame, 'ndim') and frame.ndim == 4:  # PyTorch tensor [B, H, W, C]
            # Convert PyTorch tensor to numpy array
            frame_np = frame.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to CPU numpy
            if frame_np.ndim == 3:  # Should be [H, W, C] now
                return frame_np
    except Exception:
        pass

    # C) Some classic envs draw into a pyglet viewer; grab its buffer
    try:
        # one render call to populate buffer if needed
        try:
            env.render(mode="rgb_array")
        except Exception:
            env.render()
        viewer = getattr(env, "viewer", None)
        get_array = getattr(viewer, "get_array", None) if viewer is not None else None
        if callable(get_array):
            buf = get_array()
            if isinstance(buf, np.ndarray) and buf.ndim == 3:
                return buf
    except Exception:
        pass

    # D) pygame surface fallback
    try:
        import pygame
        surf = pygame.display.get_surface()
        if surf is not None:
            # (W,H,3) -> (H,W,3)
            f = pygame.surfarray.array3d(surf)
            if isinstance(f, np.ndarray) and f.ndim == 3:
                return np.transpose(f, (1, 0, 2)).copy()
    except Exception:
        pass

    raise RuntimeError("safe_rgb_frame: RGB frame not available on this env.")

class _RandomEncoder(torch.nn.Module):
    def __init__(self, obs_shape, embed_dim=512):
        super().__init__()
        self.is_image = len(obs_shape) == 3  # (C,H,W) vs flat
        if self.is_image:
            C, H, W = obs_shape
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(C, 32, 3, 2, 1), torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, 3, 2, 1), torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, 2, 1), torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, 2, 1), torch.nn.ReLU(),
                torch.nn.Flatten(),
            )
            # defer Linear init to first forward (need flatten size)
            self.proj = None
            self.embed_dim = embed_dim
        else:
            D = obs_shape[0]
            self.encoder = torch.nn.Linear(D, embed_dim)

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x)).float()
        # ensure x is on the same device as the encoder
        dev = next(self.parameters()).device
        x = x.to(dev)
        if self.is_image:
            if x.ndim == 3:
                x = x.unsqueeze(0)
            h = self.encoder(x)
            if self.proj is None:
                self.proj = torch.nn.Linear(h.shape[-1], self.embed_dim).to(dev)
                for p in self.proj.parameters():
                    p.requires_grad = False
            y = self.proj(h)
        else:
            if x.ndim == 1:
                x = x.unsqueeze(0)
            y = self.encoder(x)
        return y.squeeze(0).cpu()


class _PretrainedEncoder(torch.nn.Module):
    """Wrapper for pretrained models."""
    
    def __init__(self, pretrained_model, device, embed_dim=None):
        super().__init__()
        self.pretrained_model = pretrained_model.to(device).eval()
        self.device = device
        self.embed_dim = embed_dim
        
        # Freeze pretrained model parameters
        for p in self.pretrained_model.parameters():
            p.requires_grad = False
        
    @torch.no_grad()
    def forward(self, x):
        """Forward pass through pretrained model.
        
        Args:
            x: Input tensor or numpy array. For images, expects (C,H,W) format.
               For bbox data, expects appropriate format for the pretrained model.
        
        Returns:
            Embedding tensor of shape [embed_dim]
        """
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x)).float()
        
        x = x.to(self.device)
        
        # Handle different input formats based on model type
        if hasattr(self.pretrained_model, 'infer'):
            # For XIRL models that have an infer method
            if x.ndim == 3:  # Image input (C,H,W)
                x = x.unsqueeze(0).unsqueeze(0)  # Add batch and time dimensions
            elif x.ndim == 4:  # Already has batch dimension
                x = x.unsqueeze(0)  # Add time dimension
            
            # Run inference
            output = self.pretrained_model.infer(x)
            
            # Extract embedding from output
            if hasattr(output, 'embs'):
                emb = output.embs
            elif hasattr(output, 'embedding'):
                emb = output.embedding
            else:
                emb = output
                
            # Convert to numpy and flatten
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            
            # Ensure we return a 1D array
            if emb.ndim > 1:
                emb = emb.flatten()
                
            return emb
        else:
            # For other model types, try direct forward pass
            if x.ndim == 3:  # Image input (C,H,W)
                x = x.unsqueeze(0)  # Add batch dimension
            
            output = self.pretrained_model(x)
            
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()
            
            if output.ndim > 1:
                output = output.flatten()
                
            return output

def _knn_bonus(batch_embed, memory_embeds, k=3):
    # batch_embed: [D] or [B,D]; memory_embeds: [N,D]
    be = np.atleast_2d(np.asarray(batch_embed, dtype=np.float32))
    me = np.asarray(memory_embeds, dtype=np.float32)
    if me.shape[0] == 0:
        # maximal novelty
        return np.ones(be.shape[0], dtype=np.float32)

    # compute squared distances; for small N a brute-force is fine
    # (B,N) = ||x||^2 + ||m||^2 - 2 x·m
    x2 = (be**2).sum(-1, keepdims=True)          # (B,1)
    m2 = (me**2).sum(-1, keepdims=True).T        # (1,N)
    xm = be @ me.T                                # (B,N)
    d2 = x2 + m2 - 2.0 * xm

    # get k-th smallest per row
    k_eff = min(k, d2.shape[1])
    kth = np.partition(d2, kth=k_eff-1, axis=1)[:, k_eff-1]  # (B,)
    r_int = np.log1p(kth)  # log(eps^2 + 1)
    return r_int.astype(np.float32)


class StateEntropyTracker:
    """Tracks state entropy and computes intrinsic rewards based on state novelty."""
    
    def __init__(
        self,
        state_dim: int,
        max_states: int = 10000,
        novelty_threshold: float = 0.1,
        intrinsic_weight: float = 0.1,
        method: str = "kmeans",
        n_clusters: int = 100,
        memory_decay: float = 0.99,
        device: str = "cpu",
        re3_cfg: dict = None,
        pretrained_model: torch.nn.Module = None
    ):
        """Initialize the state entropy tracker.
        
        Args:
            state_dim: Dimension of the state space
            max_states: Maximum number of states to store in memory
            novelty_threshold: Threshold for considering a state novel
            intrinsic_weight: Weight for the intrinsic reward component
            method: Method for clustering states ("kmeans", "hash", "distance", "re3", "pretrained")
            n_clusters: Number of clusters for k-means clustering
            memory_decay: Decay factor for old state memories
            device: Device to use for computations
            re3_cfg: Configuration for RE3 method
            pretrained_model: Pretrained model for pretrained method
        """
        self.state_dim = state_dim
        self.max_states = max_states
        self.novelty_threshold = novelty_threshold
        self.intrinsic_weight = intrinsic_weight
        self.method = method
        self.n_clusters = n_clusters
        self.memory_decay = memory_decay
        self.device = device
        
        # State storage
        self.states = deque(maxlen=max_states)
        self.state_counts = defaultdict(int)
        self.state_hashes = set()
        
        # Clustering components
        self.kmeans = None
        self.cluster_centers = None
        self.cluster_counts = defaultdict(int)
        
        # Statistics
        self.total_states = 0
        self.unique_states = 0

        #RE3 components
        self.re3_cfg = {
            "k": getattr(re3_cfg, "k", 3) if re3_cfg else 3,
            "embed_dim": getattr(re3_cfg, "embed_dim", 512) if re3_cfg else 512,
            "beta0": getattr(re3_cfg, "beta0", intrinsic_weight) if re3_cfg else intrinsic_weight,
            "beta_decay": getattr(re3_cfg, "beta_decay", 0.0) if re3_cfg else 0.0,
            "memory_subsample": getattr(re3_cfg, "memory_subsample", 4096) if re3_cfg else 4096,
            "use_images": getattr(re3_cfg, "use_images", True) if re3_cfg else True,
        }

        #Pretrained encoder components (similar to RE3 but with pretrained model)
        self.pretrained_cfg = {
            "k": getattr(re3_cfg, "k", 3) if re3_cfg else 3,
            "embed_dim": getattr(re3_cfg, "embed_dim", 512) if re3_cfg else 512,
            "beta0": getattr(re3_cfg, "beta0", intrinsic_weight) if re3_cfg else intrinsic_weight,
            "beta_decay": getattr(re3_cfg, "beta_decay", 0.0) if re3_cfg else 0.0,
            "memory_subsample": getattr(re3_cfg, "memory_subsample", 4096) if re3_cfg else 4096,
            "use_images": getattr(re3_cfg, "use_images", True) if re3_cfg else True,
        }
        self._step_t = 0
        self.embeds = deque(maxlen=max_states)
        self.encoder = None
        self._env_for_images = None  # set by wrapper if images are used
        
        # Pretrained model components
        self.pretrained_model = pretrained_model
        self.pretrained_encoder = None
        self._bbox_extractor = None  # Will be set by wrapper if needed (for ManiSkill)
        self._state_to_bboxes_func = None  # Will be set by wrapper if needed (for X-Magical)
        self._obj_detector = None   # Will be set by wrapper if needed (for X-Magical with images)
        if self.method == "pretrained" and pretrained_model is not None:
            self.pretrained_encoder = _PretrainedEncoder(pretrained_model, device)
    
    
    def set_image_env(self, env):
        """Set the environment for image-based observations."""
        self._env_for_images = env
        
    def set_bbox_extractor(self, bbox_extractor):
        """Set the bbox extractor for pretrained method."""
        self._bbox_extractor = bbox_extractor
        
    def set_obj_detector(self, obj_detector):
        """Set the object detector for pretrained method (X-Magical)."""
        self._obj_detector = obj_detector
        
    def set_state_to_bboxes_func(self, state_to_bboxes_func):
        """Set the state-to-bboxes function for pretrained method (X-Magical)."""
        self._state_to_bboxes_func = state_to_bboxes_func
        
    def _extract_bboxes_from_obs(self, obs):
        """Extract bboxes from observation using the bbox extractor. (ManiSkill PutShoesInBox)"""
        if self._bbox_extractor is None:
            raise RuntimeError("Bbox extractor not set")
        
        # Handle case where obs might be a flattened array instead of dict
        if not isinstance(obs, dict):
            raise ValueError(f"Expected observation to be a dict for bbox extraction, got {type(obs)}")
        
        # Add shoe poses to obs for bbox extraction if they're missing
        if 'extra' not in obs:
            print(f"❌ Extra not found in observation")
            obs['extra'] = {}
        
        # Get shoe poses directly from environment objects if not in obs
        if 'left_shoe_pose' not in obs['extra']:
            pose = self._env_for_images.unwrapped.left_shoe.pose.raw_pose
            obs['extra']['left_shoe_pose'] = pose
        
        if 'right_shoe_pose' not in obs['extra']:
            pose = self._env_for_images.unwrapped.right_shoe.pose.raw_pose
            obs['extra']['right_shoe_pose'] = pose
        
        # Get all bboxes from the environment
        all_bboxes = self._bbox_extractor.get_all_object_bboxes(obs, self._env_for_images)
        
        # Extract bboxes in the correct order with proper format
        ordered_bboxes = []
        object_order = [
            'gripper_left_finger',
            'gripper_right_finger', 
            'gripper_body',
            'left_shoe',
            'box_base',
            'box_lid',
            'right_shoe'
        ]
        object_ids = [31, 34, 35, 82, 87, 93, 98]
        
        for i, obj_name in enumerate(object_order):
            if obj_name in all_bboxes:
                bbox = all_bboxes[obj_name]
                # Format: [x_min, y_min, z_min, x_max, y_max, z_max, obj_id]
                min_corner = bbox[0]  # [x_min, y_min, z_min]
                max_corner = bbox[1]  # [x_max, y_max, z_max]
                obj_id = object_ids[i]
                
                # Convert to individual values
                x_min, y_min, z_min = min_corner[0].item(), min_corner[1].item(), min_corner[2].item()
                x_max, y_max, z_max = max_corner[0].item(), max_corner[1].item(), max_corner[2].item()
                
                bbox_7d = torch.tensor([x_min, y_min, z_min, x_max, y_max, z_max, obj_id], dtype=torch.float32, device=self.device)
                ordered_bboxes.append(bbox_7d)
            else:
                # Use zero bbox with correct obj_id if object not found
                obj_id = object_ids[i]
                bbox_7d = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, obj_id], dtype=torch.float32, device=self.device)
                ordered_bboxes.append(bbox_7d)
                print(f"❌ Object {obj_name} not found in observation")
        
        # Stack into tensor: [num_objects, 7]
        features = torch.stack(ordered_bboxes)
        
        # Reshape for model input: [batch_size, num_frames, num_classes, max_objects, num_features]
        # We need: [1, 1, 1, num_objects, 7] for the model
        model_input = features.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Add batch, time, and class dimensions
        
        return model_input
        
    def _extract_bboxes_from_state(self, obs):
        """Extract bboxes from state using state-to-bboxes function (X-Magical)."""
        if self._state_to_bboxes_func is None:
            raise RuntimeError("State-to-bboxes function not set")
        
        # Use the provided function to convert state to bboxes
        bboxes = self._state_to_bboxes_func(obs)
        
        # Convert to tensor format expected by GNNMultiTaskNet
        # Format: [x_min, y_min, x_max, y_max, obj_id]
        num_objects = bboxes.shape[1]  # bboxes shape: [1, num_objects, 4]
        features = []
        
        for i in range(num_objects):
            x_min, y_min, x_max, y_max = bboxes[0, i]
            obj_id = i + 1  # Object IDs start from 1
            bbox_5d = torch.tensor([x_min, y_min, x_max, y_max, obj_id], dtype=torch.float32, device=self.device)
            features.append(bbox_5d)
        
        # Stack into tensor: [num_objects, 5]
        features_tensor = torch.stack(features)
        
        # Reshape for model input: [batch_size, num_frames, num_classes, max_objects, num_features]
        # We need: [1, 1, 1, num_objects, 5] for the model
        model_input = features_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Add batch, time, and class dimensions
        
        return model_input
    
    def _create_xmagical_state_to_bboxes(self, obs):
        """Create bboxes from X-Magical state (similar to DistanceToGoalBboxReward.state_to_bboxes)."""
        # Parameters from DistanceToGoalBboxReward
        gripper_size = 60   
        target_1_size = 25
        target_2_size = 20
        goal_h = 0.65 * 384 / 2 
        goal_w = 0.55 * 384 / 2 
        goal_x = 0.1
        goal_y = 0.7
        dist_1_size = 25
        dist_2_size = 25
        dist_3_size = 25
        
        # Position adjustment offsets
        gripper_offset = np.array([0, 0])
        target_1_offset = np.array([0, 0])
        target_2_offset = np.array([0, 0])
        goal_offset = np.array([0, 0])
        
        def convert_coordinates(pos):
            pos[0] = (pos[0] + 1.1) / 2.2 * 384
            pos[1] = 384 - ((pos[1] + 1.1) / 2.2) * 384
            return pos
        
        state = np.copy(obs)
        bboxes = np.zeros((1, 4))
        
        # Extract positions from state
        gripper_pos = state[:2] 
        target_1_pos = state[2:4]  # STAR
        target_2_pos = state[4:6]  # RECT
        goal_pos = np.array([goal_x, goal_y])
        dist_1_pos = state[6:8]   # PENT (blue)
        dist_2_pos = state[8:10]  # CIRC (yellow)
        dist_3_pos = state[10:12] # PENT (yellow)

        gripper_pos = convert_coordinates(gripper_pos) + gripper_offset
        target_1_pos = convert_coordinates(target_1_pos) + target_1_offset
        target_2_pos = convert_coordinates(target_2_pos) + target_2_offset
        goal_pos = convert_coordinates(goal_pos) + goal_offset
        dist_1_pos = convert_coordinates(dist_1_pos)
        dist_2_pos = convert_coordinates(dist_2_pos)
        dist_3_pos = convert_coordinates(dist_3_pos)
        
        # Create bboxes
        bboxes[0] = goal_pos[0], goal_pos[1], goal_pos[0] + goal_w, goal_pos[1] + goal_h
        
        # Add gripper bbox
        bboxes = np.append(bboxes, [[gripper_pos[0] - gripper_size, gripper_pos[1] - gripper_size, 
                                   gripper_pos[0] + gripper_size, gripper_pos[1] + gripper_size]], axis=0)
        
        # Add target bboxes
        bboxes = np.append(bboxes, [[target_1_pos[0] - target_1_size, target_1_pos[1] - target_1_size, 
                                   target_1_pos[0] + target_1_size, target_1_pos[1] + target_1_size]], axis=0)
        bboxes = np.append(bboxes, [[target_2_pos[0] - target_2_size, target_2_pos[1] - target_2_size, 
                                   target_2_pos[0] + target_2_size, target_2_pos[1] + target_2_size]], axis=0)

        # Add distractor bboxes
        bboxes = np.append(bboxes, [[dist_1_pos[0] - dist_1_size, dist_1_pos[1] - dist_1_size, 
                                   dist_1_pos[0] + dist_1_size, dist_1_pos[1] + dist_1_size]], axis=0)
        bboxes = np.append(bboxes, [[dist_2_pos[0] - dist_2_size, dist_2_pos[1] - dist_2_size, 
                                   dist_2_pos[0] + dist_2_size, dist_2_pos[1] + dist_2_size]], axis=0)
        bboxes = np.append(bboxes, [[dist_3_pos[0] - dist_3_size, dist_3_pos[1] - dist_3_size, 
                                   dist_3_pos[0] + dist_3_size, dist_3_pos[1] + dist_3_size]], axis=0)
        
        bboxes = np.reshape(bboxes, (1, len(bboxes), 4)) 
        return bboxes
        
    def _extract_bboxes_from_image(self, obs):
        """Extract bboxes from image using object detector (X-Magical)."""
        if self._obj_detector is None:
            raise RuntimeError("Object detector not set")
        
        if self._env_for_images is None:
            raise RuntimeError("Environment not set for image rendering")
        
        # Render image from environment
        frame = safe_rgb_frame(self._env_for_images)
        
        # Perform object detection
        results = self._obj_detector.predict(frame, iou=0.4, conf=0.20, verbose=False)
        
        # Extract bounding boxes and labels
        bboxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        labels = results[0].boxes.cls.cpu().numpy()  # Class labels
        
        # Reorder bboxes (similar to DistanceToGoalBboxReward)
        ordered_bboxes, ordered_labels = self._reorder_bboxes_xmagical(bboxes, labels, frame)
        
        # Convert to tensor format expected by GNNMultiTaskNet
        # Format: [x_min, y_min, x_max, y_max, obj_id]
        num_objects = len(ordered_bboxes)
        features = []
        
        for i, (bbox, label) in enumerate(zip(ordered_bboxes, ordered_labels)):
            x_min, y_min, x_max, y_max = bbox
            obj_id = int(label)
            bbox_5d = torch.tensor([x_min, y_min, x_max, y_max, obj_id], dtype=torch.float32, device=self.device)
            features.append(bbox_5d)
        
        # Stack into tensor: [num_objects, 5]
        features_tensor = torch.stack(features)
        
        # Reshape for model input: [batch_size, num_frames, num_classes, max_objects, num_features]
        # We need: [1, 1, 1, num_objects, 5] for the model
        model_input = features_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Add batch, time, and class dimensions
        
        return model_input
    
    def _reorder_bboxes_xmagical(self, bboxes, labels, image):
        """Reorder bboxes for X-Magical MatchRegions task."""
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
            raise RuntimeError(f"Detection fails: detected {len(bboxes)} objects, expected {len(ordered_names.keys())}")

        # Process each bounding box and label
        for bbox, label in zip(bboxes, labels):
            class_name = self._get_class_name(label, names)

            # Reduce bbox size for specific classes
            if class_name in ['penthagon', 'circle']:
                bbox = self._reduce_bbox(bbox, reduction_percent=15, img_shape=image.shape)
            elif class_name == 'star':
                bbox = self._reduce_bbox(bbox, reduction_percent=20, img_shape=image.shape)
            
            if class_name == 'penthagon':
                # Determine the specific pentagon type by color
                center_y = int((bbox[3] + bbox[1]) / 2)
                center_x = int((bbox[2] + bbox[0]) / 2)
                color = image[center_y, center_x]
                
                if np.array_equal(color, [254, 213, 123]):  # penthagon2 color
                    class_name = 'penthagon2'
                elif np.array_equal(color, [135, 185, 211]):  # penthagon1 color
                    class_name = 'penthagon1'
                else:
                    raise RuntimeError("Detected wrongly a penthagon, skipping this bbox.")
                    continue  # Skip this bounding box and move to the next one
            
            # Append to ordered list
            ordered_bboxes.append((bbox, order[class_name]))

        # Sort the bounding boxes based on the desired order
        ordered_bboxes.sort(key=lambda x: x[1])

        # Separate the bboxes and labels
        sorted_bboxes = [item[0] for item in ordered_bboxes]
        sorted_labels = [item[1] for item in ordered_bboxes]

        return np.array(sorted_bboxes), np.array(sorted_labels)
    
    def _get_class_name(self, label, names):
        """Get the class name from the label."""
        return names[int(label)]

    def _reduce_bbox(self, bbox, reduction_percent, img_shape):
        """Reduce bbox size."""
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
        
    def _hash_state(self, state: np.ndarray) -> str:
        """Create a hash of the state for exact matching."""
        # Normalize state to reduce noise
        state_normalized = np.round(state, decimals=3)
        state_bytes = pickle.dumps(state_normalized)
        return hashlib.md5(state_bytes).hexdigest()
    
    def _compute_state_similarity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Compute similarity between two states using Euclidean distance."""
        return np.linalg.norm(state1 - state2)
    
    def _find_nearest_state(self, state: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """Find the nearest state in memory and return distance."""
        if len(self.states) == 0:
            return float('inf'), None
        
        min_distance = float('inf')
        nearest_state = None
        
        for stored_state in self.states:
            distance = self._compute_state_similarity(state, stored_state)
            if distance < min_distance:
                min_distance = distance
                nearest_state = stored_state
        
        return min_distance, nearest_state
    
    def _update_clusters(self):
        """Update clustering if using k-means method."""
        if self.method == "kmeans" and len(self.states) >= self.n_clusters:
            states_array = np.array(list(self.states))
            
            # Initialize or update k-means
            if self.kmeans is None:
                self.kmeans = KMeans(n_clusters=min(self.n_clusters, len(self.states)), random_state=42, n_init="auto")
                self.kmeans.fit(states_array)
            else:
                # Retrain with new data
                self.kmeans.fit(states_array)
            
            self.cluster_centers = self.kmeans.cluster_centers_
            self.cluster_counts.clear()
            
            # Count states in each cluster
            if len(states_array) > 0:
                cluster_labels = self.kmeans.predict(states_array)
                for label in cluster_labels:
                    self.cluster_counts[label] += 1
    
    def _compute_cluster_novelty(self, state: np.ndarray) -> float:
        """Compute novelty based on cluster analysis."""
        if self.cluster_centers is None or len(self.cluster_centers) == 0:
            return 1.0  # Maximum novelty if no clusters exist
        
        # Find nearest cluster center
        distances = euclidean_distances([state], self.cluster_centers)[0]
        nearest_cluster = np.argmin(distances)
        nearest_distance = distances[nearest_cluster]
        
        # Novelty based on distance to nearest cluster and cluster size
        cluster_size = self.cluster_counts.get(nearest_cluster, 1)
        
        # For small datasets, use distance-based novelty more heavily
        if len(self.states) < self.n_clusters:
            # Use exponential decay based on distance
            novelty = np.exp(-nearest_distance / self.novelty_threshold)
        else:
            # Use both distance and cluster size
            distance_factor = np.exp(-nearest_distance / self.novelty_threshold)
            size_factor = 1.0 / (1.0 + cluster_size)
            novelty = distance_factor * size_factor
        
        return novelty
    
    def _compute_hash_novelty(self, state: np.ndarray) -> float:
        """Compute novelty based on exact state matching."""
        state_hash = self._hash_state(state)
        if state_hash in self.state_hashes:
            return 0.0  # Not novel
        else:
            return 1.0  # Novel
    
    def _compute_distance_novelty(self, state: np.ndarray) -> float:
        """Compute novelty based on distance to nearest state."""
        min_distance, nearest_state = self._find_nearest_state(state)
        
        if min_distance == float('inf'):
            return 1.0  # Maximum novelty if no states in memory
        
        # Check for exact duplicates
        if min_distance == 0.0:
            return 0.0  # No novelty for exact duplicates
        
        # Convert distance to novelty score (inverse relationship)
        # Use a more robust scaling that doesn't decay too quickly
        # Scale the distance by the novelty threshold for better control
        scaled_distance = min_distance / self.novelty_threshold
        novelty = 1.0 / (1.0 + scaled_distance)
        return novelty

    def _ensure_encoder(self, obs):
        if self.encoder is None and self.method == "re3":
            # infer shape: image-like vs flat
            obs_arr = np.asarray(obs)
            if obs_arr.ndim == 3:  # (H,W,C) or (C,H,W)
                # ensure channel-first
                if obs_arr.shape[-1] in (1,3):
                    obs_arr = obs_arr.transpose(2,0,1)
            shape = obs_arr.shape
            if obs_arr.ndim == 1:
                shape = (shape[0],)
            elif obs_arr.ndim == 3:
                shape = (shape[0], shape[1], shape[2])  # assume channel-first
            self.encoder = _RandomEncoder(shape, embed_dim=self.re3_cfg["embed_dim"]).to(torch.device(self.device))

    
    def add_state(self, state: np.ndarray) -> float:
        """Add a state to the tracker and return intrinsic reward.
        
        Args:
            state: State to add
            
        Returns:
            Intrinsic reward based on state novelty
        """
        if self.method == "re3":
            # 1) choose input to the encoder
            if self.re3_cfg["use_images"]:
                assert self._env_for_images is not None, "Call set_image_env(env) for RE3 with images."
                # Note: this works for Gym ≤0.21. For Gymnasium (or Gym ≥0.26), you must create the env with render_mode="rgb_array" and call env.render() without args.
                
                try:
                    frame = safe_rgb_frame(self._env_for_images)
                    img = frame.astype(np.float32) / 255.0
                    img = img.transpose(2, 0, 1)  # C,H,W
                    enc_input = img
                except RuntimeError:
                    # graceful fallback to raw state
                    if not self.re3_cfg.get("_warned_no_rgb", False):
                        print("[RE3] Warning: RGB frames unavailable; falling back to raw-state embeddings.")
                        self.re3_cfg["_warned_no_rgb"] = True
                    enc_input = np.asarray(state, dtype=np.float32)

            else:
                # fall back to raw vector state
                enc_input = np.asarray(state, dtype=np.float32)

            # 2) lazy init encoder with proper shape
            self._ensure_encoder(enc_input)

            # 3) embedding
            with torch.no_grad():
                y = self.encoder(enc_input).numpy()  # [D]

            # 4) kNN log distance bonus
            if len(self.embeds) == 0:
                r_int = 1.0
            else:
                cand = np.array(self.embeds, dtype=np.float32)
                if cand.shape[0] > self.re3_cfg["memory_subsample"]:
                    idx = np.random.choice(cand.shape[0], self.re3_cfg["memory_subsample"], replace=False)
                    cand = cand[idx]
                r_int = float(_knn_bonus(y, cand, k=self.re3_cfg["k"])[0])

            # 5) beta schedule
            beta = float(self.re3_cfg["beta0"] * (1.0 - self.re3_cfg["beta_decay"])**self._step_t)
            self._step_t += 1

            # 6) push to memory & return beta-weighted bonus as the "raw" tracker output
            self.embeds.append(y)

            # Hash the embedding (rounded for stability) to count uniques
            emb_hash = hashlib.md5(np.round(y, 4).tobytes()).hexdigest()
            if emb_hash not in self.state_hashes:
                self.state_hashes.add(emb_hash)
                self.unique_states += 1
            self.state_counts[emb_hash] += 1
            self.total_states += 1

            return beta * r_int

        elif self.method == "pretrained":
            # Use pretrained encoder for intrinsic reward with bbox extraction
            assert self.pretrained_encoder is not None, "Pretrained encoder not initialized for pretrained method"
            
            # OPTIMIZATION: Check if embedding is already cached (from ManiSkillBBoxRewardWrapper)
            # to avoid duplicate GNN inference. This saves ~0.0019s per step!
            if isinstance(state, dict) and "cached_embedding" in state:
                # Use cached embedding from ManiSkillBBoxRewardWrapper (saves GNN inference!)
                y = state["cached_embedding"]
                # Convert to numpy if needed (for kNN computation)
                if isinstance(y, torch.Tensor):
                    y = y.detach().cpu().numpy()
                else:
                    y = np.asarray(y, dtype=np.float32)
                # Handle batch/time dimensions - flatten if needed
                if y.ndim > 1:
                    y = y.reshape(-1, y.shape[-1])[0]  # Flatten and take first
            else:
                # 1) Extract bboxes from observation (similar to DistanceToGoalBboxReward)
                # OPTIMIZATION: Check if bboxes are already cached (from ManiSkillBBoxRewardWrapper)
                # to avoid duplicate extraction. The cached bboxes are passed via state dict.
                enc_input = None
                if isinstance(state, dict) and "cached_bboxes_tensor" in state:
                    # Use cached bboxes from ManiSkillBBoxRewardWrapper (saves ~0.002s per step!)
                    enc_input = state["cached_bboxes_tensor"]
                elif hasattr(self, '_bbox_extractor') and self._bbox_extractor is not None:
                    # Use bbox extractor if available (for ManiSkill)
                    # The state parameter should be the original dict observation (from info["original_obs"])
                    # when ManiSkillPutShoesInBoxWrapper converts dict obs to numpy array
                    obs_for_bbox = state if isinstance(state, dict) else state
                    bboxes_tensor = self._extract_bboxes_from_obs(obs_for_bbox)
                    enc_input = bboxes_tensor
                elif hasattr(self, '_state_to_bboxes_func') and self._state_to_bboxes_func is not None:
                    # Use state-to-bboxes function if available (for X-Magical)
                    obs_for_bbox = state
                    bboxes_tensor = self._extract_bboxes_from_state(obs_for_bbox)
                    enc_input = bboxes_tensor
                elif hasattr(self, '_obj_detector') and self._obj_detector is not None:
                    # Use object detector if available (for X-Magical with images)
                    obs_for_bbox = state
                    bboxes_tensor = self._extract_bboxes_from_image(obs_for_bbox)
                    enc_input = bboxes_tensor
                elif self.pretrained_cfg["use_images"]:
                    # Use image input if available
                    # First check if an image was extracted from observation dict
                    if hasattr(self, '_extracted_image') and self._extracted_image is not None:
                        # Use the extracted image (already in C,H,W format with values in [0,1])
                        enc_input = self._extracted_image
                    else:
                        # Fallback to env.render()
                        assert self._env_for_images is not None, "Call set_image_env(env) for pretrained with images."
                        frame = safe_rgb_frame(self._env_for_images)
                        img = frame.astype(np.float32) / 255.0
                        img = img.transpose(2, 0, 1)  # C,H,W
                        enc_input = img
                else:
                    # Use raw vector state
                    enc_input = np.asarray(state, dtype=np.float32)
                
                # 2) embedding using pretrained encoder
                with torch.no_grad():
                    y = self.pretrained_encoder(enc_input)  # [D]
                    # Convert to numpy for kNN computation
                    if isinstance(y, torch.Tensor):
                        y = y.detach().cpu().numpy()
                    else:
                        y = np.asarray(y, dtype=np.float32)
                    # Handle batch/time dimensions - flatten if needed
                    if y.ndim > 1:
                        y = y.reshape(-1, y.shape[-1])[0]  # Flatten and take first

            # 3) kNN log distance bonus
            if len(self.embeds) == 0:
                r_int = 1.0
            else:
                cand = np.array(self.embeds, dtype=np.float32)
                if cand.shape[0] > self.pretrained_cfg["memory_subsample"]:
                    idx = np.random.choice(cand.shape[0], self.pretrained_cfg["memory_subsample"], replace=False)
                    cand = cand[idx]
                r_int = float(_knn_bonus(y, cand, k=self.pretrained_cfg["k"])[0])

            # 4) beta schedule
            beta = float(self.pretrained_cfg["beta0"] * (1.0 - self.pretrained_cfg["beta_decay"])**self._step_t)
            self._step_t += 1

            # 5) push to memory & return beta-weighted bonus as the "raw" tracker output
            self.embeds.append(y)

            # Hash the embedding (rounded for stability) to count uniques
            emb_hash = hashlib.md5(np.round(y, 4).tobytes()).hexdigest()
            if emb_hash not in self.state_hashes:
                self.state_hashes.add(emb_hash)
                self.unique_states += 1
            self.state_counts[emb_hash] += 1
            self.total_states += 1

            return beta * r_int

        else:
            state = np.array(state, dtype=np.float32).flatten()
            
            # Compute novelty based on method
            if self.method == "kmeans":
                novelty = self._compute_cluster_novelty(state)
            elif self.method == "hash":
                novelty = self._compute_hash_novelty(state)
            elif self.method == "distance":
                novelty = self._compute_distance_novelty(state)
            else:
                raise ValueError(f"Unknown cluster method: {self.method}")
            
            # Add state to memory
            state_hash = self._hash_state(state)
            if state_hash not in self.state_hashes:
                self.states.append(state)
                self.state_hashes.add(state_hash)
                self.unique_states += 1
                
                # Update clustering if needed
                if self.method == "kmeans":
                    self._update_clusters()
            
            self.state_counts[state_hash] += 1
            self.total_states += 1
            
            # Compute intrinsic reward
            intrinsic_reward = float(novelty)  # raw signal; weight is applied in the wrapper
            
            return intrinsic_reward
    
    def get_entropy_stats(self) -> Dict[str, float]:
        """Get statistics about state entropy."""
        if len(self.states) == 0:
            return {
                "total_states": 0,
                "unique_states": 0,
                "entropy": 0.0,
                "novelty_rate": 0.0
            }
        
        # Compute entropy based on state distribution
        total_count = sum(self.state_counts.values())
        if total_count == 0:
            entropy = 0.0
        else:
            probs = [count / total_count for count in self.state_counts.values()]
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        
        novelty_rate = self.unique_states / max(self.total_states, 1)
        
        return {
            "total_states": self.total_states,
            "unique_states": self.unique_states,
            "entropy": entropy,
            "novelty_rate": novelty_rate,
            "memory_size": len(self.states)
        }
    
    def save(self, filepath: str):
        """Save the tracker state to a file."""
        state = {
            "states": list(self.states),
            "state_counts": dict(self.state_counts),
            "state_hashes": self.state_hashes,
            "cluster_centers": self.cluster_centers,
            "cluster_counts": dict(self.cluster_counts),
            "total_states": self.total_states,
            "unique_states": self.unique_states,
            "kmeans": self.kmeans,
            "re3_embeds": np.array(self.embeds, dtype=np.float32),
            "re3_step_t": self._step_t,
            "re3_cfg": self.re3_cfg,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: str):
        """Load the tracker state from a file."""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.states = deque(state["states"], maxlen=self.max_states)
        self.state_counts = defaultdict(int, state["state_counts"])
        self.state_hashes = state["state_hashes"]
        self.cluster_centers = state["cluster_centers"]
        self.cluster_counts = defaultdict(int, state["cluster_counts"])
        self.total_states = state["total_states"]
        self.unique_states = state["unique_states"]
        self.kmeans = state["kmeans"]
        re3_embeds = state.get("re3_embeds", None)
        if re3_embeds is not None:
            self.embeds = deque(re3_embeds, maxlen=self.max_states)
        self._step_t = state.get("re3_step_t", 0)
        self.re3_cfg.update(state.get("re3_cfg", {}))


class IntrinsicRewardWrapper(gym.Wrapper):
    """Wrapper that adds intrinsic rewards based on state entropy.
    
    This wrapper combines intrinsic rewards (exploration bonus) with extrinsic rewards
    from the wrapped environment. The extrinsic reward is whatever reward signal comes
    from the wrapped environment - this could be:
    - Environmental reward (original task reward) when using env_reward=True
    - Learned reward (from pretrained model) when using learned rewards
    
    When using learned rewards, this wrapper explicitly checks for info['learned_reward']
    (set by ManiSkillBBoxRewardWrapper) and uses that as the extrinsic reward signal,
    ensuring the correct learned reward is used even if intermediate wrappers modify
    the reward value. The original environmental reward is preserved in info['env_reward']
    for tracking purposes.
    """
    
    def __init__(
        self,
        env,
        state_entropy_tracker: StateEntropyTracker,
        intrinsic_weight: float = 0.1,
        extrinsic_weight: float = 1.0
    ):
        """Initialize the intrinsic reward wrapper.
        
        Args:
            env: Environment to wrap
            state_entropy_tracker: Tracker for computing intrinsic rewards
            intrinsic_weight: Weight for intrinsic reward component
            extrinsic_weight: Weight for extrinsic reward component
        """
        super().__init__(env)
        self.state_entropy_tracker = state_entropy_tracker
        self.intrinsic_weight = intrinsic_weight
        self.extrinsic_weight = extrinsic_weight
        
        # Statistics
        self.episode_intrinsic_rewards = []
        self.episode_extrinsic_rewards = []

        # Timing tracking for intrinsic reward computation
        self._intrinsic_reward_times = []
        self._step_count = 0
        
    def reset(self):
        """Reset the environment and clear episode statistics."""
        result = self.env.reset()
        
        # Handle different return formats (Gymnasium vs Gym)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}
        
        self.episode_intrinsic_rewards = []
        self.episode_extrinsic_rewards = []
        return obs, info
    
    def step(self, action):
        """Take a step in the environment and compute combined reward."""
        result = self.env.step(action)
        
        # Handle different return formats (Gymnasium vs Gym)
        if len(result) == 5:
            obs, extrinsic_reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4:
            obs, extrinsic_reward, done, info = result
        else:
            raise ValueError(f"Unexpected number of return values: {len(result)}")
        
        # IMPORTANT: When using learned rewards, ManiSkillBBoxRewardWrapper returns the learned_reward
        # as the main reward value and stores env_reward in info['env_reward'].
        # ManiSkillPutShoesInBoxWrapper preserves these keys in info.
        # If info['learned_reward'] exists, use it as the authoritative source.
        # Otherwise, use the reward value (which should be the learned reward from ManiSkillBBoxRewardWrapper).
        if "learned_reward" in info:
            extrinsic_reward = info["learned_reward"]
        
        # Use original_obs from info if available (for bbox extraction with pretrained method)
        # Also pass cached bboxes and embedding if available to avoid duplicate computation
        if self.state_entropy_tracker.method == "pretrained":
            obs_for_intrinsic = info.get("original_obs", obs)
            # Add cached bboxes and embedding to the state dict if available
            if isinstance(obs_for_intrinsic, dict):
                obs_for_intrinsic = obs_for_intrinsic.copy()
                if "cached_bboxes_tensor" in info:
                    obs_for_intrinsic["cached_bboxes_tensor"] = info["cached_bboxes_tensor"]
                if "cached_embedding" in info:
                    # Prefer cached embedding over cached bboxes (saves GNN inference!)
                    obs_for_intrinsic["cached_embedding"] = info["cached_embedding"]
        else:
            obs_for_intrinsic = obs
        
        # Time intrinsic reward computation
        t0 = time.time()
        intrinsic_reward = self.state_entropy_tracker.add_state(obs_for_intrinsic)
        self._intrinsic_reward_times.append(time.time() - t0)
        
        # Combine rewards
        total_reward = (self.extrinsic_weight * extrinsic_reward + 
                       self.intrinsic_weight * intrinsic_reward)
        
        # Store statistics
        self.episode_intrinsic_rewards.append(intrinsic_reward)
        self.episode_extrinsic_rewards.append(extrinsic_reward)
        
        # Add statistics to info
        info["intrinsic_reward"] = intrinsic_reward
        info["extrinsic_reward"] = extrinsic_reward  # This is the learned reward when using learned rewards
        info["total_reward"] = total_reward
        
        # Preserve env_reward and learned_reward if they exist (from ManiSkillBBoxRewardWrapper)
        # This allows tracking the original task reward separately from the learned reward
        if "env_reward" in info or "learned_reward" in info:
            # env_reward and learned_reward are already in info from ManiSkillBBoxRewardWrapper
            # Ensure learned_reward is set to extrinsic_reward for consistency
            if "learned_reward" not in info:
                print("WARNING: learned_reward not in info, setting it to extrinsic_reward")
                info["learned_reward"] = extrinsic_reward
        
        # Print timing every 100 steps (check ENABLE_TIMING from maniskill_wrappers if available)
        self._step_count += 1
        
        if ENABLE_TIMING and self._step_count % 100 == 0 and self._intrinsic_reward_times:
            print(f"\n[IntrinsicRewardWrapper Step {self._step_count}] Timing (last 100 steps):")
            print(f"  Intrinsic reward computation: {np.mean(self._intrinsic_reward_times[-100:]):.4f}s (mean)")
            # Keep only last 200 for memory
            if len(self._intrinsic_reward_times) > 200:
                self._intrinsic_reward_times[:] = self._intrinsic_reward_times[-200:]
        
        if done:
            info["episode_intrinsic_reward"] = sum(self.episode_intrinsic_rewards)
            info["episode_extrinsic_reward"] = sum(self.episode_extrinsic_rewards)
            info["episode_total_reward"] = sum(self.episode_intrinsic_rewards) + sum(self.episode_extrinsic_rewards)
        
        return obs, total_reward, done, info
    
    def render(self, mode="rgb_array"):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)
        else:
            # Return a blank image if the underlying env doesn't support rendering
            import numpy as np
            return np.zeros((64, 64, 3), dtype=np.uint8) 