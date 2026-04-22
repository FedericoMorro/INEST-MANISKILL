"""
Modified from https://github.com/haosulab/ManiSkill/blob/8c8d33916e07984057cf3d30e5cb9d5c26377b03/mani_skill/envs/tasks/tabletop/stack_pyramid.py
"""


from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


HORIZON = 100
DEFAULT_RANDOMIZE_CUBES = True
MAX_SUBGOAL = 4

N_STEP_DENSE_REWARD = 4
SUCCESS_REWARD = 2.0


@register_env("StackPyramid-v1custom", max_episode_steps=HORIZON)
class StackPyramidEnv(BaseEnv):
    """
    **Task Description:**
    - The goal is to pick up a red cube, place it next to the green cube, and stack the blue cube on top of the red and green cube without it falling off.

    **Randomizations:**
    - all cubes have their z-axis rotation randomized
    - all cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    **Success Conditions:**
    - the blue cube is static
    - the blue cube is on top of both the red and green cube (to within half of the cube size)
    - none of the red, green, blue cubes are grasped by the robot (robot must let go of the cubes)

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackPyramid-v1_rt.mp4"

    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["none", "sparse", "dense", "normalized_dense"]

    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        seed=None,
        env_reward_type="sparse",
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        env_randomization=DEFAULT_RANDOMIZE_CUBES,
        enforce_full_episodes=True,
        **kwargs
    ):
        print("Initializing custom StackPyramid environment")
        if seed is None:
            seed = np.random.randint(0, 10000)
            print(f"No seed provided for environment initialization. Results may not be reproducible. Seed selected randomly: {seed}")
        self.seed = seed
        
        self.randomize_cubes = env_randomization
        self.enforce_full_episodes = enforce_full_episodes
        
        kwargs["reward_mode"] = env_reward_type
        self.max_subgoal = MAX_SUBGOAL
        self.curr_subgoal = None  # will be initialized to 0 at the beginning of the episode in reset()

        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        
    

    @property
    def _default_sensor_configs(self):
        #pose = sapien_utils.look_at(eye=[0.3, 0, 0.4], target=[-0.05, 0, 0.1])
        #return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("base_camera", pose, 128, 128, 1, 0.01, 100)

    @property
    def _default_human_render_camera_configs(self):
        #pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        #return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 128, 128, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, 0.2]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.2]),
        )
        self.cubeC = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 0, 1, 1],
            name="cubeC",
            initial_pose=sapien.Pose(p=[-1, 0, 0.2]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if self.randomize_cubes:
                # Randomized positions and rotations
                xyz = torch.zeros((b, 3), device=self.device)
                xyz[:, 2] = 0.02
                xy = xyz[:, :2]
                region = [[-0.1, -0.2], [0.1, 0.2]]
                sampler = randomization.UniformPlacementSampler(
                    bounds=region, batch_size=b, device=self.device
                )
                radius = torch.linalg.norm(torch.tensor([0.02, 0.02]))
                cubeA_xy = xy + sampler.sample(radius, 100)
                cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)
                cubeC_xy = xy + sampler.sample(radius, 100, verbose=False)

                # Cube A
                xyz[:, :2] = cubeA_xy
                qs = randomization.random_quaternions(
                    b,
                    lock_x=True,
                    lock_y=True,
                    lock_z=False,
                )
                self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

                # Cube B
                xyz[:, :2] = cubeB_xy
                qs = randomization.random_quaternions(
                    b,
                    lock_x=True,
                    lock_y=True,
                    lock_z=False,
                )
                self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

                # Cube C
                xyz[:, :2] = cubeC_xy
                qs = randomization.random_quaternions(
                    b,
                    lock_x=True,
                    lock_y=True,
                    lock_z=False,
                )
                self.cubeC.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            else:
                # Fixed positions with small randomization
                # Small position noise
                pos_noise_A = torch.randn((b, 3), device=self.device) * 0.005
                pos_noise_A[:, 2] = 0  # no z noise
                pos_noise_B = torch.randn((b, 3), device=self.device) * 0.005
                pos_noise_B[:, 2] = 0
                pos_noise_C = torch.randn((b, 3), device=self.device) * 0.005
                pos_noise_C[:, 2] = 0
                
                # Small rotation noise (only z-axis)
                qs_A = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi/24, np.pi/24), device=self.device)
                qs_B = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi/24, np.pi/24), device=self.device)
                qs_C = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi/24, np.pi/24), device=self.device)
                
                # Cube A
                cubeA_pos = torch.tensor([[-0.08, 0.08, 0.02]], device=self.device).repeat(b, 1) + pos_noise_A
                self.cubeA.set_pose(Pose.create_from_pq(p=cubeA_pos, q=qs_A))
                
                # Cube B
                cubeB_pos = torch.tensor([[0.08, 0.00, 0.02]], device=self.device).repeat(b, 1) + pos_noise_B
                self.cubeB.set_pose(Pose.create_from_pq(p=cubeB_pos, q=qs_B))
                
                # Cube C
                cubeC_pos = torch.tensor([[-0.08, -0.08, 0.02]], device=self.device).repeat(b, 1) + pos_noise_C
                self.cubeC.set_pose(Pose.create_from_pq(p=cubeC_pos, q=qs_C))

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        pos_C = self.cubeC.pose.p

        offset_AB = pos_A - pos_B
        offset_BC = pos_B - pos_C
        offset_AC = pos_A - pos_C

        success_A_B = self._evaluate_cube_distance(
            offset_AB, self.cubeA, self.cubeB, "next_to"
        )
        success_C_B = self._evaluate_cube_distance(offset_BC, self.cubeC, self.cubeB, "top")
        success_C_A = self._evaluate_cube_distance(offset_AC, self.cubeC, self.cubeA, "top")
        success = torch.logical_and(
            success_A_B, torch.logical_and(success_C_B, success_C_A)
        )
        return {
            "success": success,
        }

    def _evaluate_cube_distance(self, offset, cube_1, cube_2, top_or_next):
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(2 * self.cube_half_size[:2]) + 0.005   # 0.0616
        )
        z_flag = torch.abs(offset[..., 2]) > 0.02
        if top_or_next == "top":
            is_cube1_on_cube2 = torch.logical_and(xy_flag, z_flag)
        elif top_or_next == "next_to":
            is_cube1_on_cube2 = xy_flag
        else:
            return NotImplementedError(
                f"Expect top_or_next to be either 'top' or 'next_to', got {top_or_next}"
            )

        is_cube1_static = cube_1.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cube1_grasped = self.agent.is_grasping(cube_1)

        success = is_cube1_on_cube2 & is_cube1_static & (~is_cube1_grasped)
        return success.bool()

    def _success_per_cubes(self, cubes_type: str):
        if cubes_type == "A_B":
            return self._evaluate_cube_distance(
                self.cubeA.pose.p - self.cubeB.pose.p, self.cubeA, self.cubeB, "next_to"
            )
        elif cubes_type == "C_B":
            return self._evaluate_cube_distance(
                self.cubeC.pose.p - self.cubeB.pose.p, self.cubeC, self.cubeB, "top"
            )
        elif cubes_type == "C_A":
            return self._evaluate_cube_distance(
                self.cubeC.pose.p - self.cubeA.pose.p, self.cubeC, self.cubeA, "top"
            )

    def _update_subgoal_success(self):

        def _is_equal_tensor(a, b, eps=1e-3):
            return torch.norm(a - b) < eps

        curr_cubeA_pos = self.cubeA.pose.p
        curr_cubeC_pos = self.cubeC.pose.p

        # detect subgoal transitions based on cube position changes and success conditions
        if self.curr_subgoal == 0 and not _is_equal_tensor(curr_cubeA_pos, self.prev_cubeA_pos):    # pose changed from init
            self.curr_subgoal = 1

        elif self.curr_subgoal == 1 and (_is_equal_tensor(curr_cubeA_pos, self.prev_cubeA_pos)      # cubeA stopped moving after moving
                and self._success_per_cubes("A_B")):            # and success A_B (A next to B)
            self.curr_subgoal = 2

        elif self.curr_subgoal == 2 and self.agent.is_grasping(self.cubeC):        # cubeC is grasped by the robot
            self.curr_subgoal = 3
            self.prev_cubeC_pos = self.cubeC.pose.p   # initialize prev_cubeC_pos to check when it stops moving in the next step

        elif self.curr_subgoal == 3 and (_is_equal_tensor(curr_cubeC_pos, self.prev_cubeC_pos)     # cubeC stopped moving after moving
                and self._success_per_cubes("C_B") and self._success_per_cubes("C_A")):        #  and success C_B and success C_A (C on top of A and B)
            self.curr_subgoal = 4

        # update prev positions for next step, if the current subgoal is the one that requires movement, to check when it stops moving in the next step
        if self.curr_subgoal == 1:    # update prev_cubeA_pos to check when it stops moving
            self.prev_cubeA_pos = curr_cubeA_pos
        elif self.curr_subgoal == 3:  # update prev_cubeC_pos to check when it stops moving
            self.prev_cubeC_pos = curr_cubeC_pos
    
    def get_current_subgoal(self):
        return self.curr_subgoal


    def set_seed(self, seed):
        self.seed = seed

    def reset(self, **kwargs):
        self.step_count = 0

        #! equal seed at reset => same env reset
        # seed is used only one (common practice) to ensure reproducibility of environment initialization
        if self.seed is not None:
            # remove seed from kwargs to avoid multiple assignments error in BaseEnv reset
            if "seed" in kwargs:
                del kwargs["seed"]
            obs, info = super().reset(seed=self.seed, **kwargs)
            self._reset_seed()
        else:
            obs, info = super().reset(**kwargs)

        # init subgoal tracking at the beginning of the episode
        self.prev_cubeA_pos = self.cubeA.pose.p
        self.curr_subgoal = 0

        return obs, info
    
    def _reset_seed(self):
        self.seed = None


    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.step_count += 1

        # update subgoal success and add it to info
        self._update_subgoal_success()
        info["subgoal"] = self.curr_subgoal

        if self.enforce_full_episodes and self.step_count < HORIZON:
            terminated = torch.tensor([False], device=self.device)
            
        return obs, reward, terminated, truncated, info


    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeC_pos=self.cubeC.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                cubeB_to_cubeC_pos=self.cubeC.pose.p - self.cubeB.pose.p,
                cubeA_to_cubeC_pos=self.cubeC.pose.p - self.cubeA.pose.p,
                is_cubeA_grasped=self.agent.is_grasping(self.cubeA),
                is_cubeB_grasped=self.agent.is_grasping(self.cubeB),
                is_cubeC_grasped=self.agent.is_grasping(self.cubeC),
                is_cubeC_static=self.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            )
        return obs

    def _distance_to_reward(self, d,
            alpha=0.006, b=500.0, beta=0.001,
        ):
        norm_c = - b * torch.log(torch.tensor(beta, device=d.device))   # f(d=0), so that reward is (-inf,1] normalized
        rew = - alpha * d**2 - b * torch.log(d**2 + beta)
        return rew / norm_c
        #return 1.0 - d / 0.3   # linear version (need assumption 0.3 is max distance)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # get cube positions  
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        pos_C = self.cubeC.pose.p

        # get eef position
        eef_pos = self.agent.tcp.pose.p
        
        # compute offsets like in evaluate function
        offset_AB = pos_A - pos_B
        offset_BC = pos_B - pos_C
        offset_AC = pos_A - pos_C

        # perform evaluate to compute reward components and success bonuses
        success_A_B = self._evaluate_cube_distance(offset_AB, self.cubeA, self.cubeB, "next_to")
        success_C_B = self._evaluate_cube_distance(offset_BC, self.cubeC, self.cubeB, "top")
        success_C_A = self._evaluate_cube_distance(offset_AC, self.cubeC, self.cubeA, "top")
        success = torch.logical_and(success_A_B, torch.logical_and(success_C_B, success_C_A))
        
        if N_STEP_DENSE_REWARD == 3.0:
            # identify current step
            if success:
                current_step = 2
            elif success_A_B:
                current_step = 1
            else:
                current_step = 0

            # compute reward based on the step
            if current_step == 0:
                # reward based on distance of eef to cube A, and distance of cube A to cube B
                distance_eef_A = torch.linalg.norm(eef_pos - pos_A)
                distance_AB = torch.linalg.norm(offset_AB)
                
                # in [-1,0], equal component contribution
                reward = -1.0 + (
                    self._distance_to_reward(distance_eef_A) +
                    self._distance_to_reward(distance_AB)
                ) / 1.5
                reward -= 0.7

            elif current_step == 1:
                # reward based on distance of eef to cube C, and distance of cube C to cube A and B,
                #   with bonus reward for grasping and lifting cube C
                distance_eef_C = torch.linalg.norm(eef_pos - pos_C)
                distance_AC = torch.linalg.norm(offset_AC)
                distance_BC = torch.linalg.norm(offset_BC)
                z_flag = torch.abs(offset_BC[..., 2]) > 0.02
                grasp_flag = self.agent.is_grasping(self.cubeC) or z_flag
                # give reward even if not grasping but above threshold to avoid decrease when releasing cube C after stacking

                # in [0,1], equal contribution eef-C, AB-to-C, grasp+height bonus
                reward = (
                    self._distance_to_reward(distance_eef_C) * 2.0 +
                    (self._distance_to_reward(distance_AC) + self._distance_to_reward(distance_BC)) +
                    (z_flag.float() + grasp_flag.float()) / 2.0
                ) / 3.0
                reward -= 0.7

            else:
                reward = SUCCESS_REWARD
                
                
        elif N_STEP_DENSE_REWARD == 4.0:
            z_flag = torch.abs(offset_BC[..., 2]) > 0.02
            
            # identify current step
            if success:
                current_step = 3
            elif z_flag and success_A_B:
                current_step = 2
            elif success_A_B:
                current_step = 1
            else:
                current_step = 0

            # compute reward based on the step
            if current_step == 0:
                # reward based on distance of eef to cube A, and distance of cube A to cube B
                distance_eef_A = torch.linalg.norm(eef_pos - pos_A)
                distance_AB = torch.linalg.norm(offset_AB)
                
                # in [-1.5,-0.5], equal component contribution
                reward = -1.0 + (
                    self._distance_to_reward(distance_eef_A) +
                    self._distance_to_reward(distance_AB)
                ) / 1.5
                reward += - 0.7 - 0.5

            elif current_step == 1:
                # reward based on distance of eef to cube C, with grasping and lifting bonus
                distance_eef_C = torch.linalg.norm(eef_pos - pos_C)
                z_flag = torch.abs(offset_BC[..., 2]) > 0.02
                grasp_flag = self.agent.is_grasping(self.cubeC) or z_flag
                # give reward even if not grasping but above threshold to avoid decrease when releasing cube C after stacking

                # in [-0.5,0.5], equal contribution eef-C, AB-to-C, grasp+height bonus
                reward = (
                    self._distance_to_reward(distance_eef_C) * 2.0 +
                    (z_flag.float() + grasp_flag.float()) / 2.0
                ) / 2.0
                reward += - 0.7 - 0.5

            elif current_step == 2:
                # reward based on distance of cube C to cube A and B, with bonus reward for ungrasping
                distance_AC = torch.linalg.norm(offset_AC)
                distance_BC = torch.linalg.norm(offset_BC)
                grasp_flag = self.agent.is_grasping(self.cubeC)
                
                # in [0.5,1.5], equal contribution AC-to-C, BC-to-C, ungrasping bonus
                reward = (
                    (self._distance_to_reward(distance_AC) + self._distance_to_reward(distance_BC)) +
                    (1.0 - grasp_flag.float())
                ) / 2.0
                reward += 0.5
                
            else:
                reward = 2.5
            
            
        return np.array(reward)
    

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ): 
        return self.compute_dense_reward(obs=obs, action=action, info=info)