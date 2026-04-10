"""Test script to create StackPyramid environment and display first state."""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import gymnasium as gym
import numpy as np

# Import the stack pyramid env to register it
from inest_irl.maniskill3.stack_pyramid import StackPyramidEnv


def main():
    """Create and test StackPyramid environment without randomization."""
    print("Creating StackPyramid-v1custom environment without cube randomizations...")
    
    # Create environment with randomize_cubes=False
    env = gym.make(
        "StackPyramid-v1custom",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
        randomize_cubes=False,
    )
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset environment to get initial state
    print("\nResetting environment...")
    obs, info = env.reset()
    
    print("\nInitial state captured!")
    print(f"\nObservation keys: {obs.keys() if isinstance(obs, dict) else 'array'}")
    
    # Display observation details
    if isinstance(obs, dict):
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if value.size <= 12:
                    print(f"    value: {value}")
                else:
                    print(f"    first 3 elements: {value.flat[:3]}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"Observation shape: {obs.shape}")
        print(f"First few obs values: {obs[:5]}")
    
    # Display info details
    print(f"\nInfo keys: {info.keys()}")
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")
    
    # Render initial state
    print("\nRendering initial state...")
    env.render()
    
    # Run a few steps to see environment in action
    print("\nRunning 5 random action steps...")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
        env.render()
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    env.close()
    print("\nTest completed!")


if __name__ == "__main__":
    main()
