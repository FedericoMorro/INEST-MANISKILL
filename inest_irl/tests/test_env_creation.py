"""Test script to create StackPyramid environment and display first state."""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import gymnasium as gym
import numpy as np
from PIL import Image

# Import the stack pyramid env to register it
from inest_irl.maniskill3.stack_pyramid import StackPyramidEnv
from inest_irl.utils import utils


def main():
    """Create and test StackPyramid environment as it's created in SAC training."""
    print("Creating StackPyramid-v1custom environment using utils.make_env (like in sb3_sac.py)...")
    
    # Create output directory for rendered images
    output_dir = "out/test_env_creation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment the same way as in sb3_sac.py
    # Note: EpisodeMonitor wrapper changes the return format, so we disable it for this test
    # In actual training, it's wrapped by Monitor/VecMonitor and GymCompatibilityWrapper in sb3_sac.py
    env = utils.make_env(
        env_name="StackPyramid-v1custom",
        seed=np.random.randint(0, 10000),
        obs_mode="state",
        action_repeat=1,
        frame_stack=1,  # Using frame_stack=1 for clearer testing
        add_episode_monitor=False,  # Disabled to see raw gymnasium output
    )
    
    # Unwrap to the base environment and set render mode
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    if hasattr(base_env, 'set_render_mode'):
        base_env.set_render_mode("rgb_array")
    else:
        base_env.render_mode = "rgb_array"
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("This matches SAC training environment setup.")
    
    # Reset environment to get initial state
    print("\nResetting environment...")
    reset_result = env.reset()
    
    # Handle different wrapper return formats
    if isinstance(reset_result, tuple):
        if len(reset_result) == 2:
            obs, info = reset_result
        elif len(reset_result) == 3:
            # FrameStack returns (obs, info, state) or similar
            obs, info, _ = reset_result
        else:
            obs = reset_result[0]
            info = reset_result[1] if len(reset_result) > 1 else {}
    else:
        obs = reset_result
        info = {}
    
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
    img = env.render()
    if img is not None:
        print(f"  Rendered image shape: {img.shape if hasattr(img, 'shape') else type(img)}")
        if hasattr(img, 'shape'):
            # Convert torch tensor to numpy and handle batch dimension
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            else:
                img = np.array(img)
            
            # Remove batch dimension if present
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            
            # Save initial frame
            img = img.astype(np.uint8) if img.dtype != np.uint8 else img
            pil_img = Image.fromarray(img)
            output_path = os.path.join(output_dir, "initial.png")
            pil_img.save(output_path)
            print(f"  Saved to {output_path}")
    else:
        print("  No render output available")
    
    # Run a few steps to see environment in action
    print("\nRunning 5 random action steps...")
    for step in range(5):
        action = env.action_space.sample()
        step_result = env.step(action)
        
        # Handle different wrapper return formats
        if isinstance(step_result, tuple):
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            elif len(step_result) == 6:
                # Some wrappers might return 6 values
                obs, reward, terminated, truncated, info, _ = step_result
            else:
                obs = step_result[0]
                reward = step_result[1] if len(step_result) > 1 else 0.0
                terminated = step_result[2] if len(step_result) > 2 else False
                truncated = step_result[3] if len(step_result) > 3 else False
                info = step_result[4] if len(step_result) > 4 else {}
        
        print(f"Step {step+1}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
        print(info)
        img = env.render()
        if img is not None and hasattr(img, 'shape'):
            # Convert torch tensor to numpy and handle batch dimension
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            else:
                img = np.array(img)
            
            # Remove batch dimension if present
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            
            # Save frame
            img = img.astype(np.uint8) if img.dtype != np.uint8 else img
            pil_img = Image.fromarray(img)
            output_path = os.path.join(output_dir, f"step_{step+1}.png")
            pil_img.save(output_path)
            print(f"  Rendered and saved to {output_path}")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    env.close()
    print("\nTest completed!")


if __name__ == "__main__":
    main()
