"""Test script to create StackPyramid environment and display first state."""

import sys
import os
import argparse

import copy
import gymnasium as gym
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import the stack pyramid env to register it
from inest_irl.maniskill3.stack_pyramid import StackPyramidEnv
from inest_irl.utils import utils


NUM_STEPS = 10


def print_structure(obj, indent=0, max_depth=10, max_items=5):
    """Recursively print data structure down to basic types.
    
    Args:
        obj: Object to print
        indent: Current indentation level
        max_depth: Maximum recursion depth to prevent infinite loops
        max_items: Maximum number of items to show before ellipsis
    """
    prefix = "  " * indent
    
    if indent > max_depth:
        print(f"{prefix}[max depth reached]")
        return
    
    # Handle None
    if obj is None:
        print(f"{prefix}None")
        return
    
    # Handle torch tensors
    if hasattr(obj, 'shape') and hasattr(obj, 'dtype') and hasattr(obj, 'device'):
        # PyTorch tensor
        shape_str = str(obj.shape)
        dtype_str = str(obj.dtype)
        if obj.numel() <= 5:  # Show small tensors directly
            print(f"{prefix}Tensor | shape={shape_str}, dtype={dtype_str}, value={obj}")
        else:
            print(f"{prefix}Tensor | shape={shape_str}, dtype={dtype_str}")
        return
    
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        if obj.size <= 5:  # Show small arrays directly
            print(f"{prefix}ndarray | shape={obj.shape}, dtype={obj.dtype}, value={obj}")
        else:
            print(f"{prefix}ndarray | shape={obj.shape}, dtype={obj.dtype}")
        return
    
    # Handle dictionaries
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{prefix}dict | keys={keys}")
        items_to_show = keys[:max_items]
        for key in items_to_show:
            print(f"{prefix}  [{key}]:")
            print_structure(obj[key], indent + 2, max_depth, max_items)
        if len(keys) > max_items:
            print(f"{prefix}  ... ({len(keys) - max_items} more keys)")
        return
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        list_type = type(obj).__name__
        print(f"{prefix}{list_type} | len={len(obj)}")
        items_to_show = obj[:max_items]
        for i, item in enumerate(items_to_show):
            print(f"{prefix}  [{i}]:")
            print_structure(item, indent + 2, max_depth, max_items)
        if len(obj) > max_items:
            print(f"{prefix}  ... ({len(obj) - max_items} more items)")
        return
    
    # Handle basic types
    type_name = type(obj).__name__
    obj_str = str(obj)
    if len(obj_str) > 80:
        obj_str = obj_str[:77] + "..."
    print(f"{prefix}{type_name} | {obj_str}")



def main(render=False):
    """Create and test StackPyramid environment as it's created in SAC training.
    Args:
        render: If True, display rendered frames to the screen.
    """
    # Create output directory for rendered images
    output_dir = "out/test_env_creation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment the same way as in sb3_sac.py
    # Note: EpisodeMonitor wrapper changes the return format, so we disable it for this test
    # In actual training, it's wrapped by Monitor/VecMonitor and GymCompatibilityWrapper in sb3_sac.py
    print("Creating StackPyramid-v1custom environment...")
    env = utils.make_env(
        env_name="StackPyramid-v1custom",
        seed=np.random.randint(0, 10000),
        obs_mode="state+rgb",
        action_repeat=1,
        frame_stack=1,  # Using frame_stack=1 for clearer testing
        add_episode_monitor=False,  # Disabled to see raw gymnasium output
        base_camera="base_camera",
        render_camera="base_camera",
        cameras_resolution=(512, 512),  # Use higher resolution for testing
    )

    print(f"Environment created: obs_space={env.observation_space}, action_space={env.action_space}\n")
    
    # Reset environment to get initial state
    obs, info = env.reset()
    init_obs = copy.deepcopy(obs)
    
    print(f"{'='*80}")
    print("OBSERVATION FORMAT ANALYSIS:")
    print(f"{'='*80}\n")
    print_structure(obs)
    
    # Print info
    print(f"\n{'='*80}")
    print("INFO:")
    print(f"{'='*80}")
    if info:
        print_structure(info)
    else:
        print("None")
    
    # Save initial sensor images (moved to info) if available, then render initial state
    print(f"\n{'='*80}")
    print("RENDERING TEST:")
    print(f"{'='*80}")

    sensor_data = None
    if isinstance(info, dict) and "sensor_data" in info:
        sensor_data = info["sensor_data"]
    elif isinstance(obs, dict) and "sensor_data" in obs:
        sensor_data = obs["sensor_data"]

    if isinstance(sensor_data, dict):
        if "base_camera" in sensor_data and isinstance(sensor_data["base_camera"], dict) and "rgb" in sensor_data["base_camera"]:
            base_img = sensor_data["base_camera"]["rgb"]
            if hasattr(base_img, 'shape'):
                base_arr = base_img.cpu().numpy() if hasattr(base_img, 'cpu') else np.array(base_img)
                if base_arr.ndim == 4 and base_arr.shape[0] == 1:
                    base_arr = base_arr[0]
                base_arr = base_arr.astype(np.uint8) if base_arr.dtype != np.uint8 else base_arr
                Image.fromarray(base_arr).save(os.path.join(output_dir, "initial_base_camera.png"))

        if "hand_camera" in sensor_data and isinstance(sensor_data["hand_camera"], dict) and "rgb" in sensor_data["hand_camera"]:
            hand_img = sensor_data["hand_camera"]["rgb"]
            if hasattr(hand_img, 'shape'):
                hand_arr = hand_img.cpu().numpy() if hasattr(hand_img, 'cpu') else np.array(hand_img)
                if hand_arr.ndim == 4 and hand_arr.shape[0] == 1:
                    hand_arr = hand_arr[0]
                hand_arr = hand_arr.astype(np.uint8) if hand_arr.dtype != np.uint8 else hand_arr
                Image.fromarray(hand_arr).save(os.path.join(output_dir, "initial_hand_camera.png"))

    # Render initial state: if render flag is set, call env.render() to show on screen; otherwise save rgb_array
    if render:
        env.render()
    else:
        img = env.render()
        if img is not None and hasattr(img, 'shape'):
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            else:
                img = np.array(img)
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            img = img.astype(np.uint8) if img.dtype != np.uint8 else img
            pil_img = Image.fromarray(img)
            output_path = os.path.join(output_dir, "initial.png")
            pil_img.save(output_path)
            print(f"Rendered image: shape={img.shape}, saved to {output_path}")
    
    # Run steps to see environment in action
    print(f"\n{'='*80}")
    print(f"RUNNING {NUM_STEPS} RANDOM ACTION STEPS:")
    print(f"{'='*80}\n")
    
    # Create matplotlib figure for real-time visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Camera Views")
    axes[0].set_title("Base Camera")
    axes[1].set_title("Hand Camera")
    for ax in axes:
        ax.axis('off')
    
    img_base_display = axes[0].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    img_hand_display = axes[1].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    
    plt.tight_layout()
    plt.ion()  # Enable interactive mode
    
    for step in range(NUM_STEPS):
        action = env.action_space.sample()
        step_result = env.step(action)
        
        # Simplified handling of step return formats
        if isinstance(step_result, tuple):
            if len(step_result) >= 5:
                obs, reward, terminated, truncated, info = step_result[:5]
            else:
                obs = step_result[0]
                reward = step_result[1] if len(step_result) > 1 else 0.0
                terminated = step_result[2] if len(step_result) > 2 else False
                truncated = step_result[3] if len(step_result) > 3 else False
                info = step_result[4] if len(step_result) > 4 else {}
        else:
            obs = step_result
            reward = 0.0
            terminated = False
            truncated = False
            info = {}
        
        # Extract camera images if available (sensor_data moved to info)
        base_rgb = None
        hand_rgb = None
        sensor_data = None
        if isinstance(info, dict) and "sensor_data" in info:
            sensor_data = info["sensor_data"]
        elif isinstance(obs, dict) and "sensor_data" in obs:
            sensor_data = obs["sensor_data"]

        if isinstance(sensor_data, dict):
            if "base_camera" in sensor_data and isinstance(sensor_data["base_camera"], dict) and "rgb" in sensor_data["base_camera"]:
                base_rgb = sensor_data["base_camera"]["rgb"]
            if "hand_camera" in sensor_data and isinstance(sensor_data["hand_camera"], dict) and "rgb" in sensor_data["hand_camera"]:
                hand_rgb = sensor_data["hand_camera"]["rgb"]
        
        # Update matplotlib displays
        if base_rgb is not None and hasattr(base_rgb, 'shape'):
            img = base_rgb
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            else:
                img = np.array(img)
            img = img.astype(np.uint8) if img.dtype != np.uint8 else img
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            img_base_display.set_data(img)
        
        if hand_rgb is not None and hasattr(hand_rgb, 'shape'):
            img = hand_rgb
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            else:
                img = np.array(img)
            img = img.astype(np.uint8) if img.dtype != np.uint8 else img
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            img_hand_display.set_data(img)
        
        # Print progress every 25 steps
        if (step + 1) % 25 == 0:
            print(f"Step {step+1}/{NUM_STEPS}: reward={reward:.4f}")
        
        # Save camera images at specific steps
        if step in [0, 50, 99] or (step < 10):
            if base_rgb is not None and hasattr(base_rgb, 'shape'):
                img = base_rgb
                # Convert torch tensor to numpy if needed
                if hasattr(img, 'cpu'):
                    img = img.cpu().numpy()
                else:
                    img = np.array(img)
                img = img.astype(np.uint8) if img.dtype != np.uint8 else img
                if img.ndim == 4 and img.shape[0] == 1:
                    img = img[0]
                pil_img = Image.fromarray(img)
                output_path = os.path.join(output_dir, f"step_{step+1}_base_camera.png")
                pil_img.save(output_path)
                # Avoid Image.show(); rely on env.render() when render=True
            
            if hand_rgb is not None and hasattr(hand_rgb, 'shape'):
                img = hand_rgb
                # Convert torch tensor to numpy if needed
                if hasattr(img, 'cpu'):
                    img = img.cpu().numpy()
                else:
                    img = np.array(img)
                img = img.astype(np.uint8) if img.dtype != np.uint8 else img
                if img.ndim == 4 and img.shape[0] == 1:
                    img = img[0]
                pil_img = Image.fromarray(img)
                output_path = os.path.join(output_dir, f"step_{step+1}_hand_camera.png")
                pil_img.save(output_path)
                # Avoid Image.show(); rely on env.render() when render=True
        
        # Update figure display
        fig.canvas.draw()
        plt.pause(0.01)  # Small pause to allow figure to update and remain responsive
        
        if render:
            env.render()
        
        if terminated or truncated:
            break
        
    # wait for user to close the figure if rendering, otherwise just close it
    if render:
        print("Episode ended. Close the figure window to finish.")
        plt.ioff()  # Disable interactive mode
        plt.show()  # Keep the figure open until user closes it
    else:
        plt.close(fig)  # Close the figure after loop completes

    env.close()
    print("\nTest completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test environment creation and rendering")
    parser.add_argument("-r", "--render", action="store_true", help="Display rendered frames to screen")
    args = parser.parse_args()
    main(render=args.render)
