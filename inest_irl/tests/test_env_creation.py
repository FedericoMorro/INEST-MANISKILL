"""Test script to create StackPyramid environment and display first state."""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import copy
import gymnasium as gym
import numpy as np
from PIL import Image

# Import the stack pyramid env to register it
from inest_irl.maniskill3.stack_pyramid import StackPyramidEnv
from inest_irl.utils import utils


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



def main():
    """Create and test StackPyramid environment as it's created in SAC training."""
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
    )
    
    # Unwrap to the base environment and set render mode
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    if hasattr(base_env, 'set_render_mode'):
        base_env.set_render_mode("rgb_array")
    else:
        base_env.render_mode = "rgb_array"
    
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
    
    # Render initial state
    print(f"\n{'='*80}")
    print("RENDERING TEST:")
    print(f"{'='*80}")
    
    img = env.render()
    if img is not None:
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
            print(f"Rendered image: shape={img.shape}, saved to {output_path}")
    else:
        print("No render output available")
    
    # Run steps to see environment in action
    print(f"\n{'='*80}")
    print("RUNNING 100 RANDOM ACTION STEPS:")
    print(f"{'='*80}\n")
    for step in range(100):
        action = env.action_space.sample()
        step_result = env.step(action)
        
        # Handle different wrapper return formats
        if isinstance(step_result, tuple):
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            elif len(step_result) == 6:
                obs, reward, terminated, truncated, info, _ = step_result
            else:
                obs = step_result[0]
                reward = step_result[1] if len(step_result) > 1 else 0.0
                terminated = step_result[2] if len(step_result) > 2 else False
                truncated = step_result[3] if len(step_result) > 3 else False
                info = step_result[4] if len(step_result) > 4 else {}
        
        # Extract camera images if available
        base_rgb = None
        hand_rgb = None
        
        if isinstance(obs, dict) and "sensor_data" in obs:
            sensor_data = obs["sensor_data"]
            if isinstance(sensor_data, dict):
                if "base_camera" in sensor_data and isinstance(sensor_data["base_camera"], dict):
                    if "rgb" in sensor_data["base_camera"]:
                        base_rgb = sensor_data["base_camera"]["rgb"]
                
                if "hand_camera" in sensor_data and isinstance(sensor_data["hand_camera"], dict):
                    if "rgb" in sensor_data["hand_camera"]:
                        hand_rgb = sensor_data["hand_camera"]["rgb"]
        
        # Print progress every 25 steps
        if (step + 1) % 25 == 0:
            print(f"Step {step+1}/100: reward={reward:.4f}")
        
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
        
        if terminated or truncated:
            break

    obs, info = env.reset()
    init_obs_after_episode = copy.deepcopy(obs)

    print(f"{'='*80}")
    print("OBSERVATION CONSISTENCY CHECK:")
    print(f"{'='*80}")
    
    def compare_values(v1, v2):
        """Helper function to compare values that might be tensors or arrays."""
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            return np.array_equal(v1, v2)
        elif hasattr(v1, 'numpy'):
            # Torch tensor
            v1_np = v1.cpu().numpy() if hasattr(v1, 'cpu') else v1.numpy()
            v2_np = v2.cpu().numpy() if hasattr(v2, 'cpu') else v2.numpy()
            return np.array_equal(v1_np, v2_np)
        elif isinstance(v1, dict) and isinstance(v2, dict):
            return all(compare_values(v1.get(k), v2.get(k)) for k in v1.keys() if k in v2)
        else:
            try:
                return v1 == v2
            except:
                return False
    
    if isinstance(init_obs, dict) and isinstance(init_obs_after_episode, dict):
        all_match = True
        mismatches = []
        
        for key in init_obs.keys():
            if key in init_obs_after_episode:
                init_val = init_obs[key]
                after_val = init_obs_after_episode[key]
                
                # Handle nested dicts
                if isinstance(init_val, dict) and isinstance(after_val, dict):
                    sub_matches = []
                    for subkey in init_val.keys():
                        if subkey in after_val:
                            match = compare_values(init_val[subkey], after_val[subkey])
                            sub_matches.append((subkey, match))
                            if not match:
                                all_match = False
                                mismatches.append(f"{key}.{subkey}")
                        else:
                            sub_matches.append((subkey, False))
                            all_match = False
                            mismatches.append(f"{key}.{subkey} (missing)")
                    
                    match_count = sum(1 for _, m in sub_matches if m)
                    print(f"  {key}: {match_count}/{len(sub_matches)} fields match")
                else:
                    match = compare_values(init_val, after_val)
                    print(f"  {key}: {'✓' if match else '✗'}")
                    if not match:
                        all_match = False
                        mismatches.append(key)
            else:
                print(f"  {key}: ✗ (missing in after-episode)")
                all_match = False
                mismatches.append(key)
        
        print(f"\nResult: {'✓ PASS' if all_match else '✗ FAIL'} - All observations match")
        if mismatches:
            print(f"  Mismatches: {', '.join(mismatches[:5])}{'...' if len(mismatches) > 5 else ''}")
    else:
        match = compare_values(init_obs, init_obs_after_episode)
        print(f"Initial obs == After-episode obs: {'✓ PASS' if match else '✗ FAIL'}")
    
    env.close()
    print("\nTest completed!")


if __name__ == "__main__":
    main()
