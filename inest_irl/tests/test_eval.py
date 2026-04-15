#!/usr/bin/env python
"""
Test script for evaluating a trained RL model using the stable-baselines3 evaluation callback.
Prints the same metrics that are logged during training.

Usage:
    python test_eval.py <model_path> [--num_episodes 5] [--config_path <config_path>] [--device cpu]
"""

import argparse
import json
import logging
import numpy as np
import os
import sys
import torch
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from inest_irl.utils import utils


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GymCompatibilityWrapper(gym.Wrapper):
    """Wrapper to ensure environment returns (obs, info) tuple from reset/step."""
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def reset(self, seed=None, **kwargs):
        """Normalize reset output to (obs, info)."""
        result = self.env.reset(seed=seed, **kwargs)
        
        # Handle different return formats
        if isinstance(result, tuple):
            if len(result) == 2:
                return result  # (obs, info)
            else:
                return result[0], {}
        else:
            # Single value (obs only)
            return result, {}
    
    def step(self, action):
        """Normalize step output to (obs, reward, terminated, truncated, info)."""
        result = self.env.step(action)
        
        # Handle different return formats
        if isinstance(result, tuple):
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                # Ensure scalar values, not arrays
                reward = float(np.asarray(reward).squeeze())
                terminated = bool(np.asarray(terminated).squeeze())
                truncated = bool(np.asarray(truncated).squeeze())
                return obs, reward, terminated, truncated, info
            elif len(result) == 4:
                # Old Gym API: (obs, reward, done, info)
                obs, reward, done, info = result
                # Ensure scalar values
                reward = float(np.asarray(reward).squeeze())
                done = bool(np.asarray(done).squeeze())
                # Convert done to terminated/truncated
                return obs, reward, done, False, info
            else:
                raise ValueError(f"Unexpected step return length: {len(result)}")
        else:
            raise ValueError(f"Step returned non-tuple: {type(result)}")


def load_config(config_path):
    """Load configuration from Python file."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    if hasattr(config_module, 'get_config'):
        return config_module.get_config()
    elif hasattr(config_module, 'config'):
        return config_module.config
    else:
        raise ValueError(f"Could not find config in {config_path}")


def test_eval(model_path, config_path, num_episodes, device):
    """
    Test evaluation of a trained model.
    
    Args:
        model_path: Path to the trained model checkpoint
        config_path: Path to the configuration file
        num_episodes: Number of evaluation episodes
        device: Device to use (cpu, cuda:0, etc.)
    """
    
    logger.info(f"=" * 60)
    logger.info("Model Evaluation Test")
    logger.info(f"=" * 60)
    
    # Validate checkpoint path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    logger.info(f"Model path: {model_path}")
    
    # Setup device
    if torch.cuda.is_available():
        device_obj = torch.device(device)
    else:
        logger.warning("No GPU found, using CPU")
        device_obj = torch.device("cpu")
    logger.info(f"Using device: {device_obj}")
    
    # Load configuration
    logger.info(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Create evaluation environment
    logger.info(f"Creating environment: {config.env_name}")
    eval_env = utils.make_env(
        config.env_name,
        seed=42,
        action_repeat=config.action_repeat,
        frame_stack=config.frame_stack,
    )
    eval_env = GymCompatibilityWrapper(eval_env)
    eval_env = Monitor(eval_env)
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = SAC.load(model_path, device=device_obj)
    logger.info(f"Model loaded successfully")
    
    # Run evaluation
    logger.info(f"Running {num_episodes} evaluation episodes...")
    logger.info(f"-" * 60)
    
    try:
        rewards, lengths, subgoals_dict = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=num_episodes,
            deterministic=True,
            return_episode_rewards=True,
            return_episode_subgoals=True,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    
    # Compute statistics
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_length = float(np.mean(lengths))
    std_length = float(np.std(lengths))
    min_reward = float(np.min(rewards))
    max_reward = float(np.max(rewards))
    
    # Log results
    logger.info(f"-" * 60)
    logger.info("Evaluation Results")
    logger.info(f"-" * 60)
    logger.info(f"eval/mean_reward: {mean_reward:.4f} ± {std_reward:.4f}")
    logger.info(f"eval/std_reward: {std_reward:.4f}")
    logger.info(f"eval/mean_length: {mean_length:.2f} ± {std_length:.2f}")
    logger.info(f"eval/std_length: {std_length:.2f}")
    logger.info(f"eval/min_reward: {min_reward:.4f}")
    logger.info(f"eval/max_reward: {max_reward:.4f}")
    
    # Log subgoal metrics
    if subgoals_dict:
        logger.info(f"-" * 60)
        logger.info("Subgoal Completion Rates")
        logger.info(f"-" * 60)
        for subgoal_idx, rate in sorted(subgoals_dict.items()):
            logger.info(f"eval/subgoal_{subgoal_idx}: {rate:.4f}")
    
    # Log per-episode rewards
    logger.info(f"-" * 60)
    logger.info("Per-Episode Rewards")
    logger.info(f"-" * 60)
    for i, (reward, length) in enumerate(zip(rewards, lengths)):
        logger.info(f"  Episode {i}: reward={reward:.4f}, length={length}")
    
    logger.info(f"=" * 60)
    logger.info("Evaluation Complete!")
    logger.info(f"=" * 60)
    
    # Prepare results dictionary
    results = {
        "model_path": model_path,
        "num_episodes": num_episodes,
        "eval/mean_reward": mean_reward,
        "eval/std_reward": std_reward,
        "eval/mean_length": mean_length,
        "eval/std_length": std_length,
        "eval/min_reward": min_reward,
        "eval/max_reward": max_reward,
        "subgoal_completion_rates": subgoals_dict,
        "individual_rewards": rewards,
        "episode_lengths": lengths,
    }
    
    eval_env.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test evaluation of a trained RL model using stable-baselines3"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 5)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/fmorro/INEST-MANISKILL/scripts/configs/sb3_sac.py",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (e.g., 'cpu', 'cuda:0')"
    )
    
    args = parser.parse_args()
    
    # Run test
    results = test_eval(
        args.model_path,
        args.config_path,
        args.num_episodes,
        args.device
    )
    
    # Optionally save results to JSON
    output_path = os.path.join(
        os.path.dirname(args.model_path),
        "test_eval_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
