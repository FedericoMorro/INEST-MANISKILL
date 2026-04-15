#!/usr/bin/env python3
"""Example script demonstrating intrinsic rewards based on state entropy.

This script shows how to enable and configure intrinsic rewards to encourage
exploration by rewarding novel states.
"""

import os
import sys
import numpy as np
import torch
from absl import app
from absl import flags
from ml_collections import config_flags
import json

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils
from sac.state_entropy_tracker import StateEntropyTracker, IntrinsicRewardWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "MatchRegions-Gripper-State-Allo-Demo-v0", "The environment name.")
flags.DEFINE_integer("seed", 42, "RNG seed.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")
flags.DEFINE_boolean("enable_intrinsic", True, "Enable intrinsic rewards.")
flags.DEFINE_float("intrinsic_weight", 0.1, "Weight for intrinsic reward component.")
flags.DEFINE_string("method", "kmeans", "Novelty method: kmeans|hash|distance|re3")
flags.DEFINE_integer("re3_k", 3, "RE3: k for k-NN")
flags.DEFINE_integer("re3_embed_dim", 512, "RE3: embedding dimension")
flags.DEFINE_float("re3_beta0", 0.1, "RE3: initial beta")
flags.DEFINE_float("re3_beta_decay", 0.0, "RE3: per-step beta decay")
flags.DEFINE_integer("re3_memory_subsample", 4096, "RE3: candidate set size")
flags.DEFINE_boolean("re3_use_images", True, "RE3: use env.render('rgb_array') frames")

config_flags.DEFINE_config_file(
    "config",
    "base_configs/rl.py",
    "File path to the training hyperparameter configuration.",
)


def main(_):
    """Run the intrinsic reward example."""
    
    # Load config
    config = FLAGS.config
    
    # Enable intrinsic rewards
    if FLAGS.enable_intrinsic:
        config.intrinsic_reward.enabled = True
        config.intrinsic_reward.intrinsic_weight = FLAGS.intrinsic_weight
        config.intrinsic_reward.method = FLAGS.method
        if FLAGS.method == "re3":
            from ml_collections import ConfigDict
            config.intrinsic_reward.re3 = ConfigDict()
            config.intrinsic_reward.re3.k = FLAGS.re3_k
            config.intrinsic_reward.re3.embed_dim = FLAGS.re3_embed_dim
            config.intrinsic_reward.re3.beta0 = FLAGS.re3_beta0
            config.intrinsic_reward.re3.beta_decay = FLAGS.re3_beta_decay
            config.intrinsic_reward.re3.memory_subsample = FLAGS.re3_memory_subsample
            config.intrinsic_reward.re3.use_images = FLAGS.re3_use_images

        print(f"Intrinsic rewards ON (weight={config.intrinsic_reward.intrinsic_weight})")
        print(f"Method: {FLAGS.method}")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(FLAGS.device)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    with open(f'env_configs/mr_0.json', 'r') as file:
        env_config = json.load(file)
    
    # Create environment
    env = utils.make_env(
        FLAGS.env_name,
        FLAGS.seed,
        action_repeat=config.action_repeat,
        frame_stack=config.frame_stack,
        use_dense_reward=False,
        config=env_config
    )
    
    # Wrap with intrinsic rewards
    env = utils.wrap_intrinsic_reward(env, config, device)
    
    print(f"Environment observation space: {env.observation_space}")
    print(f"Environment action space: {env.action_space}")

    if FLAGS.method == "re3":
        print("RE3 enabled: using random frozen encoder + k-NN log-distance; wrapper weight = 1.0")
    
    # Run a few episodes to demonstrate intrinsic rewards
    num_episodes = 5
    max_steps_per_episode = 100
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1} ===")
        
        obs = env.reset()
        episode_reward = 0
        episode_intrinsic_reward = 0
        episode_extrinsic_reward = 0
        
        for step in range(max_steps_per_episode):
            # Take random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            if 'intrinsic_reward' in info:
                episode_intrinsic_reward += info['intrinsic_reward']
            if 'extrinsic_reward' in info:
                episode_extrinsic_reward += info['extrinsic_reward']
            
            # Print step info
            if step % 20 == 0:
                print(f"Step {step}: reward={reward:.3f}, "
                      f"intrinsic={info.get('intrinsic_reward', 0):.3f}, "
                      f"extrinsic={info.get('extrinsic_reward', 0):.3f}")
            
            if done:
                break
        
        # Print episode summary
        print(f"Episode {episode + 1} finished:")
        print(f"  Total steps: {step + 1}")
        print(f"  Total reward: {episode_reward:.3f}")
        print(f"  Intrinsic reward: {episode_intrinsic_reward:.3f}")
        print(f"  Extrinsic reward: {episode_extrinsic_reward:.3f}")
        
        # Print entropy statistics if available
        if hasattr(env, 'state_entropy_tracker'):
            stats = env.state_entropy_tracker.get_entropy_stats()
            print(f"  Entropy stats:")
            print(f"    Total states: {stats['total_states']}")
            print(f"    Unique states: {stats['unique_states']}")
            print(f"    Entropy: {stats['entropy']:.3f}")
            print(f"    Novelty rate: {stats['novelty_rate']:.3f}")
    
    print("\n=== Intrinsic Reward Demo Complete ===")
    print("The intrinsic reward system encourages exploration by:")
    print("1. Tracking visited states in memory")
    print("2. Computing novelty based on distance to known states")
    print("3. Adding intrinsic rewards for novel states")
    print("4. Combining intrinsic and extrinsic rewards")
    print("\nThis should lead to more diverse exploration and faster learning!")


if __name__ == "__main__":
    app.run(main) 