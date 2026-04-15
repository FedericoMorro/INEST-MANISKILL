#!/usr/bin/env python3
"""Test script for intrinsic rewards implementation."""

import numpy as np
import torch
from sac.state_entropy_tracker import StateEntropyTracker, IntrinsicRewardWrapper
import gym


def test_state_entropy_tracker():
    """Test the StateEntropyTracker class."""
    print("Testing StateEntropyTracker...")
    
    # Test different clustering methods
    methods = ["distance", "hash", "kmeans"]
    
    for method in methods:
        print(f"  Testing {method} method...")
        
        # Create tracker
        tracker = StateEntropyTracker(
            state_dim=4,
            max_states=1000,
            novelty_threshold=0.1,
            intrinsic_weight=0.1,
            method=method
        )
        
        # Test with some states
        states = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.1, 2.1, 3.1, 4.1]),  # Similar to first
            np.array([1.2, 2.2, 3.2, 4.2]),  # Medium distance (closer than state 2)
            np.array([1.0, 2.0, 3.0, 4.0]),  # Exact duplicate
        ]
        
        rewards = []
        for i, state in enumerate(states):
            reward = tracker.add_state(state)
            rewards.append(reward)
            print(f"    State {i+1}: reward = {reward:.3f}")
        
        # Basic checks
        assert len(rewards) == 4, f"Should have 4 rewards for {method}"
        assert all(r >= 0 for r in rewards), f"All rewards should be non-negative for {method}"
        
        # For hash method, exact duplicates should have zero reward
        if method == "hash":
            assert rewards[3] == 0.0, "Exact duplicates should have zero reward with hash method"
        
        # For distance method, check that rewards are reasonable
        if method == "distance":
            assert rewards[0] > 0, "First state should have positive novelty"
            assert rewards[1] < rewards[0], "Similar state should have lower novelty than first"
            assert rewards[3] < rewards[0], "Exact duplicate should have lower novelty than first"
        
        # Check statistics
        stats = tracker.get_entropy_stats()
        print(f"    Entropy stats: {stats}")
        assert stats['total_states'] == 4, f"Should have 4 total states for {method}"
        assert stats['unique_states'] == 3, f"Should have 3 unique states for {method}"  # One duplicate
        
        print(f"    ✓ {method} method passed!")
    
    print("✓ StateEntropyTracker tests passed!")


def test_novelty_behavior():
    """Test that novelty detection works as expected."""
    print("Testing novelty behavior...")
    
    # Test distance method specifically
    tracker = StateEntropyTracker(
        state_dim=2,
        max_states=100,
        novelty_threshold=1.0,  # Larger threshold for clearer results
        intrinsic_weight=0.1,
        method="distance"
    )
    
    # Add states in sequence
    states = [
        np.array([0.0, 0.0]),  # First state - should get max novelty
        np.array([0.1, 0.1]),  # Close to first - should get low novelty
        np.array([0.05, 0.05]),  # Closer than state 2 - should get higher novelty
        np.array([0.0, 0.0]),  # Exact duplicate - should get low novelty
    ]
    
    rewards = []
    for i, state in enumerate(states):
        reward = tracker.add_state(state)
        rewards.append(reward)
        print(f"  State {i+1} {state}: reward = {reward:.3f}")
    
    # Verify expected behavior
    assert rewards[0] > 0, "First state should have positive novelty"
    assert rewards[1] < rewards[0], "Similar state should have lower novelty"
    assert rewards[2] > rewards[1], "Closer state should have higher novelty than farther state"
    assert rewards[3] < rewards[0], "Exact duplicate should have lower novelty"
    
    print("✓ Novelty behavior tests passed!")

# Create a simple environment
class SimpleEnv(gym.Env):
    # Classic Gym (<=0.21) uses this for render dispatch
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # Add the attribute Gym’s Wrapper expects:
        self.reward_range = (-float("inf"), float("inf"))
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        # Classic Gym API: return obs only
        return self._sample_obs()

    def step(self, action):
        self.step_count += 1
        obs = self._sample_obs()
        reward = 1.0
        done = self.step_count >= 10
        info = {}
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        x = (self.step_count * 5) % 64
        img[32, x, :] = 255
        return img

    def _sample_obs(self):
        return np.random.randn(4).astype(np.float32)

def test_intrinsic_reward_wrapper():
    """Test the IntrinsicRewardWrapper class."""
    print("Testing IntrinsicRewardWrapper...")
    
    # Create tracker and wrapper
    tracker = StateEntropyTracker(
        state_dim=4,
        max_states=100,
        novelty_threshold=0.1,
        intrinsic_weight=0.1
    )
    
    env = SimpleEnv()
    wrapped_env = IntrinsicRewardWrapper(
        env=env,
        state_entropy_tracker=tracker,
        intrinsic_weight=0.1,
        extrinsic_weight=1.0
    )
    
    # Test reset
    obs = wrapped_env.reset()
    assert obs.shape == (4,)
    
    # Test step
    action = np.array([0.5, -0.3])
    obs, reward, done, info = wrapped_env.step(action)
    
    assert obs.shape == (4,)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert 'intrinsic_reward' in info
    assert 'extrinsic_reward' in info
    assert 'total_reward' in info
    
    print(f"Reward breakdown:")
    print(f"  Intrinsic: {info['intrinsic_reward']:.3f}")
    print(f"  Extrinsic: {info['extrinsic_reward']:.3f}")
    print(f"  Total: {info['total_reward']:.3f}")
    
    print("✓ IntrinsicRewardWrapper tests passed!")


def test_different_clustering_methods():
    """Test different clustering methods."""
    print("Testing different clustering methods...")
    
    methods = ["kmeans", "hash", "distance"]
    state = np.array([1.0, 2.0, 3.0, 4.0])
    
    for method in methods:
        tracker = StateEntropyTracker(
            state_dim=4,
            max_states=100,
            novelty_threshold=0.1,
            intrinsic_weight=0.1,
            method=method
        )
        
        reward = tracker.add_state(state)
        print(f"  {method}: reward = {reward:.3f}")
        assert reward >= 0, f"Reward should be non-negative for {method}"
        assert np.isfinite(reward), f"Reward should be finite for {method}"
    
    print("✓ Clustering methods tests passed!")

def test_re3_raw_state():
    print("Testing RE3 (raw-state) ...")

    tracker = StateEntropyTracker(
        state_dim=4,
        max_states=200,
        novelty_threshold=0.1,
        intrinsic_weight=0.1,
        method="re3",
        re3_cfg=type("Cfg",(object,),{
            "k":3, "embed_dim":64, "beta0":0.1, "beta_decay":0.0,
            "memory_subsample":256, "use_images":False,
        })()
    )

    states = [np.random.randn(4).astype(np.float32) for _ in range(20)]
    rewards = [tracker.add_state(s) for s in states]
    assert all(np.isfinite(r) for r in rewards), "RE3 rewards must be finite"
    assert rewards[0] > 0, "First state should have positive novelty"
    print("✓ RE3 raw-state passed")

# --- NEW: add a RE3 test with images via the wrapper ---
def test_re3_with_images_via_wrapper():
    print("Testing RE3 (images via wrapper) ...")

    env = SimpleEnv()
    tracker = StateEntropyTracker(
        state_dim=4,
        max_states=500,
        novelty_threshold=0.1,
        intrinsic_weight=0.1,
        method="re3",
        re3_cfg=type("Cfg",(object,),{
            "k":3, "embed_dim":64, "beta0":0.1, "beta_decay":0.0,
            "memory_subsample":512, "use_images":True,
        })()
    )
    tracker.set_image_env(env)  # allow tracker to call env.render()

    wrapped_env = IntrinsicRewardWrapper(
        env=env,
        state_entropy_tracker=tracker,
        # IMPORTANT: avoid double scaling; tracker already applies beta
        intrinsic_weight=1.0,
        extrinsic_weight=1.0,
    )

    obs = wrapped_env.reset()
    for _ in range(5):
        a = wrapped_env.action_space.sample()
        obs, r, done, info = wrapped_env.step(a)
        assert "intrinsic_reward" in info and "extrinsic_reward" in info

    print("✓ RE3 images via wrapper passed")

def main():
    """Run all tests."""
    print("Running intrinsic rewards tests...\n")
    
    try:
        test_state_entropy_tracker()
        print()
        
        test_novelty_behavior()
        print()
        
        test_intrinsic_reward_wrapper()
        print()
        
        test_different_clustering_methods()
        print()

        test_re3_raw_state()    
        print()

        test_re3_with_images_via_wrapper()
        print()
        
        print("🎉 All tests passed! The intrinsic reward system is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 