from absl import app, flags, logging
import copy
import functools
import json
from ml_collections import config_dict, config_flags
import mani_skill.envs
import numpy as np
import os
from pathlib import Path
import time
import torch
import wandb
import gymnasium as gym

from stable_baselines3 import SAC
from tensorboard.backend.event_processing import event_accumulator
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from inest_irl.utils import utils

FLAGS = flags.FLAGS


class GymCompatibilityWrapper(gym.Wrapper):
    """Wrapper to ensure environment returns (obs, info) tuple from reset/step."""
    
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def reset(self, seed=None, **kwargs):
        """Normalize reset output to (obs, info)."""
        result = self.env.reset(seed=seed, **kwargs)
        
        # Handle different return formats
        if isinstance(result, tuple):
            if len(result) == 2:
                return result  # (obs, info)
            elif len(result) == 1:
                return result[0], {}  # obs only, add empty info
            else:
                # More than 2 values - take first as obs, rest as info dict
                return result[0], {"extra": result[1:]}
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

flags.DEFINE_string("experiment_name", None, "Name of the experiment.")
flags.DEFINE_integer("seed", 22, "RNG seed.")
flags.DEFINE_string("device", "cuda:0", "Device to use for training (e.g., 'cpu', 'cuda:0').")
flags.DEFINE_boolean("wandb", False, "Log on W&B.")
flags.DEFINE_boolean("no_progress_bar", False, "Disable training progress bar (useful when stdout->file).")

config_flags.DEFINE_config_file(
    "config",
    "/home/fmorro/INEST-MANISKILL/scripts/configs/sb3_sac.py",
    "Path to the configuration file.",
)


class WandbCallback(BaseCallback):
    """Callback to log TensorBoard metrics to Weights & Biases."""
    def __init__(self, exp_dir, log_freq, verbose=0):
        super().__init__(verbose)
        self.exp_dir = exp_dir
        self.log_freq = log_freq
        self.last_log_time = None
        self.last_log_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            metrics = {}
            
            # Estimate remaining training time based on iteration speed
            current_time = time.time()
            if self.last_log_time is not None:
                iter_per_sec = (self.num_timesteps - self.last_log_step) / (current_time - self.last_log_time + 1e-8)
                remaining_time = (self.model._total_timesteps - self.num_timesteps) / (iter_per_sec + 1e-8)
                metrics["train/remaining_time"] = float(remaining_time / 3600)  # convert to hours
            self.last_log_time = current_time
            self.last_log_step = self.num_timesteps
            
            
            # Extract metrics from the model's logger if available
            try:
                logger_dict = self.model.logger.name_to_value
                for key, value in logger_dict.items():
                    if isinstance(value, (int, float)):
                        metrics[f"{key}"] = float(value)
            except Exception:
                pass

            # Extract rollout stats from ep_info_buffer
            try:
                if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                    rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                    lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]
                    metrics['rollout/ep_rew_mean'] = float(np.mean(rewards))
                    metrics['rollout/ep_len_mean'] = float(np.mean(lengths))
            except Exception:
                pass

            try:
                wandb.log(metrics, step=self.num_timesteps)
            except Exception:
                pass
        return True

class EvalSaveCallback(BaseCallback):
    """Periodic evaluation callback that saves returns, saves models, and logs to wandb."""
    def __init__(self, eval_env, exp_dir, eval_freq, checkpoint_freq, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.exp_dir = exp_dir
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.n_eval_episodes = n_eval_episodes
        self._last_eval = 0
        self._last_checkpoint = 0
        self.best_mean_reward = float("-inf")

    def _on_step(self) -> bool:
        try:
            step = int(self.num_timesteps)
        except Exception:
            return True

        # Evaluation and logging at eval_frequency
        if step - self._last_eval >= self.eval_freq:
            self._last_eval = step

            try:
                rewards, lengths, subgoals_dict = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=True,
                    return_episode_rewards=True,
                    return_episode_subgoals=True,
                )
                mean_reward = float(np.mean(rewards))
                std_reward = float(np.std(rewards))
                mean_length = float(np.mean(lengths))
            except Exception as e:
                logging.warning(f"Evaluation failed at step {step}: {e}")
                mean_reward = float("nan")
                std_reward = float("nan")
                mean_length = float("nan")
                subgoals_dict = {}

            # Save evaluation results to JSON
            eval_dir = os.path.join(self.exp_dir, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)
            results_path = os.path.join(eval_dir, f"eval_{step}.json")
            try:
                with open(results_path, "w") as f:
                    json.dump({
                        "mean_reward": mean_reward,
                        "std_reward": std_reward,
                        "mean_length": mean_length,
                        "rewards": rewards if isinstance(rewards, list) else list(rewards),
                        "subgoals": subgoals_dict,
                    }, f, indent=2)
            except Exception:
                pass

            # Update best mean reward for checkpoint saving
            if not np.isnan(mean_reward):
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward

            # Log to wandb
            if wandb.run is not None:
                try:
                    log_dict = {
                        "eval/mean_reward": mean_reward,
                        "eval/std_reward": std_reward,
                        "eval/mean_length": mean_length,
                        "train/step": step,
                    }

                    for subgoal, val in subgoals_dict.items():
                        if subgoal == 0:
                            log_dict[f"eval/failed_%"] = val
                        else:
                            log_dict[f"eval/subgoal_{subgoal}_%"] = val

                    wandb.log(log_dict, step=step)
                except Exception:
                    pass

        # Checkpoint saving at checkpoint_frequency
        if step - self._last_checkpoint >= self.checkpoint_freq:
            self._last_checkpoint = step

            checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save latest checkpoint
            step_checkpoint_path = os.path.join(checkpoint_dir, f"{step}")
            latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_model")
            try:
                self.model.save(step_checkpoint_path)
                self.model.save(latest_checkpoint_path)
                if self.verbose >= 1:
                    logging.info(f"Saved latest model checkpoint at step {step}")
            except Exception as e:
                logging.warning(f"Failed to save latest model checkpoint: {e}")

            # Save best model based on mean_reward (if we have evaluated)
            if self.best_mean_reward > float("-inf"):
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_model")
                try:
                    self.model.save(best_checkpoint_path)
                    if self.verbose >= 1:
                        logging.info(f"Updated best model checkpoint at step {step} (best reward: {self.best_mean_reward:.4f})")
                except Exception as e:
                    logging.warning(f"Failed to save best model checkpoint: {e}")

        return True


def _make_env_wrapper(env_name, seed, reward_wrapper_type, action_repeat, frame_stack):
    """Factory function for creating environments in subprocesses."""
    env = utils.make_env(
        env_name,
        seed=seed,
        env_reward_type="normalized_dense" if reward_wrapper_type == "env" else "sparse",
        action_repeat=action_repeat,
        frame_stack=frame_stack,
        obs_mode="state" if reward_wrapper_type != "sparse" else "state_dict",
    )
    return GymCompatibilityWrapper(env)


def main(_):
    # get config
    config = FLAGS.config
    exp_dir = os.path.join(
        config.save_dir,
        FLAGS.experiment_name,
        str(FLAGS.seed),
    )
    utils.setup_experiment(exp_dir, config)

    # setup compute device
    if torch.cuda.is_available():
        device = torch.device(FLAGS.device)
    else:
        logging.warning("No GPU device found. Falling back to CPU.")
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # set random seeds
    if FLAGS.seed is not None:
        logging.info(f"Setting random seed to {FLAGS.seed}")
        torch.manual_seed(FLAGS.seed)
        torch.cuda.manual_seed_all(FLAGS.seed)
        torch.backends.cudnn.deterministic = config.cudnn_deterministic
        torch.backends.cudnn.benchmark = not config.cudnn_deterministic
        np.random.seed(FLAGS.seed)
    else:
        logging.warning("No random seed specified. Results may not be reproducible.")

    # setup W&B logging
    if FLAGS.wandb:
        wandb.init(
            project="StackPyramid-SAC",
            group="SAC-baseline",
            name=FLAGS.experiment_name,
            config={
                "env_name": config.env_name,
                "num_envs": config.num_envs,
                "seed": FLAGS.seed,
                "device": str(device),
            },
            mode="online",
        )
        wandb.config.update(config.to_dict(), allow_val_change=True)
        wandb.run.log_code(".")

    # Create environment function for vec_env
    env_name = copy.deepcopy(config.env_name)
    action_repeat = copy.deepcopy(config.action_repeat)
    frame_stack = copy.deepcopy(config.frame_stack)
    base_seed = copy.deepcopy(FLAGS.seed)
    reward_wrapper_type = copy.deepcopy(config.reward_wrapper.type)

    # Load environments
    logging.info(f"Creating {config.num_envs} environment(s)...")
    
    if config.num_envs > 1:
        # Multiple parallel environments - pass factory function with partial args
        env_fns = [
            functools.partial(_make_env_wrapper, env_name, base_seed + i, reward_wrapper_type, action_repeat, frame_stack)
            for i in range(config.num_envs)
        ]
        env = SubprocVecEnv(env_fns)
        # Wrap with VecMonitor to track episode statistics
        env = VecMonitor(env, os.path.join(exp_dir, "train_monitor"))
    else:
        # Single environment
        env = _make_env_wrapper(env_name, base_seed, reward_wrapper_type, action_repeat, frame_stack)
        # Wrap with Monitor for episode statistics
        env = Monitor(env, os.path.join(exp_dir, "train_monitor"))
    
    # Create evaluation environment
    base_eval_env = utils.make_env(
        config.env_name,
        seed=FLAGS.seed + 100,
        action_repeat=config.action_repeat,
        frame_stack=config.frame_stack,
    )
    eval_env = GymCompatibilityWrapper(base_eval_env)
    eval_env = Monitor(eval_env, os.path.join(exp_dir, "eval_monitor"))

    # Get observation and action space dimensions
    obs_space = env.observation_space if config.num_envs > 1 else env.observation_space
    action_space = env.action_space if config.num_envs > 1 else env.action_space
    
    logging.info(f"Observation space: {obs_space}")
    logging.info(f"Action space: {action_space}")

    # Configure policy network architecture
    policy_kwargs = {
        "net_arch": {
            "pi": config.sac.actor.net_arch,
            "qf": config.sac.critic.net_arch,
        }
    }
    logging.info(f"Policy network architecture: {policy_kwargs}")

    # Configure action noise
    if config.sac.action_noise_std is not None and config.sac.action_noise_std > 0:
        action_noise = VectorizedActionNoise(
            NormalActionNoise(
                mean=np.zeros(action_space.shape[0]),
                sigma=config.sac.action_noise_std * np.ones(action_space.shape[0]),
            ),
            n_envs=config.num_envs,
        )
        logging.info(f"Using action noise: {action_noise}")
    else:   
        action_noise = None
        logging.info("No action noise will be used.")

    # Define target entropy annealing function
    def target_entropy_anneal(current_step):
        final_target_entropy = config.sac.target_entropy if config.sac.target_entropy is not None else -action_space.shape[0]
        if current_step < config.sac.start_entropy_anneal:
            return config.initial_target_entropy
        elif current_step > config.sac.end_entropy_anneal:
            return final_target_entropy
        else:
            # linear annealing from initial_target_entropy to final_target_entropy
            progress = (current_step - config.sac.start_entropy_anneal) / (config.sac.end_entropy_anneal - config.sac.start_entropy_anneal)
            return config.initial_target_entropy + progress * (final_target_entropy - config.initial_target_entropy)

    # Create SAC model with TensorBoard logging
    tb_log_dir = os.path.join(exp_dir, "tensorboard")
    logging.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
    logging.info("Creating SAC agent...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=config.sac.actor_lr,
        buffer_size=int(config.replay_buffer_capacity),
        learning_starts=int(config.num_seed_steps),
        batch_size=config.sac.batch_size,
        tau=config.sac.critic_tau,
        gamma=config.sac.discount,
        action_noise=action_noise,
        ent_coef=f'auto_{config.sac.init_temperature}',
        train_freq=1,
        gradient_steps=1,
        target_entropy=config.sac.target_entropy if config.sac.target_entropy is not None else -action_space.shape[0],
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_log_dir,
        device=device,
        verbose=1,
        target_entropy_anneal=target_entropy_anneal if config.sac.anneal_target_entropy else None,
        replay_buffer_class=None if config.reward_wrapper.type != "sparse" else HerReplayBuffer,
    )

    # Override optimizers with component-specific learning rates and betas
    logging.info("Setting up component-specific optimizers...")
    model.actor.optimizer = torch.optim.Adam(
        model.actor.parameters(),
        lr=config.sac.actor_lr,
        betas=config.sac.actor_betas,
    )
    model.critic.optimizer = torch.optim.Adam(
        model.critic.parameters(),
        lr=config.sac.critic_lr,
        betas=config.sac.critic_betas,
    )
    logging.info(f"Actor LR: {config.sac.actor_lr}, betas: {config.sac.actor_betas}")
    logging.info(f"Critic LR: {config.sac.critic_lr}, betas: {config.sac.critic_betas}")
    
    if model.ent_coef_optimizer is not None:
        model.ent_coef_optimizer = torch.optim.Adam(
            [model.log_ent_coef],
            lr=config.sac.alpha_lr,
            betas=config.sac.alpha_betas,
        )
        logging.info(f"Alpha LR: {config.sac.alpha_lr}, betas: {config.sac.alpha_betas}")

    # Setup callbacks
    callbacks = []
    if FLAGS.wandb:
        callbacks.append(WandbCallback(exp_dir, config.log_frequency))
    
    # Add periodic evaluation and save callback
    callbacks.append(
        EvalSaveCallback(
            eval_env=eval_env,
            exp_dir=exp_dir,
            eval_freq=int(config.eval_frequency),
            checkpoint_freq=int(config.checkpoint_frequency),
            n_eval_episodes=int(config.num_eval_episodes),
            verbose=1,
        )
    )
    logging.info(f"Evaluation frequency: {config.eval_frequency} steps")
    logging.info(f"Checkpoint frequency: {config.checkpoint_frequency} steps")

    # Training loop
    logging.info(f"Starting training for {config.num_train_steps} steps...")
    
    try:
        model.learn(
            total_timesteps=int(config.num_train_steps),
            callback=callbacks,
            progress_bar=not FLAGS.no_progress_bar,
        )
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")

    # Final evaluation
    logging.info("Running final evaluation...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=int(config.num_eval_episodes),
        deterministic=True,
    )
    
    eval_stats = {
        "eval/mean_reward": float(mean_reward),
        "eval/std_reward": float(std_reward),
    }
    
    logging.info(f"Final evaluation results:")
    for key, value in eval_stats.items():
        logging.info(f"  {key}: {value:.4f}")

    # Save final model
    model_path = os.path.join(exp_dir, "final_model")
    logging.info(f"Saving final model to {model_path}...")
    model.save(model_path)

    # Save evaluation results
    results_path = os.path.join(exp_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_stats, f, indent=2)
    
    logging.info("Training complete!")
    
    if FLAGS.wandb:
        wandb.finish()

    env.close()
    eval_env.close()


if __name__ == "__main__":
    app.run(main)