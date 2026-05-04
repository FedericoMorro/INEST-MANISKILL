from absl import app, flags, logging
import copy
import functools
import json
import matplotlib.pyplot as plt
from ml_collections import config_flags
import numpy as np
import os
import random
import time
import torch
import wandb

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from inest_irl.utils import utils
from inest_irl.utils.loggers import CSVLogger

from eval_policy import generate_reward_plot

FLAGS = flags.FLAGS


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


class LogWandbCallback(BaseCallback):
    """Callback to log TensorBoard metrics to Weights & Biases."""
    
    TRAIN_LOG_CSV_KEYS = [
        "step",
        "actor_loss", "critic_loss", "ent_coef_loss",
        "ent_coef", "learning_rate", "target_entropy",
        "episodes", "n_updates", "total_timesteps", "time_elapsed", "fps", "remaining_time",
    ]
    ROLLOUT_LOG_CSV_KEYS = [
        "step",
        "ep_rew_mean", "ep_rew_std", "ep_len_mean", "ep_success_mean",
        "failed_%", "subgoal_1_%", "subgoal_2_%", "subgoal_3_%", "subgoal_4_%",
        "ep_env_rew_mean", "ep_env_rew_std",
        "detected_failed_%", "detected_subgoal_1_%", "detected_subgoal_2_%", "detected_subgoal_3_%", "detected_subgoal_4_%",
    ]
    
    def __init__(self, exp_dir, log_freq, verbose=0):
        super().__init__(verbose)
        self.exp_dir = exp_dir
        self.log_freq = log_freq
        self.last_log_time = None
        self.last_log_step = 0
        
        self.train_csv_logger = CSVLogger(os.path.join(exp_dir, "train_log.csv"))
        self.rollout_csv_logger = CSVLogger(os.path.join(exp_dir, "rollout_log.csv"))
        
        self.train_csv_logger.init_logging(self.TRAIN_LOG_CSV_KEYS)
        self.rollout_csv_logger.init_logging(self.ROLLOUT_LOG_CSV_KEYS)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            log_dict = {
                "train/step": self.num_timesteps,
            }
            
            # Estimate remaining training time based on iteration speed
            current_time = time.time()
            if self.last_log_time is not None:
                iter_per_sec = (self.num_timesteps - self.last_log_step) / (current_time - self.last_log_time + 1e-8)
                remaining_time = (self.model._total_timesteps - self.num_timesteps) / (iter_per_sec + 1e-8)
                log_dict["train/remaining_time"] = float(remaining_time / 3600)  # convert to hours
            self.last_log_time = current_time
            self.last_log_step = self.num_timesteps
            
            
            # Extract metrics from the model's logger if available
            try:
                logger_dict = self.model.logger.name_to_value
                for key, value in logger_dict.items():
                    if isinstance(value, (int, float)):
                        log_dict[f"{key}"] = float(value)
            except Exception:
                pass

            # Extract rollout stats from ep_info_buffer and ep_success_buffer if available
            try:
                if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                    rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                    lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]
                    log_dict['rollout/ep_rew_mean'] = float(np.mean(rewards))
                    log_dict['rollout/ep_rew_std'] = float(np.std(rewards))
                    log_dict['rollout/ep_len_mean'] = float(np.mean(lengths))
                    if hasattr(self.model, 'ep_success_buffer') and len(self.model.ep_success_buffer) > 0:
                        successes = [float(s) for s in self.model.ep_success_buffer]
                        log_dict['rollout/ep_success_mean'] = float(np.mean(successes))
                    else:
                        log_dict['rollout/ep_success_mean'] = float(0)
                        
                # Extract additional episode info: subgoals, env_reward, detected_subgoals
                if hasattr(self.model, 'ep_add_info_buffer') and len(self.model.ep_add_info_buffer) > 0:
                    
                    print(self.model.ep_add_info_buffer)
                    
                    if "subgoal" in self.model.ep_add_info_buffer:
                        for subgoal, count in self.model.ep_add_info_buffer.get("subgoal", {}).items():
                            if subgoal == 0:
                                log_dict[f"rollout/failed_%"] = float(count)
                            else:
                                log_dict[f"rollout/subgoal_{subgoal}_%"] = float(count)
                    
                    if "cum_env_reward" in self.model.ep_add_info_buffer:
                        log_dict["rollout/ep_env_rew_mean"] = float(np.mean(self.model.ep_add_info_buffer["cum_env_reward"]))
                        log_dict["rollout/ep_env_rew_std"] = float(np.std(self.model.ep_add_info_buffer["cum_env_reward"]))
                        
                    if "detected_subgoal" in self.model.ep_add_info_buffer:
                        for subgoal, count in self.model.ep_add_info_buffer.get("detected_subgoal", {}).items():
                            if subgoal == 0:
                                log_dict[f"rollout/detected_failed_%"] = float(count)
                            else:
                                log_dict[f"rollout/detected_subgoal_{subgoal}_%"] = float(count)
                                
                    self.ep_add_info_buffer.clear()
                        
                    
            except Exception:
                pass

            # Extract time logs from model if available
            try:
                if hasattr(self.model, 'log_time_buffer'):
                    for key, value in self.model.log_time_buffer.items():
                        log_dict[key] = float(value)
            except Exception:
                pass

            try:
                # log to csv, train and rollout logs separated (removing prefixes)
                train_log_dict = {k.split("/", 1)[1]: v for k, v in log_dict.items() if k.startswith("train/") or k.startswith("time/")}
                rollout_log_dict = {k.split("/", 1)[1]: v for k, v in log_dict.items() if k.startswith("rollout/")}
                rollout_log_dict["step"] = self.num_timesteps  # add step to rollout log dict as well
                
                self.train_csv_logger.log_and_flush(train_log_dict)
                self.rollout_csv_logger.log_and_flush(rollout_log_dict)

                if wandb.run is not None:
                    wandb.log(log_dict, step=self.num_timesteps)
                    
            except Exception as e:
                logging.warning(f"Failed to log metrics at step {self.num_timesteps}: {e}")
                
        return True

class EvalSaveCallback(BaseCallback):
    """Periodic evaluation callback that saves returns, saves models, and logs to wandb."""
    
    CSV_LOG_KEYS = [
        "step",
        "mean_reward", "std_reward", "mean_length",
        "failed_%", "subgoal_1_%", "subgoal_2_%", "subgoal_3_%", "subgoal_4_%",
        "mean_env_reward", "std_env_reward",
        "detected_failed_%", "detected_subgoal_1_%", "detected_subgoal_2_%", "detected_subgoal_3_%", "detected_subgoal_4_%",
    ]

    def __init__(self,
                 eval_env, exp_dir,
                 eval_freq, checkpoint_freq,
                 n_eval_episodes=5, verbose=0, save_video=False,
                 learned_reward=False, subgoal_reward=False):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.exp_dir = exp_dir
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.n_eval_episodes = n_eval_episodes
        self.learned_reward = learned_reward
        self.subgoal_reward = subgoal_reward
        self.save_video = save_video
        
        self._last_eval = 0
        self._last_checkpoint = 0
        self.best_mean_reward = float("-inf")
        
        self.csv_logger = CSVLogger(os.path.join(exp_dir, "eval_log.csv"))
        self.csv_logger.init_logging(self.CSV_LOG_KEYS)
        
    def _on_step(self) -> bool:
        try:
            step = int(self.num_timesteps)
        except Exception:
            return True

        # Evaluation and logging at eval_frequency
        if step - self._last_eval >= self.eval_freq:
            self._last_eval = step

            try:
                rewards, lengths, additional_info = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=True,
                    return_episode_rewards=True,
                    return_episode_subgoals=True,
                    return_env_reward=self.learned_reward,
                    return_detected_subgoals=self.subgoal_reward,
                    return_video=self.save_video,
                )
                mean_reward = float(np.mean([np.sum(r) for r in rewards]))
                std_reward = float(np.std([np.sum(r) for r in rewards]))
                mean_length = float(np.mean(lengths))
                subgoals_dict = additional_info.get("episode_subgoals", {})
                subgoals_idxs = additional_info.get("episode_subgoal_idxs", [])
                env_rewards = additional_info.get("episode_env_rewards", [])
                if env_rewards:
                    if isinstance(env_rewards[0], list):
                        # List of lists - sum each episode's rewards
                        mean_env_reward = float(np.mean([np.sum(r) for r in env_rewards]))
                        std_env_reward = float(np.std([np.sum(r) for r in env_rewards]))
                    else:
                        # Flat list of scalars
                        mean_env_reward = float(np.mean(env_rewards))
                        std_env_reward = float(np.std(env_rewards))
                else:
                    mean_env_reward = None
                    std_env_reward = None
                detected_subgoals_dict = additional_info.get("episode_detected_subgoals", {})
                detected_subgoals_idxs = additional_info.get("episode_detected_subgoal_idxs", [])
                videos = additional_info.get("episode_videos", [])
            except Exception as e:
                logging.warning(f"Evaluation failed at step {step}: {e}")
                mean_reward = float("nan")
                std_reward = float("nan")
                mean_length = float("nan")
                subgoals_dict = {}
                subgoals_idxs = []
                env_rewards = []
                mean_env_reward = float("nan")
                std_env_reward = float("nan")
                detected_subgoals_dict = {}
                detected_subgoals_idxs = []
                videos = []
                
            # save evaluation results to JSON
            eval_dir = os.path.join(self.exp_dir, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)
            videos_path = os.path.join(eval_dir, "videos")
            os.makedirs(videos_path, exist_ok=True)
            results_path = os.path.join(eval_dir, f"{step}.json")
            try:
                # Convert rewards to serializable format (handle list of lists)
                rewards_serializable = []
                for r in rewards:
                    if isinstance(r, (list, np.ndarray)):
                        rewards_serializable.append([float(x) for x in r])
                    else:
                        rewards_serializable.append(float(r))
                
                # Convert env_rewards to serializable format (handle list of lists)
                env_rewards_serializable = []
                for r in env_rewards:
                    if isinstance(r, (list, np.ndarray)):
                        env_rewards_serializable.append([float(x) for x in r])
                    else:
                        env_rewards_serializable.append(float(r))
                
                with open(results_path, "w") as f:
                    json.dump({
                        "mean_reward": mean_reward,
                        "std_reward": std_reward,
                        "mean_length": mean_length,
                        "rewards": rewards_serializable,
                        "subgoals": subgoals_dict,
                        "subgoal_idxs": subgoals_idxs,
                        "env_rewards": env_rewards_serializable,
                        "mean_env_reward": mean_env_reward,
                        "std_env_reward": std_env_reward,
                        "detected_subgoals": detected_subgoals_dict,
                        "detected_subgoal_idxs": detected_subgoals_idxs,
                    }, f, indent=2)
            except Exception as e:
                logging.warning(f"Failed to save JSON evaluation results at step {step}: {e}")
            
            # save reward plot
            plot_path = os.path.join(eval_dir, f"{step}.png")
            try:
                fig, axs = plt.subplots(4, 4, figsize=(40, 20))
                for i in range(min(len(rewards), 16)):
                    generate_reward_plot(
                        ax=axs[i // 4, i % 4],
                        rewards=rewards[i],
                        subgoal_idxs=subgoals_idxs[i] if i < len(subgoals_idxs) else None,
                        env_rewards=env_rewards[i] if i < len(env_rewards) else None,
                        detected_subgoal_idxs=detected_subgoals_idxs[i] if i < len(detected_subgoals_idxs) else None,
                        title=f"Episode {i}",
                    )
                plt.tight_layout()
                plt.savefig(plot_path, dpi=300)
                plt.close(fig)
            except Exception as e:
                logging.warning(f"Failed to save PNG evaluation reward plot at step {step}: {e}")
                
            # save video of first evaluation episode
            if self.save_video and videos:
                try:
                    import imageio
                    import cv2
                    
                    for i, video in enumerate(videos):
                        if video is None:
                            logging.warning(f"Video {i} is None at step {step}. Skipping video saving.")
                            continue
                        
                        video_path = os.path.join(videos_path, f"{step}_env{i}.mp4")
                        try:
                            # resize frames to 512x512 if needed
                            video_rs = [v if v.shape[0] == 512 else cv2.resize(v, (512, 512)) for v in video]
                            imageio.mimwrite(video_path, video_rs, fps=10)
                        except Exception as e:
                            logging.warning(f"Failed to save video {i} at step {step}: {e}")
                except Exception as e:
                    logging.warning(f"Failed to save evaluation video at step {step}: {e}")

            # Update best mean reward for checkpoint saving
            if not np.isnan(mean_reward):
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward

            # Log
            try:
                log_dict = {
                    "eval/mean_reward": mean_reward,
                    "eval/std_reward": std_reward,
                    "eval/mean_length": mean_length,
                }

                for subgoal, val in subgoals_dict.items():
                    if subgoal == 0:
                        log_dict[f"eval/failed_%"] = val
                    else:
                        log_dict[f"eval/subgoal_{subgoal}_%"] = val

                if env_rewards:
                    log_dict["eval/mean_env_reward"] = mean_env_reward
                    log_dict["eval/std_env_reward"] = std_env_reward
                    
                if detected_subgoals_dict:
                    for subgoal, val in detected_subgoals_dict.items():
                        if subgoal == 0:
                            log_dict[f"eval/detected_failed_%"] = val
                        else:
                            log_dict[f"eval/detected_subgoal_{subgoal}_%"] = val
                            
                # log to csv
                csv_log_dict = {k.split("/", 1)[1]: v for k, v in log_dict.items()}
                csv_log_dict["step"] = step  # add step to csv log dict
                self.csv_logger.log_and_flush(csv_log_dict)
                            
                # add media for wandb logging
                log_dict["eval_viz/reward_plot"] = wandb.Image(plot_path)
                if os.path.exists(f"{videos_path}/{step}_env0.mp4"):
                    log_dict["eval_viz/video_env0"] = wandb.Video(f"{videos_path}/{step}_env0.mp4", format="mp4")

                # log to wandb
                if wandb.run is not None:
                    wandb.log(log_dict, step=step)
                
            except Exception as e:
                logging.warning(f"Failed to log to wandb at step {step}: {e}")

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


def _make_env_wrapper(config, seed, rank, train_flag, learned_reward_data, exp_dir):
    """Factory function for creating environments in subprocesses."""
    return utils.make_env(
        env_name=config.env_name,
        seed=seed,
        reward_type=config.reward_wrapper.type,
        obs_mode="state" if config.reward_wrapper.type != "sparse" else "state_dict",
        frame_stack=config.frame_stack,
        action_repeat=config.action_repeat,
        env_randomization=config.env_randomization,
        render_camera=config.render_camera,
        reward_scaling=config.reward_scaling,
        rank=rank,
        train_flag=train_flag,
        exp_dir=exp_dir,
        learned_reward_data=learned_reward_data,
        add_episode_monitor = True,
        save_video=False,
    )


def main(_):
    # get config
    config = FLAGS.config
    config.save_dir = f"{config.save_dir}-lr" if config.reward_wrapper.type not in ["sparse", "env", "env_state-intrinsic"] else config.save_dir
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
        random.seed(FLAGS.seed)
    else:
        logging.warning("No random seed specified. Results may not be reproducible.")

    # setup W&B logging
    if FLAGS.wandb:
        if config.reward_wrapper.type in ["sparse", "env", "env_state-intrinsic"]:
            wandb_project = "StackPyramid-SAC"
        else:
            wandb_project = "StackPyramid-SAC-LearnedReward"
        
        wandb.init(
            project=wandb_project,
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

    # Copying flags to local variables to avoid issues with VecEnv subprocesses creation
    base_seed = copy.deepcopy(FLAGS.seed)
    eval_seed = base_seed + 1000
    
    # re-set seeds if same-seed randomization is used to ensure the same environment initialization across episodes
    if config.env_randomization == "same-seed":
        logging.info(f"Using same-seed randomization with seed {config.same_seed_randomization}. Setting base and eval seeds to {config.same_seed_randomization}.")
        base_seed = config.same_seed_randomization
        eval_seed = config.same_seed_randomization

    # Load environments
    logging.info(f"Creating {config.num_envs} environment(s)...")
    
    # Populate learned_reward_data if using learned reward wrappers
    learned_reward_data = None
    if config.reward_wrapper.type not in ["sparse", "env", "env_state-intrinsic"]:
        logging.info("Loading learned reward model and data...")
        if config.reward_wrapper.pretrained_path is None:
            raise ValueError(f"config.reward_wrapper.pretrained_path must be provided for learned reward wrapper types (specified: {config.reward_wrapper.type}).")
        learned_reward_data = utils.load_learned_reward_data(config.reward_wrapper.pretrained_path, device)
        logging.info("Learned reward data loaded successfully.")
    
    # Create environment function for vec_env
    if config.num_envs > 1:
        # Multiple parallel environments - pass factory function with partial args
        env_fns = []
        for i in range(config.num_envs):
            env_seed = base_seed + i if not config.env_randomization == "same-seed" else base_seed
            env_fns.append(
                functools.partial(_make_env_wrapper, config, env_seed, rank=i, train_flag=True, learned_reward_data=learned_reward_data, exp_dir=exp_dir)
            )
        env = SubprocVecEnv(env_fns)
        # Wrap with VecMonitor to track episode statistics
        env = VecMonitor(env, os.path.join(exp_dir, "train_monitor"))
    else:
        # Single environment
        env = _make_env_wrapper(config, base_seed, rank=0, train_flag=True, learned_reward_data=learned_reward_data, exp_dir=exp_dir)
        # Wrap with Monitor for episode statistics
        env = Monitor(env, os.path.join(exp_dir, "train_monitor"))
    
    # Create evaluation environment (with different seed)
    base_eval_env = _make_env_wrapper(config, eval_seed, rank=0, train_flag=False, learned_reward_data=learned_reward_data, exp_dir=exp_dir)
    eval_env = Monitor(base_eval_env, os.path.join(exp_dir, "eval_monitor"))

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
    
    # Append custom callback for logging to CSV and W&B
    callbacks.append(LogWandbCallback(exp_dir, config.log_frequency))
    
    # Add periodic evaluation and save callback
    callbacks.append(
        EvalSaveCallback(
            eval_env=eval_env,
            exp_dir=exp_dir,
            eval_freq=int(config.eval_frequency),
            checkpoint_freq=int(config.checkpoint_frequency),
            n_eval_episodes=int(config.num_eval_episodes),
            verbose=1,
            save_video=config.save_video,
            learned_reward=(config.reward_wrapper.type not in ["sparse", "env", "env_state-intrinsic"]),
            subgoal_reward=(config.reward_wrapper.type in ["subgoal_dist"]),
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