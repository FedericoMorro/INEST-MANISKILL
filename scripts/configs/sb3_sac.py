import ml_collections


def get_config():
  """Returns default config."""
  config = ml_collections.ConfigDict()

  # ================================================= #
  # Placeholders.
  # ================================================= #
  # These values will be filled at runtime once the gym.Env is loaded.
  obs_dim = ml_collections.FieldReference(None, field_type=int)
  action_dim = ml_collections.FieldReference(None, field_type=int)
  action_range = ml_collections.FieldReference(None, field_type=tuple)

  # ================================================= #
  # Main parameters.
  # ================================================= #
  config.save_dir = "/home/fmorro/INEST-MANISKILL/experiments/sb3"

  # Set this to True to allow CUDA to find the best convolutional algorithm to
  # use for the given parameters. When False, cuDNN will deterministically
  # select the same algorithm at a possible cost in performance.
  config.cudnn_benchmark = True
  # Enforce CUDA convolution determinism. The algorithm itself might not be
  # deterministic so setting this to True ensures we make it repeatable.
  config.cudnn_deterministic = False

  config.env_name = "StackPyramid-v1custom"
  config.seed = 22

  # ================================================= #
  # Wrappers.
  # ================================================= #
  # Observation mode passed to ManiSkill gym.make (e.g., 'state', 'state_dict', 'rgbd').
  config.obs_mode = "state"

  config.action_repeat = 1
  config.frame_stack = 3

  config.reward_wrapper = ml_collections.ConfigDict()
  # config.reward_wrapper.pretrained_path = "/home/liannello/xirl/experiment_results/6Subtask/Allocentric_Pretrain/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper_ALLO_6Subtasks"
  config.reward_wrapper.pretrained_path = "/home/fmorro/INEST-MANISKILL/experiments/pretrain/batch-4"
  # Can be one of ['distance_to_goal', 'goal_classifier', 'inest', 'inest_knn', 'state_intrinsic', 'reds', 'env'].
  # Use 'env' for baseline SAC with standard environment rewards.
  config.reward_wrapper.type = "env"

  # Vector environment parameters for DDP
  config.num_envs = 32

  # ================================================= #
  # Training parameters.
  # ================================================= #
  config.num_train_steps = 30_000_000
  config.replay_buffer_capacity = 1_000_000
  config.num_seed_steps = 30_000
  config.num_eval_episodes = 50 #150
  config.eval_frequency = 1_000_000
  config.checkpoint_frequency = 1_000_000
  config.log_frequency = 1_000 #20_000
  config.save_video = True

  # ================================================= #
  # SAC parameters.
  # ================================================= #
  config.sac = ml_collections.ConfigDict()

  config.sac.obs_dim = obs_dim
  config.sac.action_dim = action_dim
  config.sac.action_range = action_range
  config.sac.discount = 0.995
  config.sac.init_temperature = 1.0
  config.sac.alpha_lr = 1e-4
  config.sac.alpha_betas = [0.9, 0.999]
  config.sac.actor_lr = 1e-4
  config.sac.actor_betas = [0.9, 0.999]
  config.sac.actor_update_frequency = 1
  config.sac.critic_lr = 3e-5
  config.sac.critic_betas = [0.9, 0.999]
  config.sac.critic_tau = 0.005
  config.sac.critic_target_update_frequency = 2
  config.sac.batch_size = 256
  config.sac.learnable_temperature = True
  config.sac.target_entropy = -3.5  # set to -|A| if None, if annealing it is the final value
  config.sac.action_noise_std = 0.0

  config.sac.anneal_target_entropy = True
  config.sac.start_entropy_anneal = 50_000
  config.sac.end_entropy_anneal = 40_000_000
  config.initial_target_entropy = 0.0

  # ================================================= #
  # Critic parameters.
  # ================================================= #
  config.sac.critic = ml_collections.ConfigDict()

  config.sac.critic.obs_dim = obs_dim
  config.sac.critic.action_dim = action_dim
  config.sac.critic.net_arch = [512, 256, 256]

  # ================================================= #
  # Actor parameters.
  # ================================================= #
  config.sac.actor = ml_collections.ConfigDict()

  config.sac.actor.obs_dim = obs_dim
  config.sac.actor.action_dim = action_dim
  config.sac.actor.net_arch = [512, 256, 256]
  config.sac.actor.log_std_bounds = [-5, 2]

  # ================================================= #

  return config
