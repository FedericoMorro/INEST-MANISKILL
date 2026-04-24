# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch script for pre-training representations."""

import os
import os.path as osp

from absl import app
from absl import flags
from absl import logging
import copy
from ml_collections import config_flags
import torch
from torchkit import CheckpointManager
from torchkit import experiment
from torchkit import Logger
from torchkit.utils.py_utils import Stopwatch
from tqdm.rich import tqdm as rich_tqdm
from xirl import common
import wandb

from configs import validate_config
from inest_irl.utils.utils import setup_experiment

# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_name", None, "Experiment name.")
flags.DEFINE_integer("seed", 22, "RNG seed for experiment. Set to `none` to disable seeding.")
flags.DEFINE_boolean("resume", False, "Whether to resume training.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")
flags.DEFINE_boolean("raw_imagenet", False, "")
flags.DEFINE_boolean("wandb", False, "Log on W&B.")
flags.DEFINE_boolean("verbose", True, "Whether to print out training logs.")

config_flags.DEFINE_config_file(
    "config",
    "/home/fmorro/INEST-MANISKILL/scripts/configs/pretrain.py",
    "File path to the training hyperparameter configuration.",
)


@experiment.pdb_fallback
def main(_):
  # Make sure we have a valid config that inherits all the keys defined in the
  # base config.
  #validate_config(FLAGS.config, mode="pretrain")

  config = FLAGS.config
  exp_dir = osp.join(config.root_dir, FLAGS.experiment_name)
  setup_experiment(exp_dir, config, FLAGS.resume)

  # No need to do any pretraining if we're loading the raw pretrained
  # ImageNet baseline.
  if FLAGS.raw_imagenet:
    return
  
  if FLAGS.wandb:
    wandb.init(project="StackPyramidPretrain", group="Pretrain", name=FLAGS.experiment_name, mode="online")
    wandb.config.update(FLAGS)
    wandb.run.log_code(".")
    wandb.config.update(config.to_dict(), allow_val_change=True)

  # Setup compute device.
  if torch.cuda.is_available():
    device = torch.device(FLAGS.device)
  else:
    logging.info("No GPU device found. Falling back to CPU.")
    device = torch.device("cpu")
  logging.info("Using device: %s", device)

  # Set RNG seeds.
  config.seed = copy.deepcopy(FLAGS.seed)
  if config.seed is not None:
    logging.info("Pretraining experiment seed: %d", config.seed)
    experiment.seed_rngs(config.seed)
    experiment.set_cudnn(config.cudnn_deterministic, config.cudnn_benchmark)
  else:
    logging.info("No RNG seed has been set for this pretraining experiment.")

  logger = Logger(osp.join(exp_dir, "tb"), FLAGS.resume)
  
  # set num_frames_per_sequence to max number of frames in a video sequence in the dataset if None
  if config.frame_sampler.num_frames_per_sequence is None:
    max_num_frames = 0
    train_data_path = osp.join(config.data.root, "train")
    for class_name in os.listdir(train_data_path):
      class_path = osp.join(train_data_path, class_name)
      for traj_path in os.listdir(class_path):
        for frame_path in os.listdir(osp.join(class_path, traj_path)):
          if frame_path.endswith(".png"):
            num_frames = int(frame_path.split("_")[-1].split(".")[0])
            max_num_frames = max(max_num_frames, num_frames)
    config.frame_sampler.num_frames_per_sequence = max_num_frames
    logging.info(f"Set frame_sampler.num_frames_per_sequence to {max_num_frames} based on the maximum number of frames in a video sequence in the dataset.")

  # Load factories.
  (
      model,
      optimizer,
      pretrain_loaders,
      downstream_loaders,
      trainer,
      eval_manager,
  ) = common.get_factories(config, device)
  
  # Create checkpoint manager.
  checkpoint_dir = osp.join(exp_dir, "checkpoints")
  checkpoint_manager = CheckpointManager(
      checkpoint_dir,
      model=model,
      optimizer=optimizer,
  )
  
  if not FLAGS.verbose:
    prog_bar = rich_tqdm(total=config.optim.train_max_iters, desc="Pretraining", unit="iter")

  global_step = checkpoint_manager.restore_or_initialize()
  total_batches = max(1, len(pretrain_loaders["train"]))
  epoch = int(global_step / total_batches)
  complete = False
  stopwatch = Stopwatch()
  try:
    while not complete:
      for batch in pretrain_loaders["train"]:
        train_loss = trainer.train_one_iter(batch)

        if not global_step % config.logging_frequency:
          for k, v in train_loss.items():
            logger.log_scalar(v, global_step, k, "pretrain")
            if FLAGS.wandb:
              wandb.log(data={f"pretrain_train/{k}": v}, step=global_step)
          logger.flush()

        if not global_step % config.eval.eval_frequency:
          # Evaluate the model on the pretraining validation dataset.
          valid_loss = trainer.eval_num_iters(
              pretrain_loaders["valid"],
              config.eval.val_iters,
          )
          for k, v in valid_loss.items():
            logger.log_scalar(v, global_step, k, "pretrain")
            if FLAGS.wandb:
              wandb.log(data={f"pretrain_valid/{k}": v}, step=global_step)

          # Evaluate the model on the downstream datasets.
          for split, downstream_loader in downstream_loaders.items():
            eval_to_metric = eval_manager.evaluate(
                model,
                downstream_loader,
                device,
                config.eval.val_iters,
            )
            for eval_name, eval_out in eval_to_metric.items():
              eval_out.log(
                  logger,
                  global_step,
                  eval_name,
                  f"downstream/{split}",
              )
              if FLAGS.wandb:
                eval_out.log_wandb(
                    wandb, global_step, eval_name, f"downstream_{split}"
                )
              
        # Save model checkpoint.
        if not global_step % config.checkpointing_frequency:
          checkpoint_manager.save(global_step)
          
        if not FLAGS.verbose:
          prog_bar.update(1)
          
        # Exit if complete.
        global_step += 1
        if global_step > config.optim.train_max_iters:
          complete = True
          break

        if FLAGS.verbose:
          time_per_iter = stopwatch.elapsed()
          remaining_time = time_per_iter * (config.optim.train_max_iters - global_step)
          logging.info(
              "Iter[{}/{}] (Epoch {}), {:.6f}s/iter (rem: {:.0f}m{:02.0f}s), Loss: {:.3f}".format(
                  global_step,
                  config.optim.train_max_iters,
                  epoch,
                  time_per_iter,
                  remaining_time // 60,
                  remaining_time % 60,
                  train_loss["train/total_loss"],
              ))
          
        if FLAGS.wandb:
          wandb.log({
              "train/total_loss": train_loss["train/total_loss"],
              "step": global_step,
              "epoch": epoch,
          }, step=global_step)
          if "reds" in FLAGS.experiment_name:
            wandb.log({
                "train_reds/epic_loss": train_loss["train/epic_loss"],
                "train_reds/supcon_loss": train_loss["train/supcon_loss"],
                "step": global_step,
                "epoch": epoch,
            }, step=global_step)
          if "holdr" in FLAGS.experiment_name:
            wandb.log({
                "train_holdr/contrastive_loss": train_loss["train/contrastive_loss"],
                "train_holdr/holdr_loss": train_loss["train/holdr_loss"],
                "train_holdr/distance_frames_before_subtask_loss": train_loss["train/distance_frames_before_subtask_loss"],
                "train_holdr/distance_subtask_means_loss": train_loss["train/distance_subtask_means_loss"],
                "step": global_step,
                "epoch": epoch,
            }, step=global_step)

          if not global_step % config.eval.eval_frequency:
            wandb.log({
              "evaluation loss": valid_loss["valid/total_loss"],
              "step": global_step,
              "epoch": epoch,
            }, step=global_step)
            if "reds" in FLAGS.experiment_name:
              wandb.log({
                  "valid_reds/epic_loss": valid_loss["valid/epic_loss"],
                  "valid_reds/supcon_loss": valid_loss["valid/supcon_loss"],
                  "step": global_step,
                  "epoch": epoch,
              }, step=global_step)
            if "holdr" in FLAGS.experiment_name:
              wandb.log({
                  "valid_holdr/contrastive_loss": valid_loss["valid/contrastive_loss"],
                  "valid_holdr/holdr_loss": valid_loss["valid/holdr_loss"],
                  "valid_holdr/distance_frames_before_subtask_loss": valid_loss["valid/distance_frames_before_subtask_loss"],
                  "valid_holdr/distance_subtask_means_loss": valid_loss["valid/distance_subtask_means_loss"],
                  "step": global_step,
                  "epoch": epoch,
              }, step=global_step)
              
        stopwatch.reset()
      epoch += 1

  except KeyboardInterrupt:
    logging.info("Caught keyboard interrupt. Saving model before quitting.")

  finally:
    checkpoint_manager.save(global_step)
    logger.close()
    if "reds" in FLAGS.experiment_name:     
      # --- SAVE THE FULL MODEL (including CLIP) ---
      # Save the model's state_dict (recommended)
      torch.save(model.state_dict(), osp.join(exp_dir, "reds_model.pth"))
      # Optionally, save the entire model (not recommended for long-term use)
      # torch.save(model, osp.join(exp_dir, "pretrained_model_full.pt"))


if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_name")
  app.run(main)
