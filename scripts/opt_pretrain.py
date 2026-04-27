"""
Optuna hyperparameter sweep for pretraining learned reward function.
"""

import math
import os
import subprocess
import sys
import time
from typing import Dict, List

from absl import app
from absl import flags
import numpy as np
import optuna
import yaml

from inest_irl.utils.csv_logger import CSVLogger


FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_name", None,
                    "Path to pretrained representation run.")
flags.DEFINE_string("train_script_path", "/home/fmorro/INEST-MANISKILL/scripts/pretrain.py",
                    "Path to pretrain.py script to execute.")
flags.DEFINE_string("exp_root_dir", "/home/fmorro/INEST-MANISKILL/experiments/opt_pretrain",
                    "Root directory for experiments.")
flags.DEFINE_string("data_path", "/data/fmorro/inest-maniskill/dataset-rc-1000-states",
                    "Dataset root passed to trainer.")
flags.DEFINE_string("storage_path", None,
                    "Optional sqlite file path for Optuna storage. Defaults to <sweep_output_path>/optuna.db.")

flags.DEFINE_bool("wandb", False, "Enable W&B in each trial.")
flags.DEFINE_string("wandb_project_name", "StackPyramid-PretrainOptuna", "W&B project.")

flags.DEFINE_integer("seed", 22, "Seed for Optuna sampler.")
flags.DEFINE_integer("n_trials", 60, "Number of Optuna trials.")
flags.DEFINE_string("study_name", "pretrain_sweep", "Optuna study name.")
flags.DEFINE_boolean("overwrite", False, "Whether to overwrite existing experiment directory.")

#flags.DEFINE_integer("epochs_min", 50, "Lower bound for epochs.")
#flags.DEFINE_integer("epochs_max", 400, "Upper bound for epochs.")
flags.DEFINE_integer("epochs_step", 25, "Epoch step.")
flags.DEFINE_integer("batch_size_min", 4, "Lower bound power-of-two batch size.")
flags.DEFINE_integer("batch_size_max", 4, "Upper bound power-of-two batch size.")
flags.DEFINE_float("learning_rate_min", 1e-6, "Lower log-uniform bound.")
flags.DEFINE_float("learning_rate_max", 1e-3, "Upper log-uniform bound.")
flags.DEFINE_float("weight_decay_min", 1e-6, "Lower log-uniform bound.")
flags.DEFINE_float("weight_decay_max", 1e-3, "Upper log-uniform bound.")
flags.DEFINE_integer("emb_size_min", 32, "Lower bound for model embedding size.")
flags.DEFINE_integer("emb_size_max", 256, "Upper bound for model embedding size.")

flags.DEFINE_bool("continue_on_trial_failure", True,
    "If True, continue study when a trial raises RuntimeError/IO/schema errors; failed trial is recorded by Optuna.")

flags.DEFINE_bool("enable_pruning", True,
    "Enable Optuna pruning by polling eval_loss from eval_log.csv during training.")
flags.DEFINE_integer("prune_check_interval", 30,
    "Seconds between pruning checks when --enable_pruning is True.")


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _power_of_two_choices(lower: int, upper: int) -> List[int]:
    if lower > upper:
        raise ValueError(f"Invalid power-of-two range: lower={lower}, upper={upper}.")
    if not _is_power_of_two(lower) or not _is_power_of_two(upper):
        raise ValueError(f"Power-of-two bounds required, got lower={lower}, upper={upper}. Use powers of two (e.g. 128..2048).")
    low_exp = int(math.log2(lower))
    high_exp = int(math.log2(upper))
    return [2**exp for exp in range(low_exp, high_exp + 1)]


def _read_log_excerpt(log_path: str, max_chars: int = 5000) -> str:
    if not os.path.isfile(log_path):
        return f"[missing log file: {log_path}]"
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _objective(train_script: str, out_dir: str, batch_choices: List[int], emb_size_choices: List[int]):
    
    def _run(trial: optuna.trial.Trial) -> float:
        params: Dict[str, object] = {
            #"epochs": trial.suggest_int("epochs", FLAGS.epochs_min, FLAGS.epochs_max, step=FLAGS.epochs_step),
            "batch_size": trial.suggest_categorical("batch_size", batch_choices),
            "learning_rate": trial.suggest_float(
                "learning_rate", FLAGS.learning_rate_min, FLAGS.learning_rate_max, log=True
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", FLAGS.weight_decay_min, FLAGS.weight_decay_max, log=True
            ),
            "embedding_size": trial.suggest_categorical("embedding_size", emb_size_choices),
        }
        
        print(f"[TRIAL {trial.number}] Starting trial with params: {params}")
        cmd = [
            sys.executable,
            train_script,
            f"--config.root_dir={out_dir}",
            f"--experiment_name={trial.number:04d}",
            f"--seed={FLAGS.seed}",
            f"--config.data.root={FLAGS.data_path}",
            "--overwrite=True",
            #f"--config.optim.num_epochs={params['epochs']}",
            f"--config.data.batch_size={params['batch_size']}",
            f"--config.optim.lr={params['learning_rate']}",
            f"--config.optim.weight_decay={params['weight_decay']}",
            f"--config.model.embedding_size={params['embedding_size']}",
        ]
        if FLAGS.wandb:
            cmd.extend(
                [
                    "--wandb",
                    f"--wandb_project_name={FLAGS.wandb_project_name}",
                ]
            )

        trial_out = os.path.join(out_dir, f"{trial.number:04d}")
        trial_log_path = os.path.join(trial_out, "trial_stdout_stderr.log")
        eval_csv_path = os.path.join(trial_out, "eval_log.csv")
        os.makedirs(trial_out, exist_ok=True)
        
        # launch training as subprocess so we can poll for pruning
        with open(trial_log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        
        # poll eval_log.csv and report loss to Optuna for pruning decisions
        last_reported_step = -1
        while proc.poll() is None:
            try:
                logger = CSVLogger(eval_csv_path)
                latest_loss = logger.get_latest_value("loss", float)
                latest_step = logger.get_latest_value("step", int)
                if latest_loss is not None and latest_step is not None:
                    # only report when step advances
                    if latest_step > last_reported_step:
                        last_reported_step = latest_step
                        trial.report(latest_loss, latest_step)
                        if FLAGS.enable_pruning and trial.should_prune():
                            proc.terminate()
                            try:
                                proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            raise optuna.exceptions.TrialPruned()
            except FileNotFoundError:
                # eval_log.csv not created yet; keep polling.
                pass
            time.sleep(FLAGS.prune_check_interval)
        
        # check exit code after process completes.
        return_code = proc.returncode
        if return_code != 0:
            excerpt = _read_log_excerpt(trial_log_path)
            raise RuntimeError(
                f"Trial {trial.number} failed with return code {return_code}. "
                f"See log: {trial_log_path}\n"
                f"Recent log excerpt:\n"
                f"{excerpt}"
            )
            
        # read final metrics from eval CSV.
        eval_csv = CSVLogger(eval_csv_path)
        eval_steps = eval_csv.get_field_data("step", int)
        eval_loss = eval_csv.get_field_data("loss", float)
        
        # score combination of final validation loss and stdev of validation loss of last 2k steps
        filtered_losses = [l for s, l in zip(eval_steps, eval_loss) if s >= eval_steps[-1] - 2000]
        loss_std = np.std(filtered_losses)
        objective_val = 0.8 * eval_loss[-1] + 0.2 * loss_std
        
        trial.set_user_attr("final_val_loss", eval_loss[-1])
        trial.set_user_attr("val_loss_std_last_2k_steps", loss_std)
        trial.set_user_attr("objective_val", objective_val)
        trial.set_user_attr("trial_output_path", trial_out)
        
        return objective_val

    return _run


def main(_):
    out_dir = os.path.join(FLAGS.exp_root_dir, FLAGS.experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    train_script = FLAGS.train_script_path
    if not os.path.isfile(train_script):
        raise FileNotFoundError(f"Training script not found at {train_script}")

    batch_choices = _power_of_two_choices(FLAGS.batch_size_min, FLAGS.batch_size_max)
    emb_size_choices = _power_of_two_choices(FLAGS.emb_size_min, FLAGS.emb_size_max)

    storage_path = (
        os.path.abspath(FLAGS.storage_path)
        if FLAGS.storage_path
        else os.path.join(out_dir, "optuna.db")
    )
    
    if os.path.exists(storage_path) and FLAGS.overwrite:
        os.remove(storage_path)
        print(f"[SWEEP] Existing Optuna storage at {storage_path} removed due to --overwrite flag.")
    
    sampler = optuna.samplers.TPESampler(seed=FLAGS.seed, multivariate=True)
    study = optuna.create_study(
        direction="minimize",
        study_name=FLAGS.study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=sampler,
    )
    catch_exceptions = (
        (RuntimeError, FileNotFoundError, KeyError, TypeError, ValueError)
        if FLAGS.continue_on_trial_failure
        else ()
    )
    if FLAGS.enable_pruning:
        print("[SWEEP] Pruning enabled. Trials will be stopped if the current trial loss is significantly worse than the median of previous trials.")
    study.optimize(
        _objective(train_script=train_script, out_dir=out_dir, batch_choices=batch_choices, emb_size_choices=emb_size_choices),
        n_trials=FLAGS.n_trials,
        show_progress_bar=True,
        catch=catch_exceptions,
    )

    best = study.best_trial
    result = {
        "best_trial_number": int(best.number),
        "best_objective_value": float(best.value),
        "best_params": dict(best.params),
        "best_objective_value": float(best.user_attrs["objective_val"]),
        "best_final_val_loss": float(best.user_attrs["final_val_loss"]),
        "best_val_loss_std_last_2k_steps": float(best.user_attrs["val_loss_std_last_2k_steps"]),
        "trial_output_path": str(best.user_attrs["trial_output_path"]),
        "total_trials_in_study": int(len(study.trials)),
        "study_name": FLAGS.study_name,
        "storage_path": storage_path,
    }
    out_path = os.path.join(out_dir, "best_result.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    print(f"[SWEEP] Done. Best objective={result['best_objective_value']:.6f}")
    print(f"[SWEEP] Best final validation loss={result['best_final_val_loss']:.6f}")
    print(f"[SWEEP] Best validation loss std (last 2k steps)={result['best_val_loss_std_last_2k_steps']:.6f}")
    print(f"[SWEEP] Best params={result['best_params']}")
    print(f"[SWEEP] Saved summary to {out_path}")


if __name__ == "__main__":
    flags.mark_flags_as_required([
        "experiment_name",
        "train_script_path",
        "exp_root_dir",
        "data_path"
    ])
    app.run(main)