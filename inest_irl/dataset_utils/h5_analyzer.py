"""
Example usage:

python inest_irl/dataset_utils/h5_analyzer.py
	../data/maniskill/StackPyramid-v1_.../trajectory...h5
    [--print_structure]         file structure inspection, it will exit (it is printed by default even in other modes)
	[--vis]
	[--sample_traj]
	[--stats]
	[--rewards]
    [--no_plot_all_rew]
	[--output_path out]
	[--subgoals path/to/subgoal/otherwise/search/data/folder]
"""

import argparse
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from tqdm import tqdm
import warnings
from PIL import Image, ImageDraw, ImageFont

try:
    import imageio.v2 as imageio
except Exception:
    import imageio

IMAGE_KEY = 'obs/sensor_data/base_camera/rgb'
FPS = 10

# report-friendly plotting defaults (compact figure size with readable text)
FIGSIZE_TRAJ = (7.0, 3.6)
FIGSIZE_MEAN = (7.0, 3.6)
FS_LABEL = 12
FS_TITLE = 13
FS_LEGEND = 9


def _display_item_recursive(item, key, indent=6):
    """Recursively display item structure."""
    if isinstance(item, h5py.Dataset):
        print(f"{' ' * indent}{key}: shape={item.shape}, dtype={item.dtype}")
    elif isinstance(item, h5py.Group):
        print(f"{' ' * indent}{key}: Group with keys {list(item.keys())}")
        for subkey in item.keys():
            subitem = item[subkey]
            _display_item_recursive(subitem, subkey, indent + 2)


def _dataset_to_timeseries(array):
    """Convert an array to shape [T, D] where T is timesteps and D is dimensions."""
    if array.ndim == 0:
        return None
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array.reshape(array.shape[0], -1)


def _is_numerical_dataset(path, dataset, exclude_visual_tokens=False):
    """Check if a dataset should be treated as numerical trajectory data."""
    name = path.lower()
    if exclude_visual_tokens and any(token in name for token in ["rgb", "depth", "seg", "image", "camera", "pointcloud"]):
        return False

    if not np.issubdtype(dataset.dtype, np.number):
        return False

    # Most image tensors are high-dimensional and/or uint8.
    if dataset.ndim >= 3 and dataset.dtype == np.uint8:
        return False

    return dataset.ndim >= 1


def _collect_numeric_datasets(group, prefix, exclude_visual_tokens=False):
    """Recursively collect numerical leaf datasets from a group."""
    collected = {}
    for key in group.keys():
        item = group[key]
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Group):
            collected.update(_collect_numeric_datasets(item, prefix=path, exclude_visual_tokens=exclude_visual_tokens))
        elif isinstance(item, h5py.Dataset) and _is_numerical_dataset(path, item, exclude_visual_tokens=exclude_visual_tokens):
            arr = item[()]
            if isinstance(arr, np.ndarray):
                ts = _dataset_to_timeseries(arr)
                if ts is not None:
                    collected[path] = ts
    return collected


def _plot_single_series(title, series, label_prefix, output_dir):
    """Plot one figure for a single numerical key, with one line per dimension."""
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_TRAJ)
    timesteps = np.arange(series.shape[0])
    for dim in range(series.shape[1]):
        ax.plot(timesteps, series[:, dim], label=f"{label_prefix}[{dim}]", linewidth=1.6)

    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold')
    ax.set_xlabel("Timestep", fontsize=FS_LABEL)
    ax.set_ylabel("Value", fontsize=FS_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND)
    plt.tight_layout()
    filename = f"{title.lower().replace(':', '').replace('/', '-').replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()


def _plot_trajectory_stats(demo, output_dir):
    """Plot one figure per numerical key for a single demo trajectory."""
    series_by_name = _collect_all_numerical_series(demo)
    if not series_by_name:
        print("No numerical trajectory data found for plotting.")
        return

    for signal_name in sorted(series_by_name.keys()):
        _plot_single_series(f"Trajectory: {signal_name}", series_by_name[signal_name], signal_name, output_dir=output_dir)
    print(f"Saved {len(series_by_name)} trajectory plot image(s) to: {output_dir}")


def _collect_all_numerical_series(demo):
    """Collect all numerical trajectory series in one map keyed by signal name."""
    series = {}

    if "actions" in demo and isinstance(demo["actions"], h5py.Dataset):
        ts = _dataset_to_timeseries(np.asarray(demo["actions"][()]))
        if ts is not None:
            series["actions"] = ts
    elif "actions" in demo and isinstance(demo["actions"], h5py.Group):
        series.update(_collect_numeric_datasets(demo["actions"], prefix="actions"))

    if "env_states" in demo and isinstance(demo["env_states"], h5py.Dataset):
        ts = _dataset_to_timeseries(np.asarray(demo["env_states"][()]))
        if ts is not None:
            series["env_states"] = ts
    elif "env_states" in demo and isinstance(demo["env_states"], h5py.Group):
        series.update(_collect_numeric_datasets(demo["env_states"], prefix="env_states"))

    if "obs" in demo and isinstance(demo["obs"], h5py.Group):
        series.update(_collect_numeric_datasets(demo["obs"], prefix="obs", exclude_visual_tokens=True))

    if "rewards" in demo and isinstance(demo["rewards"], h5py.Dataset):
        ts = _dataset_to_timeseries(np.asarray(demo["rewards"][()]))
        if ts is not None:
            series["rewards"] = ts

    return series


def _write_dataset_mse_stats(h5_file, output_dir):
    """Write dataset-level MSE stats for numerical signals to a text file."""
    out_file = os.path.join(output_dir, "dataset_mse_stats.txt")
    out_f = open(out_file, "w")

    demo_names = list(h5_file.keys())
    if not demo_names:
        out_f.write("No demos found in the H5 file.\n")
        return

    mse_by_signal = {}

    for demo_name in demo_names:
        demo = h5_file[demo_name]
        series_by_name = _collect_all_numerical_series(demo)
        if not series_by_name:
            continue

        for signal_name in sorted(series_by_name.keys()):
            series = np.asarray(series_by_name[signal_name], dtype=np.float64)
            if series.shape[0] == 0:
                continue

            mse_per_dim = np.mean(np.square(series), axis=0)
            mse_by_signal.setdefault(signal_name, []).append(mse_per_dim)

    out_f.write("\nDataset-level MSE stats:\n")
    if not mse_by_signal:
        out_f.write("  No numerical MSE stats available.\n")
        out_f.close()
        return

    for signal_name in sorted(mse_by_signal.keys()):
        values = mse_by_signal[signal_name]
        dim_sizes = {v.shape[0] for v in values}

        out_f.write(f"\n  {signal_name}:")
        if len(dim_sizes) != 1:
            out_f.write("    Skipped dataset aggregation due to inconsistent dimensionality across demos.\n")
            continue

        stacked = np.stack(values, axis=0)
        dataset_mse = np.mean(stacked, axis=0)

        # MSE computed again on the per-trajectory MSE values.
        mse_of_mse = np.mean(np.square(stacked), axis=0)

        # Also report dispersion of per-trajectory MSE around dataset-level MSE.
        mse_of_mse_around_mean = np.mean(np.square(stacked - dataset_mse), axis=0)

        overall_dataset_mse = float(np.mean(dataset_mse))
        overall_mse_of_mse = float(np.mean(mse_of_mse))
        overall_mse_of_mse_around_mean = float(np.mean(mse_of_mse_around_mean))

        out_f.write(f"    demos counted = {stacked.shape[0]}\n")
        out_f.write(f"    dataset MSE per dim = {_format_array(dataset_mse)}\n")
        out_f.write(f"    overall dataset MSE (mean over dims) = {overall_dataset_mse:.6f}\n")
        out_f.write(f"    MSE of resulting MSE per dim = {_format_array(mse_of_mse)}\n")
        out_f.write(f"    overall MSE of resulting MSE (mean over dims) = {overall_mse_of_mse:.6f}\n")
        out_f.write(f"    MSE of resulting MSE around dataset mean per dim = {_format_array(mse_of_mse_around_mean)}\n")
        out_f.write(
            f"    overall MSE of resulting MSE around dataset mean (mean over dims) = "
            f"{overall_mse_of_mse_around_mean:.6f}\n"
        )

    out_f.close()
    print(f"Saved dataset stats to: {out_file}")

def _format_array(arr):
    """Format a 1D array as a string with 6 decimal places."""
    return "[" + ", ".join(f"{x:.3f}" for x in arr) + "]"


def _get_nested(h5obj, path):
    keys = path.split('/')
    obj = h5obj
    for k in keys:
        if k not in obj:
            return None
        obj = obj[k]
    return obj


def _save_demo_videos(h5_file, demo_names, output_dir, subgoals_path=None):
    print("Saving demo videos with subgoal overlays (if provided)...")
    out_dir = os.path.join(output_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)

    subgoals_data = json.load(open(subgoals_path, 'r')) if subgoals_path is not None else None

    for demo_name in tqdm(demo_names, desc="Demos"):
        demo = h5_file[demo_name]
        demo_idx = demo_name.split("_")[-1]
        imgs = _get_nested(demo, IMAGE_KEY)
        imgs = np.asarray(imgs)

        # write video
        video_path = os.path.join(out_dir, f"{demo_idx}.mp4")
        writer = imageio.get_writer(video_path, fps=FPS)

        sub_idxs = subgoals_data.get(demo_idx, []) if subgoals_data is not None else []
        # create frames and write
        for i in range(imgs.shape[0]):
            frame = imgs[i]
            # frame: numpy array HxWxC (uint8)
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            # resize frame to 512x512 for better visualization
            frame = np.array(Image.fromarray(frame).resize((512, 512)))

            # add subgoal overlay if this frame is a subgoal frame
            if len(sub_idxs) == 0:
                writer.append_data(frame)
            else:
                subgoal_num = sum(1 for idx in sub_idxs if idx <= i)
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 36)
                except Exception:
                    font = ImageFont.load_default()
                draw.text((16, 16), f"Subgoal: {subgoal_num}", fill=(255, 255, 255), font=font)
                img = np.asarray(img)
                
                writer.append_data(img.astype(np.uint8))

        writer.close()


def _plot_reward_curves(h5_file, demo_names, subgoals_data=None, output_dir=None, no_plot_all_rew=False):
    print("Plotting reward curves for all demos...")
    out_dir = os.path.join(output_dir, "reward_curves")
    os.makedirs(out_dir, exist_ok=True)

    all_rewards = []
    for demo_name in tqdm(demo_names, desc="Demos"):
        demo = h5_file[demo_name]
        demo_idx = demo_name.split("_")[-1]
        rewards = _get_nested(demo, "rewards")
        if rewards is None:
            continue
        rewards = np.asarray(rewards)
        all_rewards.append(rewards)

        if no_plot_all_rew:
            continue

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_TRAJ)
        timesteps = np.arange(rewards.shape[0])
        ax.plot(timesteps, rewards, label="Reward", linewidth=2)

        sub_idxs = subgoals_data.get(demo_idx, []) if subgoals_data is not None else []
        for idx in sub_idxs:
            ax.axvline(x=idx, color='crimson', linestyle=':', linewidth=1.8, alpha=0.9, label="Subgoal" if idx == sub_idxs[0] else None)

        ax.set_title(f"Trajectory {demo_idx} (Length: {rewards.shape[0]})", fontsize=FS_TITLE, fontweight='bold')
        ax.set_xlabel("Timestep", fontsize=FS_LABEL)
        ax.set_ylabel("Reward", fontsize=FS_LABEL)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FS_LEGEND)
        plt.tight_layout()
        filename = f"{demo_idx}.png"
        plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()

    if len(all_rewards) > 0:
        all_rewards = np.array([r for r in all_rewards], dtype=object)
        max_len = max(len(r) for r in all_rewards)
        rewards_padded = np.full((len(all_rewards), max_len), np.nan)
        for i, r in enumerate(all_rewards):
            rewards_padded[i, :len(r)] = r
        
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_MEAN)
        timesteps = np.arange(max_len)
        mean_rewards = np.nanmean(rewards_padded, axis=0)
        std_rewards = np.nanstd(rewards_padded, axis=0)
        
        ax.plot(timesteps, mean_rewards, label="Mean Reward", linewidth=2)
        ax.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, label="+-1 Std Dev")
        
        ax.set_title(f"Mean Reward per Timestep (N={len(all_rewards)} trajectories)", fontsize=FS_TITLE, fontweight='bold')
        ax.set_xlabel("Timestep", fontsize=FS_LABEL)
        ax.set_ylabel("Reward", fontsize=FS_LABEL)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FS_LEGEND)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "_avg_reward.png"), bbox_inches='tight', dpi=300)
        plt.close()

        min_rew, max_rew = np.inf, -np.inf
        end_traj_rews = []
        for r in all_rewards:
            r_final = r[-1] if len(r) > 0 else np.nan
            r_no_final = [x for x in r if x != r_final]
            min_rew = min(min_rew, np.nanmin(r_no_final))
            max_rew = max(max_rew, np.nanmax(r_no_final))
            end_traj_rews.append(r_no_final[-1])


        with open(os.path.join(out_dir, "_reward_summary.txt"), "w") as f:
            f.write(f"OVERALL\n")
            f.write(f"  # of trajs: {len(all_rewards)}\n")
            f.write(f"  Min reward: {min_rew:.3f}\n")
            f.write(f"  Max reward: {max_rew:.3f}\n")
            f.write(f"  Avg reward: {np.nanmean(mean_rewards):.3f}\n")
            f.write(f"  Std reward: {np.nanmean(std_rewards):.3f}\n")
            f.write(f"\nINIT - END\n")
            f.write(f"  Avg reward t = 0:   {mean_rewards[0]:.3f}\n")
            f.write(f"  Std reward t = 0:   {std_rewards[0]:.3f}\n")
            f.write(f"  Avg reward t = max: {np.nanmean(end_traj_rews):.3f}\n")
            f.write(f"  Std reward t = max: {np.nanstd(end_traj_rews):.3f}\n")
            f.write(f"\nNote: reward==1.0 removed, since it is the terminal reward\n")

    print(f"Saved reward curve plots to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect H5 file structure and optionally create visualizations/videos")
    parser.add_argument("filepath", type=str, help="Path to H5 file")
    parser.add_argument("--print_structure", action="store_true", help="Print the structure of the H5 file")
    parser.add_argument("--vis", action="store_true", help="Save per-trajectory videos")
    parser.add_argument("--sample_traj", action="store_true", help="Create trajectory plot images")
    parser.add_argument("--stats", action="store_true", help="Create txt file with dataset stats")
    parser.add_argument("--rewards", action="store_true", help="Plot reward curves for all demos")
    parser.add_argument("--no_plot_all_rew", action="store_true", help="Skip plotting reward curves for all demos (compute only summary stats and mean plot)")
    parser.add_argument("--output_path", type=str, default="out", help="Output directory for visualizations and stats")
    parser.add_argument("--subgoals", type=str, default=None, help="Path to subgoal_frame.json (optional, if not specified, same folder as h5 file will be checked)")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    subgoals_path = None
    if args.subgoals is None:
        # check if in same folder as h5 file
        dir_name = os.path.dirname(args.filepath)
        expected_path = os.path.join(dir_name, "subgoal_frames.json")
        if os.path.exists(expected_path):
            subgoals_path = expected_path

    if subgoals_path is not None:
        with open(subgoals_path, 'r') as f:
            subgoals_data = json.load(f)
        print(f"Loaded subgoal frame data from: {subgoals_path}")
    else:
        print("No subgoal frame data found: not specified nor found in same folder as h5 file")

    with h5py.File(args.filepath, 'r') as f:
        print(f"H5 File: {args.filepath}")
        print(f"Number of demos: {len(f.keys())}")
        print("\nDemo structure:")
        demo_names = list(f.keys())
        if len(demo_names) == 0:
            print("No demos found in the H5 file.")
        else:
            # inspect first demo structure
            demo_name = demo_names[0]
            demo = f[demo_name]
            print(f"\n  {demo_name}:")
            print(f"    Keys: {list(demo.keys())}")
            for key in demo.keys():
                item = demo[key]
                _display_item_recursive(item, key, indent=6)

        if args.print_structure:
            print("\nFile structure inspection complete. Exiting as --print_structure flag is set.")
            exit(0)

        # save videos if requested
        if args.vis:
            _save_demo_videos(f, demo_names, output_dir=args.output_path, subgoals_data=subgoals_path)

        # create output path for plots and stats
        dir_name = os.path.dirname(args.filepath).split("/")[-1]
        out_path = os.path.join(args.output_path, f"{dir_name}_viz")
        os.makedirs(out_path, exist_ok=True)
        print(f"Output path for visualizations and stats: {out_path}")

        # plot trajectory statistics if requested
        if args.sample_traj:
            plots_dir = os.path.join(out_path, "sample_plots")
            os.makedirs(plots_dir, exist_ok=True)
            # plot for first demo only (preserve prior behavior)
            _plot_trajectory_stats(f[demo_names[0]], output_dir=plots_dir)

        # write dataset stats if requested
        if args.stats:
            _write_dataset_mse_stats(h5_file=f, output_dir=out_path)

        # plot reward curves for all demos
        if args.rewards:
            _plot_reward_curves(f, demo_names,
                subgoals_data=subgoals_data, output_dir=out_path, no_plot_all_rew=args.no_plot_all_rew)

