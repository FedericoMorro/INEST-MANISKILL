import h5py
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

def inspect_h5_file(filepath, visualize=False, plot_traj=False, write_stats=False, output_dir="out"):
    """Inspect H5 file structure and display demo information."""
    
    with h5py.File(filepath, 'r') as f:
        print(f"H5 File: {filepath}")
        print(f"Number of demos: {len(f.keys())}")
        print("\nDemo structure:")
        
        # Get first demo only
        demo_name = list(f.keys())[0]
        demo = f[demo_name]
        print(f"\n  {demo_name}:")
        print(f"    Keys: {list(demo.keys())}")
        
        for key in demo.keys():
            item = demo[key]
            _display_item_recursive(item, key, indent=6)
        
        # Visualize images if requested
        if visualize:
            _visualize_images_from_demo(demo, output_dir=output_dir)

        # Plot trajectory statistics if requested
        if plot_traj:
            _plot_trajectory_stats(demo, output_dir=output_dir)

        # Write dataset stats if requested
        if write_stats:
            _write_dataset_mse_stats(h5_file=f, output_dir=output_dir)


def _display_item_recursive(item, key, indent=6):
    """Recursively display item structure."""
    if isinstance(item, h5py.Dataset):
        print(f"{' ' * indent}{key}: shape={item.shape}, dtype={item.dtype}")
    elif isinstance(item, h5py.Group):
        print(f"{' ' * indent}{key}: Group with keys {list(item.keys())}")
        for subkey in item.keys():
            subitem = item[subkey]
            _display_item_recursive(subitem, subkey, indent + 2)


def _visualize_images_from_demo(demo, output_dir):
    """Visualize the first image from each camera observation."""
    try:
        import matplotlib.pyplot as plt

        imgs = []
        
        # Navigate to sensor_data -> base_camera and hand_camera
        if 'obs' in demo and isinstance(demo['obs'], h5py.Group):
            obs = demo['obs']
            if 'sensor_data' in obs and isinstance(obs['sensor_data'], h5py.Group):
                sensor_data = obs['sensor_data']
                for camera_name in ['base_camera', 'hand_camera']:
                    if camera_name in sensor_data:
                        camera = sensor_data[camera_name]
                        if 'rgb' in camera:
                            rgb_data = camera['rgb'][()]
                            img = rgb_data[0]  # First frame
                            imgs.append((camera_name, img))
        
        if imgs:
            fig, axes = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 5))
            if len(imgs) == 1:
                axes = [axes]  # Ensure axes is iterable
            for ax, (camera_name, img) in zip(axes, imgs):
                ax.imshow(img)
                ax.set_title(camera_name)
                ax.axis('off')
            plt.savefig(os.path.join(output_dir, "demo_images.png"), bbox_inches='tight', dpi=300)
            plt.close()
        else:
            print("No images found in the demo for visualization.")

    except ImportError:
        print("matplotlib not installed for visualization")


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
    plt.figure(figsize=(12, 6))
    timesteps = np.arange(series.shape[0])
    for dim in range(series.shape[1]):
        plt.plot(timesteps, series[:, dim], label=f"{label_prefix}[{dim}]")

    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect H5 file structure")
    parser.add_argument("filepath", type=str, help="Path to H5 file")
    parser.add_argument("--vis", action="store_true", help="Visualize images")
    parser.add_argument("--traj", action="store_true", help="Create trajectory plot images")
    parser.add_argument("--stats", action="store_true", help="Create txt file with dataset stats")
    parser.add_argument("--output_path", type=str, default="out", help="Output path for replayed trajectory data and videos (default: same directory as input file)")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    out_path = os.path.join(args.output_path, f"{args.filepath.replace('/', '_').replace('.h5', '_visualizations')}")
    os.makedirs(out_path, exist_ok=True)
    args.output_path = out_path
    print(f"Output path for visualizations and stats: {args.output_path}")

    inspect_h5_file(
        args.filepath,
        visualize=args.vis,
        plot_traj=args.traj,
        write_stats=args.stats,
        output_dir=args.output_path,
    )
