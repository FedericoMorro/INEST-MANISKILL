import json
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_actions(path):
    p = Path(path)
    with p.open("r") as f:
        data = json.load(f)
    acts = data.get("actions", None)
    if acts is None:
        raise KeyError("JSON does not contain key 'actions'")
    return np.array(acts, dtype=float)


def plot_actions(actions, out=None, title=None):
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[:, None]
    steps = np.arange(actions.shape[0])
    plt.figure(figsize=(10, 4))
    for i in range(actions.shape[1]):
        plt.plot(steps, actions[:, i], label=f"a{i}")
    plt.xlabel("step")
    plt.ylabel("action")
    if title:
        plt.title(title)
    plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=150)
        print(f"Saved plot to {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot actions from JSON file")
    parser.add_argument("file", help="path to JSON file containing 'actions'")
    parser.add_argument("--out", help="output image path (png). If omitted, same folder and same basename of input.")
    args = parser.parse_args()
    acts = load_actions(args.file)
    out = args.out
    if out is None:
        # save next to input JSON, same basename, .png extension
        out = Path(args.file).with_suffix(".png")
    plot_actions(acts, out=str(out), title=Path(args.file).name)


if __name__ == "__main__":
    main()
