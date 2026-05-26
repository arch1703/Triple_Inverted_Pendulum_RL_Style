# plot training curves from SB3 monitor CSVs
# usage: python local/plot_local.py [--runs dir1 dir2] [--mean_window 200] [--out local/figures]

from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


plt.rcParams.update({
    "figure.dpi": 200,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
})

_COLOURS_V1 = ["#F44336", "#E91E63", "#FF5722"]
_COLOURS_V2 = ["#2196F3", "#009688", "#3F51B5"]

_M_FORMATTER = mticker.FuncFormatter(
    lambda x, _: (f"{x/1e6:.1f}M" if x >= 1e6
                  else (f"{x/1e3:.0f}k" if x >= 1e3 else str(int(x))))
)


def parse_args():
    p = argparse.ArgumentParser(description="Plot SB3 training curves")
    p.add_argument("--runs", type=str, nargs="+", default=None,
                   help="paths to run dirs (default: all in local/runs/)")
    p.add_argument("--mean_window", type=int, default=200)
    p.add_argument("--out", type=str, default="local/figures")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _rolling_mean(series: np.ndarray, window: int) -> np.ndarray:
    w = min(window, len(series))
    kernel = np.ones(w) / w
    padded = np.pad(series, w // 2, mode="edge")
    result = np.convolve(padded, kernel, mode="valid")
    return result[:len(series)]


def load_monitor_csvs(run_dir: str):
    csv_files = glob.glob(os.path.join(run_dir, "**", "*.monitor.csv"), recursive=True)
    if not csv_files:
        csv_files = glob.glob(os.path.join(run_dir, "*.csv"))
    if not csv_files:
        return None

    dfs = []
    for f in sorted(csv_files):
        try:
            df = pd.read_csv(f, comment="#")
            if {"r", "l", "t"}.issubset(df.columns):
                dfs.append(df[["r", "l", "t"]])
        except Exception:
            pass

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values("t", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    # X-axis: cumulative simulator timesteps (sum of episode lengths)
    combined["steps"] = combined["l"].cumsum()
    return combined


def _plot_single_run(ax, steps, values, colour, label, mean_window):
    ax.plot(steps, values, color=colour, alpha=0.12, linewidth=0.6, zorder=1)
    mean_line = _rolling_mean(values, mean_window)
    ax.plot(steps, mean_line, color=colour, linewidth=2.2, label=label, zorder=3)


def plot_metric(run_data, col, ylabel, title, out_path, mean_window, show, colour_pool=None):
    if colour_pool is None:
        colour_pool = _COLOURS_V2

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title(title)
    ax.set_xlabel("Simulator Timesteps")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(_M_FORMATTER)

    for idx, (label, df) in enumerate(run_data.items()):
        colour = colour_pool[idx % len(colour_pool)]
        nice = label.replace("triple_pendulum_ppo_", "").replace("_", " ")
        _plot_single_run(ax, df["steps"].values, df[col].values, colour, nice, mean_window)

    ax.legend(loc="upper left", framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_comparison(v1_data, v2_data, col, ylabel, title, out_path, mean_window, show):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title(title)
    ax.set_xlabel("Simulator Timesteps")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(_M_FORMATTER)

    for idx, (label, df) in enumerate(v1_data.items()):
        colour = _COLOURS_V1[idx % len(_COLOURS_V1)]
        nice = label.replace("triple_pendulum_ppo_", "").replace("_", " ")
        _plot_single_run(ax, df["steps"].values, df[col].values, colour, f"v1 - {nice}", mean_window)

    for idx, (label, df) in enumerate(v2_data.items()):
        colour = _COLOURS_V2[idx % len(_COLOURS_V2)]
        nice = label.replace("triple_pendulum_ppo_v2_", "").replace("_", " ")
        _plot_single_run(ax, df["steps"].values, df[col].values, colour, f"v2 - {nice}", mean_window)

    ax.legend(loc="upper left", framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()

    run_dirs = args.runs or sorted(
        d for d in glob.glob(os.path.join("local", "runs", "*")) if os.path.isdir(d)
    )

    if not run_dirs:
        print("no run dirs found under local/runs/ - train first")
        sys.exit(0)

    v1_data = {}
    v2_data = {}

    for d in run_dirs:
        label = os.path.basename(d)
        df = load_monitor_csvs(d)
        if df is not None and len(df) > 0:
            if "v2" in label:
                v2_data[label] = df
            else:
                v1_data[label] = df
            print(f"  loaded {len(df):,} episodes from {label}")
        else:
            print(f"  warning: no monitor csv in {d}")

    if not v1_data and not v2_data:
        print("no monitor data found")
        sys.exit(0)

    os.makedirs(args.out, exist_ok=True)

    for version_label, data, colours, tag in [
        ("Baseline (v1)", v1_data, _COLOURS_V1, "v1"),
        ("Improved (v2)", v2_data, _COLOURS_V2, "v2"),
    ]:
        if not data:
            continue
        os.makedirs(os.path.join(args.out, tag), exist_ok=True)
        for col, ylabel, stem in [
            ("r", "Episode Reward", "training_reward"),
            ("l", "Episode Length (steps)", "training_length"),
        ]:
            plot_metric(
                data,
                col=col,
                ylabel=ylabel,
                title=f"Triple Inverted Pendulum - {ylabel} [{version_label}]",
                out_path=os.path.join(args.out, tag, f"{stem}.png"),
                mean_window=args.mean_window,
                show=args.show,
                colour_pool=colours,
            )

    if v1_data and v2_data:
        for col, ylabel, stem in [
            ("r", "Episode Reward", "comparison_reward"),
            ("l", "Episode Length (steps)", "comparison_length"),
        ]:
            plot_comparison(
                v1_data, v2_data,
                col=col,
                ylabel=ylabel,
                title=f"Triple Inverted Pendulum - {ylabel}: v1 vs v2",
                out_path=os.path.join(args.out, f"{stem}.png"),
                mean_window=args.mean_window,
                show=args.show,
            )

    print(f"\ndone - figures in {args.out}/")


if __name__ == "__main__":
    main()
