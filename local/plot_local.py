"""
Plot Training Curves – Triple Inverted Pendulum (local MuJoCo runs)
====================================================================
Reads SB3 VecMonitor CSV logs from local/runs/ and produces
presentation-quality figures.

Usage (from repo root):
    python local/plot_local.py                        # all seeds found in local/runs/
    python local/plot_local.py --runs local/runs/triple_pendulum_ppo_seed42
    python local/plot_local.py --smooth 20 --out local/figures/

Outputs:
    training_reward.png   – episode reward vs environment steps
    training_length.png   – episode length vs environment steps
"""

from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import pandas as pd


# ---- matplotlib style -------------------------------------------------------
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   1.5,
})

# Colour palette – one per seed/run
_COLOURS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot SB3 training curves")
    p.add_argument("--runs",   type=str, nargs="+", default=None,
                   help="Paths to run directories (default: all in local/runs/)")
    p.add_argument("--smooth", type=int, default=10,
                   help="Exponential moving-average window for smoothing")
    p.add_argument("--out",    type=str, default="local/figures",
                   help="Output directory for saved figures")
    p.add_argument("--show",   action="store_true",
                   help="Also open interactive plot window")
    return p.parse_args()


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (mimics pandas ewm)."""
    alpha = 2.0 / (span + 1)
    result = np.zeros_like(series, dtype=float)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def load_monitor_csvs(run_dir: str) -> pd.DataFrame | None:
    """Load and concatenate all monitor.csv files inside a run directory."""
    csv_files = glob.glob(os.path.join(run_dir, "**", "*.monitor.csv"),
                          recursive=True)
    # SB3 VecMonitor writes e.g. monitor.monitor.csv
    if not csv_files:
        csv_files = glob.glob(os.path.join(run_dir, "*.csv"), recursive=False)
    if not csv_files:
        return None

    dfs = []
    for f in sorted(csv_files):
        try:
            df = pd.read_csv(f, comment="#")
            # VecMonitor columns: r, l, t
            if {"r", "l", "t"}.issubset(df.columns):
                dfs.append(df[["r", "l", "t"]])
        except Exception:
            pass

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values("t", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    # Approximate env steps (cumulative episode lengths)
    combined["steps"] = combined["l"].cumsum()
    return combined


def plot_metric(
    run_data: dict[str, pd.DataFrame],
    col: str,
    ylabel: str,
    title: str,
    out_path: str,
    smooth: int,
    show: bool,
):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)

    for idx, (label, df) in enumerate(run_data.items()):
        colour = _COLOURS[idx % len(_COLOURS)]
        steps  = df["steps"].values
        values = df[col].values

        # Raw (faint)
        ax.plot(steps, values, color=colour, alpha=0.15, linewidth=0.8)
        # Smoothed
        smoothed = _ema(values, smooth)
        ax.plot(steps, smoothed, color=colour, label=label)

    if len(run_data) > 1:
        ax.legend(loc="upper left")

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()

    # Discover run directories
    if args.runs:
        run_dirs = args.runs
    else:
        run_dirs = sorted(
            d for d in glob.glob(os.path.join("local", "runs", "*"))
            if os.path.isdir(d)
        )

    if not run_dirs:
        print("No run directories found under local/runs/. Train first.")
        sys.exit(0)

    # Load data
    run_data: dict[str, pd.DataFrame] = {}
    for d in run_dirs:
        label = os.path.basename(d)
        df = load_monitor_csvs(d)
        if df is not None and len(df) > 0:
            run_data[label] = df
            print(f"  Loaded {len(df):,} episodes from {label}")
        else:
            print(f"  WARNING: no monitor CSV found in {d}")

    if not run_data:
        print("No monitor data found. Did training finish at least one episode?")
        sys.exit(0)

    os.makedirs(args.out, exist_ok=True)

    plot_metric(
        run_data,
        col      = "r",
        ylabel   = "Episode reward",
        title    = "Triple Inverted Pendulum – Training Reward",
        out_path = os.path.join(args.out, "training_reward.png"),
        smooth   = args.smooth,
        show     = args.show,
    )

    plot_metric(
        run_data,
        col      = "l",
        ylabel   = "Episode length (steps)",
        title    = "Triple Inverted Pendulum – Episode Length",
        out_path = os.path.join(args.out, "training_length.png"),
        smooth   = args.smooth,
        show     = args.show,
    )

    print(f"\nFigures written to: {args.out}/")


if __name__ == "__main__":
    main()
