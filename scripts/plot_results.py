# Generate figures for training curves and eval metrics (Isaac Lab pipeline)
# reads TensorBoard events and results/eval/*.json
# usage: python scripts/plot_results.py --run_dirs results/runs/seed0 results/runs/seed1 --eval_dir results/eval

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # headless rendering

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font="DejaVu Serif", font_scale=1.2)
PALETTE = sns.color_palette("colorblind", n_colors=4)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.figsize": (6.0, 4.0),
    "axes.spines.top": False,
    "axes.spines.right": False,
})

parser = argparse.ArgumentParser(description="Generate paper figures")
parser.add_argument("--run_dirs", type=str, nargs="+", required=True)
parser.add_argument("--eval_dir", type=str, default="results/eval")
parser.add_argument("--output_dir", type=str, default="results/plots")
parser.add_argument("--smooth", type=int, default=20)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    print("tensorboard not installed - training curves will be skipped")
    _TENSORBOARD_AVAILABLE = False


def read_tb_scalar(run_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays for a given TensorBoard scalar tag."""
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events], dtype=float)
    values = np.array([e.value for e in events], dtype=float)
    return steps, values


def ema_smooth(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Exponential moving average smoothing."""
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def align_to_grid(
    steps_list: list[np.ndarray], values_list: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate multiple runs onto a common step grid and return mean ± std."""
    max_step  = min(s[-1] for s in steps_list)
    grid      = np.linspace(0, max_step, 500)
    interped  = np.stack(
        [np.interp(grid, s, v) for s, v in zip(steps_list, values_list)], axis=0
    )
    return grid, interped.mean(axis=0), interped.std(axis=0)


# ---------------------------------------------------------------------------
# Figure 1 & 2 – Training curves
# ---------------------------------------------------------------------------
RETURN_TAG = "Reward / Total reward (mean)"
SURVIVAL_TAG = "Episode / Total timesteps (mean)"

def plot_training_curve(
    tag: str,
    ylabel: str,
    fig_name: str,
    scale: float = 1.0,
) -> None:
    if not _TENSORBOARD_AVAILABLE:
        return

    fig, ax = plt.subplots()
    run_dirs = [Path(d) for d in args.run_dirs]

    steps_all, vals_all = [], []
    for i, rdir in enumerate(run_dirs):
        # Find events file recursively
        event_files = list(rdir.rglob("events.out.tfevents.*"))
        if not event_files:
            print(f"no TensorBoard events in {rdir}")
            continue
        try:
            s, v = read_tb_scalar(event_files[0].parent, tag)
        except KeyError:
            print(f"tag '{tag}' not found in {rdir} - skipping")
            continue

        v = ema_smooth(v * scale, alpha=2 / (args.smooth + 1))
        ax.plot(s / 1e6, v, alpha=0.35, color=PALETTE[i], linewidth=0.8)
        steps_all.append(s)
        vals_all.append(v)

    if steps_all:
        grid, mean, std = align_to_grid(steps_all, vals_all)
        ax.plot(grid / 1e6, mean, color=PALETTE[0], linewidth=2.0, label="Mean (3 seeds)")
        ax.fill_between(grid / 1e6, mean - std, mean + std,
                        color=PALETTE[0], alpha=0.20, label="±1 std")

    ax.set_xlabel("Environment steps (x1e6)")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{fig_name}.{ext}")
    plt.close(fig)
    print(f"saved {fig_name}")


plot_training_curve(RETURN_TAG, "Mean episode return", "fig1_training_curve")
plot_training_curve(SURVIVAL_TAG, "Mean survival time (steps)", "fig2_survival_curve")

# Figure 3 - reward component breakdown at convergence
def plot_reward_breakdown() -> None:
    """
    Reads the final metrics JSON from each seed and plots the reward
    components.  Because component-level data isn't in the JSON by default,
    this figure uses the aggregate return values as a proxy bar chart.
    To add per-component logging, instrument _get_rewards() in the env and
    log via skrl's experiment writer.
    """
    eval_dir = Path(args.eval_dir)
    # Find the latest checkpoint metric file per seed
    all_json = sorted(eval_dir.glob("*_metrics.json"))

    # Group by seed (expect naming: ..._seed{N}_..._<step>_metrics.json)
    seed_final: dict[str, dict] = {}
    for jf in all_json:
        match = re.search(r"seed(\d+)", jf.stem)
        key   = match.group(0) if match else "seed0"
        seed_final[key] = json.loads(jf.read_text())  # last write wins → latest ckpt

    if not seed_final:
        print("no eval JSON found - skipping reward breakdown figure")
        return

    labels  = list(seed_final.keys())
    returns = [d["mean_return"] for d in seed_final.values()]
    errors  = [d["std_return"]  for d in seed_final.values()]

    fig, ax = plt.subplots(figsize=(max(4.0, len(labels) * 1.2), 4.0))
    bars = ax.bar(labels, returns, yerr=errors, capsize=4,
                  color=PALETTE[:len(labels)], alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Mean episode return ± std")
    ax.set_title("Convergence performance across seeds")
    for bar, val in zip(bars, returns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig3_reward_breakdown.{ext}")
    plt.close(fig)
    print("saved fig3_reward_breakdown")


plot_reward_breakdown()

# Figure 4 - robustness: impulse magnitude vs. recovery rate
def plot_robustness():
    eval_dir = Path(args.eval_dir)
    all_json = sorted(eval_dir.glob("*_metrics.json"))
    if not all_json:
        print("no eval JSON found - skipping robustness figure")
        return

    # Aggregate recovery rates across checkpoints / seeds
    records = []
    for jf in all_json:
        d = json.loads(jf.read_text())
        if "robustness_recovery_rate" in d and not np.isnan(d["robustness_recovery_rate"]):
            records.append({
                "recovery_rate": d["robustness_recovery_rate"],
                "mean_impulse":  d.get("mean_impulse_N_s", 10.0),
            })

    if not records:
        print("no robustness data found - skipping figure 4")
        return

    impulses   = np.array([r["mean_impulse"]  for r in records])
    recoveries = np.array([r["recovery_rate"] for r in records]) * 100  # %

    fig, ax = plt.subplots()
    ax.scatter(impulses, recoveries, color=PALETTE[1], s=60, zorder=3)
    # Trend line
    if len(impulses) > 2:
        z = np.polyfit(impulses, recoveries, 1)
        xfit = np.linspace(impulses.min(), impulses.max(), 100)
        ax.plot(xfit, np.polyval(z, xfit), "--", color=PALETTE[1], alpha=0.6)
    ax.set_xlabel("Applied impulse magnitude (N-s)")
    ax.set_ylabel("Recovery rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Robustness to random impulse on pole 3")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig4_robustness.{ext}")
    plt.close(fig)
    print("saved fig4_robustness")


plot_robustness()

print(f"all figures written to {output_dir}")
