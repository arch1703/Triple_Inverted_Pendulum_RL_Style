# Triple Inverted Pendulum – PPO

ROB-GY 6203 Reinforcement Learning II · NYU Tandon · Spring 2026  
Arnav Chopra

PPO agent trained to balance a triple inverted pendulum on a cart using MuJoCo. Two versions are compared: a baseline (v1) and an improved agent (v2) that hits 100% episode success.

## Results

| Agent | Mean Reward | Mean Length | Success |
|---|---|---|---|
| v1 (baseline) | 58.3 ± 41.2 | 112 ± 89 steps | 14% |
| v2 (improved) | 316.8 ± 2.1 | 500 ± 0 steps | **100%** |

Evaluated over 20 deterministic episodes from the best checkpoint.

## Project Structure

```
local/                      # MuJoCo pipeline (main codebase, all results from here)
  envs/triple_pendulum_mujoco.py   # Gymnasium environment
  models/triple_pendulum.xml       # MuJoCo MJCF model
  train_local.py            # PPO training
  play_local.py             # visualise / record a policy
  plot_local.py             # training curve figures
  requirements.txt
  checkpoints/              # saved model checkpoints
  runs/                     # VecMonitor CSV logs and TensorBoard events
  eval_logs/                # evaluation results (.npz)
  figures/                  # generated plots
  videos/                   # recorded MP4 rollouts
source/                     # Isaac Lab environment (HPC, not used for final results)
scripts/                    # Isaac Lab training / eval entry points
slurm/                      # NYU HPC SLURM job scripts
configs/ppo_cfg.yaml        # PPO hyperparameter config
report/report.tex           # project report
setup.py
```

## Environment

`TriplePendulumMuJoCoEnv` wraps MuJoCo for a cart-triple-pole system.

| Parameter | Value |
|---|---|
| Cart mass | 1.0 kg |
| Link mass (each) | 0.1 kg |
| Link length (each) | 0.5 m |
| Physics timestep | 0.01 s (100 Hz) |
| Policy frequency | 50 Hz (decimation 2) |
| Max cart force | 20 N |
| Episode horizon | 500 steps (10 s) |

**Observation** (8-dim): `[x, x_dot, theta1, theta1_dot, theta2, theta2_dot, theta3, theta3_dot]`  
**Action** (1-dim): normalised force in `[-1, 1]` scaled to N  
**Termination**: any `|theta_i| >= theta_max` or `|x| >= 2 m`

## v1 to v2 Changes

1. Tighter termination threshold (30 deg -> 15 deg) - forces precision
2. Linear angle penalty added (weight 0.1) - non-zero gradient at small angles
3. Curriculum: reset range widens from 1 deg to 5 deg at 500k steps
4. Entropy decay: linearly annealed from 0.005 to 0.0001 over 6M steps

## Installation

```bash
pip install -r local/requirements.txt
pip install -e .
```

Python 3.9+, no GPU required.

## Usage

**Train:**
```bash
python local/train_local.py                          # v2 agent, seed 42, 6M steps
python local/train_local.py --seed 0 --timesteps 6000000 --num_envs 16
tensorboard --logdir local/runs/                     # monitor live
```

Checkpoints saved every 100k steps to `local/checkpoints/<run_name>/`.

**Play / record:**
```bash
python local/play_local.py \
    --checkpoint local/checkpoints/triple_pendulum_ppo_v2_seed42/best_model

python local/play_local.py \
    --checkpoint local/checkpoints/triple_pendulum_ppo_v2_seed42/best_model \
    --record --episodes 5 --out local/videos/demo.mp4
```

**Plot training curves:**
```bash
python local/plot_local.py
python local/plot_local.py --mean_window 200 --out local/figures/
```

Outputs: `local/figures/v1/`, `v2/`, and comparison figures.

## Hyperparameters

| Parameter | v1 | v2 |
|---|---|---|
| Total timesteps | 3M | 6M |
| Parallel envs | 8 | 8 |
| Rollout steps | 24 | 24 |
| Mini-batches | 4 | 4 |
| Epochs per rollout | 5 | 5 |
| Discount | 0.99 | 0.99 |
| GAE lambda | 0.95 | 0.95 |
| Clip range | 0.2 | 0.2 |
| Learning rate | 3e-4 | 3e-4 |
| Entropy coeff | 0.01 (const.) | 0.005 -> 0.0001 |
| Network | 2x256 ELU | 2x256 ELU |
| Angle limit | 30 deg | 15 deg |
| Curriculum | none | 1 deg -> 5 deg at 500k |
| Linear angle penalty | 0 | 0.1 |

## Report

Full report is in `report/report.tex`. Compile with:
```bash
cd report && pdflatex report.tex
```

## HPC (Isaac Lab)

SLURM scripts for NYU HPC are in `slurm/`. The `source/` directory has the Isaac Lab environment and SKRL config. Due to Windows compatibility issues with Isaac Sim, all reported results use the MuJoCo pipeline above.

## Dependencies

| Package | Purpose |
|---|---|
| `mujoco >= 3.0` | Physics simulation |
| `gymnasium >= 0.29` | Environment API |
| `stable-baselines3 >= 2.3` | PPO implementation |
| `tensorboard >= 2.14` | Training monitoring |
| `matplotlib >= 3.7` | Plotting |
| `pandas >= 2.0` | Log parsing |
| `imageio[ffmpeg]` | Video recording |
