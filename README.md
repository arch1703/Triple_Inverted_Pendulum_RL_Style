# Triple Inverted Pendulum Reinforcement Learning

PPO-based stabilization of a simulated triple inverted pendulum on a cart.

This repository includes:
- A local MuJoCo training pipeline (used for all reported results)
- An Isaac Lab + SKRL pipeline scaffold
- Plotting, evaluation, and video generation scripts

Important note: Isaac Sim integration did not run successfully on the HPC platform attempted, and it has not been tested locally in this project setup. All reported training/evaluation numbers, figures, and videos in this repository come from the MuJoCo pipeline.

## Results Snapshot

| Agent | Mean Reward | Mean Length | Success Rate |
| --- | --- | --- | --- |
| v1 (baseline) | 58.3 +/- 41.2 | 112 +/- 89 steps | 14% |
| v2 (improved) | 316.8 +/- 2.1 | 500 +/- 0 steps | 100% |

Evaluation was run over 20 deterministic episodes using the best checkpoint.

## Project Layout

```text
local/                      # Main MuJoCo workflow
  envs/triple_pendulum_mujoco.py
  models/triple_pendulum.xml
  train_local.py            # Train PPO locally
  play_local.py             # Visualize / record rollouts
  plot_local.py             # Generate learning curves
  requirements.txt
  figures/
  videos/

source/triple_pendulum/     # Isaac Lab task + agent config
scripts/                    # Isaac Lab train/play/eval scripts
configs/ppo_cfg.yaml        # PPO config used by Isaac Lab pipeline
slurm/                      # HPC (SLURM) helper scripts
setup.py
```

## Environment (MuJoCo)

The local environment models a cart with three serial links.

| Parameter | Value |
| --- | --- |
| Cart mass | 1.0 kg |
| Link mass (each) | 0.1 kg |
| Link length (each) | 0.5 m |
| Physics timestep | 0.01 s |
| Control frequency | 50 Hz (decimation 2) |
| Max force | 20 N |
| Episode horizon | 500 steps |

- Observation (8D): [x, x_dot, theta1, theta1_dot, theta2, theta2_dot, theta3, theta3_dot]
- Action (1D): normalized force in [-1, 1] scaled to Newtons
- Termination: angle limit exceeded or cart position out of bounds

## What Changed From v1 to v2

1. Tighter termination threshold (30 deg -> 15 deg)
2. Added linear angle penalty (weight 0.1)
3. Reset-range curriculum (1 deg -> 5 deg at 500k steps)
4. Entropy coefficient decay (0.005 -> 0.0001 over training)

## Quick Start (Local MuJoCo)

### 1. Install dependencies

```bash
pip install -r local/requirements.txt
pip install -e .
```

Recommended Python version: 3.10+

### 2. Train

```bash
python local/train_local.py
```

Custom run example:

```bash
python local/train_local.py --seed 0 --timesteps 6000000 --num_envs 16
```

Monitor training:

```bash
tensorboard --logdir local/runs/
```

### 3. Play or record

```bash
python local/play_local.py --checkpoint local/checkpoints/triple_pendulum_ppo_v2_seed42/best_model
```

```bash
python local/play_local.py --checkpoint local/checkpoints/triple_pendulum_ppo_v2_seed42/best_model --record --episodes 5 --out local/videos/demo.mp4
```

### 4. Plot learning curves

```bash
python local/plot_local.py
```

```bash
python local/plot_local.py --mean_window 200 --out local/figures/
```

## Hyperparameters (v1 vs v2)

| Parameter | v1 | v2 |
| --- | --- | --- |
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
| Curriculum | none | 1 deg -> 5 deg |
| Linear angle penalty | 0 | 0.1 |

## Isaac Lab / HPC Path (Status)

The Isaac Lab path is included as code scaffolding, but is currently unverified in this repository:
- It did not run successfully on the HPC platform that was attempted.
- It has not been tested locally.
- All project data/results are from MuJoCo (`local/` pipeline).

Reference commands (for future validation):

```bash
python scripts/train.py --num_envs 512 --headless --seed 42
```

Evaluate and record:

```bash
python scripts/play.py --checkpoint results/runs/<experiment>/checkpoints/agent_3000000.pt --num_episodes 10
```

SLURM helpers are available in the slurm/ directory.

## Dependencies

Key packages used by the local pipeline:
- mujoco
- gymnasium
- stable-baselines3
- tensorboard
- matplotlib
- pandas
- imageio[ffmpeg]

## Author / Course Context

Developed for ROB-GY 6203 (Reinforcement Learning II), NYU Tandon, Spring 2026.
