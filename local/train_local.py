"""
Local PPO training – Triple Inverted Pendulum (MuJoCo)
=======================================================
No Isaac Sim required. Uses stable-baselines3 + MuJoCo.

Install:
    pip install -r local/requirements.txt

Run (from repo root):
    python local/train_local.py                         # default seed 42
    python local/train_local.py --seed 0 --num_envs 16
    python local/train_local.py --seed 1 --timesteps 5000000

Monitor live:
    tensorboard --logdir local/runs/

Checkpoints saved to: local/checkpoints/
"""

from __future__ import annotations
import argparse
import os
import sys

# Ensure repo root is on the path so 'local.envs' imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local.envs.triple_pendulum_mujoco import TriplePendulumMuJoCoEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train triple-pendulum PPO locally with MuJoCo")
    p.add_argument("--seed",       type=int,   default=42,        help="Random seed")
    p.add_argument("--timesteps",  type=int,   default=3_000_000, help="Total env steps")
    p.add_argument("--num_envs",   type=int,   default=8,         help="Parallel envs (CPU workers)")
    p.add_argument("--lr",         type=float, default=3e-4,      help="Learning rate")
    p.add_argument("--run_name",   type=str,   default=None,      help="Override run directory name")
    return p.parse_args()


def main():
    args = parse_args()

    run_name = args.run_name or f"triple_pendulum_ppo_seed{args.seed}"
    log_dir  = os.path.join("local", "runs",        run_name)
    ckpt_dir = os.path.join("local", "checkpoints", run_name)
    eval_dir = os.path.join("local", "eval_logs",   run_name)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"[train_local] seed={args.seed}  envs={args.num_envs}  "
          f"timesteps={args.timesteps:,}  run={run_name}")

    # ------------------------------------------------------------------
    # Vectorised training environments
    # ------------------------------------------------------------------
    def _make(rank: int):
        def _init():
            env = TriplePendulumMuJoCoEnv(render_mode=None)
            env.reset(seed=args.seed + rank)
            return env
        return _init

    vec_cls = SubprocVecEnv if args.num_envs > 1 else None
    if vec_cls is not None:
        train_env = SubprocVecEnv([_make(i) for i in range(args.num_envs)])
    else:
        train_env = make_vec_env(
            lambda: TriplePendulumMuJoCoEnv(render_mode=None),
            n_envs=1, seed=args.seed
        )
    train_env = VecMonitor(train_env, filename=os.path.join(log_dir, "monitor"))

    # ------------------------------------------------------------------
    # Evaluation environment (single, no sub-process)
    # ------------------------------------------------------------------
    eval_env = VecMonitor(
        make_vec_env(
            lambda: TriplePendulumMuJoCoEnv(render_mode=None),
            n_envs=1, seed=args.seed + 9999
        ),
        filename=os.path.join(eval_dir, "monitor"),
    )

    # ------------------------------------------------------------------
    # PPO hyperparameters (match Isaac Lab / skrl_ppo_cfg.py)
    # rollouts=24, epochs=5, mini_batches=4, γ=0.99, λ=0.95, lr=3e-4
    # entropy=0.005, 2×256 ELU (SB3 uses tanh by default; override below)
    # ------------------------------------------------------------------
    n_steps    = 24
    n_minibatch = 4
    batch_size = (n_steps * args.num_envs) // n_minibatch

    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        n_steps         = n_steps,
        batch_size      = batch_size,
        n_epochs        = 5,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.005,
        max_grad_norm   = 1.0,
        learning_rate   = args.lr,
        policy_kwargs   = dict(
            net_arch        = [256, 256],
            activation_fn   = __import__("torch.nn", fromlist=["ELU"]).ELU,
        ),
        tensorboard_log = os.path.join("local", "runs"),
        seed            = args.seed,
        verbose         = 1,
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(100_000 // args.num_envs, 1),
        save_path   = ckpt_dir,
        name_prefix = "triple_ppo",
        verbose     = 1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = ckpt_dir,
        log_path             = eval_dir,
        eval_freq            = max(50_000 // args.num_envs, 1),
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model.learn(
        total_timesteps  = args.timesteps,
        callback         = [checkpoint_cb, eval_cb],
        tb_log_name      = run_name,
        progress_bar     = True,
        reset_num_timesteps = True,
    )

    final_path = os.path.join(ckpt_dir, "triple_ppo_final")
    model.save(final_path)
    print(f"\n[train_local] Training complete. Model saved to {final_path}.zip")
    print(f"[train_local] View results:  tensorboard --logdir local/runs/")


if __name__ == "__main__":
    main()
