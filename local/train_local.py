# PPO training script for triple inverted pendulum (MuJoCo, no Isaac Sim)
# usage: python local/train_local.py [--seed N] [--timesteps N] [--num_envs N]
# tensorboard: tensorboard --logdir local/runs/

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
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)


class EntropyDecayCallback(BaseCallback):
    def __init__(self, start_val: float = 0.005, end_val: float = 0.0001):
        super().__init__()
        self.start_val = start_val
        self.end_val = end_val

    def _on_step(self) -> bool:
        p = 1.0 - self.num_timesteps / self.model._total_timesteps
        self.model.ent_coef = float(self.end_val + (self.start_val - self.end_val) * p)
        return True


class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum_steps: int = 500_000, final_range: float = 0.087):
        super().__init__()
        self.curriculum_steps = curriculum_steps
        self.final_range = final_range
        self._widened = False

    def _on_step(self) -> bool:
        if not self._widened and self.num_timesteps >= self.curriculum_steps:
            self.training_env.env_method("set_reset_range", self.final_range)
            self._widened = True
            import numpy as _np
            print(f"[curriculum] widened reset range to ±{_np.degrees(self.final_range):.1f} deg at step {self.num_timesteps:,}")
        return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train triple-pendulum PPO locally with MuJoCo")
    p.add_argument("--seed",       type=int,   default=42,        help="Random seed")
    p.add_argument("--timesteps",  type=int,   default=6_000_000, help="Total env steps")
    p.add_argument("--num_envs",   type=int,   default=8,         help="Parallel envs (CPU workers)")
    p.add_argument("--lr",         type=float, default=3e-4,      help="Learning rate")
    p.add_argument("--run_name",   type=str,   default=None,      help="Override run directory name")
    return p.parse_args()


def main():
    args = parse_args()

    run_name = args.run_name or f"triple_pendulum_ppo_v2_seed{args.seed}"
    log_dir = os.path.join("local", "runs", run_name)
    ckpt_dir = os.path.join("local", "checkpoints", run_name)
    eval_dir = os.path.join("local", "eval_logs", run_name)
    for d in (log_dir, ckpt_dir, eval_dir):
        os.makedirs(d, exist_ok=True)

    print(f"seed={args.seed}  envs={args.num_envs}  steps={args.timesteps:,}  run={run_name}")

    def _make(rank: int):
        def _init():
            env = TriplePendulumMuJoCoEnv(render_mode=None)
            env.reset(seed=args.seed + rank)
            return env
        return _init

    if args.num_envs > 1:
        train_env = SubprocVecEnv([_make(i) for i in range(args.num_envs)])
    else:
        train_env = make_vec_env(lambda: TriplePendulumMuJoCoEnv(render_mode=None), n_envs=1, seed=args.seed)
    train_env = VecMonitor(train_env, filename=os.path.join(log_dir, "monitor"))

    eval_env = VecMonitor(
        make_vec_env(lambda: TriplePendulumMuJoCoEnv(render_mode=None), n_envs=1, seed=args.seed + 9999),
        filename=os.path.join(eval_dir, "monitor"),
    )

    n_steps = 24
    batch_size = (n_steps * args.num_envs) // 4

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        max_grad_norm=1.0,
        learning_rate=args.lr,
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=__import__("torch.nn", fromlist=["ELU"]).ELU,
        ),
        tensorboard_log=os.path.join("local", "runs"),
        seed=args.seed,
        verbose=1,
        device="cpu",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // args.num_envs, 1),
        save_path=ckpt_dir,
        name_prefix="triple_ppo",
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=ckpt_dir,
        log_path=eval_dir,
        eval_freq=max(50_000 // args.num_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb, EntropyDecayCallback(), CurriculumCallback()],
        tb_log_name=run_name,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    final_path = os.path.join(ckpt_dir, "triple_ppo_final")
    model.save(final_path)
    print(f"saved to {final_path}.zip")
    print(f"tensorboard --logdir local/runs/")


if __name__ == "__main__":
    main()
