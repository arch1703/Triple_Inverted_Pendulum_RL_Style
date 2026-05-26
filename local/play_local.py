# load a trained PPO checkpoint and either render live or record an MP4
# usage: python local/play_local.py --checkpoint local/checkpoints/<run>/best_model [--record]

from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local.envs.triple_pendulum_mujoco import TriplePendulumMuJoCoEnv

import numpy as np
from stable_baselines3 import PPO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--record", action="store_true")
    p.add_argument("--out", type=str, default="local/videos/triple_pendulum_demo.mp4")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", default=True)
    return p.parse_args()


def run_live(model, episodes, seed):
    env = TriplePendulumMuJoCoEnv(render_mode="human")
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        step = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            time.sleep(0.02)
        print(f"  ep {ep+1}: {step} steps, reward={total_reward:.2f}")
    env.close()


def run_record(model, episodes, seed, out_path, fps, width, height):
    try:
        import imageio
    except ImportError:
        print("imageio not installed - run: pip install imageio[ffmpeg]")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    env = TriplePendulumMuJoCoEnv(render_mode="rgb_array")
    # Override renderer resolution for higher-quality video
    env._render_width  = width
    env._render_height = height

    frames: list[np.ndarray] = []
    ep_stats: list[tuple[int, float]] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        # Grab initial frame
        frame = env._render_rgb_array(width=width, height=height)
        frames.append(frame)

        total_reward = 0.0
        step = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            frame = env._render_rgb_array(width=width, height=height)
            frames.append(frame)

        ep_stats.append((step, total_reward))
        print(f"  ep {ep+1}: {step} steps, reward={total_reward:.2f}")

    env.close()

    print(f"writing {len(frames)} frames to {out_path}...")
    imageio.mimsave(out_path, frames, fps=fps, quality=9)
    print(f"saved: {out_path}")

    avg_steps = sum(s for s, _ in ep_stats) / len(ep_stats)
    avg_reward = sum(r for _, r in ep_stats) / len(ep_stats)
    print(f"avg: {avg_steps:.0f} steps/ep, {avg_reward:.2f} reward/ep")


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint) and not os.path.exists(args.checkpoint + ".zip"):
        print(f"checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"loading {args.checkpoint}")
    model = PPO.load(args.checkpoint, device="cpu")

    if args.record:
        run_record(model, args.episodes, args.seed, args.out, args.fps, args.width, args.height)
    else:
        run_live(model, args.episodes, args.seed)


if __name__ == "__main__":
    main()
