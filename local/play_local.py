"""
Play & Record – Triple Inverted Pendulum (MuJoCo)
==================================================
Loads a trained SB3 PPO checkpoint and either renders it live or
records a polished MP4 for presentations.

Usage (from repo root):
    # Live window (requires a display)
    python local/play_local.py --checkpoint local/checkpoints/<run>/triple_ppo_final

    # Record MP4 (no display needed, works headless)
    python local/play_local.py --checkpoint local/checkpoints/<run>/triple_ppo_final \\
        --record --episodes 3 --out local/videos/triple_pendulum_demo.mp4

    # Record with the best eval checkpoint
    python local/play_local.py \\
        --checkpoint local/checkpoints/<run>/best_model \\
        --record --episodes 5 --fps 50

Dependencies:
    pip install imageio[ffmpeg]
"""

from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local.envs.triple_pendulum_mujoco import TriplePendulumMuJoCoEnv

import numpy as np
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play / record trained triple-pendulum policy")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to SB3 .zip checkpoint (omit the .zip extension)")
    p.add_argument("--record", action="store_true",
                   help="Record to MP4 instead of opening a live window")
    p.add_argument("--out", type=str, default="local/videos/triple_pendulum_demo.mp4",
                   help="Output video path (used with --record)")
    p.add_argument("--episodes", type=int, default=3,
                   help="Number of episodes to run")
    p.add_argument("--fps", type=int, default=50,
                   help="Video frames per second (should match policy rate)")
    p.add_argument("--width",  type=int, default=1280, help="Video width  in pixels")
    p.add_argument("--height", type=int, default=720,  help="Video height in pixels")
    p.add_argument("--seed",   type=int, default=0,    help="Environment seed")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Use deterministic (mean) actions")
    return p.parse_args()


def run_live(model: PPO, episodes: int, seed: int):
    """Run policy with a live MuJoCo viewer window."""
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
            time.sleep(0.02)   # ~50 Hz
        print(f"  Episode {ep+1}: {step} steps, reward = {total_reward:.2f}")
    env.close()


def run_record(model: PPO, episodes: int, seed: int,
               out_path: str, fps: int, width: int, height: int):
    """Record episodes to MP4 using imageio."""
    try:
        import imageio
    except ImportError:
        print("ERROR: imageio not installed. Run:  pip install imageio[ffmpeg]")
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
        print(f"  Episode {ep+1}: {step} steps, reward = {total_reward:.2f}")

    env.close()

    print(f"\nWriting {len(frames)} frames to {out_path} ...")
    imageio.mimsave(out_path, frames, fps=fps, quality=9)
    print(f"Saved: {out_path}")

    avg_steps  = sum(s for s, _ in ep_stats) / len(ep_stats)
    avg_reward = sum(r for _, r in ep_stats) / len(ep_stats)
    print(f"Average: {avg_steps:.0f} steps/episode, {avg_reward:.2f} reward/episode")


def main():
    args = parse_args()

    # Load model
    ckpt = args.checkpoint
    if not ckpt.endswith(".zip") and not os.path.exists(ckpt):
        ckpt_zip = ckpt + ".zip"
    else:
        ckpt_zip = ckpt

    if not os.path.exists(ckpt_zip if ckpt_zip != ckpt else ckpt):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("       Provide the path without .zip extension.")
        sys.exit(1)

    print(f"[play_local] Loading: {args.checkpoint}")
    model = PPO.load(args.checkpoint, device="cpu")

    if args.record:
        print(f"[play_local] Recording {args.episodes} episode(s) → {args.out}")
        run_record(model, args.episodes, args.seed,
                   args.out, args.fps, args.width, args.height)
    else:
        print(f"[play_local] Live render – {args.episodes} episode(s)")
        run_live(model, args.episodes, args.seed)


if __name__ == "__main__":
    main()
