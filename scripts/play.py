"""
play.py – Load a checkpoint and record evaluation videos
=========================================================
Runs N evaluation episodes with a trained policy and saves an MP4 for each
checkpoint.  Designed to be called at three training stages:

    Early   (~100k env steps)  – policy is still learning
    Mid     (~1M  env steps)  – partial stabilisation
    Late    (~3M  env steps)  – converged policy

Usage:
    python scripts/play.py \\
        --checkpoint results/runs/<exp>/checkpoints/agent_100000.pt \\
        --num_episodes 10 \\
        --video_dir results/videos

The Isaac AppLauncher MUST be constructed first; see note in train.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Windows DLL fix – must run before any Isaac Sim imports (see train.py)
if sys.platform == "win32":
    _env_root = os.path.dirname(sys.executable)
    for _dll_dir in [
        os.path.join(_env_root, "Lib", "site-packages", "torch", "lib"),
        os.path.join(_env_root, "Library", "bin"),
        os.path.join(_env_root, "Library", "mingw-w64", "bin"),
        _env_root,
    ]:
        if os.path.isdir(_dll_dir):
            os.add_dll_directory(_dll_dir)
            os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate triple pendulum and record video")
parser.add_argument("--checkpoint",   type=str, required=True, help="Path to .pt checkpoint file")
parser.add_argument("--num_episodes", type=int, default=10,    help="Number of evaluation episodes")
parser.add_argument("--video_dir",    type=str, default="",    help="Output directory for MP4 files")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Enable URDF importer extension (Isaac Sim 4.x does not auto-enable it)
import omni.kit.app as _kit_app
_ext_mgr = _kit_app.get_app().get_extension_manager()
if not _ext_mgr.is_extension_enabled("isaacsim.asset.importer.urdf"):
    _ext_mgr.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
del _ext_mgr, _kit_app

# ---------------------------------------------------------------------------
import numpy as np
import torch
import imageio

sys.path.insert(0, str(Path(__file__).parent.parent / "source"))
import triple_pendulum  # noqa: E402

from triple_pendulum.agents.skrl_ppo_cfg import PolicyNetwork, ValueNetwork
from triple_pendulum.tasks.triple_pendulum_env import TriplePendulumEnv
from triple_pendulum.tasks.triple_pendulum_env_cfg import TriplePendulumEnvCfg

from skrl.envs.wrappers.torch import wrap_env

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
checkpoint_path = Path(args_cli.checkpoint)
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

video_dir = Path(args_cli.video_dir) if args_cli.video_dir else (
    Path(__file__).parent.parent / "results" / "videos"
)
video_dir.mkdir(parents=True, exist_ok=True)
video_path = video_dir / f"{checkpoint_path.stem}.mp4"

# ---------------------------------------------------------------------------
# Environment  (single env, rendering enabled)
# ---------------------------------------------------------------------------
env_cfg = TriplePendulumEnvCfg()
env_cfg.scene.num_envs = 1

env = TriplePendulumEnv(cfg=env_cfg, render_mode="rgb_array")
env = wrap_env(env)
device = env.device

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
policy = PolicyNetwork(env.observation_space, env.action_space, device)
value  = ValueNetwork(env.observation_space,  env.action_space, device)

# Load checkpoint – skrl saves a dict keyed by model role
ckpt = torch.load(str(checkpoint_path), map_location=device)
if "policy" in ckpt:
    policy.load_state_dict(ckpt["policy"])
else:
    # Fallback: try loading the full file as a state-dict
    policy.load_state_dict(ckpt)

policy.eval()

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
all_frames:       list[np.ndarray] = []
episode_returns:  list[float]      = []
episode_lengths:  list[int]        = []

print(f"\n[play] Running {args_cli.num_episodes} episodes from {checkpoint_path.name}")

for ep in range(args_cli.num_episodes):
    obs, _ = env.reset()
    done        = False
    ep_return   = 0.0
    ep_length   = 0
    ep_frames:  list[np.ndarray] = []

    while not done:
        with torch.no_grad():
            action, _, _ = policy.act({"states": obs}, role="policy")

        obs, reward, terminated, truncated, _ = env.step(action)
        done = (terminated | truncated).any().item()

        ep_return += reward.sum().item()
        ep_length += 1

        frame = env.render()
        if frame is not None:
            ep_frames.append(frame if isinstance(frame, np.ndarray) else np.array(frame))

    all_frames.extend(ep_frames)
    episode_returns.append(ep_return)
    episode_lengths.append(ep_length)
    print(f"  Episode {ep+1:2d}: return={ep_return:.2f}  length={ep_length} steps")

# ---------------------------------------------------------------------------
# Save video
# ---------------------------------------------------------------------------
if all_frames:
    # 50 Hz rendering (policy frequency = 50 Hz)
    imageio.mimsave(str(video_path), all_frames, fps=50, quality=8)
    print(f"\n[play] Video saved → {video_path}")
else:
    print("[play] WARNING: no frames captured; render_mode may not be supported headlessly")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
mean_return = np.mean(episode_returns)
std_return  = np.std(episode_returns)
mean_length = np.mean(episode_lengths)
std_length  = np.std(episode_lengths)
dt_policy   = env_cfg.sim.dt * env_cfg.decimation  # seconds per policy step

print(
    f"\n[play] Summary ({args_cli.num_episodes} episodes)"
    f"\n  Mean return : {mean_return:.2f} ± {std_return:.2f}"
    f"\n  Mean length : {mean_length:.1f} ± {std_length:.1f} steps"
    f"  ({mean_length * dt_policy:.2f} ± {std_length * dt_policy:.2f} s)"
)

import os as _os
if _os.name == "nt":
    _os._exit(0)
else:
    simulation_app.close()
