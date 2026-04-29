"""
eval.py – Quantitative evaluation of trained checkpoints
=========================================================
Runs 100 deterministic evaluation episodes per checkpoint and records:
  - Mean / std episode return
  - Mean / std survival time (seconds)
  - Success rate  (fraction of episodes where ALL poles stayed upright ≥ 5 s)
  - Robustness test: a random lateral impulse is applied to the tip of pole_3
    at t=2 s (well after the policy has stabilised).  The impulse magnitude is
    drawn from Uniform(5, 15) N·s.  Recovery rate = fraction of episodes that
    remain upright for ≥ 3 s after the impulse.

All results are saved to  results/eval/<checkpoint_stem>_metrics.json

Usage:
    # Evaluate a single checkpoint
    python scripts/eval.py --checkpoint results/runs/.../agent_3000000.pt

    # Evaluate all checkpoints in a run directory (batch mode)
    python scripts/eval.py --run_dir results/runs/<exp>/checkpoints

The AppLauncher MUST be constructed before any Isaac Lab imports.
"""

from __future__ import annotations

import argparse
import json
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

parser = argparse.ArgumentParser(description="Evaluate trained triple-pendulum policy")
parser.add_argument("--checkpoint",    type=str, default="",   help="Single .pt file to evaluate")
parser.add_argument("--run_dir",       type=str, default="",   help="Directory containing .pt files")
parser.add_argument("--num_episodes",  type=int, default=100,  help="Episodes per checkpoint")
parser.add_argument("--success_secs",  type=float, default=5.0, help="Minimum upright seconds for success")
parser.add_argument("--impulse_time",  type=float, default=2.0, help="Seconds into episode to apply impulse")
parser.add_argument("--impulse_recovery_secs", type=float, default=3.0,
                    help="Seconds post-impulse required for recovery")
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

sys.path.insert(0, str(Path(__file__).parent.parent / "source"))
import triple_pendulum  # noqa: E402

from triple_pendulum.agents.skrl_ppo_cfg import PolicyNetwork, ValueNetwork
from triple_pendulum.tasks.triple_pendulum_env import TriplePendulumEnv
from triple_pendulum.tasks.triple_pendulum_env_cfg import TriplePendulumEnvCfg

from skrl.envs.wrappers.torch import wrap_env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_policy(checkpoint_path: Path, obs_space, act_space, device: torch.device) -> PolicyNetwork:
    policy = PolicyNetwork(obs_space, act_space, device)
    ckpt   = torch.load(str(checkpoint_path), map_location=device)
    policy.load_state_dict(ckpt["policy"] if "policy" in ckpt else ckpt)
    policy.eval()
    return policy


def run_evaluation(
    policy: PolicyNetwork,
    env: TriplePendulumEnv,
    wrapped_env,
    cfg: TriplePendulumEnvCfg,
    n_episodes: int,
    success_steps: int,
) -> dict:
    """Run n_episodes and return aggregate metrics (no impulse)."""
    dt       = cfg.sim.dt * cfg.decimation
    returns  = []
    lengths  = []
    survived = 0  # episodes where the policy stayed up for ≥ success_steps steps

    for _ in range(n_episodes):
        obs, _ = wrapped_env.reset()
        done       = False
        ep_return  = 0.0
        ep_length  = 0

        while not done:
            with torch.no_grad():
                action, _, _ = policy.act({"states": obs}, role="policy")
            obs, reward, terminated, truncated, _ = wrapped_env.step(action)
            done       = (terminated | truncated).any().item()
            ep_return += reward.sum().item()
            ep_length += 1

        returns.append(ep_return)
        lengths.append(ep_length)
        if ep_length >= success_steps:
            survived += 1

    return {
        "mean_return":       float(np.mean(returns)),
        "std_return":        float(np.std(returns)),
        "mean_survival_s":   float(np.mean(lengths) * dt),
        "std_survival_s":    float(np.std(lengths)  * dt),
        "success_rate":      float(survived / n_episodes),
        "n_episodes":        n_episodes,
    }


def run_robustness_test(
    policy: PolicyNetwork,
    env: TriplePendulumEnv,
    wrapped_env,
    cfg: TriplePendulumEnvCfg,
    n_episodes: int,
    impulse_step: int,
    recovery_steps: int,
) -> dict:
    """Apply a random lateral impulse to pole_3 at step ``impulse_step``.

    The impulse is modelled as a single-step force on the outermost link.
    Recovery = episode survived for ≥ ``recovery_steps`` steps after impulse.
    """
    dt             = cfg.sim.dt * cfg.decimation
    impulse_mags   = np.random.uniform(5.0, 15.0, n_episodes)  # N·s
    recoveries     = 0

    # Find pole_3 body index for applying external wrench
    try:
        pole3_body_ids, _ = env.robot.find_bodies("pole_3")
        pole3_body_id     = pole3_body_ids[0]
        can_apply_impulse = True
    except Exception:
        print("[eval] WARNING: Could not find pole_3 body – robustness test skipped")
        can_apply_impulse = False

    for ep_idx in range(n_episodes):
        obs, _ = wrapped_env.reset()
        done        = False
        step        = 0
        impulse_applied = False
        post_impulse_steps = 0

        while not done:
            with torch.no_grad():
                action, _, _ = policy.act({"states": obs}, role="policy")

            # Apply impulse at the designated time step
            if can_apply_impulse and step == impulse_step and not impulse_applied:
                impulse_force = float(impulse_mags[ep_idx]) / dt  # convert N·s → N (one step)
                forces  = torch.zeros(1, 3, device=env.device)
                torques = torch.zeros(1, 3, device=env.device)
                forces[0, 0] = impulse_force  # lateral force along x-axis
                env.robot.set_external_force_and_torque(
                    forces=forces,
                    torques=torques,
                    body_ids=[pole3_body_id],
                    env_ids=torch.tensor([0], device=env.device),
                )
                impulse_applied = True

            obs, reward, terminated, truncated, _ = wrapped_env.step(action)
            done  = (terminated | truncated).any().item()
            step += 1

            if impulse_applied:
                post_impulse_steps += 1

        if impulse_applied and post_impulse_steps >= recovery_steps:
            recoveries += 1

    recovery_rate = float(recoveries / n_episodes) if can_apply_impulse else float("nan")
    return {
        "robustness_recovery_rate": recovery_rate,
        "impulse_time_s":   float(impulse_step * dt),
        "recovery_window_s": float(recovery_steps * dt),
        "mean_impulse_N_s": float(np.mean(impulse_mags)),
    }


# ---------------------------------------------------------------------------
# Resolve checkpoint list
# ---------------------------------------------------------------------------
ckpt_paths: list[Path] = []
if args_cli.checkpoint:
    ckpt_paths = [Path(args_cli.checkpoint)]
elif args_cli.run_dir:
    ckpt_paths = sorted(Path(args_cli.run_dir).glob("*.pt"))
else:
    raise ValueError("Provide --checkpoint or --run_dir")

if not ckpt_paths:
    raise FileNotFoundError(f"No checkpoints found in {args_cli.run_dir}")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
env_cfg = TriplePendulumEnvCfg()
env_cfg.scene.num_envs = 1

raw_env  = TriplePendulumEnv(cfg=env_cfg, render_mode=None)
env      = wrap_env(raw_env)
device   = env.device

dt_policy      = env_cfg.sim.dt * env_cfg.decimation
success_steps  = int(args_cli.success_secs / dt_policy)
impulse_step   = int(args_cli.impulse_time / dt_policy)
recovery_steps = int(args_cli.impulse_recovery_secs / dt_policy)

output_dir = Path(__file__).parent.parent / "results" / "eval"
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Evaluation loop over checkpoints
# ---------------------------------------------------------------------------
for ckpt_path in ckpt_paths:
    print(f"\n[eval] Evaluating {ckpt_path.name} …")
    policy = load_policy(ckpt_path, env.observation_space, env.action_space, device)

    base_metrics = run_evaluation(
        policy, raw_env, env, env_cfg, args_cli.num_episodes, success_steps
    )
    robustness   = run_robustness_test(
        policy, raw_env, env, env_cfg, args_cli.num_episodes, impulse_step, recovery_steps
    )

    metrics = {**base_metrics, **robustness, "checkpoint": str(ckpt_path)}
    out_path = output_dir / f"{ckpt_path.stem}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"  Return  : {base_metrics['mean_return']:.2f} ± {base_metrics['std_return']:.2f}"
        f"\n  Survival: {base_metrics['mean_survival_s']:.2f} ± {base_metrics['std_survival_s']:.2f} s"
        f"\n  Success : {base_metrics['success_rate']*100:.1f}%"
        f"\n  Recovery: {robustness['robustness_recovery_rate']*100:.1f}%"
        f"\n  → {out_path}"
    )

import os as _os
if _os.name == "nt":
    _os._exit(0)
else:
    simulation_app.close()
