"""
train.py – PPO training for the triple inverted pendulum
=========================================================
Usage (local, single env for debugging):
    python scripts/train.py --num_envs 4 --headless --seed 42

Usage (HPC, full training):
    python scripts/train.py --num_envs 512 --headless --seed 0 --timesteps 3000000

The AppLauncher MUST be constructed before any Isaac Sim / Isaac Lab modules
are imported.  All Isaac-related imports therefore appear AFTER app launch.

Outputs (under results/):
  runs/<experiment_name>/                  ← TensorBoard event files
  checkpoints/<experiment_name>/           ← .pt checkpoints every 100k steps
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 0.  Windows DLL fix  (MUST run before ANY Isaac Sim / omni imports)
# ---------------------------------------------------------------------------
# On Windows, Isaac Sim's extension loader tries to import torch while loading
# isaaclab_assets / isaaclab_tasks extensions during SimulationApp startup.
# At that point CUDA DLLs have already been touched by Isaac Sim's renderer,
# causing c10.dll to fail with WinError 1114 ("DLL init routine failed").
#
# Fix: register torch's DLL directory AND pre-import torch (CPU only) so its
# DLLs are cached in sys.modules before Isaac Sim's extension loader runs.
# Python's import cache means the second import inside the extension loader is
# instant and requires no new DLL loading.
if sys.platform == "win32":
    _env_root = os.path.dirname(sys.executable)          # conda env root
    for _dll_dir in [
        os.path.join(_env_root, "Lib", "site-packages", "torch", "lib"),
        os.path.join(_env_root, "Library", "bin"),        # conda CUDA/cuDNN DLLs
        os.path.join(_env_root, "Library", "mingw-w64", "bin"),
        _env_root,                                        # env root itself
    ]:
        if os.path.isdir(_dll_dir):
            os.add_dll_directory(_dll_dir)
            os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")
    # Pre-load torch so DLLs are already in memory when Isaac Sim extensions
    # try to import it.  CPU-only zeros call forces DLL init without a CUDA ctx.
    import torch as _torch_win_preload          # noqa: E402
    _torch_win_preload.zeros(1)                 # forces c10.dll init now
    del _torch_win_preload

# ---------------------------------------------------------------------------
# 1.  AppLauncher  (MUST come before all Isaac Sim / Lab imports)
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train triple pendulum PPO")
parser.add_argument("--num_envs",   type=int,   default=512,     help="Number of parallel environments")
parser.add_argument("--seed",       type=int,   default=42,      help="Random seed")
parser.add_argument("--timesteps",  type=int,   default=3_000_000, help="Total environment steps")
parser.add_argument("--experiment", type=str,   default="",      help="Experiment name suffix (e.g. seed_0)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher    = AppLauncher(args_cli)
simulation_app  = app_launcher.app

# ---------------------------------------------------------------------------
# 1b. Enable URDF importer extension  (Isaac Sim 4.x does not auto-enable it)
# ---------------------------------------------------------------------------
import omni.kit.app as _kit_app
_ext_mgr = _kit_app.get_app().get_extension_manager()
if not _ext_mgr.is_extension_enabled("isaacsim.asset.importer.urdf"):
    _ext_mgr.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
del _ext_mgr, _kit_app

# ---------------------------------------------------------------------------
# 2.  All other imports  (safe after AppLauncher)
# ---------------------------------------------------------------------------
import torch

# Add project source to path so ``triple_pendulum`` package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "source"))
import triple_pendulum  # noqa: E402  – registers gymnasium task

from triple_pendulum.agents.skrl_ppo_cfg import PPO_CONFIG, PolicyNetwork, ValueNetwork
from triple_pendulum.tasks.triple_pendulum_env import TriplePendulumEnv
from triple_pendulum.tasks.triple_pendulum_env_cfg import TriplePendulumEnvCfg

from skrl.agents.torch.ppo import PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# ---------------------------------------------------------------------------
# 3.  Seed
# ---------------------------------------------------------------------------
set_seed(args_cli.seed)

# ---------------------------------------------------------------------------
# 4.  Environment
# ---------------------------------------------------------------------------
env_cfg = TriplePendulumEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs

env = TriplePendulumEnv(cfg=env_cfg, render_mode=None)
env = wrap_env(env)

device = env.device

# ---------------------------------------------------------------------------
# 5.  Models
# ---------------------------------------------------------------------------
models = {
    "policy": PolicyNetwork(env.observation_space, env.action_space, device),
    "value":  ValueNetwork(env.observation_space,  env.action_space, device),
}

# ---------------------------------------------------------------------------
# 6.  Memory  (on-policy rollout buffer – size = rollout horizon)
# ---------------------------------------------------------------------------
memory = RandomMemory(
    memory_size=PPO_CONFIG.rollouts,
    num_envs=env.num_envs,
    device=device,
)

# ---------------------------------------------------------------------------
# 7.  Agent  (skrl 2.0: PPO_CONFIG is a dataclass, not a dict)
# ---------------------------------------------------------------------------
# Build a unique experiment name that encodes key run parameters
exp_name = f"triple_pendulum_ppo_seed{args_cli.seed}"
if args_cli.experiment:
    exp_name += f"_{args_cli.experiment}"

PPO_CONFIG.experiment.experiment_name   = exp_name
PPO_CONFIG.experiment.directory         = str(Path(__file__).parent.parent / "results" / "runs")
PPO_CONFIG.experiment.checkpoint_interval = 100_000
PPO_CONFIG.experiment.write_interval    = 1000

agent = PPO(
    models=models,
    memory=memory,
    cfg=PPO_CONFIG,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# ---------------------------------------------------------------------------
# 7b.  Auto-resume from latest checkpoint (handles spot-instance preemption)
# ---------------------------------------------------------------------------
# On HPC Cloud Bursting, GCP spot instances can be preempted mid-run.  The
# job is automatically requeued (#SBATCH --requeue).  When it restarts here,
# we detect the most-recent checkpoint, load it, and reduce the remaining
# timestep budget so we don't overtrain.
_ckpt_dir = Path(PPO_CONFIG.experiment.directory) / exp_name / "checkpoints"
_completed_steps = 0
if _ckpt_dir.exists():
    _ckpts = sorted(
        _ckpt_dir.glob("agent_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if _ckpts:
        _latest_ckpt = _ckpts[-1]
        _completed_steps = int(_latest_ckpt.stem.rsplit("_", 1)[-1])
        print(
            f"[train] Auto-resuming from: {_latest_ckpt.name} "
            f"({_completed_steps:,} / {args_cli.timesteps:,} steps done)"
        )
        agent.load(str(_latest_ckpt))
        args_cli.timesteps = max(0, args_cli.timesteps - _completed_steps)
        if args_cli.timesteps == 0:
            print("[train] Training already complete. Exiting.")
            import os as _os2; _os2._exit(0) if _os2.name == "nt" else exit(0)

# ---------------------------------------------------------------------------
# 8.  Trainer
# ---------------------------------------------------------------------------
trainer_cfg = {
    "timesteps": args_cli.timesteps,
    "headless":  True,
}

trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

print(
    f"\n[train] Starting PPO training"
    f"\n  Envs      : {args_cli.num_envs}"
    f"\n  Timesteps : {args_cli.timesteps:,}"
    f"\n  Seed      : {args_cli.seed}"
    f"\n  Device    : {device}"
    f"\n  Exp name  : {exp_name}"
    f"\n  Resuming  : {'yes, from step ' + str(_completed_steps) if _completed_steps else 'no'}\n"
)

trainer.train()

# ---------------------------------------------------------------------------
# 9.  Cleanup
# ---------------------------------------------------------------------------
# Isaac Sim's Omniverse/D3D12 teardown causes a hard C-level crash on Windows,
# which overrides any Python-level exit code.  All training data (TFEvents,
# checkpoints) are already flushed by skrl's trainer before this point, so we
# skip simulation_app.close() and force a clean exit via os._exit(0).
# On Linux/HPC (Greene A100) simulation_app.close() works fine, so we only
# skip it on Windows.
import os as _os
if _os.name == "nt":
    _os._exit(0)
else:
    simulation_app.close()
