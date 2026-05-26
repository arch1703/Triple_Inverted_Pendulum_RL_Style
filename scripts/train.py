# PPO training for triple inverted pendulum via Isaac Lab / skrl
# usage: python scripts/train.py --num_envs 512 --headless --seed 42
# HPC: python scripts/train.py --num_envs 512 --headless --seed 0 --timesteps 3000000

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Windows: pre-load torch DLLs before Isaac Sim does to avoid WinError 1114
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
    import torch as _torch_win_preload
    _torch_win_preload.zeros(1)
    del _torch_win_preload

# AppLauncher must come before all Isaac Sim / Lab imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train triple pendulum PPO")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timesteps", type=int, default=3_000_000)
parser.add_argument("--experiment", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# enable URDF importer (Isaac Sim 4.x doesn't auto-enable it)
import omni.kit.app as _kit_app
_ext_mgr = _kit_app.get_app().get_extension_manager()
if not _ext_mgr.is_extension_enabled("isaacsim.asset.importer.urdf"):
    _ext_mgr.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
del _ext_mgr, _kit_app

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "source"))
import triple_pendulum

from triple_pendulum.agents.skrl_ppo_cfg import PPO_CONFIG, PolicyNetwork, ValueNetwork
from triple_pendulum.tasks.triple_pendulum_env import TriplePendulumEnv
from triple_pendulum.tasks.triple_pendulum_env_cfg import TriplePendulumEnvCfg

from skrl.agents.torch.ppo import PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

set_seed(args_cli.seed)

env_cfg = TriplePendulumEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs

env = TriplePendulumEnv(cfg=env_cfg, render_mode=None)
env = wrap_env(env)
device = env.device

models = {
    "policy": PolicyNetwork(env.observation_space, env.action_space, device),
    "value": ValueNetwork(env.observation_space, env.action_space, device),
}

memory = RandomMemory(
    memory_size=PPO_CONFIG.rollouts,
    num_envs=env.num_envs,
    device=device,
)

exp_name = f"triple_pendulum_ppo_seed{args_cli.seed}"
if args_cli.experiment:
    exp_name += f"_{args_cli.experiment}"

PPO_CONFIG.experiment.experiment_name = exp_name
PPO_CONFIG.experiment.directory = str(Path(__file__).parent.parent / "results" / "runs")
PPO_CONFIG.experiment.checkpoint_interval = 100_000
PPO_CONFIG.experiment.write_interval = 1000

agent = PPO(
    models=models,
    memory=memory,
    cfg=PPO_CONFIG,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# auto-resume from latest checkpoint if the job was preempted
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
        print(f"resuming from {_latest_ckpt.name} ({_completed_steps:,}/{args_cli.timesteps:,} done)")
        agent.load(str(_latest_ckpt))
        args_cli.timesteps = max(0, args_cli.timesteps - _completed_steps)
        if args_cli.timesteps == 0:
            print("training already complete")
            import os as _os2; _os2._exit(0) if _os2.name == "nt" else exit(0)

trainer = SequentialTrainer(
    cfg={"timesteps": args_cli.timesteps, "headless": True},
    env=env,
    agents=agent,
)

print(f"starting: envs={args_cli.num_envs} steps={args_cli.timesteps:,} seed={args_cli.seed} device={device}")
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
