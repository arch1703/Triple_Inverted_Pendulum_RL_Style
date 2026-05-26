# registers the gymnasium env and exports env + cfg

import gymnasium as gym

from .triple_pendulum_env import TriplePendulumEnv
from .triple_pendulum_env_cfg import TriplePendulumEnvCfg

gym.register(
    id="Isaac-TriplePendulum-Direct-v0",
    entry_point="triple_pendulum.tasks.triple_pendulum_env:TriplePendulumEnv",
    kwargs={"cfg": TriplePendulumEnvCfg()},
    disable_env_checker=True,
)

__all__ = ["TriplePendulumEnv", "TriplePendulumEnvCfg"]
