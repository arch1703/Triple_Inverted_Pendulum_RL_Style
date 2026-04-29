"""Triple pendulum tasks package.

Importing this module registers the gymnasium environment so that both
``gymnasium.make("Isaac-TriplePendulum-Direct-v0")`` and direct instantiation
via ``TriplePendulumEnv(cfg=...)`` work correctly.
"""

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
