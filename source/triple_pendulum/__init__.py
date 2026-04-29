"""Triple pendulum top-level package.

Importing ``triple_pendulum`` triggers task registration via the tasks subpackage.
"""

from triple_pendulum.tasks import TriplePendulumEnv, TriplePendulumEnvCfg

__all__ = ["TriplePendulumEnv", "TriplePendulumEnvCfg"]
