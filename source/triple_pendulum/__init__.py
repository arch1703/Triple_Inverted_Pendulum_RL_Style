# triple_pendulum: importing this package triggers task registration via the tasks subpackage

from triple_pendulum.tasks import TriplePendulumEnv, TriplePendulumEnvCfg

__all__ = ["TriplePendulumEnv", "TriplePendulumEnvCfg"]
