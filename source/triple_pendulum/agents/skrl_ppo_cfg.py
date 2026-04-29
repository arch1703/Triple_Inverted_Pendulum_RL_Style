"""PPO agent configuration and neural network models for the triple pendulum.

Architecture
------------
Both the policy and value networks are two-layer MLPs with 256 hidden units.
The policy outputs a diagonal Gaussian distribution over the cart-force action;
the value network outputs a scalar state-value estimate.

PPO hyperparameters are tuned for Isaac Lab parallel environments (512+ envs):
  - Short rollout horizon  (24 steps) to stay on-policy
  - 5 learning epochs per rollout for sample efficiency
  - Entropy bonus (0.005) encourages early exploration
  - Linear LR decay prevents overshooting in late training
"""

from __future__ import annotations

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO_CFG
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.schedulers.torch import KLAdaptiveLR


# ---------------------------------------------------------------------------
# Neural network models
# ---------------------------------------------------------------------------

class PolicyNetwork(GaussianMixin, Model):
    """Stochastic actor: outputs mean and learned log-std for the cart force.

    The log-std is a free parameter (not state-dependent), which is standard
    for continuous-control PPO.  ``clip_actions=True`` maps the sampled action
    into [-1, 1] via tanh.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
    ) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std, min_log_std=min_log_std, max_log_std=max_log_std, reduction=reduction)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, self.num_actions),
        )
        # Shared log-std parameter initialised to 0 (std ≈ 1)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs: dict, role: str):
        mean = self.net(inputs["observations"])
        return mean, {"log_std": self.log_std_parameter}


class ValueNetwork(DeterministicMixin, Model):
    """Critic: outputs a scalar state-value estimate V(s)."""

    def __init__(self, observation_space, action_space, device, clip_actions: bool = False) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs: dict, role: str):
        return self.net(inputs["observations"]), {}


# ---------------------------------------------------------------------------
# PPO hyperparameter configuration  (skrl 2.0 dataclass API)
# ---------------------------------------------------------------------------

PPO_CONFIG = PPO_CFG(
    # ---- Rollout -----------------------------------------------------------
    rollouts=24,            # steps collected per env before each update
    learning_epochs=5,      # gradient passes over the collected rollout
    mini_batches=4,         # mini-batch count per learning epoch

    # ---- Discount & advantage ----------------------------------------------
    discount_factor=0.99,
    gae_lambda=0.95,        # GAE-λ  (was "lambda" in skrl <2.0)

    # ---- Learning rate -----------------------------------------------------
    learning_rate=3e-4,
    # KL-adaptive scheduler automatically shrinks lr if the policy update
    # becomes too aggressive (standard in Isaac Lab PPO examples).
    learning_rate_scheduler=KLAdaptiveLR,
    learning_rate_scheduler_kwargs={"kl_threshold": 0.008},

    # ---- PPO clipping -------------------------------------------------------
    ratio_clip=0.2,
    value_clip=0.2,

    # ---- Loss coefficients --------------------------------------------------
    value_loss_scale=1.0,
    entropy_loss_scale=0.005,

    # ---- Gradient -----------------------------------------------------------
    grad_norm_clip=1.0,
)
