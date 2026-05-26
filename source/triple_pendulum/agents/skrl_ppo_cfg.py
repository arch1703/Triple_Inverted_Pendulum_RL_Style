# PPO networks and hyperparameters for the triple pendulum (skrl 2.0)
# policy/value: 2-layer MLP with 256 hidden units and ELU activations

from __future__ import annotations

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO_CFG
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.schedulers.torch import KLAdaptiveLR


class PolicyNetwork(GaussianMixin, Model):
    """Stochastic actor: Gaussian policy with state-independent log-std."""

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
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs: dict, role: str):
        mean = self.net(inputs["observations"])
        return mean, {"log_std": self.log_std_parameter}


class ValueNetwork(DeterministicMixin, Model):
    """Critic: scalar state-value estimate V(s)."""

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


PPO_CONFIG = PPO_CFG(
    rollouts=24,
    learning_epochs=5,
    mini_batches=4,
    discount_factor=0.99,
    gae_lambda=0.95,
    learning_rate=3e-4,
    learning_rate_scheduler=KLAdaptiveLR,
    learning_rate_scheduler_kwargs={"kl_threshold": 0.008},
    ratio_clip=0.2,
    value_clip=0.2,
    value_loss_scale=1.0,
    entropy_loss_scale=0.005,
    grad_norm_clip=1.0,
)
