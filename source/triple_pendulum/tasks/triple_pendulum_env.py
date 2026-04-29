"""
Triple Inverted Pendulum on a Cart – Isaac Lab DirectRLEnv
==========================================================
Implements a 4-DOF system:
  DOF 0  slider_to_cart   prismatic, actuated (cart force)
  DOF 1  cart_to_pole1    revolute,  passive
  DOF 2  pole1_to_pole2   revolute,  passive
  DOF 3  pole2_to_pole3   revolute,  passive

Observation  (8-dim): [x, ẋ, θ₁, θ̇₁, θ₂, θ̇₂, θ₃, θ̇₃]
Action       (1-dim): cart force  ∈ [-1, 1]  → scaled to ±action_scale N

Reward (per step):
  r = Σᵢ exp(-k·θᵢ²)                (upright bonus, max = 3 when all θ=0)
    − 0.01 · Σᵢ θ̇ᵢ²                (angular velocity penalty)
    − 0.01 · x²                     (cart displacement penalty)
    − 0.001 · ẋ²                    (cart velocity penalty)
    + 0.5                            (survival bonus per step)

Episode terminates if:
  - |θᵢ| > 30°  for any link         (non-recoverable fall)
  - |x|  > 2.0 m                     (cart leaves track)
  - step count ≥ max_episode_length  (time limit / truncation)

Reset:
  - θᵢ   ~ Uniform(−0.087, +0.087) rad  (±5°)
  - θ̇ᵢ  ~ Uniform(−0.05,  +0.05)  rad/s
  - cart pos/vel = 0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

if TYPE_CHECKING:
    from .triple_pendulum_env_cfg import TriplePendulumEnvCfg


class TriplePendulumEnv(DirectRLEnv):
    """Isaac Lab DirectRLEnv for a triple inverted pendulum on a cart."""

    cfg: TriplePendulumEnvCfg

    def __init__(self, cfg: TriplePendulumEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ------------------------------------------------------------------ #
        # Resolve DOF indices by joint name (done once after scene creation)  #
        # ------------------------------------------------------------------ #
        cart_ids, _   = self.robot.find_joints(self.cfg.cart_dof_name)
        pole1_ids, _  = self.robot.find_joints(self.cfg.pole1_dof_name)
        pole2_ids, _  = self.robot.find_joints(self.cfg.pole2_dof_name)
        pole3_ids, _  = self.robot.find_joints(self.cfg.pole3_dof_name)

        # Store as plain Python ints for clean tensor indexing
        self._cart_dof_idx  = cart_ids[0]
        self._pole1_dof_idx = pole1_ids[0]
        self._pole2_dof_idx = pole2_ids[0]
        self._pole3_dof_idx = pole3_ids[0]

        # Pre-allocate observation cache (reused across _get_rewards / _get_dones)
        self._cart_pos   = torch.zeros(self.num_envs, device=self.device)
        self._cart_vel   = torch.zeros(self.num_envs, device=self.device)
        self._pole1_ang  = torch.zeros(self.num_envs, device=self.device)
        self._pole1_vel  = torch.zeros(self.num_envs, device=self.device)
        self._pole2_ang  = torch.zeros(self.num_envs, device=self.device)
        self._pole2_vel  = torch.zeros(self.num_envs, device=self.device)
        self._pole3_ang  = torch.zeros(self.num_envs, device=self.device)
        self._pole3_vel  = torch.zeros(self.num_envs, device=self.device)

    # ---------------------------------------------------------------------- #
    # Scene setup                                                             #
    # ---------------------------------------------------------------------- #
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)

        # Ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                )
            ),
        )

        # Clone envs, filter collisions so carts don't collide across clones
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # Dome lighting for clean renders
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/skyLight", light_cfg, translation=(0.0, 0.0, 5.0))

        # Register robot with the scene manager
        self.scene.articulations["robot"] = self.robot

    # ---------------------------------------------------------------------- #
    # Action handling                                                         #
    # ---------------------------------------------------------------------- #
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Cache and clamp the incoming policy actions (called once per policy step)."""
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        """Apply scaled force to the cart joint (called every physics sub-step)."""
        efforts = self.cfg.action_scale * self.actions  # shape (N, 1)
        self.robot.set_joint_effort_target(efforts, joint_ids=[self._cart_dof_idx])

    # ---------------------------------------------------------------------- #
    # Observation                                                             #
    # ---------------------------------------------------------------------- #
    def _get_observations(self) -> dict:
        """Return observation dict and cache state for reward / done computation."""
        jp = self.robot.data.joint_pos  # (N, 4)
        jv = self.robot.data.joint_vel  # (N, 4)

        # Cache state components (reused in _get_rewards and _get_dones)
        self._cart_pos  = jp[:, self._cart_dof_idx]
        self._cart_vel  = jv[:, self._cart_dof_idx]
        self._pole1_ang = jp[:, self._pole1_dof_idx]
        self._pole1_vel = jv[:, self._pole1_dof_idx]
        self._pole2_ang = jp[:, self._pole2_dof_idx]
        self._pole2_vel = jv[:, self._pole2_dof_idx]
        self._pole3_ang = jp[:, self._pole3_dof_idx]
        self._pole3_vel = jv[:, self._pole3_dof_idx]

        obs = torch.stack(
            [
                self._cart_pos,
                self._cart_vel,
                self._pole1_ang,
                self._pole1_vel,
                self._pole2_ang,
                self._pole2_vel,
                self._pole3_ang,
                self._pole3_vel,
            ],
            dim=-1,
        )  # shape (N, 8)

        return {"policy": obs}

    # ---------------------------------------------------------------------- #
    # Reward                                                                  #
    # ---------------------------------------------------------------------- #
    def _get_rewards(self) -> torch.Tensor:
        """
        Shaped reward designed to encourage upright balance and small motions.

        r = rew_upright_scale  * Σᵢ exp(-upright_kernel · θᵢ²)
          + rew_ang_vel_scale  * Σᵢ θ̇ᵢ²
          + rew_cart_pos_scale * x²
          + rew_cart_vel_scale * ẋ²
          + rew_survive_scale

        Maximum possible value (all terms at optimum): 3 * 1 + 0 + 0 + 0 + 0.5 = 3.5
        """
        k = self.cfg.upright_kernel

        r_upright = self.cfg.rew_upright_scale * (
            torch.exp(-k * self._pole1_ang ** 2)
            + torch.exp(-k * self._pole2_ang ** 2)
            + torch.exp(-k * self._pole3_ang ** 2)
        )

        r_ang_vel = self.cfg.rew_ang_vel_scale * (
            self._pole1_vel ** 2
            + self._pole2_vel ** 2
            + self._pole3_vel ** 2
        )

        r_cart_pos = self.cfg.rew_cart_pos_scale * self._cart_pos ** 2
        r_cart_vel = self.cfg.rew_cart_vel_scale * self._cart_vel ** 2
        r_survive  = torch.full(
            (self.num_envs,), self.cfg.rew_survive_scale, device=self.device
        )

        return r_upright + r_ang_vel + r_cart_pos + r_cart_vel + r_survive

    # ---------------------------------------------------------------------- #
    # Termination                                                             #
    # ---------------------------------------------------------------------- #
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            terminated: env failed due to physics condition (fall / out-of-bounds)
            time_out  : env reached max episode length (success/truncation)
        """
        pole_fall = (
            (torch.abs(self._pole1_ang) > self.cfg.max_pole_angle)
            | (torch.abs(self._pole2_ang) > self.cfg.max_pole_angle)
            | (torch.abs(self._pole3_ang) > self.cfg.max_pole_angle)
        )
        cart_oob  = torch.abs(self._cart_pos) > self.cfg.max_cart_pos
        terminated = pole_fall | cart_oob
        time_out   = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ---------------------------------------------------------------------- #
    # Reset                                                                   #
    # ---------------------------------------------------------------------- #
    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """Reset selected environments to a near-upright initial state with noise."""
        if len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        n = len(env_ids)

        # Start from the articulation's default joint state
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()  # (n, 4)
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()  # (n, 4)

        # Cart: zero position and velocity
        joint_pos[:, self._cart_dof_idx] = 0.0
        joint_vel[:, self._cart_dof_idx] = 0.0

        # Pole angles: uniform noise within ±5° around upright (0 rad)
        angle_noise = self.cfg.init_angle_noise
        joint_pos[:, self._pole1_dof_idx] = (
            torch.rand(n, device=self.device) * 2 * angle_noise - angle_noise
        )
        joint_pos[:, self._pole2_dof_idx] = (
            torch.rand(n, device=self.device) * 2 * angle_noise - angle_noise
        )
        joint_pos[:, self._pole3_dof_idx] = (
            torch.rand(n, device=self.device) * 2 * angle_noise - angle_noise
        )

        # Pole velocities: small uniform noise
        vel_noise = self.cfg.init_vel_noise
        joint_vel[:, self._pole1_dof_idx] = (
            torch.rand(n, device=self.device) * 2 * vel_noise - vel_noise
        )
        joint_vel[:, self._pole2_dof_idx] = (
            torch.rand(n, device=self.device) * 2 * vel_noise - vel_noise
        )
        joint_vel[:, self._pole3_dof_idx] = (
            torch.rand(n, device=self.device) * 2 * vel_noise - vel_noise
        )

        # Write directly to simulation state (bypasses actuator targets)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
