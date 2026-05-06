"""
Triple Inverted Pendulum – MuJoCo Gymnasium Environment
========================================================
Physics is identical to the Isaac Lab version (same URDF parameters, same
reward function, same termination conditions, same reset distribution).

Observation (8-dim): [x, ẋ, θ₁, θ̇₁, θ₂, θ̇₂, θ₃, θ̇₃]
Action     (1-dim):  [F_norm] ∈ [-1, 1]  →  F = F_norm × 20 N on cart

Install:  pip install mujoco>=3.0 gymnasium>=0.29
"""

from __future__ import annotations
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco.viewer

# ---------------------------------------------------------------------------
# Constants (match triple_pendulum_env_cfg.py exactly)
# ---------------------------------------------------------------------------
_XML_PATH   = os.path.join(os.path.dirname(__file__), "..", "models", "triple_pendulum.xml")
ACTION_SCALE = 20.0          # N  (action_scale in cfg)
MAX_CART     = 2.0           # m  (max_cart_pos)
MAX_ANGLE    = np.pi / 6     # rad  (max_pole_angle ≈ 0.5236 rad, 30°)
MAX_STEPS    = 500           # episode_length_s=10s @ 50Hz policy
_SUBSTEPS    = 2             # decimation (100 Hz physics / 50 Hz policy)

# Reward weights (rew_*_scale in cfg)
_K_UPRIGHT   = 10.0          # upright_kernel
_W_ANG_VEL   = 0.01          # |rew_ang_vel_scale|
_W_CART_POS  = 0.01          # |rew_cart_pos_scale|
_W_CART_VEL  = 0.001         # |rew_cart_vel_scale|
_W_SURVIVE   = 0.5           # rew_survive_scale


class TriplePendulumMuJoCoEnv(gym.Env):
    """
    MuJoCo-backed triple inverted pendulum on a cart.

    render_mode options
    -------------------
    None        – no rendering (fastest, use for training)
    "human"     – live OpenGL window (interactive, requires display)
    "rgb_array" – returns (H, W, 3) uint8 frames; use for video recording
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        xml_path = os.path.abspath(_XML_PATH)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # Enforce timestep (MJCF sets 0.01, but be explicit)
        self.model.opt.timestep = 0.01

        # ----- spaces -----
        obs_high = np.array([3.0, 20.0, np.pi, 20.0,
                              np.pi, 20.0, np.pi, 20.0], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(
            np.array([-1.0], dtype=np.float32),
            np.array([ 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ----- render state -----
        assert render_mode in (None, "human", "rgb_array"), \
            f"Unsupported render_mode: {render_mode}"
        self.render_mode = render_mode
        self._viewer:    mujoco.viewer.Handle | None = None
        self._renderer:  mujoco.Renderer      | None = None

        self._step_count = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        rng = self.np_random
        # Small random angles near upright (±5°) – matches Isaac Lab reset
        self.data.qpos[1] = rng.uniform(-0.087, 0.087)
        self.data.qpos[2] = rng.uniform(-0.087, 0.087)
        self.data.qpos[3] = rng.uniform(-0.087, 0.087)
        # Small random angular velocities
        self.data.qvel[1] = rng.uniform(-0.05, 0.05)
        self.data.qvel[2] = rng.uniform(-0.05, 0.05)
        self.data.qvel[3] = rng.uniform(-0.05, 0.05)

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        if self.render_mode == "human":
            self._render_human()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        F = float(np.clip(action[0], -1.0, 1.0)) * ACTION_SCALE
        self.data.ctrl[0] = F

        for _ in range(_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        # qpos = [x, θ₁, θ₂, θ₃],  qvel = [ẋ, θ̇₁, θ̇₂, θ̇₃]
        x,  t1, t2, t3  = self.data.qpos
        xd, t1d, t2d, t3d = self.data.qvel

        # ---- Reward (identical to Isaac Lab) ----
        reward = float(
            np.exp(-_K_UPRIGHT * t1**2)
            + np.exp(-_K_UPRIGHT * t2**2)
            + np.exp(-_K_UPRIGHT * t3**2)
            - _W_ANG_VEL  * (t1d**2 + t2d**2 + t3d**2)
            - _W_CART_POS * x**2
            - _W_CART_VEL * xd**2
            + _W_SURVIVE
        )

        # ---- Termination ----
        pole_fall  = abs(t1) > MAX_ANGLE or abs(t2) > MAX_ANGLE or abs(t3) > MAX_ANGLE
        cart_out   = abs(x) > MAX_CART
        terminated = bool(pole_fall or cart_out)
        truncated  = self._step_count >= MAX_STEPS

        if self.render_mode == "human":
            self._render_human()

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        if self.render_mode == "human":
            self._render_human()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Return [x, ẋ, θ₁, θ̇₁, θ₂, θ̇₂, θ₃, θ̇₃] as float32."""
        x,  t1, t2, t3  = self.data.qpos
        xd, t1d, t2d, t3d = self.data.qvel
        return np.array([x, xd, t1, t1d, t2, t2d, t3, t3d], dtype=np.float32)

    def _render_human(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def _render_rgb_array(self, width: int = 640, height: int = 480) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera="side_track")
        return self._renderer.render()  # (H, W, 3) uint8
