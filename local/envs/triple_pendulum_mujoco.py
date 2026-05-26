# MuJoCo gymnasium env for triple inverted pendulum on a cart.
# obs: [x, xd, t1, t1d, t2, t2d, t3, t3d]  action: force in [-1,1] * 20N

from __future__ import annotations
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco.viewer

_XML_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "triple_pendulum.xml")

ACTION_SCALE = 20.0       # N
MAX_CART = 2.0            # m
MAX_ANGLE = np.pi / 12    # 15 deg
MAX_STEPS = 500
_SUBSTEPS = 2             # 100Hz physics, 50Hz policy

_K_UPRIGHT = 10.0
_W_ANG_VEL = 0.01
_W_CART_POS = 0.01
_W_CART_VEL = 0.001
_W_SURVIVE = 0.5
_W_ANGLE = 0.1


class TriplePendulumMuJoCoEnv(gym.Env):
    # render_mode: None (training), "human" (live window), "rgb_array" (recording)

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        xml_path = os.path.abspath(_XML_PATH)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.model.opt.timestep = 0.01

        obs_high = np.array([3.0, 20.0, np.pi, 20.0, np.pi, 20.0, np.pi, 20.0], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            np.array([-1.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        assert render_mode in (None, "human", "rgb_array"), f"bad render_mode: {render_mode}"
        self.render_mode = render_mode
        self._viewer = None
        self._renderer = None

        self._step_count = 0
        self._reset_range = 0.017  # ~1 deg, widened by curriculum

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        rng = self.np_random
        for i in range(1, 4):
            self.data.qpos[i] = rng.uniform(-self._reset_range, self._reset_range)
            self.data.qvel[i] = rng.uniform(-0.05, 0.05)

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

        upright = sum(np.exp(-_K_UPRIGHT * t**2) for t in (t1, t2, t3))
        ang_pen = _W_ANGLE * (abs(t1) + abs(t2) + abs(t3))
        vel_pen = _W_ANG_VEL * (t1d**2 + t2d**2 + t3d**2)
        reward = float(upright - ang_pen - vel_pen - _W_CART_POS * x**2 - _W_CART_VEL * xd**2 + _W_SURVIVE)

        pole_fall = abs(t1) > MAX_ANGLE or abs(t2) > MAX_ANGLE or abs(t3) > MAX_ANGLE
        terminated = bool(pole_fall or abs(x) > MAX_CART)
        truncated = self._step_count >= MAX_STEPS

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

    def set_reset_range(self, range_rad: float):
        self._reset_range = range_rad

    def _get_obs(self) -> np.ndarray:
        x, t1, t2, t3 = self.data.qpos
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
        return self._renderer.render()
