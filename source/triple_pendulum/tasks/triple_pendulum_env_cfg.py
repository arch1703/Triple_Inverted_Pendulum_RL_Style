"""
Triple Pendulum Cart – Isaac Lab Environment Configuration
==========================================================
All hyperparameters that control the environment (physics, rewards, termination,
observation/action spaces) are gathered here.  The training script and the env
class both import from this module, so there is a single source of truth.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Absolute path to the URDF so the package can be run from any working directory
_ASSET_DIR = Path(__file__).parent.parent / "assets"
_URDF_PATH = str(_ASSET_DIR / "triple_pendulum_cart.urdf")


# ---------------------------------------------------------------------------
# Robot articulation configuration
# ---------------------------------------------------------------------------
TRIPLE_PENDULUM_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/TriplePendulum",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_URDF_PATH,
        activate_contact_sensors=False,
        # USD conversion output dir – /tmp doesn't exist on Windows, use system temp
        usd_dir=os.path.join(tempfile.gettempdir(), "IsaacLab"),
        # Required by UrdfConverterCfg: base is NOT fixed (cart can slide)
        fix_base=False,
        # Required by UrdfConverterCfg: set joint drive to none so all DOFs
        # are purely physics-driven; our ImplicitActuatorCfg handles the cart.
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="none",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Cart sits 0.15 m above ground so its bottom edge is flush with floor
        pos=(0.0, 0.0, 0.15),
        joint_pos={
            "slider_to_cart": 0.0,
            "cart_to_pole1":  0.0,
            "pole1_to_pole2": 0.0,
            "pole2_to_pole3": 0.0,
        },
        joint_vel={
            "slider_to_cart": 0.0,
            "cart_to_pole1":  0.0,
            "pole1_to_pole2": 0.0,
            "pole2_to_pole3": 0.0,
        },
    ),
    actuators={
        # Cart is force-controlled; stiffness=0 means pure torque/force control
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_cart"],
            effort_limit=30.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        # Pole joints are passive – physics-only with small viscous damping
        "passive_poles": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole1", "pole1_to_pole2", "pole2_to_pole3"],
            effort_limit=0.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.005,
        ),
    },
)


# ---------------------------------------------------------------------------
# Main environment configuration
# ---------------------------------------------------------------------------
@configclass
class TriplePendulumEnvCfg(DirectRLEnvCfg):
    """Configuration for the triple inverted pendulum on a cart.

    Observation vector (8 dims, in order):
        [cart_pos, cart_vel, θ₁, θ̇₁, θ₂, θ̇₂, θ₃, θ̇₃]

    Action vector (1 dim):
        [cart_force]  ∈ [-1, 1]  scaled by ``action_scale`` to Newtons.
    """

    # ------------------------------------------------------------------
    # Spaces  (class variables, not dataclass fields – Isaac Lab convention)
    # ------------------------------------------------------------------
    observation_space: int = 8
    action_space: int = 1
    state_space: int = 0  # no asymmetric critic

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    # Physics runs at 100 Hz; policy runs at 50 Hz (decimation = 2)
    decimation: int = 2
    episode_length_s: float = 10.0  # → max 500 policy steps per episode

    # ------------------------------------------------------------------
    # Action scaling
    # ------------------------------------------------------------------
    action_scale: float = 20.0  # policy output [-1,1] → force in N

    # ------------------------------------------------------------------
    # Termination thresholds
    # ------------------------------------------------------------------
    max_cart_pos: float = 2.0            # |x|  > this → failure
    max_pole_angle: float = math.pi / 6  # |θᵢ| > 30°  → failure

    # ------------------------------------------------------------------
    # Reward weights
    # ------------------------------------------------------------------
    rew_upright_scale: float = 1.0    # coefficient on exp(-10 θᵢ²) sum
    rew_ang_vel_scale: float = -0.01  # penalise pole angular velocities
    rew_cart_pos_scale: float = -0.01  # penalise cart displacement
    rew_cart_vel_scale: float = -0.001  # penalise cart velocity
    rew_survive_scale: float = 0.5    # per-step survival bonus

    # Exponent inside the upright Gaussian kernel: exp(-k * θ²)
    upright_kernel: float = 10.0

    # ------------------------------------------------------------------
    # Initialisation noise (reset distribution)
    # ------------------------------------------------------------------
    init_angle_noise: float = 0.087   # ±5° in radians
    init_vel_noise: float = 0.05      # ±0.05 rad/s for pole velocities

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,          # 100 Hz physics
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
    )

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # ------------------------------------------------------------------
    # Robot
    # ------------------------------------------------------------------
    robot_cfg: ArticulationCfg = TRIPLE_PENDULUM_CFG.replace(
        prim_path="/World/envs/env_.*/TriplePendulum"
    )

    # ------------------------------------------------------------------
    # Joint name keys (used in env to look up DOF indices)
    # ------------------------------------------------------------------
    cart_dof_name: str = "slider_to_cart"
    pole1_dof_name: str = "cart_to_pole1"
    pole2_dof_name: str = "pole1_to_pole2"
    pole3_dof_name: str = "pole2_to_pole3"
