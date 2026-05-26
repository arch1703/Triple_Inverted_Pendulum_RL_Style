# Isaac Lab environment configuration for the triple inverted pendulum on a cart
# all hyperparameters (physics, reward, termination, spaces) are in one place

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

_ASSET_DIR = Path(__file__).parent.parent / "assets"
_URDF_PATH = str(_ASSET_DIR / "triple_pendulum_cart.urdf")


TRIPLE_PENDULUM_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/TriplePendulum",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_URDF_PATH,
        activate_contact_sensors=False,
        # USD conversion output dir – /tmp doesn't exist on Windows, use system temp
        usd_dir=os.path.join(tempfile.gettempdir(), "IsaacLab"),
        fix_base=False,
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
        pos=(0.0, 0.0, 0.15),
        joint_pos={
            "slider_to_cart": 0.0,
            "cart_to_pole1": 0.0,
            "pole1_to_pole2": 0.0,
            "pole2_to_pole3": 0.0,
        },
        joint_vel={
            "slider_to_cart": 0.0,
            "cart_to_pole1": 0.0,
            "pole1_to_pole2": 0.0,
            "pole2_to_pole3": 0.0,
        },
    ),
    actuators={
        # cart is force-controlled; stiffness=0 means pure torque/force control
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_cart"],
            effort_limit=30.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        # pole joints are passive - physics-only with small viscous damping
        "passive_poles": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole1", "pole1_to_pole2", "pole2_to_pole3"],
            effort_limit=0.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.005,
        ),
    },
)

@configclass
class TriplePendulumEnvCfg(DirectRLEnvCfg):
    """Configuration for the triple inverted pendulum on a cart.

    obs: [cart_pos, cart_vel, theta1, theta1_dot, theta2, theta2_dot, theta3, theta3_dot]
    action: cart force in [-1, 1] scaled by action_scale
    """

    observation_space: int = 8
    action_space: int = 1
    state_space: int = 0

    decimation: int = 2
    episode_length_s: float = 10.0

    action_scale: float = 20.0

    max_cart_pos: float = 2.0
    max_pole_angle: float = math.pi / 6

    rew_upright_scale: float = 1.0
    rew_ang_vel_scale: float = -0.01
    rew_cart_pos_scale: float = -0.01
    rew_cart_vel_scale: float = -0.001
    rew_survive_scale: float = 0.5
    upright_kernel: float = 10.0

    init_angle_noise: float = 0.087
    init_vel_noise: float = 0.05

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=3.0,
        replicate_physics=True,
    )

    robot_cfg: ArticulationCfg = TRIPLE_PENDULUM_CFG.replace(
        prim_path="/World/envs/env_.*/TriplePendulum"
    )

    cart_dof_name: str = "slider_to_cart"
    pole1_dof_name: str = "cart_to_pole1"
    pole2_dof_name: str = "pole1_to_pole2"
    pole3_dof_name: str = "pole2_to_pole3"
