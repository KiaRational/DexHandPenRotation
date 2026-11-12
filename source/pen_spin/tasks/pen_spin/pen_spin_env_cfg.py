"""Pen spinning with the Sharpa Wave (22-DoF, right hand) -- Isaac Lab 3.0 / Isaac Sim 6.2.

The hand asset, joint names, init pose and PD scheme follow the official sharpa-rl-lab
in-hand rotation task (Apache-2.0). The pen is a procedural cylinder (no mesh files).

Conventions (Isaac Lab 3.0): quaternions are (x, y, z, w). The policy observation gives the pen
quaternion as (w, x, y, z), in the joint order listed in ACTUATED_JOINTS.
"""

from __future__ import annotations

import math
import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SHARPA_USD = os.path.join(_PROJ_ROOT, "assets", "SharpaWave", "right_sharpa_wave.usda")

# Policy action / observation joint order (PhysX breadth-first order), pinned so the
# pretrained checkpoint stays valid.
ACTUATED_JOINTS = [
    "right_index_MCP_FE", "right_middle_MCP_FE", "right_pinky_CMC", "right_ring_MCP_FE", "right_thumb_CMC_FE",
    "right_index_MCP_AA", "right_middle_MCP_AA", "right_pinky_MCP_FE", "right_ring_MCP_AA", "right_thumb_CMC_AA",
    "right_index_PIP", "right_middle_PIP", "right_pinky_MCP_AA", "right_ring_PIP", "right_thumb_MCP_FE",
    "right_index_DIP", "right_middle_DIP", "right_pinky_PIP", "right_ring_DIP", "right_thumb_MCP_AA",
    "right_pinky_DIP", "right_thumb_IP",
]
FINGERTIPS = [
    "right_thumb_fingertip", "right_index_fingertip", "right_middle_fingertip",
    "right_ring_fingertip", "right_pinky_fingertip",
]

_deg = math.pi / 180.0
# Default grasp posture from sharpa-rl-lab (degrees -> rad).
DEFAULT_JOINT_POS = {
    "right_thumb_CMC_FE": 95.12771 * _deg, "right_thumb_CMC_AA": -3.11244 * _deg,
    "right_thumb_MCP_FE": 14.81626 * _deg, "right_thumb_MCP_AA": -1.03493 * _deg, "right_thumb_IP": 12.23986 * _deg,
    "right_index_MCP_FE": 65.21091 * _deg, "right_index_MCP_AA": 6.1133 * _deg,
    "right_index_PIP": 15.58495 * _deg, "right_index_DIP": 5.90325 * _deg,
    "right_middle_MCP_FE": 31.74149 * _deg, "right_middle_MCP_AA": -0.95812 * _deg,
    "right_middle_PIP": 41.88173 * _deg, "right_middle_DIP": 12.844 * _deg,
    "right_ring_MCP_FE": 31.72383 * _deg, "right_ring_MCP_AA": 9.84458 * _deg,
    "right_ring_PIP": 35.22366 * _deg, "right_ring_DIP": 18.02839 * _deg,
    "right_pinky_CMC": 10.9712 * _deg, "right_pinky_MCP_FE": 68.30895 * _deg,
    "right_pinky_MCP_AA": 7.99151 * _deg, "right_pinky_PIP": 5.89626 * _deg, "right_pinky_DIP": 5.89875 * _deg,
}

# PD gains from right_sharpa_wave.usda (drive:angular:physics:*, per degree), converted to per radian.
_R = 180.0 / math.pi
_USD_GAINS = {  # joint regex: (stiffness [Nm/deg], damping [Nms/deg])
    ".*_(index|middle|ring|pinky|thumb)_MCP_FE": (0.083078, 0.0031940),
    ".*_(index|middle|ring|pinky|thumb)_MCP_AA": (0.11557, 3.63e-3),
    ".*_(index|middle|ring|pinky)_PIP": (0.01584, 6.98e-4),
    "right_thumb_IP": (0.01584, 6.98e-4),
    ".*_DIP": (0.01578, 5.50e-4),
    "right_pinky_CMC": (0.02409, 6.85e-4),
    "right_thumb_CMC_FE": (0.12138, 4.206e-3),
    "right_thumb_CMC_AA": (0.2304, 7.8826e-3),
}
SHARPA_KP = {k: v[0] * _R for k, v in _USD_GAINS.items()}
SHARPA_KD = {k: v[1] * _R for k, v in _USD_GAINS.items()}

PEN_RADIUS = 0.0045    # 9 mm diameter
PEN_LENGTH = 0.14      # 14 cm
PEN_MASS = 0.015       # 15 g

ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=SHARPA_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, -0.5735764, 0.0, 0.819152),  # (x,y,z,w) == sharpa-rl-lab (w,x,y,z)=(0.819,0,-0.574,0)
        joint_pos=DEFAULT_JOINT_POS,
    ),
    # Explicit PD with Sharpa's tuned gains. In Isaac Lab 3.0, stiffness/damping=None resolved to 0
    # (limp hand), so the USD drive values are given here explicitly (USD: per degree -> x 180/pi).
    actuators={"joints": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=SHARPA_KP, damping=SHARPA_KD)},
    soft_joint_pos_limit_factor=1.0,
)

PEN_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/object",
    spawn=sim_utils.CylinderCfg(
        radius=PEN_RADIUS,
        height=PEN_LENGTH,
        axis="X",  # pen long axis = local +X
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            max_depenetration_velocity=1000.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=PEN_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.7),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.22, 0.75), roughness=0.25, metallic=0.1),
    ),
    # same spot sharpa-rl-lab places its cylinder in the grasp
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.09559, -0.00517, 0.61906), rot=(0.0, 0.0, 0.0, 1.0)),
)


@configclass
class PenSpinSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", collision_group=-1, spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = ROBOT_CFG
    object: RigidObjectCfg = PEN_CFG
    # net contact force on each fingertip elastomer (tactile-like signal, 5 values)
    contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_.*_elastomer", history_length=1)
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )


@configclass
class EventCfg:
    """Domain randomization. Hand friction is NOT randomized: the USD has tuned elastomer vs metal materials."""

    pen_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    pen_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.7, 1.4),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class PenSpinEnvCfg(DirectRLEnvCfg):
    # ---------------- env ----------------
    episode_length_s = 20.0
    decimation = 6                 # 120 Hz physics / 6 = 20 Hz policy
    action_space = 22
    observation_space = 130        # see PenSpinEnv._get_observations
    state_space = 0
    clip_actions = 1.0
    clip_obs = 5.0
    action_scale = 1.0 / 24.0      # relative joint-target delta per step (rad)
    dof_limits_scale = 0.9

    # ---------------- sim ----------------
    # Isaac Sim's PhysX pinned (not the "physx" auto-preset, which switches engines between
    # headless training and rendering -- see microduck notes).
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=5 * 2**18,
        ),
    )
    scene: PenSpinSceneCfg = PenSpinSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=True)
    events: EventCfg = EventCfg()

    actuated_joint_names = ACTUATED_JOINTS
    fingertip_body_names = FINGERTIPS
    pen_radius = PEN_RADIUS
    pen_length = PEN_LENGTH

    # ---------------- reset / init ----------------
    # Grasp cache from scripts/gen_grasp.py: rows = [22 joints (ACTUATED_JOINTS order), pen pos (3),
    # pen quat (w,x,y,z)]. If missing, reset = default pose + noise.
    grasp_cache_path = os.path.join(_PROJ_ROOT, "cache", "pen_grasps.npy")
    reset_joint_noise = 0.05       # rad, used only without a cache
    reset_yaw_range = math.pi      # random initial pen yaw (+/-), used only without a cache
    fall_height = 0.05             # terminate if pen centre drops this far below its start
    max_pen_drift = 0.08           # terminate if pen centre moves this far (m) from its start
    max_tilt = 50.0 * _deg         # terminate if pen axis tilts more than this from horizontal

    # ---------------- reward ----------------
    # Spin reward: heading change of the pen's long axis over each policy step, divided by step_dt,
    # clipped symmetrically to +-rotate_clip (rad/s).
    rotate_clip = 3.0
    rotate_reward_scale = 1.0
    tilt_penalty_scale = -1.0      # |pen_axis . z|
    pos_penalty_scale = -5.0       # ||pen_pos - start||
    linvel_penalty_scale = -0.3
    pose_diff_penalty_scale = -0.1  # ||q - q_default||^2
    torque_penalty_scale = -0.01
    work_penalty_scale = -0.05
    action_rate_penalty_scale = -0.01
    fall_penalty = -10.0

    # ---------------- noise / perturbation ----------------
    joint_obs_noise = 0.01
    random_force_scale = 0.5       # in units of pen weight
    random_force_prob = 0.05

    # ---------------- curriculum ----------------
    # start with weak gravity and ramp up once the policy stops dropping the pen
    gravity_curriculum = True
    gravity_start = 2.0
    gravity_step = 0.25
    gravity_max = 9.81
    gravity_drop_rate_threshold = 0.002   # per-step fraction of envs that fell


@configclass
class PenSpinEnvCfg_PLAY(PenSpinEnvCfg):
    """Evaluation / video: full gravity, no perturbations, few envs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.gravity_curriculum = False
        self.random_force_scale = 0.0
        self.joint_obs_noise = 0.0
