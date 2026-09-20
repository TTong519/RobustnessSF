"""Named constants for the SO-ARM100 pick-and-place experiment.

Single source of truth for joint/actuator names and order, the home
keyframe, the verified ctrlrange matrix, scene element names, and the
pinned physics/task constants. Values are extracted verbatim from the
read-only upstream model (trs_so_arm100/so_arm100.xml) or pinned by the
Phase 1 decisions (see .planning/phases/01-task-scene-base-env-physics-gate/01-CONTEXT.md).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Joint / actuator names and order (upstream so_arm100.xml <actuator> order)
# ---------------------------------------------------------------------------

JOINT_NAMES: list[str] = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]

# `home` keyframe (upstream so_arm100.xml keyframe block): ctrl == qpos.
HOME_QPOS: list[float] = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
HOME_CTRL: list[float] = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]

# Verified actuator ctrlrange (RESEARCH.md Pitfall 3) — asymmetric bounds come
# from upstream inheritrange; never hand-normalize. Phase 5's noise wrapper
# clips to these bounds.
ACTUATOR_CTRLRANGE: list[list[float]] = [
    [-1.92, 1.92],    # Rotation
    [-3.32, 0.174],   # Pitch    <- asymmetric
    [-0.174, 3.14],   # Elbow    <- asymmetric
    [-1.66, 1.66],    # Wrist_Pitch
    [-2.79, 2.79],    # Wrist_Roll
    [-0.174, 1.75],   # Jaw      <- asymmetric
]

# ---------------------------------------------------------------------------
# Scene element names (src/so100rob/assets/scene_pickplace.xml)
# ---------------------------------------------------------------------------

CUBE_JOINT: str = "cube_joint"
CUBE_GEOM: str = "cube_geom"
GOAL_SITE: str = "goal_site"
CAMERA_NAME: str = "overhead"
HOME_KEY: str = "home"

SCENE_XML_PATH: Path = Path(__file__).parent / "assets" / "scene_pickplace.xml"

# ---------------------------------------------------------------------------
# Pinned physics constants (ENV-06)
# ---------------------------------------------------------------------------

PHYSICS_TIMESTEP: float = 0.002

# Default control rate: 0.002 s * 25 = 0.05 s control period = 20 Hz
# (A1 resolution; Phase 5 PERT-01 derives its magnitudes from this constant).
DEFAULT_FRAME_SKIP: int = 25

EPISODE_SIM_SECONDS: float = 10.0

# ---------------------------------------------------------------------------
# Pinned task constants (Phase 1 decisions D-02, D-03, D-04)
# ---------------------------------------------------------------------------

# "Successfully placed" = cube center within 2 cm (XY) of the goal site (D-04).
SUCCESS_TOLERANCE_XY: float = 0.02

# Fixed goal site (D-02): 10 cm to the +x side of the spawn patch center.
GOAL_POS: tuple[float, float, float] = (0.10, -0.22, 0.015)

# Spawn patch (D-01): ~5x5 cm in the arm's comfortable mid-reach band,
# directly in front of the base (arm reaches toward -y).
SPAWN_PATCH_X: tuple[float, float] = (-0.025, 0.025)
SPAWN_PATCH_Y: tuple[float, float] = (-0.252, -0.202)

# Frozen spawn protocol (D-09/D-10/D-11): the 25 evaluation spawns are
# checked-in DATA, never re-derived at runtime. The env loads this file
# verbatim; the Phase 5 protocol freeze references it by hash.
SPAWN_GRID_SIZE: int = 25
SPAWN_SEEDS_PATH: Path = Path(__file__).resolve().parents[2] / "config" / "spawn_seeds.json"

# Default spawn: patch center, ~0.5 mm above the floor (settle protocol).
SPAWN_CENTER: tuple[float, float, float] = (0.0, -0.227, 0.0155)
SPAWN_Z: float = 0.0155

# 3 cm cube (D-03), yaw fixed, aligned to the gripper approach.
CUBE_HALF_SIZE: float = 0.015

# Settle protocol (verified: 250 physics steps = 0.5 s -> cube at rest,
# qvel norm ~1.75e-9).
SETTLE_STEPS: int = 250
REST_QVEL_TOL: float = 1e-6

# "Resting on the floor" (D-04, Plan 01-03): the settled 3 cm cube sits at
# z ~= 0.0149 (verified); the accepted band brackets contact-height slop.
CUBE_REST_Z_RANGE: tuple[float, float] = (0.013, 0.017)

# ---------------------------------------------------------------------------
# Pinned stage-machine constants (Phase 2 decisions D-01, D-02, D-03)
# ---------------------------------------------------------------------------

# Multi-step grasp verification window in CONTROL STEPS (D-01/D-02): the
# grasp flag sets only after this many consecutive kinematic-hold steps, so
# a momentary contact can never satisfy it (success criterion 2). Round
# number, physically meaningful: 5 steps = 0.25 s at the 20 Hz default rate.
# NOTE (Pitfall 3): the constant is pinned in control steps per D-02 — its
# SIM-TIME meaning scales with frame_skip (0.5 s at 10 Hz, 1.0 s at 5 Hz);
# every audit/eval JSON logs frame_skip so funnel comparisons stay
# interpretable under Phase 5's control-rate perturbation.
GRASP_WINDOW_STEPS: int = 5

# Approach threshold (D-03): horizontal grasp-point-to-cube-center distance
# at or below this sets the approach flag. Round tuning value: comfortably
# above the scripted trajectory's IK residual (~mm) at the pre-grasp/descend
# waypoints, well below the home-pose stand-off (~15 cm) — a coarse
# "the gripper is over the cube" semantic, not a tight script fit.
APPROACH_DIST_M: float = 0.05

# Lift margin (D-03): the cube counts as lifted once its z exceeds the
# CUBE_REST_Z_RANGE top by this margin. Round value: one cube half-size
# step (CUBE_HALF_SIZE = 0.015 m scale) clear of contact-height slop, far
# below the scripted lift_height = 0.08 m.
LIFT_MARGIN_M: float = 0.01

# Cube-tracks-gripper tolerance (D-01): 3D distance between the cube center
# and the grasp point within which the cube counts as "staying with the
# gripper" while the jaw is commanded closed. Round value at the 2x
# CUBE_HALF_SIZE scale: loose enough to absorb pad-compliance wobble during
# the scripted close/lift, tight enough that an un-grasped cube on the floor
# (gripper 3.5+ cm away) never tracks.
GRASP_TRACK_TOL_M: float = 0.03

# Jaw-closed command threshold (D-01, A5 note): an APPLIED jaw ctrl at or
# below this counts as a closed command. Keyed on the applied action value,
# not realized jaw qpos (a closed command while the jaw is forced open
# still counts — makes the momentary-contact negative control
# deterministic). Script closed = 0.13 passes; open = 0.35 fails.
GRASP_JAW_CLOSED_CTRL_MAX: float = 0.20