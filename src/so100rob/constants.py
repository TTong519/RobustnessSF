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

# NOTE: the scene XML includes the pinned upstream robot model via a
# repo-tree-relative path (trs_so_arm100/ at the repo root), so env
# construction requires the repo tree (an editable install); the
# pyproject wheel target exists for dependency declaration, not
# deployment. See scene_pickplace.xml's header comment.
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
# verbatim; the Phase 5 protocol freeze references it by hash. The file
# lives INSIDE the package (like assets/scene_pickplace.xml) for
# consistency, and because grid mode is the default spawn_mode — a path
# outside the package would break every default env construction. This
# does NOT by itself make the package wheel-deployable: the scene XML
# still reaches out to the repo tree for the upstream robot model (see
# the SCENE_XML_PATH note above), so so100rob is a repo-tree /
# editable-install package and the wheel target is for dependency
# declaration, not deployment.
SPAWN_GRID_SIZE: int = 25
SPAWN_SEEDS_PATH: Path = Path(__file__).parent / "data" / "spawn_seeds.json"

# Patch-center representative spawn (grid entry 12); the default grid
# reset is entry 0 (patch corner). ~0.5 mm above the floor (settle protocol).
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
# waypoints. MEASURED (Task 3, Rule 1 doc fix): at the home pose the pad
# midpoint sits ~2.9 cm horizontally from the spawn-0 cube center — BELOW
# this threshold — so the approach flag legitimately sets at episode start;
# it is a coarse "the gripper is near the cube" semantic, NOT a
# from-home-to-cube progress signal (the pre-grasp stand-off claim in the
# original Phase 2 research was wrong). Downstream consumers must not read
# approach == False at t=0.
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

# ---------------------------------------------------------------------------
# Pinned reward constants (Phase 2 decisions D-05, D-06)
# ---------------------------------------------------------------------------

# Stage-indexed monotone potential Phi (D-05): index = stage_index (count of
# True flags), Phi(s_0) = 0.0, total shapable potential = 1.0.
STAGE_POTENTIALS: tuple[float, ...] = (0.0, 0.15, 0.35, 0.60, 1.0)

# Hybrid approach sub-term weight (OQ-3 resolution): ONE small continuous
# term — APPROACH_W * (1 - clamp(approach_dist / APPROACH_POTENTIAL_SCALE_M,
# 0, 1)) — active only while stage_index < 2, keeps Phi informative before
# any flag flips (RESEARCH A3) without threatening stage monotonicity (the
# guard STAGE_POTENTIALS[1] + APPROACH_W <= STAGE_POTENTIALS[2] is
# import-time validated). 0.05 = a small slice of the 1.0 shapable budget.
APPROACH_W: float = 0.05

# Distance normalizer for the approach sub-term [m]: the sub-term ramps to 0
# as the horizontal gripper-to-cube distance grows past ~2x the spawn-patch
# offset scale — round number, covers home-to-cube stand-off.
APPROACH_POTENTIAL_SCALE_M: float = 0.20

# Terminal place bonus (D-06 terminal dominance): 10x the 1.0 total shapable
# potential, paid exactly ONCE on the step the place flag first sets —
# dense shaping can never accumulate significantly without a stage flag
# advancing (success criterion 3).
TERMINAL_PLACE_BONUS: float = 10.0

# Per-step time penalty (D-05): encourages brevity; SVRC field-guide scale.
TIME_PENALTY: float = 0.01

# MUST equal the Phase 4 PPO discount (Pitfall 1): the shaping term
# gamma*Phi(s') - Phi(s) is policy-invariant ONLY when gamma matches the
# learner's discount (Ng/Harada/Russell ICML 1999). SB3 PPO default
# gamma=0.99 — Phase 4 must read its discount from THIS constant.
REWARD_GAMMA: float = 0.99

# Criterion-3 audit bounds (data consumed by 02-02's audit runner): a
# full-flags-no-place episode earns at most this fraction of the terminal
# bonus, and constant-(stage, Phi) windows accumulate at most
# AUDIT_DENSE_FRACTION of it without a flag advance.
FULL_FLAGS_NO_PLACE_MAX_FRACTION: float = 0.2
AUDIT_DENSE_FRACTION: float = 0.01

# ---------------------------------------------------------------------------
# Pinned observation constants (Phase 3 ENV-03; decisions D-01/D-02/D-03,
# D-08, D-10, D-12)
# ---------------------------------------------------------------------------

# Dict key names (planner structural choice; D-08 names the "state" key as
# the VecNormalize norm_obs_keys target so the uint8 image key is NEVER
# mean-normalized — SB3 divides image observations by 255 internally).
OBS_STATE_KEY: str = "state"
OBS_IMAGE_KEY: str = "image"

# obs_mode values (D-12): state-only is a FIRST-CLASS mode with the same
# pipeline shape as vision, not a degraded fallback. The env's default is
# state mode — every existing runner (benchmark / reward audit / physics
# gate) constructs the env with no obs_mode and must keep working with
# their Phase 1/2 measured numbers reproducible.
OBS_MODE_STATE: str = "state"
OBS_MODE_VISION: str = "vision"

# Rendered image side in pixels (D-05 candidates are 64x64 grayscale and
# 64x64 RGB; the scene's offscreen buffer allows up to 256). Phase 1's
# TRAIN-02 measured render cost as ~resolution-independent at this scale.
OBS_IMAGE_SIZE: int = 64

# Frame-stack depth (D-10): 4 frames = 0.2 s of motion history at the 20 Hz
# default control rate. Consumed by plan 03-02's image-only stacker and
# 03-03's smoke runner — pinned here ONCE, never re-literalized. SB3's
# VecFrameStack/StackedObservations, never gymnasium.wrappers.FrameStack
# (SB3 CNN incompatibility, issue #799).
OBS_N_STACK: int = 4

# Pinned state-vector dimensionality (D-01/D-02/D-03 layout):
#   [0:6]   arm qpos (incl. Jaw aperture)              passthrough
#   [6:12]  arm qvel                                     passthrough
#   [12:15] cube position (x, y, z)                     D-01
#   [15]    cube yaw (atan2 of w-first quaternion)      D-01 — never the quat
#   [16:22] cube qvel (6D linear + angular)              D-03
#   [22:25] gripper-pad-midpoint -> cube 3D offset       D-02
#   [25:28] cube -> goal 3D offset                       D-02
# COSTLY to change: VecNormalize statistics, PPO feature extractors, and
# every Phase 4/6 checkpoint consume this layout; changing it after training
# starts invalidates checkpoints.
OBS_STATE_DIM: int = 28

# Rec.601 luma weights for the grayscale image conversion (D-05 gray
# candidate) — mirrors scripts/benchmark.py's _LUMA so the env-side
# conversion and the benchmark's timed conversion apply the same
# luma -> rint -> clip transform; the benchmark additionally rescales
# its mock-CNN input to [0, 1] (matching SB3's internal uint8
# normalization).
OBS_LUMA: tuple[float, float, float] = (0.2989, 0.5870, 0.1140)