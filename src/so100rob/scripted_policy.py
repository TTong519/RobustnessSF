"""Spawn-aware scripted grasp-lift-place trajectory (D-05/D-08, ENV-06).

The Phase 1 physics-gate policy: an open-loop (in the RL sense — no
feedback control, but NOT state-blind) trajectory builder that reads the
privileged cube pose and emits a fixed labeled sequence of 6-dim
joint-target actions. It lives as a reusable module, not a throwaway gate
script: Phase 2's hand-posed trajectory verification reuses this exact
machinery (D-08).

Mechanics:
  * TrajectoryParams    — named tuning knobs (the D-05 hand-tuning surface)
  * grasp_point()       — world-position midpoint of the two opposing
                          mid-finger collision pads (names read from the
                          upstream so_arm100.xml, resolved via mj_name2id)
  * _solve_ik()         — hand-rolled damped-least-squares IK over the
                          (Rotation, Pitch, Elbow) joints via mujoco.mj_jac,
                          with Wrist_Pitch / Wrist_Roll pinned at the fixed
                          grasp orientation (3 joints, 3D position target)
  * build_trajectory()  — the waypoint sequence pre-grasp-above -> descend
                          -> close -> lift -> transit-above-goal -> descend
                          -> release, each action held for its dwell count

build_trajectory is SIDE-EFFECT-FREE on the simulator: it snapshots
data state, solves every waypoint IK on the snapshot, and restores the
state before returning. Callers step the env themselves.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from so100rob import constants as C

# The two opposing mid-finger collision pads (upstream so_arm100.xml,
# class "finger_collision") whose world-position midpoint defines the grasp
# point the IK drives. Names are read directly from the upstream XML.
GRASP_PAD_GEOMS: tuple[str, str] = ("fixed_jaw_pad_2", "moving_jaw_pad_2")

# Control steps the gate/test holds the final arm config after the release
# phase so the cube comes to rest before the D-04 predicate is evaluated
# (the jaw opens ~1-2 cm above the floor; the cube must settle).
POST_RELEASE_SETTLE_STEPS: int = 40  # 2.0 sim seconds at the 20 Hz default

# IK solver tuning (not a hand-tuning surface; leave alone unless IK itself
# misbehaves — trajectory shaping is done through TrajectoryParams).
_IK_DAMPING: float = 0.1
_MAX_IK_STEP: float = 0.2  # rad per iteration, keeps DLS from jumping
_IK_CONVERGED_TOL: float = 1e-6


@dataclass
class TrajectoryParams:
    """Named tuning knobs for the scripted trajectory (D-05).

    Every knob is a documented hand-tuning surface; the physics gate's
    closed-loop tuning protocol (Plan 01-03 Task 2, A5 resolution) adjusts
    exactly ONE knob at a time and re-runs the 25-spawn gate until 25/25.
    """

    # Height of the pre-grasp waypoint above the cube center [m].
    pre_grasp_clearance: float = 0.05
    # Height of the grasp-point target above the cube center at the
    # descend waypoint [m] (~grip the cube at mid-upper face).
    descend_height: float = 0.005
    # Jaw ctrl held during close/lift/transit/place phases. Verified Jaw
    # ctrlrange [-0.174, 1.75]; home (jaw=0) leaves a ~19.8 mm pad gap, so
    # closed values near 0.1 squeeze a 3 cm cube and open values ~0.3
    # straddle it (empirical pad-gap slope ~67 mm per unit ctrl).
    jaw_closed_ctrl: float = 0.13
    jaw_open_ctrl: float = 0.35
    # Control steps the close phase holds the closed-jaw action.
    close_dwell_steps: int = 10
    # Height the cube is lifted above its spawn z [m].
    lift_height: float = 0.08
    # Height of the transit waypoint above the goal z [m].
    transit_height: float = 0.08
    # Control steps the release phase holds the open-jaw action.
    release_dwell_steps: int = 10
    # Control steps every waypoint action is held.
    waypoint_dwell_steps: int = 20
    # Damped-least-squares iterations per waypoint.
    ik_iters: int = 100
    # Fixed grasp orientation: Wrist_Pitch / Wrist_Roll are pinned at these
    # values while (Rotation, Pitch, Elbow) solve the 3D position target.
    wrist_pitch_grasp: float = 1.57
    wrist_roll_grasp: float = -1.57


@dataclass
class _ArmAddressing:
    """Name-resolved joint/actuator addressing for one compiled model."""

    qadr: dict[str, int]  # joint name -> qpos address
    dadr: dict[str, int]  # joint name -> dof address
    act_idx: dict[str, int]  # joint name -> actuator index
    jaw_act: int  # actuator index of the Jaw
    ik_qadrs: np.ndarray  # (3,) qpos addresses of Rotation, Pitch, Elbow
    ik_dofs: np.ndarray  # (3,) dof addresses of Rotation, Pitch, Elbow
    ik_lo: np.ndarray  # (3,) ctrl lower bounds of the IK joints
    ik_hi: np.ndarray  # (3,) ctrl upper bounds of the IK joints
    fixed_jaw_body: int  # body id of Fixed_Jaw (jacobian anchor)


def _resolve_arm(model) -> _ArmAddressing:
    """Resolve arm addressing BY NAME — never by index guessing."""
    qadr, dadr, act_idx = {}, {}, {}
    for name in C.JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} not found in compiled model")
        qadr[name] = int(model.jnt_qposadr[jid])
        dadr[name] = int(model.jnt_dofadr[jid])
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise ValueError(f"actuator {name!r} not found in compiled model")
        act_idx[name] = int(aid)

    # Upstream <actuator> order must match the pinned name order: the emitted
    # action vector is indexed by actuator id, so a silent reorder upstream
    # would scramble the action mapping. Fail loudly instead.
    for i, name in enumerate(C.JOINT_NAMES):
        actual = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actual != name:
            raise ValueError(
                f"actuator order drift: position {i} is {actual!r}, expected {name!r}"
            )

    ik_names = ("Rotation", "Pitch", "Elbow")
    ik_qadrs = np.array([qadr[n] for n in ik_names])
    ik_dofs = np.array([dadr[n] for n in ik_names])
    ik_lo = model.actuator_ctrlrange[[act_idx[n] for n in ik_names], 0].copy()
    ik_hi = model.actuator_ctrlrange[[act_idx[n] for n in ik_names], 1].copy()

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    if body < 0:
        raise ValueError("body 'Fixed_Jaw' not found in compiled model")

    return _ArmAddressing(
        qadr=qadr,
        dadr=dadr,
        act_idx=act_idx,
        jaw_act=act_idx["Jaw"],
        ik_qadrs=ik_qadrs,
        ik_dofs=ik_dofs,
        ik_lo=ik_lo,
        ik_hi=ik_hi,
        fixed_jaw_body=body,
    )


def _geom_world_pos(model, data, geom_name: str) -> np.ndarray:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise ValueError(f"finger pad geom {geom_name!r} not found in compiled model")
    return np.array(data.geom_xpos[gid])


def grasp_point(model, data) -> np.ndarray:
    """World-position midpoint of the two opposing finger pads (the IK target).

    The midpoint is evaluated at the CURRENT pose: with the jaw qpos held
    fixed during a solve, the point is rigidly attached to the Fixed_Jaw
    body, so mj_jac anchored on that body is exact.
    """
    a = _geom_world_pos(model, data, GRASP_PAD_GEOMS[0])
    b = _geom_world_pos(model, data, GRASP_PAD_GEOMS[1])
    return (a + b) / 2.0


def _solve_ik(model, data, target: np.ndarray, params: TrajectoryParams,
              jaw_ctrl: float, ctx: _ArmAddressing) -> np.ndarray:
    """Damped-least-squares IK over (Rotation, Pitch, Elbow).

    Wrist_Pitch / Wrist_Roll are pinned at the fixed grasp orientation and
    the Jaw at ``jaw_ctrl``; the three arm joints solve the 3D position
    target for the grasp point. Mutates ``data.qpos`` (the caller owns the
    state snapshot); returns the 6-dim ctrlrange-clamped joint-target action
    in actuator order.
    """
    data.qpos[ctx.qadr["Wrist_Pitch"]] = params.wrist_pitch_grasp
    data.qpos[ctx.qadr["Wrist_Roll"]] = params.wrist_roll_grasp
    data.qpos[ctx.qadr["Jaw"]] = jaw_ctrl

    jacp = np.zeros((3, model.nv))
    for _ in range(params.ik_iters):
        mujoco.mj_forward(model, data)
        point = grasp_point(model, data)
        err = target - point
        mujoco.mj_jac(model, data, jacp, None, point, ctx.fixed_jaw_body)
        J = jacp[:, ctx.ik_dofs]  # 3x3 over the IK joints' dofs
        A = J @ J.T + (_IK_DAMPING**2) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err)
        step = float(np.linalg.norm(dq))
        if step < _IK_CONVERGED_TOL:
            break
        if step > _MAX_IK_STEP:
            dq *= _MAX_IK_STEP / step
        data.qpos[ctx.ik_qadrs] = np.clip(
            data.qpos[ctx.ik_qadrs] + dq, ctx.ik_lo, ctx.ik_hi
        )
    mujoco.mj_forward(model, data)

    # Emit the achieved configuration as joint-target action, in actuator
    # order, clamped to ctrlrange (the builder clamps; the env never has to).
    action = np.zeros(model.nu)
    for name in C.JOINT_NAMES:
        action[ctx.act_idx[name]] = data.qpos[ctx.qadr[name]]
    action[ctx.jaw_act] = jaw_ctrl
    return np.clip(action, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def _validate_params(params: TrajectoryParams, ctx: _ArmAddressing, model) -> None:
    if params.ik_iters <= 0:
        raise ValueError(f"ik_iters must be positive, got {params.ik_iters}")
    for knob in ("close_dwell_steps", "release_dwell_steps", "waypoint_dwell_steps"):
        if getattr(params, knob) <= 0:
            raise ValueError(f"{knob} must be positive, got {getattr(params, knob)}")
    lo, hi = model.actuator_ctrlrange[ctx.jaw_act]
    for knob in ("jaw_closed_ctrl", "jaw_open_ctrl"):
        v = getattr(params, knob)
        if not lo <= v <= hi:
            raise ValueError(f"{knob}={v} outside the Jaw ctrlrange [{lo}, {hi}]")


def build_trajectory(
    model,
    data,
    cube_pose,
    goal_pos,
    params: TrajectoryParams | None = None,
) -> list[tuple[np.ndarray, str]]:
    """Build the spawn-aware grasp-lift-place action sequence (D-05).

    Reads the privileged cube pose (open-loop means no feedback control,
    NOT state-blind), solves IK for the waypoint chain

        pre-grasp above cube -> descend to grasp height -> close jaw
        -> lift -> transit above goal -> descend -> open jaw (release)

    and returns a list of ``(action, phase_label)`` pairs, one per control
    step, in actuator order. The goal-descent reuses the "descend" label —
    the ordered first occurrences are exactly pre_grasp, descend, close,
    lift, transit, release.

    Side-effect-free on ``data``: state is snapshotted, IK runs on the
    snapshot, and the state is restored before returning.
    """
    params = params if params is not None else TrajectoryParams()
    cube = np.asarray(cube_pose, dtype=float).reshape(-1)
    if cube.size < 3:
        raise ValueError(f"cube_pose needs at least 3 coordinates, got {np.shape(cube_pose)}")
    goal = np.asarray(goal_pos, dtype=float).reshape(-1)
    if goal.size < 3:
        raise ValueError(f"goal_pos needs at least 3 coordinates, got {np.shape(goal_pos)}")
    cube_xyz = cube[:3].copy()
    goal_xyz = goal[:3].copy()

    ctx = _resolve_arm(model)
    _validate_params(params, ctx, model)

    qpos0, qvel0, ctrl0 = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    try:
        # Deterministic IK warm start: arm at the home keyframe pose.
        for name, q in zip(C.JOINT_NAMES, C.HOME_QPOS):
            data.qpos[ctx.qadr[name]] = q

        def solve(target_xyz: np.ndarray, jaw_ctrl: float) -> np.ndarray:
            return _solve_ik(model, data, target_xyz, params, jaw_ctrl, ctx)

        pre_grasp = solve(cube_xyz + np.array([0, 0, params.pre_grasp_clearance]),
                          params.jaw_open_ctrl)
        descend = solve(cube_xyz + np.array([0, 0, params.descend_height]),
                        params.jaw_open_ctrl)
        close = descend.copy()
        close[ctx.jaw_act] = params.jaw_closed_ctrl
        lift = solve(cube_xyz + np.array([0, 0, params.lift_height]),
                     params.jaw_closed_ctrl)
        transit = solve(goal_xyz + np.array([0, 0, params.transit_height]),
                        params.jaw_closed_ctrl)
        place = solve(goal_xyz + np.array([0, 0, params.descend_height]),
                      params.jaw_closed_ctrl)
        release = place.copy()
        release[ctx.jaw_act] = params.jaw_open_ctrl
    finally:
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        data.ctrl[:] = ctrl0
        mujoco.mj_forward(model, data)

    seq: list[tuple[np.ndarray, str]] = []
    seq.extend([(pre_grasp, "pre_grasp")] * params.waypoint_dwell_steps)
    seq.extend([(descend, "descend")] * params.waypoint_dwell_steps)
    seq.extend([(close, "close")] * params.close_dwell_steps)
    seq.extend([(lift, "lift")] * params.waypoint_dwell_steps)
    seq.extend([(transit, "transit")] * params.waypoint_dwell_steps)
    seq.extend([(place, "descend")] * params.waypoint_dwell_steps)
    seq.extend([(release, "release")] * params.release_dwell_steps)
    return seq