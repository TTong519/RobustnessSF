"""Unit suite for the spawn-aware scripted trajectory builder (D-05/D-08).

Six behaviors from Plan 01-03 Task 1 — all verifiable WITHOUT any physics
stepping: the builder is a pure (model, data, poses) -> action-sequence
transform that must not advance the simulator.
"""

import dataclasses

import numpy as np
import pytest

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv
from so100rob.scripted_policy import TrajectoryParams, build_trajectory

EXPECTED_KNOBS = {
    "pre_grasp_clearance",
    "descend_height",
    "jaw_closed_ctrl",
    "jaw_open_ctrl",
    "close_dwell_steps",
    "lift_height",
    "transit_height",
    "release_dwell_steps",
    "waypoint_dwell_steps",
    "ik_iters",
    "wrist_pitch_grasp",
    "wrist_roll_grasp",
}

EXPECTED_PHASE_ORDER = ["pre_grasp", "descend", "close", "lift", "transit", "release"]


def _cube_pose(env):
    adr = env._cube_qposadr
    return np.array(env.data.qpos[adr : adr + 3])


def _actuator_id(env, name):
    import mujoco

    return mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


@pytest.fixture()
def env():
    """Fresh env per test: build_trajectory may touch data during IK solving
    (it must restore state), so instances are never shared."""
    env_ = SO100PickPlaceEnv(frame_skip=C.DEFAULT_FRAME_SKIP, spawn_mode="grid")
    env_.reset(seed=0, options={"spawn_index": 12})
    yield env_
    env_.close()


def test_build_trajectory_returns_joint_target_sequence(env):
    """Test 1: non-empty sequence of 6-dim joint-target actions in actuator
    order (Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw)."""
    params = TrajectoryParams()
    cube = _cube_pose(env)
    seq = build_trajectory(env.model, env.data, cube, np.asarray(C.GOAL_POS), params)

    assert isinstance(seq, (list, tuple)) and len(seq) > 0
    for action, label in seq:
        action = np.asarray(action)
        assert action.shape == (6,), f"each action must be 6-dim, got {action.shape}"
        assert isinstance(label, str) and label
    # The builder must be side-effect-free on the simulator state.
    assert np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()


def test_spawn_awareness_first_grasp_waypoint_tracks_cube(env):
    """Test 2 (spawn-awareness, D-05): the pre-grasp waypoint derives from the
    privileged cube pose — two different cube poses produce different first
    grasp-phase actions."""
    params = TrajectoryParams()
    pose_a = np.array([-0.02, -0.21, 0.015])
    pose_b = np.array([0.02, -0.24, 0.015])

    seq_a = build_trajectory(env.model, env.data, pose_a, np.asarray(C.GOAL_POS), params)
    seq_b = build_trajectory(env.model, env.data, pose_b, np.asarray(C.GOAL_POS), params)

    first_a = next(a for a, label in seq_a if label == "pre_grasp")
    first_b = next(a for a, label in seq_b if label == "pre_grasp")
    assert not np.allclose(np.asarray(first_a), np.asarray(first_b)), (
        "pre-grasp waypoint must track the cube pose (spawn-aware, D-05)"
    )


def test_jaw_values_within_ctrlrange(env):
    """Test 3: every close-phase action's Jaw equals params.jaw_closed_ctrl and
    every release-phase action's Jaw equals params.jaw_open_ctrl; both lie in
    the verified Jaw ctrlrange [-0.174, 1.75]."""
    params = TrajectoryParams()
    jaw_idx = _actuator_id(env, "Jaw")
    lo, hi = env.model.actuator_ctrlrange[jaw_idx]
    assert (lo, hi) == (-0.174, 1.75)

    seq = build_trajectory(env.model, env.data, _cube_pose(env), np.asarray(C.GOAL_POS), params)

    close_actions = [np.asarray(a) for a, label in seq if label == "close"]
    release_actions = [np.asarray(a) for a, label in seq if label == "release"]
    assert close_actions, "sequence must contain a close phase"
    assert release_actions, "sequence must contain a release phase"

    for a in close_actions:
        assert a[jaw_idx] == params.jaw_closed_ctrl
        assert lo <= a[jaw_idx] <= hi
    for a in release_actions:
        assert a[jaw_idx] == params.jaw_open_ctrl
        assert lo <= a[jaw_idx] <= hi


def test_all_actions_within_ctrlrange(env):
    """Test 4: every emitted action lies within model.actuator_ctrlrange
    elementwise — the builder clamps, the env never has to."""
    params = TrajectoryParams()
    seq = build_trajectory(env.model, env.data, _cube_pose(env), np.asarray(C.GOAL_POS), params)

    ctrl_lo = env.model.actuator_ctrlrange[:, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:, 1]
    for action, _ in seq:
        action = np.asarray(action)
        assert np.all(action >= ctrl_lo) and np.all(action <= ctrl_hi), (
            f"action {action} outside ctrlrange"
        )


def test_phase_order(env):
    """Test 5: the sequence visits pre-grasp-above -> descend -> close -> lift
    -> transit-above-goal -> release, in that order."""
    params = TrajectoryParams()
    seq = build_trajectory(env.model, env.data, _cube_pose(env), np.asarray(C.GOAL_POS), params)

    seen = []
    for _, label in seq:
        if label not in seen:
            seen.append(label)
    for phase in EXPECTED_PHASE_ORDER:
        assert phase in seen, f"sequence must visit phase {phase!r}; got {seen}"
    first_occurrences = [l for l in seen if l in EXPECTED_PHASE_ORDER]
    assert first_occurrences == EXPECTED_PHASE_ORDER, (
        f"phases out of order: {first_occurrences}"
    )


def test_trajectory_params_exposes_named_knobs():
    """Test 6: TrajectoryParams is a dataclass exposing the named tuning knobs."""
    assert dataclasses.is_dataclass(TrajectoryParams), "TrajectoryParams must be a dataclass"
    field_names = {f.name for f in dataclasses.fields(TrajectoryParams)}
    missing = EXPECTED_KNOBS - field_names
    assert not missing, f"TrajectoryParams is missing knobs: {sorted(missing)}"

    params = TrajectoryParams()
    assert params.ik_iters > 0
    for dwell in (
        params.close_dwell_steps,
        params.release_dwell_steps,
        params.waypoint_dwell_steps,
    ):
        assert dwell > 0