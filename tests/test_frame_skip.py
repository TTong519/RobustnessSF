"""ENV-04 / D-07: frame_skip semantics proven by fast assertions.

A different control update rate is a DIFFERENT env instance: frame_skip is
construction-only, the physics timestep stays pinned at 0.002 s, and only
the decision period (control dt) changes. Episodes are defined in
simulation seconds — max_episode_steps is derived via ceil so the sim-time
horizon is identical across rates (never shorter than EPISODE_SIM_SECONDS,
never longer than one extra control step).
"""

import math

import mujoco
import numpy as np
import pytest

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv

FRAME_SKIPS = [5, 10, 25, 50]


def cube_qpos(env):
    """Full 7-dof cube freejoint qpos copy (x, y, z, quat w-first)."""
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, C.CUBE_JOINT)
    adr = env.model.jnt_qposadr[jid]
    return np.asarray(env.data.qpos[adr : adr + 7]).copy()


# ---------------------------------------------------------------------------
# Test 1: physics dt pinned, control dt = 0.002 * frame_skip, ceil-derived cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame_skip", FRAME_SKIPS)
def test_frame_skip_pins_physics_and_derives_control_dt(frame_skip):
    env = SO100PickPlaceEnv(frame_skip=frame_skip)
    try:
        assert env.model.opt.timestep == 0.002
        assert env.dt == pytest.approx(0.002 * frame_skip, abs=1e-12)
        assert env.max_episode_steps == math.ceil(C.EPISODE_SIM_SECONDS / env.dt)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Test 2: identical sim-second episodes across rates (ceil never under-covers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame_skip", FRAME_SKIPS)
def test_episode_sim_time_identical_across_rates(frame_skip):
    env = SO100PickPlaceEnv(frame_skip=frame_skip)
    try:
        sim_time = env.max_episode_steps * env.dt
        assert C.EPISODE_SIM_SECONDS <= sim_time < C.EPISODE_SIM_SECONDS + env.dt
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Test 3: per-instance render_fps metadata (Pitfall 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame_skip", FRAME_SKIPS)
def test_render_fps_matches_control_rate(frame_skip):
    env = SO100PickPlaceEnv(frame_skip=frame_skip)
    try:
        assert env.metadata["render_fps"] == round(1 / env.dt)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Test 4: DEFAULT_FRAME_SKIP pin (A1) and default-construction dt == 0.05
# ---------------------------------------------------------------------------


def test_default_frame_skip_is_25():
    assert C.DEFAULT_FRAME_SKIP == 25
    env = SO100PickPlaceEnv()
    try:
        assert env.dt == pytest.approx(0.05, abs=1e-12)  # 0.002 s * 25 = 20 Hz
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Test 5: simultaneous instances are independent
# ---------------------------------------------------------------------------


def test_instances_are_independent():
    env_a = SO100PickPlaceEnv(frame_skip=25)
    env_b = SO100PickPlaceEnv(frame_skip=25)
    try:
        env_a.reset(seed=0)
        env_b.reset(seed=0)
        b_before = cube_qpos(env_b)

        action = np.asarray(C.HOME_CTRL, dtype=np.float64)
        for _ in range(10):
            env_a.step(action)

        # stepping A never touches B's simulator state
        np.testing.assert_array_equal(cube_qpos(env_b), b_before)
    finally:
        env_a.close()
        env_b.close()


# ---------------------------------------------------------------------------
# Test 6: episode_sim_seconds constructor parameter drives the cap
# ---------------------------------------------------------------------------


def test_episode_sim_seconds_parameter():
    env = SO100PickPlaceEnv(frame_skip=25, episode_sim_seconds=12.0)
    try:
        assert env.episode_sim_seconds == 12.0
        assert env.max_episode_steps == 240  # ceil(12.0 / 0.05)
    finally:
        env.close()

    default = SO100PickPlaceEnv()
    try:
        assert default.episode_sim_seconds == C.EPISODE_SIM_SECONDS
    finally:
        default.close()