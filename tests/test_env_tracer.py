"""Tracer test for the Phase 1 walking skeleton.

Proves one real end-to-end path under the pinned venv (Python 3.12,
mujoco 3.12.0, gymnasium 1.3.0):

    construct env -> reset to a settled at-rest cube spawn -> step physics
    -> render a 64x64 frame

Every later phase builds on this env and scene; proving the full stack
end-to-end first (research guard A2) catches an architectural dead-end
after one commit instead of after a training run.
"""

import numpy as np
import pytest

from so100rob import constants
from so100rob.envs.pick_place_env import SO100PickPlaceEnv


@pytest.fixture()
def env():
    env = SO100PickPlaceEnv(frame_skip=25, render_mode="rgb_array")
    yield env
    env.close()


def test_constructs_with_pinned_timestep(env):
    """Test 1: env constructs and the scene pins physics timestep to 0.002."""
    assert env.model.opt.timestep == 0.002


def test_reset_spawns_settled_cube_at_patch_center(env):
    """Test 2: reset(seed=0) settles the cube at rest on the patch-center spawn."""
    env.reset(seed=0)

    cube_id = mujoco_id(env, constants.CUBE_JOINT)
    adr = env.model.jnt_qposadr[cube_id]
    dof_adr = env.model.jnt_dofadr[cube_id]

    x, y, z = env.data.qpos[adr : adr + 3]
    assert x == pytest.approx(constants.SPAWN_CENTER[0], abs=1e-9)
    assert y == pytest.approx(constants.SPAWN_CENTER[1], abs=1e-9)
    # resting on the floor: neither interpenetrating nor hovering/bouncing
    assert 0.014 <= z <= 0.017

    qvel = np.asarray(env.data.qvel[dof_adr : dof_adr + 6])
    assert np.linalg.norm(qvel) < 1e-6


def test_step_50_control_steps_leaves_cube_stable(env):
    """Test 3: 50 control steps at home ctrl keep the cube finite and on the floor."""
    env.reset(seed=0)
    action = np.asarray(constants.HOME_CTRL, dtype=np.float64)

    for _ in range(50):
        env.step(action)

    assert np.all(np.isfinite(env.data.qpos))
    assert np.all(np.isfinite(env.data.qvel))

    cube_id = mujoco_id(env, constants.CUBE_JOINT)
    adr = env.model.jnt_qposadr[cube_id]
    x, y, z = env.data.qpos[adr : adr + 3]
    # still on the floor, not ejected by contact blow-ups
    assert z > 0.01
    assert np.linalg.norm(np.asarray(env.data.qpos[adr : adr + 3])) < 1.0


def test_render_returns_nondegenerate_64x64_frame(env):
    """Test 4: render() returns a uint8 (64, 64, 3) frame with real dynamic range."""
    env.reset(seed=0)

    frame = env.render()

    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    assert frame.shape == (64, 64, 3)
    assert frame.max() > frame.min()


def test_action_space_matches_actuator_ctrlrange(env):
    """Test 5: action space is the auto-built Box from actuator_ctrlrange (asymmetric bounds)."""
    assert env.action_space.shape == (6,)
    np.testing.assert_array_equal(env.action_space.low, env.model.actuator_ctrlrange[:, 0])
    np.testing.assert_array_equal(env.action_space.high, env.model.actuator_ctrlrange[:, 1])


def mujoco_id(env, name):
    import mujoco

    return mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)