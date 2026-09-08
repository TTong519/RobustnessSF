"""Scene-integrity tests for the L1 task scene (ENV-01).

Doubles as the research A2 smoke re-verification: every structural fact the
Phase 1 research verified under mujoco 3.8.0 / gymnasium 1.2.3 is re-asserted
here under the pinned versions (mujoco 3.12.0 / gymnasium 1.3.0), so a
version-driven regression fails loudly in tests instead of silently in a
training run.

The contract under test:
  * the scene pins timestep + integrator while PRESERVING the upstream
    cone/impratio contact settings (include-and-extend merge semantics);
  * the compiled model has the exact expected dimensions (nq=13, nv=12,
    nu=6, nkey=2) and cube qpos address 6;
  * the "home" keyframe is the six home joint values followed by a
    zero-padded cube pose with a w-first identity quaternion;
  * every name the env code resolves by string exists in the compiled model.
"""

import mujoco
import numpy as np
import pytest

from so100rob import constants


def obj_id(model, obj_type, name):
    """Resolve a MuJoCo object name to its id, asserting validity."""
    jid = mujoco.mj_name2id(model, obj_type, name)
    assert jid >= 0, f"name {name!r} does not resolve in the compiled model"
    return jid


def test_pinned_physics_options(env):
    """Scene pins timestep 0.002 + implicitfast (attribute-level merge wins)."""
    assert env.model.opt.timestep == 0.002
    assert env.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def test_upstream_contact_settings_preserved(env):
    """Upstream cone=elliptic / impratio=10 survive the include merge."""
    assert env.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert env.model.opt.impratio == 10.0


def test_model_dimensions(env):
    """nq=13, nv=12, nu=6, nkey=2 — freejoint adds 7 qpos / 6 qvel over the 6 arm joints."""
    assert env.model.nq == 13
    assert env.model.nv == 12
    assert env.model.nu == 6
    assert env.model.nkey == 2


def test_cube_joint_qpos_address(env):
    """The cube freejoint qpos block starts at address 6 (after the 6 arm joints)."""
    cube_id = obj_id(env.model, mujoco.mjtObj.mjOBJ_JOINT, constants.CUBE_JOINT)
    assert env.model.jnt_qposadr[cube_id] == 6
    assert env.model.jnt_type[cube_id] == mujoco.mjtJoint.mjJNT_FREE


def test_home_keyframe_layout(env):
    """key_qpos["home"] = 6 home joint values + zero-padded cube pose (w-first quat (1,0,0,0))."""
    home_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_KEY, constants.HOME_KEY)
    assert home_id >= 0

    key_qpos = np.asarray(env.model.key_qpos[home_id])
    expected = np.array(
        list(constants.HOME_QPOS)            # 6 arm joint values
        + [0.0, 0.0, 0.0]                    # cube position zero-padded to origin
        + [1.0, 0.0, 0.0, 0.0]               # w-first identity quaternion
    )
    np.testing.assert_allclose(key_qpos, expected, atol=1e-12)


def test_actuator_ctrlrange_matches_constants(env):
    """Compiled ctrlrange equals the verified constants matrix elementwise."""
    np.testing.assert_allclose(
        np.asarray(env.model.actuator_ctrlrange),
        np.asarray(constants.ACTUATOR_CTRLRANGE),
        atol=1e-12,
    )


def test_joint_and_actuator_names_resolve_in_fixed_order(env):
    """All six joints and actuators resolve, in the upstream <actuator> order (ids 0..5)."""
    for i, name in enumerate(constants.JOINT_NAMES):
        jid = obj_id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = obj_id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert jid == i, f"joint {name!r} expected id {i}, got {jid}"
        assert aid == i, f"actuator {name!r} expected id {i}, got {aid}"


def test_scene_element_names_resolve(env):
    """cube_joint, cube_geom, goal_site, and camera 'overhead' all have valid ids."""
    obj_id(env.model, mujoco.mjtObj.mjOBJ_JOINT, constants.CUBE_JOINT)
    obj_id(env.model, mujoco.mjtObj.mjOBJ_GEOM, constants.CUBE_GEOM)
    obj_id(env.model, mujoco.mjtObj.mjOBJ_SITE, constants.GOAL_SITE)
    obj_id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, constants.CAMERA_NAME)


def test_scene_file_compiles_standalone(scene_path):
    """The scene XML compiles from its path alone (include + meshdir both resolve)."""
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model.opt.timestep == 0.002
    assert model.nq == 13
    assert model.nu == 6