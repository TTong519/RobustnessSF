"""ENV-03 observation-contract tests (Phase 3, Plan 03-01).

Pins the obs_mode-parameterized observation contract of SO100PickPlaceEnv
under the pinned versions (gymnasium 1.3.0, mujoco 3.12.0):

  * default (state) mode: a single flat float32 Box of shape
    (C.OBS_STATE_DIM,) == (28,) — the D-12 first-class default. Every
    existing caller (benchmark / reward audit / physics gate) constructs
    the env with no obs_mode and must keep working unchanged.
  * vision mode: spaces.Dict with exactly the keys "state" (float32
    Box(28,)) and "image" (uint8 channel-last 64x64, grayscale optional)
    — the ENV-03 contract.
  * gymnasium check_env green (no errors; no warnings beyond the three
    advisories inherent to the pinned spaces) on all three constructions:
    default, vision, grayscale vision.

28-dim state-vector layout (D-01/D-02/D-03; block-for-block tests in the
depth suite below):

    [0:6]   arm qpos (incl. Jaw aperture)             passthrough
    [6:12]  arm qvel                                  passthrough
    [12:15] cube position (x, y, z)                   D-01
    [15]    cube yaw (atan2 of w-first quaternion)    D-01 — never the quat
    [16:22] cube qvel (6D linear + angular)           D-03
    [22:25] gripper-pad-midpoint -> cube offset       D-02
    [25:28] cube -> goal offset                       D-02
"""

import math
import warnings

import mujoco
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv
from so100rob.scripted_policy import grasp_point


def _cube_addresses(env):
    """Independently re-resolve the cube freejoint qpos/qvel addresses by name."""
    cube_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, C.CUBE_JOINT)
    assert cube_id >= 0
    return env.model.jnt_qposadr[cube_id], env.model.jnt_dofadr[cube_id]


def test_default_state_mode_flat_28():
    """D-12 default: no obs_mode arg -> flat float32 Box with shape (28,).

    This is the RED anchor of Plan 03-01: it fails against the pre-refactor
    27-dim observation, proving the 28-dim pin is real before GREEN.
    """
    env = SO100PickPlaceEnv()
    try:
        assert env.observation_space.shape == (28,)
        # The pinned dimension is DATA (constants), not a scattered literal.
        assert C.OBS_STATE_DIM == 28
        assert env.observation_space.shape == (C.OBS_STATE_DIM,)
        assert env.observation_space.dtype == np.float32
        assert isinstance(env.observation_space, spaces.Box)

        obs, _info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        obs, _rew, _term, _trunc, _info = env.step(
            np.asarray(C.HOME_CTRL, dtype=np.float64)
        )
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_vision_mode_dict_space(vision_env):
    """ENV-03: obs_mode='vision' emits Dict{state, image} matching the contract.

    state: float32 Box(-inf, inf, (28,)); image: uint8 Box(0, 255, (64, 64, 3))
    channel-last. reset() and one step() both satisfy observation_space.contains.
    """
    space = vision_env.observation_space
    assert isinstance(space, spaces.Dict)
    assert set(space.keys()) == {"state", "image"}

    state_space = space["state"]
    assert isinstance(state_space, spaces.Box)
    assert state_space.dtype == np.float32
    assert state_space.shape == (C.OBS_STATE_DIM,)
    np.testing.assert_array_equal(state_space.low, -np.inf)
    np.testing.assert_array_equal(state_space.high, np.inf)

    image_space = space["image"]
    assert isinstance(image_space, spaces.Box)
    assert image_space.dtype == np.uint8
    assert image_space.shape == (64, 64, 3)
    assert image_space.low.min() == 0
    assert image_space.high.max() == 255

    obs, _info = vision_env.reset(seed=0)
    assert isinstance(obs, dict)
    assert space.contains(obs)
    obs, _rew, _term, _trunc, _info = vision_env.step(
        np.asarray(C.HOME_CTRL, dtype=np.float64)
    )
    assert space.contains(obs)


def test_check_env_all_modes():
    """ENV-03 gate: check_env passes without errors and without warnings on all 3 constructions.

    Each env is constructed OUTSIDE the warnings-capture context so only
    check_env's own emissions are recorded. The record must contain ONLY
    the three gymnasium advisories that are INHERENT to this project's
    pinned spaces (tracer resolution of the flagged planner assumption):
      * the Box ACTION-space normalization advisory — the action space is
        the auto-built asymmetric ACTUATOR_CTRLRANGE Box (Phase 1 pinned
        data, deliberately not rescaled),
      * the observation-space "-infinity is probably too low" and
        "infinity is probably too high" advisories — ENV-03 mandates the
        state Box(low=-inf, high=inf).
    These are stylistic recommendations, not API violations. Any OTHER
    warning — including the step/reset determinism "not equal although
    similar" warnings — fails this gate.
    """
    inherent_advisories = (
        "For Box action spaces, we recommend using a symmetric and normalized space",
        "A Box observation space minimum value is -infinity",
        "A Box observation space maximum value is infinity",
    )

    def is_inherent(message: str) -> bool:
        # gymnasium colorizes its WARN lines with ANSI escapes — strip them
        # before matching the advisory text.
        cleaned = message.replace("\x1b[33m", "").replace("\x1b[0m", "")
        return any(pattern in cleaned for pattern in inherent_advisories)

    envs = [
        SO100PickPlaceEnv(),
        SO100PickPlaceEnv(
            obs_mode="vision", render_mode="rgb_array", width=64, height=64
        ),
        SO100PickPlaceEnv(
            obs_mode="vision", render_mode="rgb_array", width=64, height=64,
            grayscale=True,
        ),
    ]
    try:
        for env in envs:
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                # skip_close_check: check_env's close check only runs for
                # registered envs (spec is not None) — ours is unregistered —
                # but pin the intent explicitly anyway.
                check_env(env, skip_render_check=True, skip_close_check=True)
            unexpected = [str(w.message) for w in record if not is_inherent(str(w.message))]
            assert unexpected == [], (
                f"check_env emitted unexpected warnings for {type(env).__name__}: "
                f"{unexpected}"
            )
            # Every emission was one of the three inherent advisories.
            assert all(is_inherent(str(w.message)) for w in record)
    finally:
        # Close ONLY the state-mode env (no renderer). The two vision envs
        # are deliberately NOT closed here: gymnasium 1.3.0's
        # OffScreenViewer.close() calls glfw.terminate() — a PROCESS-GLOBAL
        # teardown that silently turns every other live offscreen renderer's
        # frames black (verified empirically: the session vision_env's
        # frames max==min==0 after any vision env.close()). Their GL
        # contexts are freed by OffScreenViewer.__del__ (free() only, no
        # terminate) at GC/process exit.
        envs[0].close()


def test_vision_requires_render_mode():
    """obs_mode='vision' with render_mode=None fails fast: ValueError naming rgb_array."""
    with pytest.raises(ValueError, match="rgb_array"):
        SO100PickPlaceEnv(obs_mode="vision", render_mode=None)


def test_invalid_obs_mode_rejected():
    """An unrecognized obs_mode raises ValueError naming the accepted modes (spawn_mode pattern)."""
    with pytest.raises(ValueError, match="state"):
        SO100PickPlaceEnv(obs_mode="bogus")


# ---------------------------------------------------------------------------
# Observation depth: D-01/D-02/D-03 block layout, yaw cross-check, image
# contracts, cross-mode state identity, non-finite guard (Plan 03-01 Task 2)
# ---------------------------------------------------------------------------


def test_state_vector_block_layout():
    """D-01/D-02/D-03: the 28-dim vector matches an IN-TEST recomputation block-by-block.

    The expected vector is recomputed here directly from env.data (qpos/qvel,
    the cube qpos block, the atan2 yaw, the cube qvel block, grasp_point(),
    GOAL_POS) — anything added to or dropped from _get_obs breaks the
    equality. This is the anti-RNG / anti-noise test: an injected draw or
    noise term can never match an independent recomputation from simulator
    state.
    """
    env = SO100PickPlaceEnv()
    try:
        obs, _info = env.reset(seed=0)
        qpos = np.asarray(env.data.qpos, dtype=np.float64)
        qvel = np.asarray(env.data.qvel, dtype=np.float64)
        adr, dof_adr = _cube_addresses(env)

        cube_pos = qpos[adr : adr + 3]
        w, x, y, z = qpos[adr + 3 : adr + 7]
        cube_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cube_qvel = qvel[dof_adr : dof_adr + 6]
        pad_midpoint = np.asarray(grasp_point(env.model, env.data), dtype=np.float64)

        expected = np.concatenate(
            [
                qpos[:6],                          # [0:6]   arm qpos
                qvel[:6],                          # [6:12]  arm qvel
                cube_pos,                          # [12:15] cube position
                [cube_yaw],                        # [15]    cube yaw
                cube_qvel,                         # [16:22] cube qvel (6D)
                pad_midpoint - cube_pos,           # [22:25] gripper->cube
                cube_pos - np.asarray(C.GOAL_POS),  # [25:28] cube->goal
            ]
        )

        assert obs.dtype == np.float32
        assert obs.shape == (C.OBS_STATE_DIM,)
        np.testing.assert_allclose(obs, expected, rtol=1e-6, atol=1e-6)
        # The 4-dim quaternion never enters the vector, and the superseded
        # 2-D cube_xy-goal term is gone: the total is exactly the 7 blocks.
        assert obs.shape[0] == 6 + 6 + 3 + 1 + 6 + 3 + 3
    finally:
        env.close()


def test_cube_yaw_matches_mujoco_conversion():
    """A3: the observed yaw (index 15) matches mujoco's own quaternion conversion.

    Writes known cube orientations into data.qpos directly (identity, a
    pure z-rotation by a known angle, and a non-axis-aligned tilt), runs
    mj_forward, and asserts the observed yaw equals the heading derived
    from mujoco.mju_quat2Mat's rotation matrix (atan2(R[1,0], R[0,0])) —
    never a second hand-rolled quaternion formula.
    """
    env = SO100PickPlaceEnv()
    try:
        env.reset(seed=0)
        adr, _dof_adr = _cube_addresses(env)

        def quat_for(w, x, y, z):
            q = np.array([w, x, y, z], dtype=float)
            return q / np.linalg.norm(q)

        cases = [
            # (quaternion, known yaw or None when only the matrix cross-check applies)
            (quat_for(1.0, 0.0, 0.0, 0.0), 0.0),                        # identity
            (quat_for(np.cos(0.7 / 2), 0.0, 0.0, np.sin(0.7 / 2)), 0.7),  # pure z-rot
            (quat_for(0.8, 0.3, -0.2, 0.4), None),                     # non-axis-aligned
        ]
        for quat, known_yaw in cases:
            env.data.qpos[adr + 3 : adr + 7] = quat
            mujoco.mj_forward(env.model, env.data)

            # mujoco's own conversion: quaternion -> rotation matrix (row-
            # major: mat[0]=R[0,0], mat[3]=R[1,0]), then the ZYX heading
            # atan2(R[1,0], R[0,0]) — identical to the env's 2(wz+xy) /
            # 1-2(y^2+z^2) atan2 pair by construction.
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, quat)
            expected_yaw = math.atan2(mat[3], mat[0])
            if known_yaw is not None:
                assert expected_yaw == pytest.approx(known_yaw, abs=1e-12)

            obs = env._get_obs()
            assert obs[15] == pytest.approx(expected_yaw, abs=1e-6), (
                f"obs yaw {obs[15]!r} != mujoco-derived heading {expected_yaw!r} "
                f"for quaternion {quat.tolist()}"
            )
    finally:
        env.close()


def test_vision_image_contract(vision_env):
    """Image observations: raw contiguous uint8 (64, 64, 3) in [0, 255], non-degenerate.

    render() on the same env still returns RGB (64, 64, 3) — the channel
    contract is channel-last HWC at the env boundary (Pitfalls 2/9).
    """
    obs, _info = vision_env.reset(seed=0)
    for image in (
        obs[C.OBS_IMAGE_KEY],
        vision_env.step(np.asarray(C.HOME_CTRL, dtype=np.float64))[0][C.OBS_IMAGE_KEY],
    ):
        assert image.dtype == np.uint8
        assert image.shape == (64, 64, 3)
        assert image.flags["C_CONTIGUOUS"]
        assert image.min() >= 0
        assert image.max() <= 255
        assert image.max() > image.min()  # non-degenerate dynamic range

    frame = vision_env.render()
    assert frame.dtype == np.uint8
    assert frame.shape == (64, 64, 3)


def test_grayscale_image_contract(gray_env):
    """Grayscale image observations keep the trailing channel: uint8 (64, 64, 1).

    A squeezed 2-dim frame would fail SB3's is_image_space; the trailing
    channel is mandatory. env.render() on the same env still returns RGB
    (64, 64, 3) — the grayscale conversion lives ONLY inside _get_obs.
    """
    obs, _info = gray_env.reset(seed=0)
    image = obs[C.OBS_IMAGE_KEY]
    assert image.dtype == np.uint8
    assert image.shape == (64, 64, 1)  # trailing channel present, 3 dims total
    assert image.flags["C_CONTIGUOUS"]
    assert image.min() >= 0
    assert image.max() <= 255
    assert gray_env.observation_space.contains(obs)

    frame = gray_env.render()
    assert frame.dtype == np.uint8
    assert frame.shape == (64, 64, 3)  # render() stays RGB regardless of grayscale


def test_state_identical_across_modes():
    """D-12: the 28-dim state assembly is mode-independent — bit-identical across modes.

    The image key is purely additive: resetting a state-mode env and a
    vision env with the same seed must yield bit-identical state vectors.
    """
    state_env = SO100PickPlaceEnv()
    vision = SO100PickPlaceEnv(
        obs_mode="vision", render_mode="rgb_array", width=64, height=64
    )
    try:
        state_obs, _ = state_env.reset(seed=123)
        vision_obs, _ = vision.reset(seed=123)
        np.testing.assert_array_equal(state_obs, vision_obs[C.OBS_STATE_KEY])
    finally:
        state_env.close()
        vision.close()


def test_obs_nonfinite_fails_loudly():
    """T-03-01: NaN cube state makes _get_obs raise ValueError, never emit an observation.

    Non-finite physics must fail loudly at the simulator->observation trust
    boundary instead of silently poisoning the vector the policy trains on.
    The guard is an explicit raise (not a bare assert) so it survives
    python -O (WR-03).
    """
    env = SO100PickPlaceEnv()
    try:
        env.reset(seed=0)
        adr, _dof_adr = _cube_addresses(env)
        env.data.qpos[adr : adr + 3] = np.nan
        with pytest.raises(ValueError, match="T-03-01"):
            env._get_obs()
    finally:
        env.close()