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
  * gymnasium check_env green (no errors, no warnings-module emissions)
    on all three constructions: default, vision, grayscale vision.

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

import warnings

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv


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
    check_env's own emissions are recorded; the record must be empty.
    """
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
                check_env(env, skip_render_check=True)
            assert len(record) == 0, (
                f"check_env emitted warnings for {type(env).__name__}: "
                f"{[str(w.message) for w in record]}"
            )
    finally:
        for env in envs:
            env.close()


def test_vision_requires_render_mode():
    """obs_mode='vision' with render_mode=None fails fast: ValueError naming rgb_array."""
    with pytest.raises(ValueError, match="rgb_array"):
        SO100PickPlaceEnv(obs_mode="vision", render_mode=None)


def test_invalid_obs_mode_rejected():
    """An unrecognized obs_mode raises ValueError naming the accepted modes (spawn_mode pattern)."""
    with pytest.raises(ValueError, match="state"):
        SO100PickPlaceEnv(obs_mode="bogus")