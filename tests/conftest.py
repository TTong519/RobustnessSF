"""Shared fixtures for the SO-ARM100 env test suites.

Scene-integrity tests are read-only on the compiled model (dimensions,
keyframes, ctrlranges, name resolution), so the model-bearing fixtures are
session-scoped: compiling the scene and settling the cube once per session
keeps the suite fast. Any test that MUTATES ``data`` (stepping, resetting
mid-test with a different spawn) should construct its own env instead of
reusing these.
"""

import pytest

from so100rob import constants
from so100rob.envs.pick_place_env import SO100PickPlaceEnv


@pytest.fixture(scope="session")
def scene_path():
    """Path to the L1 task scene (include-and-extend of the upstream model)."""
    return constants.SCENE_XML_PATH


@pytest.fixture(scope="session")
def env():
    """Default-rate env (frame_skip=25 -> 20 Hz), model compiled + cube settled.

    Session-scoped: scene-integrity assertions only READ the compiled model,
    which is immutable after construction. Mutation-free by contract.
    """
    env_ = SO100PickPlaceEnv(frame_skip=25)
    env_.reset(seed=0)
    yield env_
    env_.close()


@pytest.fixture(scope="session")
def render_env():
    """rgb_array env with the 64x64 render configuration.

    Session-scoped and mutation-free: rendering reads the simulator state
    without stepping it.
    """
    env_ = SO100PickPlaceEnv(frame_skip=25, render_mode="rgb_array", width=64, height=64)
    env_.reset(seed=0)
    yield env_
    env_.close()