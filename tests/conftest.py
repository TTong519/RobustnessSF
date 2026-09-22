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

    The D-12 state-mode default (no obs_mode argument): observation is the
    flat float32 28-dim state vector (Phase 3, ENV-03). Session-scoped:
    scene-integrity assertions only READ the compiled model, which is
    immutable after construction. Mutation-free by contract.
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


@pytest.fixture()
def vision_env():
    """Vision-mode env (obs_mode='vision', ENV-03 Dict obs), 64x64 rgb_array render.

    FUNCTION-scoped (deviation from the plan's session scope, Rule 3):
    gymnasium 1.3.0's OffScreenViewer.render() never re-establishes GL
    context currency — creating a SECOND offscreen viewer anywhere in the
    process steals the "current" context and the first viewer silently
    renders all-black frames (verified empirically). A shared session
    renderer therefore cannot survive alongside any other viewer-creating
    test (e.g. check_env on vision constructions); a fresh env per test
    gives each test its own viewer. Tests reset(seed=0) first — a seeded
    grid reset is fully deterministic.
    """
    env_ = SO100PickPlaceEnv(
        frame_skip=25,
        obs_mode="vision",
        render_mode="rgb_array",
        width=64,
        height=64,
    )
    env_.reset(seed=0)
    yield env_
    env_.close()


@pytest.fixture()
def gray_env():
    """Grayscale vision env: vision_env's construction plus grayscale=True (D-05 gray candidate).

    FUNCTION-scoped for the same gymnasium GL-context-currency reason as
    vision_env. The image observations are uint8 (64, 64, 1) — trailing
    channel kept.
    """
    env_ = SO100PickPlaceEnv(
        frame_skip=25,
        obs_mode="vision",
        render_mode="rgb_array",
        width=64,
        height=64,
        grayscale=True,
    )
    env_.reset(seed=0)
    yield env_
    env_.close()


@pytest.fixture()
def grid_env():
    """Grid-mode env (frozen 25-spawn seed file; options={"spawn_index": i}).

    Function-scoped: spawn-reproducibility tests RESET with different
    indices, so instances are never shared across tests.
    """
    env_ = SO100PickPlaceEnv(spawn_mode="grid")
    env_.reset(seed=0)
    yield env_
    env_.close()


@pytest.fixture()
def continuous_env():
    """Continuous-mode env (fresh uniform patch sample per reset, D-12).

    Function-scoped: reseeding per reset mutates env state.
    """
    env_ = SO100PickPlaceEnv(spawn_mode="continuous")
    env_.reset(seed=0)
    yield env_
    env_.close()


@pytest.fixture(scope="module")
def gate_env():
    """Grid-mode env for the ENV-06 physics-gate integration suite.

    Constructed WITHOUT render_mode — the gate is metric-collection and is
    never recorded (timing discipline). Module-scoped: every parametrized
    case opens with its own reset(options={"spawn_index": i}), so sharing
    one instance across the 25 cases is safe; the reset-first contract is
    asserted by the tests themselves.
    """
    env_ = SO100PickPlaceEnv(spawn_mode="grid")
    yield env_
    env_.close()


@pytest.fixture()
def gate_params():
    """The tuned default TrajectoryParams (Plan 01-03 Task 2 closed-loop
    tuning result — the values that achieved 25/25 on the gate)."""
    from so100rob.scripted_policy import TrajectoryParams

    return TrajectoryParams()