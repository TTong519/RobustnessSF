"""Eval-mode chain contract tests (Plan 04-04 Task 1, TRAIN-03 tooling).

Pin the WR-03 eval-mode reload recipe and the ONE in-process eval chain
before the runners execute against the real bundle (04-05):

  * Test 1 — load_bundle pins BOTH VecNormalize flags at LOAD time
    (WR-03: ``norm.training=False`` AND ``norm.norm_reward=False`` —
    verified against installed SB3 2.9.0: ``.training`` alone does NOT
    stop reward normalization, and there is NO ``set_deterministic``;
    determinism is per-call ``model.predict(obs, deterministic=...)``).
  * Test 2 — the builder's chain mirrors the training order exactly over
    ONE in-process env: DummyVecEnv -> VecNormalize ->
    ImageOnlyFrameStack(4) -> VecTransposeImage, with the env kwargs
    pinning obs_mode="vision" and grayscale=True (the OBS_DECISION
    format) and the gate's rescale_minus_one_one action wrapper.
    D-07 SCOPING: eval is NOT a spawn topology — no SubprocVecEnv, no
    hardened construction — and the in-process vision env is NEVER
    closed mid-session (WR-01 / Pitfall 9; the process exit reclaims it).
  * Test 3 — run_episodes targets spawn_index through the VecEnv
    options stash (Pitfall 6: an EXPLICIT spawn per episode, never a
    default reset), and consecutive episodes never inherit the previous
    episode's spawn.
  * Test 4 — determinism is the per-call predict flag: deterministic
    mode calls ``model.predict(obs, deterministic=True)`` and stochastic
    calls ``deterministic=False`` (asserted by monkeypatching predict —
    no set_deterministic exists to call).

The bundle under test is TINY and SYNTHETIC (a fresh untrained PPO +
VecNormalize saved in exactly the 04-03 four-artifact layout) — this
plan runs in Wave 3 parallel with 04-03 and never needs the 1.5M-step
artifact.

RED contract (the established missing-artifact pattern from
tests/test_train_gate.py / tests/test_train_config.py): every test calls
the loader from its OWN BODY, so a missing artifact is an intentional
in-body failing assertion — never a fixture/setup crash, never a
collection error.

GL discipline (Phase 3 decision, WR-01): the module cache builds ONE
synthetic bundle then ONE loaded chain, in that order — the newest
in-process offscreen viewer is the one the tests step through, and no
env is ever closed mid-process (gymnasium 1.3.0's OffScreenViewer.close()
terminates process-global GLFW state). The cache holds both chains for
module lifetime, so neither env can be garbage-collected mid-process
either.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage

from so100rob import constants as C
from so100rob.vec_wrappers import ImageOnlyFrameStack

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

BUNDLE_ARTIFACTS = ("ppo_final.zip", "vecnormalize.pkl", "config.json", "gate.json")

# Module cache: ONE synthetic bundle + ONE loaded chain per test process.
_STATE: dict = {}


def load_eval_chain():
    """Import scripts/_eval_chain.py side-effect-free (missing artifact =
    intentional failing assertion, per the module docstring)."""
    if not (SCRIPTS_DIR / "_eval_chain.py").exists():
        pytest.fail(
            "scripts/_eval_chain.py does not exist yet — RED: the shared "
            "eval-mode chain builder is the missing artifact (04-04 Task 1)"
        )
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import _eval_chain as ec

    return ec


def _loaded():
    """(model, chain) from a tiny synthetic bundle in EXACTLY the 04-03
    four-artifact layout (ppo_final.zip, vecnormalize.pkl, config.json,
    gate.json) — same save shape, none of the 1.5M-step cost. Built once
    per test process; the bundle chain is referenced for module lifetime
    (GL discipline: created before the loaded chain, never closed)."""
    if "loaded" not in _STATE:
        ec = load_eval_chain()
        bundle = Path(tempfile.mkdtemp(prefix="so100rob-synthetic-bundle-"))
        _STATE["bundle_chain"] = ec.make_synthetic_bundle(bundle)
        for artifact in BUNDLE_ARTIFACTS:
            assert (bundle / artifact).exists(), (
                f"synthetic bundle is missing {artifact} — it must match the "
                "04-03 bundle layout exactly (the save shape is contract)"
            )
        model, chain = ec.load_bundle(bundle)
        _STATE["loaded"] = (model, chain)
    return _STATE["loaded"]


# ---------------------------------------------------------------------------
# Test 1 — the WR-03 load-time invariant
# ---------------------------------------------------------------------------


def test_load_bundle_pins_eval_flags_at_load():
    """load_bundle returns a chain whose VecNormalize has BOTH eval flags
    already False — asserted at load, never deferred to predict time.

    WR-03/T-04-13: ``.training`` alone does NOT stop reward normalization
    (SB3 2.9.0 step_wait gates obs_rms updates and reward-return tracking
    on ``.training`` but applies ``normalize_reward`` whenever
    ``norm_reward`` is True, independently of ``.training``) — so BOTH
    instance attributes must be False the moment the bundle is loaded.
    """
    ec = load_eval_chain()
    _model, chain = _loaded()
    norm = ec.unwrap_vec_normalize(chain)
    assert isinstance(norm, VecNormalize)
    assert norm.training is False, (
        "the loaded VecNormalize still has training=True — obs statistics "
        "would drift during evaluation (WR-03)"
    )
    assert norm.norm_reward is False, (
        "the loaded VecNormalize still has norm_reward=True — eval returns "
        "would silently be normalized-scale while the harness believes they "
        "are raw (WR-03: .training=False alone does NOT stop it)"
    )


# ---------------------------------------------------------------------------
# Test 2 — the chain mirrors the training order over ONE in-process env
# ---------------------------------------------------------------------------


def test_eval_chain_mirrors_training_order():
    """DummyVecEnv(one in-process env) -> VecNormalize -> ImageOnlyFrameStack(4)
    -> VecTransposeImage, with env kwargs obs_mode="vision", grayscale=True
    and the gate's rescale_minus_one_one action wrapper (the policy acts in
    [-1,1]; a missing RescaleAction would feed the policy's [-1,1] actions
    to the raw asymmetric ctrlrange Box and invalidate every measurement)."""
    model, chain = _loaded()

    # Wrapper order, outermost inward — identical to the training chain
    assert isinstance(chain, VecTransposeImage)
    stacked = chain.venv
    assert isinstance(stacked, ImageOnlyFrameStack)
    assert stacked.n_stack == C.OBS_N_STACK == 4
    norm = stacked.venv
    assert isinstance(norm, VecNormalize)
    dummy = norm.venv
    assert isinstance(dummy, DummyVecEnv)
    assert dummy.num_envs == 1, (
        "eval/audit use ONE in-process env (D-07 scoping: NOT a spawn "
        "topology — do not wrap eval in SubprocVecEnv)"
    )

    # Env kwargs pin the OBS_DECISION format: vision Dict obs, 64x64
    # grayscale (1-channel) uint8 image, 28-dim state key.
    obs_space = dummy.envs[0].unwrapped.observation_space
    assert obs_space[C.OBS_STATE_KEY].shape == (C.OBS_STATE_DIM,)
    assert obs_space[C.OBS_IMAGE_KEY].shape == (
        C.OBS_IMAGE_SIZE,
        C.OBS_IMAGE_SIZE,
        1,
    )
    assert obs_space[C.OBS_IMAGE_KEY].dtype == np.uint8

    # The gate's pinned action-space mode is applied around the raw env
    act_space = dummy.envs[0].action_space
    np.testing.assert_allclose(act_space.low, -1.0)
    np.testing.assert_allclose(act_space.high, 1.0)

    # The stacked+transposed image space matches the loaded policy's
    n_ch = obs_space[C.OBS_IMAGE_KEY].shape[-1] * C.OBS_N_STACK
    assert chain.observation_space[C.OBS_IMAGE_KEY].shape == (
        n_ch,
        C.OBS_IMAGE_SIZE,
        C.OBS_IMAGE_SIZE,
    )
    assert model.observation_space[C.OBS_IMAGE_KEY].shape == (
        n_ch,
        C.OBS_IMAGE_SIZE,
        C.OBS_IMAGE_SIZE,
    )
    np.testing.assert_allclose(model.action_space.low, -1.0)
    np.testing.assert_allclose(model.action_space.high, 1.0)


# ---------------------------------------------------------------------------
# Test 3 — explicit spawn targeting per episode (Pitfall 6 reset hygiene)
# ---------------------------------------------------------------------------


def test_run_episodes_targets_spawn_index():
    """set_options({'spawn_index': i, 'seed': s}) + reset reaches the env's
    _reset_options stash (the Phase 1 contract), the env reports the
    requested spawn, and consecutive targeted resets never inherit the
    previous episode's spawn."""
    ec = load_eval_chain()
    model, chain = _loaded()
    grid = json.loads(C.SPAWN_SEEDS_PATH.read_text())["spawns"]

    def assert_spawn(idx):
        cube = ec.cube_position(chain)
        np.testing.assert_allclose(
            cube[:2], grid[idx][:2], atol=1e-6,
            err_msg=f"expected grid spawn {idx}, got cube xy {cube[:2]}",
        )
        # settled rest height (spawn z is ~0.5 mm above the floor; the
        # settle protocol brings the cube to its rest height)
        assert C.CUBE_REST_Z_RANGE[0] <= cube[2] <= C.CUBE_REST_Z_RANGE[1]

    chain.set_options({"spawn_index": 3, "seed": 0})
    chain.seed(0)
    chain.reset()
    assert_spawn(3)

    # a second targeted reset does NOT inherit the previous spawn
    chain.set_options({"spawn_index": 11, "seed": 1})
    chain.seed(1)
    chain.reset()
    assert_spawn(11)

    # run_episodes drives the same mechanism per episode and returns the
    # per-episode record (raw return from Monitor, steps, stage flags)
    recs = ec.run_episodes(
        model, chain, spawn_index=7, n_episodes=1, deterministic=True, seed=0
    )
    assert len(recs) == 1
    rec = recs[0]
    assert rec["spawn_index"] == 7
    assert rec["mode"] == "deterministic"
    assert isinstance(rec["is_success"], bool)
    assert isinstance(rec["raw_return"], float)
    assert isinstance(rec["steps"], int) and rec["steps"] > 0
    assert set(rec["stage_flags"]) == {
        "stage_approach",
        "stage_grasp",
        "stage_lift",
        "stage_place",
    }


# ---------------------------------------------------------------------------
# Test 4 — determinism is the per-call predict flag (D-02)
# ---------------------------------------------------------------------------


def test_predict_determinism_is_per_call(monkeypatch):
    """deterministic mode calls model.predict(obs, deterministic=True) and
    stochastic calls deterministic=False — asserted by monkeypatching
    predict (SB3 2.9.0 has NO set_deterministic; the flag is per-call)."""
    ec = load_eval_chain()
    model, chain = _loaded()
    captured: list[bool] = []

    def fake_predict(obs, state=None, episode_start=None, deterministic=False, **kw):
        captured.append(bool(deterministic))
        action = np.zeros(
            model.action_space.shape, dtype=model.action_space.dtype
        ).reshape(1, -1)
        return action, None

    monkeypatch.setattr(model, "predict", fake_predict)

    ec.run_episodes(
        model, chain, spawn_index=0, n_episodes=1, deterministic=True, seed=0
    )
    first = list(captured)
    ec.run_episodes(
        model, chain, spawn_index=0, n_episodes=1, deterministic=False, seed=0
    )
    second = list(captured[len(first):])
    # predict is called EVERY step: every call in the deterministic episode
    # carries True, every call in the stochastic episode carries False.
    assert len(first) > 0, "the deterministic episode never called predict"
    assert all(first), (
        f"the deterministic episode must call predict with "
        f"deterministic=True on every step; captured {first[:5]}..."
    )
    assert len(second) > 0, "the stochastic episode never called predict"
    assert not any(second), (
        f"the stochastic episode must call predict with "
        f"deterministic=False on every step; captured {second[:5]}..."
    )