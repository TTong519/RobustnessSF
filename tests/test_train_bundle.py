"""Train-runner bundle contract tests (Plan 04-03 Task 1, TRAIN-01).

Pin the checkpoint-bundle save/reload shape Phase 5+ depends on, using
train.py's own save/reload functions on a TINY real bundle (2 envs,
128 steps — under 30 s, the proven tests/test_sb3_pipeline.py mini-train
pattern):

  * the bundle directory contains EXACTLY the four artifacts (policy
    zip, vecnormalize.pkl, resolved-config JSON, gate echo);
  * PPO.load + VecNormalize.load re-attach in the strict order
    (VecNormalize.load -> ImageOnlyFrameStack -> VecTransposeImage ->
    PPO.load().set_env), VecNormalize.load's pickle.load + set_venv
    semantics hold, and model.predict(obs, deterministic=True) returns
    an action within the action space — the reload-shape contract;
  * the eval-mode reload (WR-03 recipe: norm.training=False AND
    norm.norm_reward=False — SB3 2.9.0 attribute names) is available
    for the eval runners while the runner's own training-mode reload
    keeps the flags True.

Vision envs are never closed mid-run (gymnasium 1.3.0's OffScreenViewer
close() terminates process-global GLFW state); they are left to GC.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "scripts" / "train.py"


def load_train():
    """Import scripts/train.py side-effect-free (RED contract as in
    tests/test_train_config.py: missing runner = intentional failure)."""
    if not TRAIN_PATH.exists():
        pytest.fail(
            "scripts/train.py does not exist yet — RED: the production "
            "training runner is the missing artifact (04-03 Task 1)"
        )
    spec = importlib.util.spec_from_file_location("so100rob_train_runner", TRAIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_venv(train, gate, n_envs: int = 2):
    """Tiny in-process training construction: the runner's resolved env
    kwargs + rescale wrapper on a DummyVecEnv (no spawn workers, no GL
    wedge window — the D-07 hardened path is exercised by the runner's
    own tracer run, not by the test suite)."""
    wc, wk = train.resolve_action_wrapper(gate)
    return make_vec_env(
        SO100PickPlaceEnv,
        n_envs=n_envs,
        seed=0,
        env_kwargs=train.resolve_env_kwargs(gate),
        wrapper_class=wc,
        wrapper_kwargs=wk,
    )


def _tiny_bundle(tmp_path):
    """Train a tiny real PPO through train.py's chain and save the bundle
    with train.py's own save_bundle. Returns (train, gate, cfg, model)."""
    train = load_train()
    gate = train.load_gate()
    cfg = train.resolve_config(
        gate, mode="run", steps=128, lr=gate["tuning"]["tie_fallback_lr"],
        run_name="bundle-test",
    )
    venv = _tiny_venv(train, gate)
    chain, norm = train.build_chain(venv, cfg["gamma"])
    model = PPO(
        "MultiInputPolicy",
        chain,
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=train.N_EPOCHS,
        gamma=cfg["gamma"],
        seed=cfg["seed"],
        device="cpu",
        policy_kwargs=dict(net_arch=[16]),
        verbose=0,  # no tensorboard_log here: the suite stays filesystem-clean
    )
    # One rollout buffer: 128 vec steps x 2 envs = 256 env steps; batch 64
    # divides the 256-step buffer (the pinned divisibility at tiny scale).
    model.learn(total_timesteps=cfg["steps"])
    train.save_bundle(model, norm, tmp_path, cfg, gate)
    return train, gate, cfg, model


# ---------------------------------------------------------------------------
# Test 5 — the bundle contains exactly the four artifacts
# ---------------------------------------------------------------------------


def test_bundle_contains_exactly_four_artifacts(tmp_path):
    """The save shape is contract for 04-04/04-05/Phase 5+: policy zip +
    vecnormalize.pkl + resolved config + gate echo, and nothing else."""
    _train, _gate, _cfg, _model = _tiny_bundle(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {
        "ppo_final.zip",
        "vecnormalize.pkl",
        "config.json",
        "gate.json",
    }


# ---------------------------------------------------------------------------
# Test 4 — reload re-attaches and predicts within the action space
# ---------------------------------------------------------------------------


def test_bundle_reloads_reattaches_and_predicts_within_space(tmp_path):
    """PPO.load + VecNormalize.load re-attach to a fresh eval-mode chain
    (strict order, set_venv semantics, WR-03 flags) and a deterministic
    predict returns an action within the action space — bitwise-equal to
    the trained model's."""
    train, gate, cfg, model = _tiny_bundle(tmp_path)
    fresh = _tiny_venv(train, gate)
    loaded_model, loaded_norm = train.reload_bundle(tmp_path, fresh, eval_mode=True)

    # Verified SB3 2.9.0 semantics: VecNormalize.load = pickle.load +
    # set_venv — the loaded normalizer is re-attached to the fresh venv.
    assert loaded_norm.venv is fresh
    # WR-03 recipe applied on the eval-mode path (both flags, 2.9.0 names).
    assert loaded_norm.training is False
    assert loaded_norm.norm_reward is False

    # fixed_obs built in the LOADED model's observation-space shapes —
    # read from the space, never hand-counted.
    fixed_obs = {
        C.OBS_STATE_KEY: np.zeros(
            (1, *loaded_model.observation_space[C.OBS_STATE_KEY].shape),
            dtype=np.float32,
        ),
        C.OBS_IMAGE_KEY: np.full(
            (1, *loaded_model.observation_space[C.OBS_IMAGE_KEY].shape),
            128,
            dtype=np.uint8,
        ),
    }
    b, _ = loaded_model.predict(fixed_obs, deterministic=True)
    space = loaded_model.action_space
    assert np.all(b >= space.low), "reloaded predict outside the action space"
    assert np.all(b <= space.high), "reloaded predict outside the action space"

    # The reload is the SAME policy: bitwise action equality with the
    # trained model (the D-09 proof, deterministic=True).
    a, _ = model.predict(fixed_obs, deterministic=True)
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes()

    # The bundle's config records what ran (steps + action-space mode).
    j = json.loads((tmp_path / "config.json").read_text())
    assert j["steps"] == 128
    assert j["action_space_mode"] == gate["action_space_mode"] == "rescale_minus_one_one"