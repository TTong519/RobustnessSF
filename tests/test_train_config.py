"""Train-runner config contract tests (Plan 04-03 Task 1, TRAIN-01).

Pin the WR-02/T-04-07 discipline for scripts/train.py — the production
training runner — before the long runs execute:

  * every pinned number is READ from config/train_gate.json (schema
    so100rob/train_gate/1) or so100rob.constants, never restated as a
    literal in the runner (gamma from REWARD_GAMMA via the gate's
    gamma_source, 128/64/8 divisibility, cpu, spawn_mode=continuous);
  * the resolved-config JSON written into every bundle ECHOES the gate
    fields it consumed (budget, lr, action_space_mode) — the echo is how
    04-05 and Phase 5+ verify what actually ran (T-04-07 mitigation);
  * the Task-1 action-space decision (rescale_minus_one_one, pinned in
    the gate by 04-02) is applied as gymnasium RescaleAction around each
    raw env INSIDE the env factory — BELOW VecNormalize, ABOVE the raw
    env — and the raw_ctrlrange branch wraps nothing.

No training runs here: all assertions are pure config/wrapper checks
plus a tiny in-process env construction (the test_sb3_pipeline.py
pattern). Vision envs are never closed mid-run (gymnasium 1.3.0's
OffScreenViewer close() terminates process-global GLFW state — the
03-02 discipline); they are left to GC.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.env_util import make_vec_env

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "scripts" / "train.py"


def load_train():
    """Import scripts/train.py side-effect-free (the __main__ guard holds).

    RED contract (the established missing-artifact pattern from
    tests/test_hardened_venv.py / tests/test_train_gate.py): a missing
    runner is an intentional failing assertion, never a collection
    crash.
    """
    if not TRAIN_PATH.exists():
        pytest.fail(
            "scripts/train.py does not exist yet — RED: the production "
            "training runner is the missing artifact (04-03 Task 1)"
        )
    spec = importlib.util.spec_from_file_location("so100rob_train_runner", TRAIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test 1 — resolved config matches the gate + constants (WR-02, Pitfall 1/2)
# ---------------------------------------------------------------------------


def test_resolved_config_matches_gate_and_constants():
    """The runner's config object resolves EVERY pinned number from the gate
    or so100rob.constants: gamma == REWARD_GAMMA (Pitfall 2), 128/64/8
    divisibility, cpu, spawn_mode=continuous in the env kwargs (Pitfall 1),
    and the gate seed recorded."""
    train = load_train()
    gate = train.load_gate()
    assert gate["schema"] == "so100rob/train_gate/1"
    cfg = train.resolve_config(gate, mode="run", steps=None, lr=None)

    fo = gate["fixed_overrides"]
    # Pitfall 2: PPO's gamma is READ from the pinned constant via the
    # gate's gamma_source — a divergent discount silently invalidates the
    # Phase 2 reward design.
    assert cfg["gamma"] == C.REWARD_GAMMA
    assert C.REWARD_GAMMA == pytest.approx(0.99)
    assert train.resolve_gamma(gate) == C.REWARD_GAMMA

    assert cfg["n_steps"] == fo["n_steps"] == 128
    assert cfg["batch_size"] == fo["batch_size"] == 64
    assert cfg["n_envs"] == fo["n_envs"] == 8
    assert cfg["n_envs"] * cfg["n_steps"] % cfg["batch_size"] == 0
    assert cfg["device"] == fo["device"] == "cpu"

    # Pitfall 1 (CRITICAL): training env kwargs carry spawn_mode from the
    # gate — the constructor default is grid, and grid-default training
    # silently trains on one fixed cube position.
    assert cfg["env_kwargs"]["spawn_mode"] == gate["spawn_mode"] == "continuous"
    assert cfg["env_kwargs"]["obs_mode"] == C.OBS_MODE_VISION
    # OBS_DECISION D-05: vision-64x64-gray is the decided format.
    assert cfg["env_kwargs"]["grayscale"] is True

    # The seed comes from the gate and is recorded in the resolved config.
    assert cfg["seed"] == gate["tuning"]["seed"] == 0

    # D-04: run mode's default budget is the gate's training_budget_steps
    # (never a restated literal).
    assert cfg["steps"] == gate["training_budget_steps"] == 1500000
    # Explicit --steps override (the tracer slice) is honored verbatim.
    assert train.resolve_config(gate, mode="run", steps=2048, lr=None)["steps"] == 2048

    # Fail-loud divisibility: an inconsistent gate cannot train.
    bad = json.loads(json.dumps(gate))
    bad["fixed_overrides"] = dict(fo, batch_size=100)
    with pytest.raises(ValueError, match="divisib"):
        train.resolve_config(bad, mode="run", steps=None, lr=None)


# ---------------------------------------------------------------------------
# Test 2 — the resolved-config JSON echoes the gate fields it consumed
# ---------------------------------------------------------------------------


def test_resolved_config_json_echoes_gate_fields(tmp_path):
    """The bundle's config.json + gate.json are the provenance record: what
    actually ran is provable from the bundle alone (04-05 / Phase 5+)."""
    train = load_train()
    gate = train.load_gate()
    cfg = train.resolve_config(gate, mode="run", steps=None, lr=None)

    class _StubSaver:
        """Minimal save() surface (bundle layout only — the reload contract
        is pinned by tests/test_train_bundle.py on a real bundle)."""

        def __init__(self, marker: bytes):
            self._marker = marker

        def save(self, path):
            Path(path).write_bytes(self._marker)

    run_dir = tmp_path / "bundle"
    train.save_bundle(_StubSaver(b"model"), _StubSaver(b"norm"), run_dir, cfg, gate)

    j = json.loads((run_dir / "config.json").read_text())
    # The gate echo: budget, lr, action_space_mode (Test 2 contract).
    assert j["steps"] == gate["training_budget_steps"]
    assert j["learning_rate"] == gate["tuning"]["tie_fallback_lr"]
    assert j["action_space_mode"] == gate["action_space_mode"] == "rescale_minus_one_one"
    assert j["checkpoint_selection"] == gate["checkpoint_selection"] == "final-step"
    assert j["spawn_mode"] == gate["spawn_mode"] == "continuous"
    assert j["gamma"] == C.REWARD_GAMMA
    assert j["seed"] == gate["tuning"]["seed"]
    assert j["gate_echo"]["training_budget_steps"] == gate["training_budget_steps"]
    assert j["gate_echo"]["train_window_hours"] == gate["train_window_hours"]
    assert j["gate_echo"]["action_space_mode"] == gate["action_space_mode"]
    # gate.json is the verbatim committed pin.
    assert json.loads((run_dir / "gate.json").read_text()) == gate


# ---------------------------------------------------------------------------
# Test 3 — action-space mode: RescaleAction inside the env factory
# ---------------------------------------------------------------------------


def test_rescale_wrapper_wraps_env_below_normalize():
    """action_space_mode=rescale_minus_one_one wraps each env in
    RescaleAction(gate rescale_min/max) INSIDE the env factory — BELOW
    VecNormalize, ABOVE the raw env — so the whole vec chain (and PPO)
    presents the symmetric [-1,1] space; raw_ctrlrange wraps nothing;
    an unknown mode fails loudly."""
    train = load_train()
    gate = train.load_gate()

    wc, wk = train.resolve_action_wrapper(gate)
    assert gate["action_space_mode"] == "rescale_minus_one_one"
    assert wc is RescaleAction
    assert wk == dict(min_action=gate["rescale_min"], max_action=gate["rescale_max"])
    assert gate["rescale_min"] == pytest.approx(-1.0)
    assert gate["rescale_max"] == pytest.approx(1.0)

    venv = make_vec_env(
        SO100PickPlaceEnv,
        n_envs=1,
        seed=0,
        env_kwargs=train.resolve_env_kwargs(gate),
        wrapper_class=wc,
        wrapper_kwargs=wk,
    )
    # The wrapper sits below the observation normalizer: VecNormalize and
    # everything above it see the rescaled space.
    assert np.allclose(venv.action_space.low, gate["rescale_min"])
    assert np.allclose(venv.action_space.high, gate["rescale_max"])
    chain, norm = train.build_chain(venv, train.resolve_gamma(gate))
    assert np.allclose(chain.action_space.low, gate["rescale_min"])
    assert np.allclose(chain.action_space.high, gate["rescale_max"])

    # The raw_ctrlrange branch wraps nothing and records that.
    gate_raw = json.loads(json.dumps(gate))
    gate_raw["action_space_mode"] = "raw_ctrlrange"
    assert train.resolve_action_wrapper(gate_raw) == (None, None)

    # Unknown mode fails loudly instead of silently training raw.
    gate_bad = json.loads(json.dumps(gate))
    gate_bad["action_space_mode"] = "banana"
    with pytest.raises(ValueError, match="action_space_mode"):
        train.resolve_action_wrapper(gate_bad)