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
    # 04-08: the training env kwargs construct the CV observation contract
    # (the judged training chain is on the 32-dim CV state key).
    assert cfg["env_kwargs"]["cv_features"] is True

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


# ---------------------------------------------------------------------------
# Task 2 — selection rule (pure function on synthetic episode logs, D-06)
# ---------------------------------------------------------------------------


def test_selection_rule_on_synthetic_episode_logs():
    """Highest last-100-episode mean RAW episodic return wins; a <=1%
    relative tie between the best two falls back to the pinned middle lr;
    the function returns the chosen lr plus a JSON-serializable record of
    every variant's metric."""
    train = load_train()
    gate = train.load_gate()
    fallback = str(gate["tuning"]["tie_fallback_lr"])
    assert fallback == "0.0003"  # the pinned middle grid entry

    # Clear winner: 1e-3 (mean 0.0 beats -1.0 and -2.0).
    logs = {"0.0001": [-2.0] * 100, "0.0003": [-1.0] * 100, "0.001": [-1.0] * 50 + [1.0] * 50}
    chosen, rec = train.apply_selection_rule(gate, logs)
    assert chosen == "0.001"
    assert rec["chosen_lr"] == "0.001"
    assert rec["metrics"]["0.001"] == pytest.approx(0.0)
    json.dumps(rec)  # the record is JSON-serializable (selection.json)

    # Only the LAST 100 episodes count: a variant with great early
    # returns followed by a poor last-100 must NOT win.
    logs_stale = {"0.0001": [-2.0] * 100, "0.0003": [5.0] * 100 + [-2.0] * 100, "0.001": [-1.5] * 100}
    chosen2, rec2 = train.apply_selection_rule(gate, logs_stale)
    assert rec2["metrics"]["0.0003"] == pytest.approx(-2.0)
    assert chosen2 == "0.001" and chosen2 != "0.0003"

    # A <=1% relative tie between the best two -> tie_fallback_lr (3e-4).
    logs_tie = {"0.0001": [-2.0] * 100, "0.0003": [1.0] * 100, "0.001": [1.01] * 100}
    chosen3, rec3 = train.apply_selection_rule(gate, logs_tie)
    assert chosen3 == fallback
    assert set(rec3["metrics"]) == {"0.0001", "0.0003", "0.001"}

    # Just outside the band -> the highest metric wins.
    logs_no = {"0.0001": [-2.0] * 100, "0.0003": [1.0] * 100, "0.001": [1.02] * 100}
    chosen4, _ = train.apply_selection_rule(gate, logs_no)
    assert chosen4 == "0.001"


# ---------------------------------------------------------------------------
# Task 2 — retry mode: refusal + budget from the gate (D-03)
# ---------------------------------------------------------------------------


def test_retry_mode_refusal_and_budget_from_gate():
    """Retry uses 2x the budget (3M) asserted FROM THE GATE, never a
    literal, and refuses unless the gate's retry.max_retries allows it
    AND the run is an explicitly confirmed D-03 response — the runner
    does NOT loop."""
    train = load_train()
    gate = train.load_gate()

    cfg = train.resolve_config(gate, mode="retry", steps=None, lr=None)
    assert cfg["steps"] == gate["retry"]["retry_budget_steps"]
    assert cfg["steps"] == 2 * gate["training_budget_steps"] == 3000000

    # Refusal without the explicit D-03 confirmation.
    with pytest.raises(ValueError, match="confirm"):
        train.check_retry_allowed(gate, confirmed=False)
    # Refusal when the gate forbids retries.
    gate_no = json.loads(json.dumps(gate))
    gate_no["retry"]["max_retries"] = 0
    with pytest.raises(ValueError, match="max_retries"):
        train.check_retry_allowed(gate_no, confirmed=True)
    # A tampered/inconsistent retry budget fails loudly (never trains).
    gate_bad = json.loads(json.dumps(gate))
    gate_bad["retry"]["retry_budget_steps"] = 1234567
    with pytest.raises(ValueError, match="2 x"):
        train.check_retry_allowed(gate_bad, confirmed=True)
    # Allowed with confirmation + gate consent.
    assert train.check_retry_allowed(gate, confirmed=True) is True


# ---------------------------------------------------------------------------
# Task 2 — checkpoint-selection discipline (D-05)
# ---------------------------------------------------------------------------


def test_run_mode_checkpoint_discipline():
    """The resolved config records checkpoint_selection="final-step"; run
    mode adds a periodic CheckpointCallback(save_vecnormalize=True) whose
    save path is under runs/train/<ts>/checkpoints/ — crash recovery
    ONLY, no code path reads it for selection."""
    train = load_train()
    gate = train.load_gate()

    cfg = train.resolve_config(gate, mode="run", steps=None, lr=None)
    assert cfg["checkpoint_selection"] == gate["checkpoint_selection"] == "final-step"

    run_dir = Path("runs/train/20000101T000000Z")
    progress = train.TrainProgressCallback("discipline-check")
    callbacks = train.build_callbacks(cfg, run_dir, progress)
    ckpts = [c for c in callbacks if type(c).__name__ == "CheckpointCallback"]
    assert len(ckpts) == 1
    ckpt = ckpts[0]
    assert ckpt.save_path == str(run_dir / "checkpoints")
    assert ckpt.save_vecnormalize is True

    # Tune variants do not write periodic checkpoints (selection reads the
    # episode logs' final windows, never a checkpoint).
    cfg_tune = train.resolve_config(gate, mode="tune", steps=None, lr=None)
    assert not [
        c for c in train.build_callbacks(cfg_tune, run_dir, progress)
        if type(c).__name__ == "CheckpointCallback"
    ]

    # No code path reads the checkpoints dir for selection (D-05): the
    # string appears exactly once in the runner — the callback's save_path.
    src = TRAIN_PATH.read_text()
    assert src.count('"checkpoints"') == 1, (
        "the checkpoints dir must be write-only (crash recovery); a second "
        "reference would be a selection code path (D-05 violation)"
    )

def test_selection_json_line_is_repo_relative():
    """04-07 Rule 3 reconciliation pin: the tune pass's SELECTION_JSON line
    must be the REPO-RELATIVE spelling "SELECTION_JSON=runs/train/tune/
    selection.json" — the 04-07 plan/VALIDATION freshness verify greps that
    exact literal (an absolute path would never match it; the 04-05
    tooling-to-plan-spelling pattern)."""
    train = load_train()
    assert train._selection_json_line() == (
        "SELECTION_JSON=runs/train/tune/selection.json"
    )
