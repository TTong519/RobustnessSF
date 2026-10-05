"""Pre-agreement consistency tests for config/train_gate.json (04-02).

The train gate is COMMITTED DATA that precedes any training or measurement
(D-01: the Phase 3 D-04 pre-agreement pattern — the pin commit provably
precedes the results commit in git history, the config/obs_gate.json
cc1945b precedent). These tests are the consistency contract that survives
edits: any future gate change must re-derive from a decision (D-01..D-06),
not convenience (T-04-04 tampering / T-04-05 repudiation mitigations).

Pinned sources (read_first; never re-hardcoded here):
  - .planning/phases/04-baseline-training/04-CONTEXT.md — D-01..D-06 verbatim
  - config/obs_gate.json — phase4_budget_steps / window_hours (the gate must
    MATCH, never restate new numbers)
  - src/so100rob/constants.py — REWARD_GAMMA (Pitfall 1), the audit-bound
    arithmetic sources
  - src/so100rob/data/spawn_seeds.json — the frozen 25-spawn grid (the plan's
    "config/spawn_seeds.json" path predates the Phase 1 WR-03 move of the
    grid INTO the package; constants.SPAWN_SEEDS_PATH is the live location)
  - scripts/run_sb3_smoke.py — the pinned n_steps=128 / batch_size=64 /
    n_envs=8 divisibility pattern (D-06/D-07)
  - 04-02 Task 1 (user-resolved blocking decision): action_space_mode =
    "rescale_minus_one_one"
"""

import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from gymnasium.spaces import Box

from so100rob import constants as C

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "config" / "train_gate.json"
OBS_GATE_PATH = REPO_ROOT / "config" / "obs_gate.json"
SCHEMA = "so100rob/train_gate/1"


def load_gate() -> dict:
    """Load the committed train gate.

    Fails the TARGET TEST (never the collection) when the pre-agreement pin
    does not exist yet — the intentional TDD RED for a missing artifact.
    """
    if not GATE_PATH.exists():
        pytest.fail(
            f"{GATE_PATH} does not exist — the pre-agreement pin (D-01) is "
            "missing: the threshold bar must be committed data BEFORE any "
            "training step or measurement."
        )
    gate = json.loads(GATE_PATH.read_text())
    if gate.get("schema") != SCHEMA:
        pytest.fail(
            f"config/train_gate.json schema is {gate.get('schema')!r}, "
            f"expected {SCHEMA!r}"
        )
    return gate


def audit_episode_ok(gate: dict, raw_return: float, is_success: bool) -> bool:
    """The return/success agreement judgment the 04-04 audit runner will use.

    Pinned HERE so the runner and the pre-agreement cannot drift apart
    (T-04-06): ``<=`` for the no-success return max, ``>=`` for the success
    return min.
    """
    if is_success:
        return raw_return >= gate["audit_success_return_min"]
    return raw_return <= gate["audit_no_success_return_max"]


# ---------------------------------------------------------------------------
# Test 1: schema + decision coverage (D-01/D-02, D-05, Task 1 decision)
# ---------------------------------------------------------------------------


def test_schema_and_decision_coverage():
    gate = load_gate()

    # The TRAIN-03 bar (D-01): >=70% of the 25 frozen spawns, >=18/25,
    # 3 episodes per spawn, majority of 3 per-spawn rule, 75 total.
    assert gate["success_threshold_fraction"] == 0.70
    assert gate["spawns_total"] == 25
    assert gate["spawns_total"] == C.SPAWN_GRID_SIZE
    frozen_spawns = json.loads(C.SPAWN_SEEDS_PATH.read_text())["spawns"]
    assert gate["spawns_total"] == len(frozen_spawns)
    assert gate["min_spawns_successful"] == 18
    # 18/25 = 0.72 >= 0.70 threshold fraction (the plan's boundary truth)
    assert (
        gate["min_spawns_successful"] / gate["spawns_total"]
        >= gate["success_threshold_fraction"]
    )
    # 18 is exactly the smallest integer count achieving the 70% fraction
    assert gate["min_spawns_successful"] == math.ceil(
        gate["success_threshold_fraction"] * gate["spawns_total"]
    )
    assert gate["episodes_per_spawn"] == 3
    assert gate["per_spawn_majority"] == 2
    # 3 * 25 = 75 episodes (the plan's boundary truth)
    assert gate["episodes_total"] == 75
    assert gate["episodes_total"] == gate["episodes_per_spawn"] * gate["spawns_total"]
    # majority means 2 of 3 episodes (strict majority)
    assert gate["per_spawn_majority"] <= gate["episodes_per_spawn"]
    assert 2 * gate["per_spawn_majority"] > gate["episodes_per_spawn"]

    # Both eval modes, deterministic primary (D-02)
    assert gate["eval_modes"] == ["deterministic", "stochastic"]
    assert gate["primary_bar"] == "deterministic"

    # Task 1 blocking decision (resolved by the user): RescaleAction(-1, 1)
    # around the raw asymmetric ctrlrange Box — pinned with its bounds.
    assert gate["action_space_mode"] == "rescale_minus_one_one"
    assert gate["rescale_min"] == -1.0
    assert gate["rescale_max"] == 1.0

    # Final-step checkpoint selection, no best-on-eval (D-05)
    assert gate["checkpoint_selection"] == "final-step"

    # Provenance: every number traceable to a decision ID in _doc (T-04-05)
    doc_dump = json.dumps(gate["_doc"])
    for decision_id in ("D-01", "D-02", "D-03", "D-04", "D-05", "D-06"):
        assert decision_id in doc_dump, f"_doc must cite {decision_id}"


# ---------------------------------------------------------------------------
# Test 2: budget arithmetic (D-03, D-04; obs_gate consistency)
# ---------------------------------------------------------------------------


def test_budget_and_retry_arithmetic():
    gate = load_gate()
    obs_gate = json.loads(OBS_GATE_PATH.read_text())

    # The budget IS the obs_gate number — never a restatement (D-04)
    assert gate["training_budget_steps"] == 1500000
    assert gate["training_budget_steps"] == obs_gate["phase4_budget_steps"]

    # Exactly one retry at 2x budget, then stop-and-investigate (D-03)
    retry = gate["retry"]
    assert retry["max_retries"] == 1
    assert retry["retry_budget_steps"] == 3000000
    assert retry["retry_budget_steps"] == 2 * gate["training_budget_steps"]
    assert retry["on_second_miss"] == "stop-and-investigate"

    # The train window IS the obs_gate window (the 04-01 two-window scheme
    # already reads window_hours from the same pin)
    assert gate["train_window_hours"] == 12
    assert gate["train_window_hours"] == obs_gate["window_hours"]


# ---------------------------------------------------------------------------
# Test 3: pinned-constants integrity (D-06/D-07, Pitfall 1)
# ---------------------------------------------------------------------------


def test_pinned_constants_integrity():
    gate = load_gate()
    fixed = gate["fixed_overrides"]

    # gamma reads from so100rob.constants.REWARD_GAMMA (Pitfall 1 contract)
    assert fixed["gamma_source"] == "so100rob.constants.REWARD_GAMMA"
    module_path, _, attr = fixed["gamma_source"].rpartition(".")
    resolved_gamma = getattr(importlib.import_module(module_path), attr)
    assert resolved_gamma == C.REWARD_GAMMA == 0.99

    # The smoke run's pinned divisibility pattern (D-06/D-07):
    # n_envs * n_steps = 8 * 128 = 1024, divisible by batch_size 64
    assert fixed["n_steps"] == 128
    assert fixed["batch_size"] == 64
    assert fixed["n_envs"] == 8
    assert fixed["n_envs"] * fixed["n_steps"] % fixed["batch_size"] == 0

    # device=cpu always (CLAUDE.md: no MPS for training)
    assert fixed["device"] == "cpu"

    # Pitfall 1: TRAINING envs use spawn_mode="continuous" — the env default
    # is grid and a no-options grid reset spawns index 0 forever, silently
    # invalidating the run. Eval envs use grid mode + set_options.
    assert gate["spawn_mode"] == "continuous"


# ---------------------------------------------------------------------------
# Test 4: tuning protocol (D-06, pre-agreed discretion)
# ---------------------------------------------------------------------------


def test_tuning_protocol():
    gate = load_gate()
    tuning = gate["tuning"]

    assert tuning["variants"] == 3
    assert tuning["steps_per_variant"] == 100000
    assert tuning["sweep"] == "learning_rate"
    assert tuning["lr_grid"] == [1e-4, 3e-4, 1e-3]
    assert len(tuning["lr_grid"]) == tuning["variants"]

    # Selection rule: last-100-episode mean RAW episodic return, with the
    # 1%-relative-tie-falls-back-to-3e-4 rule, recorded as prose + structured
    rule = tuning["selection_rule"]
    assert "last-100-episode" in rule
    assert "raw" in rule.lower()
    assert "1%" in rule
    assert tuning["tie_band_rel"] == 0.01
    assert tuning["tie_fallback_lr"] == 3e-4
    # the tie fallback is the middle grid entry (never a restated number)
    assert tuning["tie_fallback_lr"] == tuning["lr_grid"][1]

    # Tuning seed pinned (the smoke run's SEED=0 comparability discipline)
    assert tuning["seed"] == 0


# ---------------------------------------------------------------------------
# Test 5: audit bounds arithmetic (T-04-06; recomputed from constants)
# ---------------------------------------------------------------------------


def test_audit_bounds_derivation():
    gate = load_gate()

    # Dense ceiling: a full-flags-no-place episode earns at most
    # FULL_FLAGS_NO_PLACE_MAX_FRACTION * TERMINAL_PLACE_BONUS = 0.2 * 10.0
    dense_ceiling = C.FULL_FLAGS_NO_PLACE_MAX_FRACTION * C.TERMINAL_PLACE_BONUS
    assert dense_ceiling == pytest.approx(2.0)

    # Time-penalty ceiling: TIME_PENALTY per step over the pinned episode
    # horizon — 10.0 sim s / (0.002 s * frame_skip 25) = 200 control steps
    max_ctrl_steps = C.EPISODE_SIM_SECONDS / (C.PHYSICS_TIMESTEP * C.DEFAULT_FRAME_SKIP)
    assert max_ctrl_steps == pytest.approx(200)
    time_penalty_ceiling = C.TIME_PENALTY * max_ctrl_steps
    assert time_penalty_ceiling == pytest.approx(2.0)

    # No-success bound: a no-place return is bounded above by
    # 2.0 dense ceiling - 2.0 time penalty = 0.0; the pinned 2.5 dominates
    # that bound with margin (never tighter than the arithmetic).
    computed_no_success_bound = dense_ceiling - time_penalty_ceiling
    assert gate["audit_no_success_return_max"] == 2.5
    assert gate["audit_no_success_return_max"] > computed_no_success_bound

    # Success bound: a successful return is bounded below by the 10.0
    # terminal bonus - 2.0 dense ceiling - 2.0 time penalty = 6.0; the
    # pinned minimum equals the computed bound (never looser).
    computed_success_min = (
        C.TERMINAL_PLACE_BONUS - dense_ceiling - time_penalty_ceiling
    )
    assert gate["audit_success_return_min"] == 6.0
    assert gate["audit_success_return_min"] == pytest.approx(
        computed_success_min, abs=1e-9
    )

    # The derivation is recorded in _doc (T-04-06: bounds pinned WITH
    # derivation, not as bare numbers)
    doc_dump = json.dumps(gate["_doc"])
    assert "2.5" in doc_dump
    assert "6.0" in doc_dump


# ---------------------------------------------------------------------------
# Test 6: boundary records (the audit runner's comparison operators)
# ---------------------------------------------------------------------------


def test_audit_boundary_records():
    gate = load_gate()

    # Exact-boundary synthetic episode records are compliant
    assert audit_episode_ok(gate, 2.5, is_success=False) is True
    assert audit_episode_ok(gate, 6.0, is_success=True) is True
    # One step past each boundary is a violation
    assert audit_episode_ok(gate, 2.6, is_success=False) is False
    assert audit_episode_ok(gate, 5.9, is_success=True) is False


# ---------------------------------------------------------------------------
# Test 7: the Task 1 decision, run end-to-end against the real env
# (research A4: RescaleAction was verified importable but never executed
#  against this env — this test pins the mapping)
# ---------------------------------------------------------------------------


def test_rescale_action_maps_pinned_space_to_ctrlrange(env):
    """The gate's rescale bounds over the REAL env action space map [-1] to
    ctrl low, [+1] to ctrl high, and 0 to the per-dim ctrlrange midpoint."""
    gate = load_gate()
    assert gate["action_space_mode"] == "rescale_minus_one_one"

    from gymnasium.wrappers import RescaleAction

    raw = env.action_space
    assert isinstance(raw, Box)
    assert raw.shape == (len(C.JOINT_NAMES),)

    # The auto-built space IS the verified ctrlrange (Phase 1, float32 cast)
    low = np.asarray(raw.low, dtype=np.float64)
    high = np.asarray(raw.high, dtype=np.float64)
    np.testing.assert_allclose(
        low, [r[0] for r in C.ACTUATOR_CTRLRANGE], atol=1e-6
    )
    np.testing.assert_allclose(
        high, [r[1] for r in C.ACTUATOR_CTRLRANGE], atol=1e-6
    )

    # The wrapper is built from the GATE's rescale bounds (never re-hardcoded)
    wrapped = RescaleAction(
        env, min_action=gate["rescale_min"], max_action=gate["rescale_max"]
    )
    np.testing.assert_allclose(
        np.asarray(wrapped.action_space.low, dtype=np.float64),
        gate["rescale_min"],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(wrapped.action_space.high, dtype=np.float64),
        gate["rescale_max"],
        atol=1e-6,
    )

    def mapped(action: list[float]) -> np.ndarray:
        return wrapped.action(np.asarray(action, dtype=np.float32))

    # -1 -> ctrl low, +1 -> ctrl high, 0 -> per-dim midpoint
    np.testing.assert_allclose(mapped([-1.0] * 6), low, atol=1e-5)
    np.testing.assert_allclose(mapped([1.0] * 6), high, atol=1e-5)
    np.testing.assert_allclose(mapped([0.0] * 6), (low + high) / 2.0, atol=1e-5)

    # The home pose maps INSIDE the pinned space (research A4: the home ctrl
    # must lie within [-1, 1] for the initial policy to hold home stably)
    home = np.asarray(C.HOME_CTRL, dtype=np.float64)
    home_rescaled = (home - low) / (high - low) * 2.0 - 1.0
    assert np.all(home_rescaled >= gate["rescale_min"])
    assert np.all(home_rescaled <= gate["rescale_max"])