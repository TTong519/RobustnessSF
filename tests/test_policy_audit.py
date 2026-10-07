"""Rendered reward-audit gate contract tests (Plan 04-04 Task 3, TRAIN-03).

Pin the audit layer before it gates the real bundle:

  * audit_episode_ok — the pre-agreed return bounds, READ from the gate
    (T-04-06): a no-success episode at exactly
    audit_no_success_return_max=2.5 and a success at exactly
    audit_success_return_min=6.0 are COMPLIANT (<= / >= operators); 2.6
    (no success) and 5.9 (success) are reward-hacking violations. A
    doctored-gate copy proves the bounds are not literals (WR-02).
  * audit_verdict — a conjunction over episode records; one violating
    episode fails the audit and is named by its spawn_index.
  * episode_record shape — stage flags + reward components captured from
    the final info (the audit report's per-episode evidence).
  * AUDIT_SPAWN_SET — the plan's five-spawn audit grid [12, 0, 6, 18, 24]
    (patch center first, then spread), all inside the frozen 25-spawn grid.

Pure tests over synthetic records — the RENDERED audit run against the
real bundle happens in the audit runner, judged in 04-05.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
GATE_PATH = REPO_ROOT / "config" / "train_gate.json"


def load_modules():
    """Import _eval_chain + run_policy_audit (missing artifact = the
    established intentional failing assertion)."""
    if not (SCRIPTS_DIR / "run_policy_audit.py").exists():
        pytest.fail(
            "scripts/run_policy_audit.py does not exist yet — RED: the "
            "rendered reward-audit runner is the missing artifact "
            "(04-04 Task 3)"
        )
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import _eval_chain as ec
    import run_policy_audit as pa

    return ec, pa


def load_gate() -> dict:
    return json.loads(GATE_PATH.read_text())


def audit_record(raw_return: float, is_success: bool, spawn_index: int = 0) -> dict:
    """A minimal audit episode record (the shape run_episodes returns)."""
    return {
        "mode": "deterministic",
        "spawn_index": spawn_index,
        "seed": 0,
        "is_success": is_success,
        "raw_return": raw_return,
        "steps": 200,
        "stage_flags": {
            "stage_approach": True,
            "stage_grasp": is_success,
            "stage_lift": is_success,
            "stage_place": is_success,
        },
        "reward_components": {
            "reward_shaping": 0.0,
            "reward_time_penalty": -2.0,
            "reward_terminal": 10.0 if is_success else 0.0,
            "reward_phi": 0.0,
        },
    }


# ---------------------------------------------------------------------------
# audit_episode_ok — the pre-agreed bounds, read from the gate (T-04-06)
# ---------------------------------------------------------------------------


def test_audit_bounds_boundary_from_gate():
    """2.5 (no success) and 6.0 (success) are compliant AT the bounds
    (<= / >=); 2.6 (no success) and 5.9 (success) are violations — the
    operators are the pin, not the numbers alone."""
    ec, _pa = load_modules()
    gate = load_gate()
    no_max = float(gate["audit_no_success_return_max"])
    ok_min = float(gate["audit_success_return_min"])
    assert no_max == 2.5 and ok_min == 6.0, "the frozen audit pins"

    assert ec.audit_episode_ok(gate, no_max, False) is True
    assert ec.audit_episode_ok(gate, ok_min, True) is True
    assert ec.audit_episode_ok(gate, no_max + 0.1, False) is False
    assert ec.audit_episode_ok(gate, ok_min - 0.1, True) is False


def test_audit_bounds_are_not_literals():
    """A doctored-gate copy shifts the boundary — proving audit_episode_ok
    READS the bounds instead of hardcoding 2.5/6.0 (WR-02)."""
    ec, _pa = load_modules()
    gate = load_gate()
    doctored = dict(
        gate,
        audit_no_success_return_max=3.0,
        audit_success_return_min=7.0,
    )
    # compliant under the real gate, still compliant under the wider max
    assert ec.audit_episode_ok(doctored, 2.6, False) is True
    # 6.0 success passes the real gate but VIOLATES the doctored 7.0 min
    assert ec.audit_episode_ok(doctored, 6.0, True) is False


# ---------------------------------------------------------------------------
# audit_verdict — one violation fails and names the episode
# ---------------------------------------------------------------------------


def test_audit_verdict_compliant_set_passes():
    ec, _pa = load_modules()
    gate = load_gate()

    records = [
        audit_record(2.5, False, spawn_index=i) for i in (12, 0, 6)
    ] + [audit_record(6.0, True, spawn_index=18), audit_record(6.0, True, 24)]
    verdict = ec.audit_verdict(records, gate)
    assert verdict["pass"] is True
    assert verdict["violations"] == []


def test_audit_verdict_names_violating_episode():
    """One no-success episode at 2.6 (over the max) fails the audit and is
    named by its spawn_index — the runner's nonzero exit names it."""
    ec, _pa = load_modules()
    gate = load_gate()

    records = [
        audit_record(2.5, False, spawn_index=12),
        audit_record(2.5, False, spawn_index=0),
        audit_record(2.6, False, spawn_index=6),  # the violation
        audit_record(6.0, True, spawn_index=18),
        audit_record(5.9, True, spawn_index=24),  # second violation
    ]
    verdict = ec.audit_verdict(records, gate)
    assert verdict["pass"] is False
    violating_spawns = [v["spawn_index"] for v in verdict["violations"]]
    assert violating_spawns == [6, 24], "every violating episode is named"


# ---------------------------------------------------------------------------
# episode_record — the audit's per-episode evidence shape
# ---------------------------------------------------------------------------


def test_episode_record_carries_audit_evidence():
    """The record carries the stage flags + reward components from the
    final info — the audit report's evidence for a human reviewer."""
    ec, _pa = load_modules()
    final_info = {
        "is_success": True,
        "stage_approach": True,
        "stage_grasp": True,
        "stage_lift": True,
        "stage_place": True,
        "reward_shaping": 0.5,
        "reward_time_penalty": -2.0,
        "reward_terminal": 10.0,
        "reward_phi": 1.0,
        "episode": {"r": 6.0, "l": 42},
    }
    rec = ec.episode_record("deterministic", 12, final_info, seed=0)
    assert rec["is_success"] is True
    assert rec["raw_return"] == 6.0
    assert rec["steps"] == 42
    assert rec["stage_flags"] == {
        "stage_approach": True,
        "stage_grasp": True,
        "stage_lift": True,
        "stage_place": True,
    }
    assert rec["reward_components"] == {
        "reward_shaping": 0.5,
        "reward_time_penalty": -2.0,
        "reward_terminal": 10.0,
        "reward_phi": 1.0,
    }


# ---------------------------------------------------------------------------
# AUDIT_SPAWN_SET — the plan's five-spawn audit grid
# ---------------------------------------------------------------------------


def test_audit_spawn_set_is_the_planned_five():
    """[12, 0, 6, 18, 24]: the patch-center spawn first, then the grid
    spread — all inside the frozen 25-spawn grid."""
    ec, pa = load_modules()
    gate = load_gate()

    assert pa.AUDIT_SPAWN_SET == [12, 0, 6, 18, 24]
    assert len(pa.AUDIT_SPAWN_SET) == 5
    assert all(0 <= i < int(gate["spawns_total"]) for i in pa.AUDIT_SPAWN_SET)