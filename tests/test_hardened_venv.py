"""Tests for so100rob._hardened_venv — the extracted hardened construction
(Plan 04-01 Task 2, D-07 / STATE.md Phase 4 contract).

Pin the two-window faulthandler scheme and the thin-consumer contract:

  * SETUP_TRIPWIRE_S is a COMPUTED worst-case bound over the bounded-retry
    setup path (IN-09: the old 25-min hardcode was smaller than the run's
    own ~38-min worst-case bounded-retry path, so a healthy retry-laden
    run could be killed by its own backstop) — derived from the
    construction constants, never a hardcoded 25*60.
  * A separate TRAIN window is exported for train.py (04-03): 12 h read
    from config/obs_gate.json window_hours, never re-hardcoded (WR-02
    discipline — runners read committed pins).
  * scripts/run_sb3_smoke.py is a THIN CONSUMER: the hardened construction
    (watchdogged readiness barrier, per-worker first-render warm-up,
    watched untimed reset/warm-up, bounded kill/rebuild/retry, SIGKILL
    reaping) exists exactly once, in the package module — no duplicated
    definitions and no leftover 25-min hardcode in the script.

The watchdog-discipline truths themselves (timed regions never armed,
every untimed setup exchange covered) stay pinned by
tests/test_smoke_watchdog.py, which drives the extracted functions via
the module after the same refactor.
"""

import faulthandler
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "run_sb3_smoke.py"
GATE_PATH = REPO_ROOT / "config" / "obs_gate.json"

# IN-09: the old 25-min tripwire was smaller than the bounded-retry worst
# path (~38 min). The computed bound must strictly dominate that path.
IN09_WORST_PATH_S = 38 * 60


def _hv():
    """Import the module under test; a missing module is a test FAILURE
    (the module is this task's deliverable), not a collection error —
    RED must be the target tests failing, never an import crash."""
    try:
        from so100rob import _hardened_venv as m
    except ImportError as exc:
        pytest.fail(
            "so100rob._hardened_venv is not importable — the hardened "
            "construction module is this task's deliverable: "
            f"{exc}"
        )
    return m


def test_setup_tripwire_is_computed_dominating_the_bounded_retry_worst_case():
    """Test 1: SETUP_TRIPWIRE_S > 38*60, derived from the construction
    constants (never the old 25*60 hardcode)."""
    hv = _hv()
    assert hv.SETUP_TRIPWIRE_S > IN09_WORST_PATH_S, (
        f"SETUP_TRIPWIRE_S={hv.SETUP_TRIPWIRE_S} does not dominate the "
        f"IN-09 bounded-retry worst path ({IN09_WORST_PATH_S} s)"
    )
    assert hv.SETUP_TRIPWIRE_S != 25 * 60, (
        "SETUP_TRIPWIRE_S must be the computed worst-case bound, not the "
        "old 25-min hardcode (IN-09)"
    )
    # The derivation is recomputed here from the module's own construction
    # constants, so a future revert to any hardcoded number fails loudly.
    kill_bound = 2.0 * hv._KILL_JOIN_S  # bounded join after terminate + after SIGKILL
    worst_construction = hv.CONSTRUCTION_ATTEMPTS * (
        hv.CONSTRUCTION_TIMEOUT_S + hv.SPAWN_WORKER_READY_DELAY_S + kill_bound
    )
    worst_bench_window = hv.BENCH_SETUP_TIMEOUT_S + kill_bound
    worst_bench_repeat = hv.BENCH_SETUP_ATTEMPTS * (
        worst_construction + worst_bench_window
    )
    expected = hv.SETUP_TRIPWIRE_SAFETY * (
        hv.BENCH_TOPOLOGIES * hv.TOPO_REPEATS * worst_bench_repeat
        + hv.SMOKE_CONSTRUCTION_WINDOWS * worst_construction
    )
    assert hv.SETUP_TRIPWIRE_S == pytest.approx(expected), (
        "SETUP_TRIPWIRE_S no longer matches its documented worst-case "
        "derivation from the construction constants"
    )


def test_train_window_reads_window_hours_from_the_gate_pin():
    """Test 2: the train-window helper exists and yields window_hours*3600
    (12 h) from config/obs_gate.json — never a re-hardcoded 12h."""
    hv = _hv()
    gate = json.loads(GATE_PATH.read_text())
    assert hv.train_window_s() == gate["window_hours"] * 3600.0
    assert hv.train_window_s() == 12 * 3600.0
    assert callable(hv.arm_train_tripwire)


def test_arm_train_tripwire_arms_and_returns_the_window():
    """arm_train_tripwire(seconds=None) arms the TRAIN window (after the
    setup tripwire is cancelled — train.py calls it post-construction) and
    returns the armed window; None resolves through the pin."""
    hv = _hv()
    try:
        assert hv.arm_train_tripwire(12345.0) == 12345.0
        assert hv.arm_train_tripwire(None) == 12 * 3600.0
    finally:
        faulthandler.cancel_dump_traceback_later()


def test_smoke_runner_is_a_thin_consumer_of_the_module():
    """Test 4: no duplicated construction definitions and no 25-min
    hardcode survive in scripts/run_sb3_smoke.py — the construction is
    imported from so100rob._hardened_venv (key link, D-07)."""
    src = SMOKE_SCRIPT.read_text()
    assert "def _make_topology_robust" not in src, (
        "scripts/run_sb3_smoke.py still defines _make_topology_robust — "
        "the hardened construction must live only in "
        "so100rob._hardened_venv"
    )
    assert "def _kill_venv" not in src, (
        "scripts/run_sb3_smoke.py still defines _kill_venv — the hardened "
        "construction must live only in so100rob._hardened_venv"
    )
    assert "FAULTHANDLER_TRIPWIRE_S = 25" not in src, (
        "the 25-min tripwire hardcode survives in the smoke runner (IN-09)"
    )
    assert "_hardened_venv" in src, (
        "the smoke runner must import the hardened construction from "
        "so100rob._hardened_venv"
    )