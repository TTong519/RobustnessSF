"""Subprocess smoke test for the reward audit runner (Plan 02-02, D-10).

Runs scripts/run_reward_audit.py under the same pinned venv interpreter
(sys.executable — Pitfall 5: never the Homebrew system python), asserts
the exit-code contract (exit 0 ONLY on a fully green audit) and the
"AUDIT PASSED" stdout line, then loads the newest report.json from
runs/reward_audit/ and checks its shape: 28/28 passed, frame_skip pinned
to DEFAULT_FRAME_SKIP (Pitfall 3), and every trajectory record carrying
the per-step info series with all stage flags and reward components,
first-flip indices, terminal count, and dense return.
"""

import json
import subprocess
import sys
from pathlib import Path

from so100rob import constants as C

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "run_reward_audit.py"
RUNS_ROOT = REPO_ROOT / "runs" / "reward_audit"

_STAGE_FLAG_KEYS = ("stage_approach", "stage_grasp", "stage_lift", "stage_place")
_PER_STEP_INFO_KEYS = (
    "stage_approach",
    "stage_grasp",
    "stage_lift",
    "stage_place",
    "stage_index",
    "is_success",
    "reward_shaping",
    "reward_time_penalty",
    "reward_terminal",
    "reward_phi",
)


def _latest_report() -> Path:
    reports = sorted(RUNS_ROOT.glob("*/report.json"))
    assert reports, "no runs/reward_audit/*/report.json found"
    return reports[-1]


def test_reward_audit_exit_contract_and_report_shape():
    proc = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"audit exited {proc.returncode} — the exit-code contract allows 0 "
        f"ONLY on a fully green audit\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "AUDIT PASSED: 28/28" in proc.stdout, proc.stdout

    report = json.loads(_latest_report().read_text())
    summary = report["summary"]
    assert summary["passed"] == summary["total"] == 28, summary
    assert summary["frame_skip"] == C.DEFAULT_FRAME_SKIP, (
        f"frame_skip must be pinned to DEFAULT_FRAME_SKIP="
        f"{C.DEFAULT_FRAME_SKIP} in the report (Pitfall 3), got "
        f"{summary['frame_skip']}"
    )
    # the pinned constants snapshot is the audit's reproducibility evidence
    snapshot = summary["constants"]
    for key in (
        "GRASP_WINDOW_STEPS",
        "GRASP_CUBE_OFF_FLOOR_Z_M",
        "STAGE_POTENTIALS",
        "TERMINAL_PLACE_BONUS",
        "TIME_PENALTY",
        "REWARD_GAMMA",
        "FULL_FLAGS_NO_PLACE_MAX_FRACTION",
        "AUDIT_DENSE_FRACTION",
    ):
        assert key in snapshot, f"constants snapshot missing {key}"

    trajectories = report["trajectories"]
    assert len(trajectories) == 28
    names = {t["name"] for t in trajectories}
    assert {
        "momentary_contact",
        "approach_retreat",
        "drop_outside_goal",
    } <= names, names
    positives = [t for t in trajectories if t["name"].startswith("positive_spawn")]
    assert len(positives) == 25

    for t in trajectories:
        assert "first_flip_indices" in t and "terminal_paid_count" in t
        assert "dense_return" in t and "phi_0" in t
        assert t["pass"] is True, f"{t['name']} failed its assertions"
        assert isinstance(t["steps"], list) and len(t["steps"]) == t["n_steps"] > 0
        for step in t["steps"]:
            info = step["info"]
            for key in _PER_STEP_INFO_KEYS:
                assert key in info, f"{t['name']}: per-step info missing {key}"

    # positives: all four flags flipped in strict order, is_success True,
    # and the terminal bonus paid exactly once
    for t in positives:
        flips = t["first_flip_indices"]
        assert set(flips) == set(_STAGE_FLAG_KEYS), flips
        idx = [flips[k] for k in _STAGE_FLAG_KEYS]
        assert idx == sorted(idx) and len(set(idx)) == 4, flips
        assert t["terminal_paid_count"] == 1, t["name"]
        assert t["is_success"] is True, t["name"]

    # drop-outside-goal: everything but place — and no terminal payment
    drop = next(t for t in trajectories if t["name"] == "drop_outside_goal")
    assert drop["final_flags"] == {
        "stage_approach": True,
        "stage_grasp": True,
        "stage_lift": True,
        "stage_place": False,
    }, drop["final_flags"]
    assert drop["is_success"] is False and drop["terminal_paid_count"] == 0