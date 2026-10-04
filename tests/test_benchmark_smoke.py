"""Structural smoke test for the throughput benchmark (Plan 01-04, D-15).

Runs scripts/benchmark.py --quick in a subprocess (seconds, not minutes) and
asserts the output JSON contract: all 4 config names present, state_only has
exactly 1 timing mode (raw), each render config has both timing modes
(raw + mock_cnn), and every steps_per_sec is a positive finite number.

This enforces the T-01-07 measurement-integrity surface: the structure the
Phase 3 obs-space decision reads must hold even in quick mode.

The subprocess uses sys.executable — the same pinned 3.12 venv interpreter
pytest runs under (Pitfall 5: never the Homebrew system python).
"""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark.py"

ALL_CONFIGS = {"state_only", "rgb64", "gray64", "rgb128"}
RENDER_MODES = {"raw", "mock_cnn"}


def _run_quick_benchmark() -> dict:
    proc = subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), "--quick"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"benchmark --quick exited {proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    match = re.search(r"^RESULTS_JSON=(.+)$", proc.stdout, re.MULTILINE)
    assert match is not None, f"no RESULTS_JSON line in stdout:\n{proc.stdout}"
    results_path = Path(match.group(1).strip())
    assert results_path.exists(), f"reported JSON missing: {results_path}"
    return json.loads(results_path.read_text())


def test_quick_benchmark_output_structure():
    data = _run_quick_benchmark()

    results = data["results"]
    assert isinstance(results, list) and len(results) == 7, (
        f"expected 7 (config, mode) entries, got {len(results)}"
    )

    entries: dict[str, set[str]] = {}
    for entry in results:
        config, mode = entry["config"], entry["mode"]
        entries.setdefault(config, set()).add(mode)

    # All 4 configs present.
    assert set(entries) == ALL_CONFIGS, f"configs mismatch: {set(entries)}"

    # state_only: exactly 1 timing mode (raw — it must never render).
    # Each render config: both timing modes (D-14).
    assert entries["state_only"] == {"raw"}, (
        f"state_only must have only the raw mode, got {entries['state_only']}"
    )
    for config in ALL_CONFIGS - {"state_only"}:
        assert entries[config] == RENDER_MODES, (
            f"{config} must have modes {RENDER_MODES}, got {entries[config]}"
        )

    # Every steps_per_sec is a positive finite number, and repeats are
    # persisted with the median (T-01-07: no un-warmed single-run figures).
    for entry in results:
        value = entry["steps_per_sec"]
        assert isinstance(value, (int, float)), entry
        assert math.isfinite(value) and value > 0, entry
        repeats = entry["repeats"]
        assert isinstance(repeats, list) and len(repeats) >= 1, entry
        assert entry["warmup_steps"] >= 1 and entry["n_steps"] >= 1, entry

    # Machine context and pinned package versions recorded for audit (T-01-08).
    assert "machine" in data and data["machine"].get("platform"), data.get("machine")
    versions = data["versions"]
    for pkg in ("mujoco", "gymnasium", "torch", "numpy"):
        assert pkg in versions and versions[pkg], versions

    # The Pitfall 8 sanity gate must actually gate (WR-02): a FAIL status
    # must never ride out in officially consumable JSON.
    assert data["sanity_gate"]["status"] == "OK", data["sanity_gate"]
