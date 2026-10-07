"""Full-grid threshold judgment contract tests (Plan 04-04 Task 2, TRAIN-03).

Pin the PURE judgment layer before the real bundle is judged in 04-05:

  * per_spawn_success — the pre-agreed majority rule (D-01: a spawn counts
    successful on a majority of its episodes_per_spawn=3 episodes;
    majority=2 -> [T,T,F] passes, [T,F,F] misses) — read from the gate,
    never a local constant.
  * judge — the TRAIN-03 bar: min_spawns_successful=18 of EXACTLY the
    spawns_total=25 frozen spawns, deterministic (the primary bar,
    D-02); 18 passes, 17 misses; a perfect stochastic sweep (25/25)
    CANNOT flip a deterministic miss.
  * full_grid_spawn_indices — the frozen 25-spawn grid, integrity-checked
    against the gate (a doctored pin fails loudly, T-04-04 tampering).
  * the runner's full-grid wiring — evaluate_baseline.main in --full-grid
    scope writes results.json with the verdict + rates + per-spawn
    breakdown + verbatim gate echo (verified WITHOUT env construction by
    monkeypatching the eval chain — the pure judgment is the contract).

All pinned numbers are READ from config/train_gate.json — a doctored-gate
copy proves the bounds are not literals (WR-02).
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
GATE_PATH = REPO_ROOT / "config" / "train_gate.json"


def load_modules():
    """Import _eval_chain + evaluate_baseline (missing artifact = the
    established intentional failing assertion)."""
    for name in ("_eval_chain.py", "evaluate_baseline.py"):
        if not (SCRIPTS_DIR / name).exists():
            pytest.fail(
                f"scripts/{name} does not exist yet — RED: the eval runner "
                "is the missing artifact (04-04 Task 2)"
            )
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import _eval_chain as ec
    import evaluate_baseline as eb

    return ec, eb


def load_gate() -> dict:
    return json.loads(GATE_PATH.read_text())


# ---------------------------------------------------------------------------
# Synthetic episode records + results (pure data, no env)
# ---------------------------------------------------------------------------


def episodes_for_spawn(spawn_index: int, n_success: int, mode="deterministic"):
    """episodes_per_spawn records for one spawn with exactly n_success
    successes (the rest misses)."""
    gate = load_gate()
    n = int(gate["episodes_per_spawn"])
    return [
        {
            "mode": mode,
            "spawn_index": spawn_index,
            "seed": None,
            "is_success": ep < n_success,
            "raw_return": 0.0,
            "steps": 200,
            "stage_flags": {},
            "reward_components": {},
        }
        for ep in range(n)
    ]


def make_results(successful_det: int, successful_stoch: int | None = None) -> dict:
    """Full results over the frozen grid: the FIRST successful_det spawns
    pass their deterministic majority, the rest miss; same for stochastic
    (omitted -> the mode is absent, e.g. a single-mode run)."""
    gate = load_gate()
    total = int(gate["spawns_total"])
    majority = int(gate["per_spawn_majority"])
    results = {}
    det = {
        i: episodes_for_spawn(i, majority if i < successful_det else majority - 1)
        for i in range(total)
    }
    results["deterministic"] = det
    if successful_stoch is not None:
        results["stochastic"] = {
            i: episodes_for_spawn(
                i,
                majority if i < successful_stoch else majority - 1,
                mode="stochastic",
            )
            for i in range(total)
        }
    return results


# ---------------------------------------------------------------------------
# per_spawn_success — the pre-agreed majority rule (D-01)
# ---------------------------------------------------------------------------


def test_per_spawn_majority_rule_from_gate():
    """majority=2 of 3: [T,T,F] passes, [T,T,T] passes, [T,F,F] misses,
    [F,F,F] misses — for EVERY count 0..3, judged against the gate's
    per_spawn_majority, never a local 2."""
    ec, _eb = load_modules()
    gate = load_gate()
    majority = int(gate["per_spawn_majority"])
    n = int(gate["episodes_per_spawn"])
    assert majority == 2 and n == 3, "the frozen pins: majority 2 of 3 episodes"
    for n_success in range(n + 1):
        episodes = episodes_for_spawn(0, n_success)
        expected = n_success >= majority
        assert ec.per_spawn_success(episodes, majority) is expected, (
            f"{n_success} successes of {n} episodes must be "
            f"{'success' if expected else 'miss'} at majority={majority}"
        )


# ---------------------------------------------------------------------------
# judge — the TRAIN-03 bar (18 of exactly 25, deterministic)
# ---------------------------------------------------------------------------


def test_judge_pass_bar_18_of_25():
    """18 successful deterministic spawns -> deterministic_pass True;
    17 -> False — the pinned boundary, read by judge from (results, gate)
    alone."""
    ec, _eb = load_modules()
    gate = load_gate()
    assert int(gate["min_spawns_successful"]) == 18

    verdict = ec.judge(make_results(successful_det=18), gate)
    assert verdict["deterministic_pass"] is True
    assert verdict["deterministic_rate"] == pytest.approx(18 / 25)

    verdict = ec.judge(make_results(successful_det=17), gate)
    assert verdict["deterministic_pass"] is False
    assert verdict["deterministic_rate"] == pytest.approx(17 / 25)


def test_judge_stochastic_cannot_flip_deterministic_miss():
    """D-02: the bar is deterministic-only. A perfect 25/25 stochastic
    sweep reports stochastic_rate=1.0 but cannot flip a 17-pass
    deterministic miss."""
    ec, _eb = load_modules()
    gate = load_gate()

    verdict = ec.judge(make_results(successful_det=17, successful_stoch=25), gate)
    assert verdict["deterministic_pass"] is False
    assert verdict["stochastic_rate"] == pytest.approx(1.0)


def test_judge_reports_per_spawn_breakdown():
    """The per-spawn breakdown names every spawn's success per mode — the
    04-05 report and Phase 5 comparisons consume this shape."""
    ec, _eb = load_modules()
    gate = load_gate()
    total = int(gate["spawns_total"])

    verdict = ec.judge(make_results(successful_det=18, successful_stoch=10), gate)
    assert set(verdict["per_spawn"]["deterministic"]) == set(range(total))
    assert set(verdict["per_spawn"]["stochastic"]) == set(range(total))
    assert sum(verdict["per_spawn"]["deterministic"].values()) == 18
    assert sum(verdict["per_spawn"]["stochastic"].values()) == 10
    assert verdict["spawns_evaluated"] == total


def test_judge_fails_loudly_off_grid_primary_results():
    """The bar is 18 of EXACTLY 25 — a primary-mode result set that is not
    exactly the frozen grid (missing spawn, extra spawn) fails loudly
    instead of judging a partial sweep."""
    ec, _eb = load_modules()
    gate = load_gate()

    results = make_results(successful_det=18)
    del results["deterministic"][24]  # one spawn missing
    with pytest.raises(ValueError, match="spawns_total"):
        ec.judge(results, gate)

    results = make_results(successful_det=18)
    results["deterministic"][25] = episodes_for_spawn(25, 3)  # off-grid extra
    with pytest.raises(ValueError, match="spawns_total"):
        ec.judge(results, gate)


# ---------------------------------------------------------------------------
# full_grid_spawn_indices — the frozen grid, integrity-checked
# ---------------------------------------------------------------------------


def test_full_grid_spawn_indices_matches_frozen_grid():
    """The full grid is list(range(25)) — exactly the frozen spawns file's
    entry order, integrity-checked against gate spawns_total AND the
    packaged grid constant."""
    ec, _eb = load_modules()
    gate = load_gate()

    indices = ec.full_grid_spawn_indices(gate)
    assert indices == list(range(int(gate["spawns_total"])))
    # the file behind it: same count, same order stability
    spawns = json.loads(
        (REPO_ROOT / "src" / "so100rob" / "data" / "spawn_seeds.json").read_text()
    )["spawns"]
    assert len(indices) == len(spawns)


def test_full_grid_spawn_indices_doctored_gate_fails_loudly():
    """A doctored pin (spawns_total inconsistent with the frozen grid /
    the constant) fails loudly — never a silent partial sweep (T-04-04)."""
    ec, _eb = load_modules()
    gate = load_gate()

    doctored = dict(gate, spawns_total=20)
    with pytest.raises(ValueError):
        ec.full_grid_spawn_indices(doctored)


# ---------------------------------------------------------------------------
# The runner's full-grid wiring (no env construction — pure judgment path)
# ---------------------------------------------------------------------------


def test_full_grid_scope_writes_verdict_results_json(tmp_path, monkeypatch):
    """evaluate_baseline.main --full-grid runs BOTH gate eval_modes over
    the frozen grid and writes results.json with scope=full_grid, the
    judge's verdict + rates + per-spawn breakdown, and the verbatim gate
    echo. The eval chain is monkeypatched with SYNTHETIC records — this
    pins the CLI wiring + artifact shape, not the env."""
    ec, eb = load_modules()
    gate = load_gate()
    total = int(gate["spawns_total"])
    majority = int(gate["per_spawn_majority"])

    class _FakeModel:
        pass

    def fake_run_episodes(model, chain, spawn_index, n_episodes, deterministic, seed=None, frame_hook=None):
        mode = "deterministic" if deterministic else "stochastic"
        # deterministic: 18 spawns pass their majority; stochastic: 10
        if deterministic:
            n_success = majority if spawn_index < 18 else majority - 1
        else:
            n_success = majority if spawn_index < 10 else majority - 1
        return [
            {
                "mode": mode,
                "spawn_index": spawn_index,
                "seed": seed,
                "is_success": ep < n_success,
                "raw_return": 0.0,
                "steps": 200,
                "stage_flags": {},
                "reward_components": {},
            }
            for ep in range(n_episodes)
        ]

    monkeypatch.setattr(ec, "load_bundle", lambda *a, **kw: (_FakeModel(), object()))
    monkeypatch.setattr(ec, "run_episodes", fake_run_episodes)

    out_dir = tmp_path / "fullgrid"
    exit_code = eb.main(
        [
            "--bundle",
            str(tmp_path),  # never read: load_bundle is faked
            "--full-grid",
            "--mode",
            "both",
            "--out",
            str(out_dir),
        ]
    )
    assert exit_code == 0

    results = json.loads((out_dir / "results.json").read_text())
    assert results["scope"] == "full_grid"
    assert results["modes"] == ["deterministic", "stochastic"]
    assert results["episodes_per_spawn"] == int(gate["episodes_per_spawn"])
    assert results["verdict"]["deterministic_pass"] is True
    assert results["verdict"]["deterministic_rate"] == pytest.approx(18 / 25)
    assert results["verdict"]["stochastic_rate"] == pytest.approx(10 / 25)
    assert sum(results["verdict"]["per_spawn"]["deterministic"].values()) == 18
    assert sum(results["verdict"]["per_spawn"]["stochastic"].values()) == 10
    assert results["records_count"] == 2 * total * int(gate["episodes_per_spawn"])
    # the verbatim pin echo proves which pin the run consumed
    assert results["gate"] == gate
    # the compact gate_echo is what the 04-05 plan verify globs — it must
    # mirror the committed judging pins (Rule 3 reconciliation, T-04-17)
    echo = results["gate_echo"]
    assert echo["min_spawns_successful"] == gate["min_spawns_successful"] == 18
    assert echo["spawns_total"] == gate["spawns_total"]
    assert echo["episodes_per_spawn"] == gate["episodes_per_spawn"]
    assert echo["per_spawn_majority"] == gate["per_spawn_majority"]
    assert echo["primary_bar"] == gate["primary_bar"]
    assert echo["eval_modes"] == gate["eval_modes"]
    # episodes.jsonl: one row per episode
    lines = (out_dir / "episodes.jsonl").read_text().strip().splitlines()
    assert len(lines) == results["records_count"]