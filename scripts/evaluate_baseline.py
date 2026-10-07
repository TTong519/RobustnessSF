#!/usr/bin/env python
"""75-episode dual-mode threshold evaluation runner (TRAIN-03 tooling).

Two scopes:

  * TARGETED (the Task 1 tracer): one spawn via --spawns, --episodes
    episodes, a single mode — proving the shared eval chain end-to-end
    (synthetic bundle) or spot-checking one spawn on the real bundle.
  * FULL-GRID (Task 2, the threshold scope): the frozen 25-spawn grid x
    episodes_per_spawn=3 episodes in BOTH gate eval_modes (D-02), judged
    by the pre-agreed TRAIN-03 bar (18 of 25 deterministic spawns,
    per-spawn majority) — the real bundle's judgment executes in 04-05,
    never here.

Every pinned number is READ from config/train_gate.json (WR-02) or
so100rob.constants at run time; nothing is re-hardcoded. Eval runs on
ONE in-process env (D-07 scoping) and never closes it (WR-01).

EXIT-CODE CONTRACT: 0 = the evaluation ran and its artifacts were
written (pass/fail judgment, when present, is reported in results.json —
a TRAIN-03 MISS still exits 0 because the measurement succeeded);
nonzero = the evaluation could not run (bad bundle, bad gate, IO
error). The last stdout line is always RESULTS_JSON=<path>.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # _eval_chain
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so100rob

import _eval_chain as ec

RUNS_EVAL_ROOT = Path(__file__).resolve().parents[1] / "runs" / "eval"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRAIN-03 threshold evaluation runner (04-04)"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="training bundle dir (the 04-03 four-artifact layout)",
    )
    parser.add_argument(
        "--mode",
        choices=("deterministic", "stochastic", "both"),
        default="deterministic",
        help="eval mode(s) (D-02; default: deterministic, the primary bar)",
    )
    parser.add_argument(
        "--spawns",
        type=int,
        default=None,
        help="targeted scope: the spawn index (single spawn); omitted with "
        "--full-grid means the full frozen grid (Task 2)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="episodes per spawn (default: gate episodes_per_spawn)",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="full-grid threshold scope: the frozen 25-spawn grid in both "
        "modes (Task 2) — the pre-agreed 18/25 deterministic bar",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output dir (default: runs/eval/<UTC-timestamp>)",
    )
    return parser.parse_args(argv)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["deterministic", "stochastic"]
    return [mode]


def gate_echo(gate: dict) -> dict:
    """Compact echo of the pins the verdict consumed — the 04-05 plan
    verify asserts ``gate_echo.min_spawns_successful`` against the
    committed gate (T-04-17 repudiation guard). Derived from the same
    loaded gate as the verbatim ``gate`` field, so the two can never
    disagree. Rule 3 reconciliation: 04-04 shipped the verbatim gate
    under ``gate`` only; the 04-05 plan/VALIDATION row 04-05-01 globs
    ``gate_echo``."""
    return {
        "schema": gate["schema"],
        "spawns_total": gate["spawns_total"],
        "min_spawns_successful": gate["min_spawns_successful"],
        "episodes_per_spawn": gate["episodes_per_spawn"],
        "per_spawn_majority": gate["per_spawn_majority"],
        "primary_bar": gate["primary_bar"],
        "eval_modes": gate["eval_modes"],
    }


def write_artifacts(out_dir: Path, records: list[dict], results: dict) -> Path:
    """episodes.jsonl (one row per episode, Phase 5 EVAL-02 anticipates
    this shape) + results.json (the run's summary)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episodes.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    return results_path


def main(argv=None) -> int:
    gate = ec.load_gate()
    args = parse_args(argv)

    modes = resolve_modes(args.mode)
    episodes_per_spawn = (
        args.episodes
        if args.episodes is not None
        else int(gate["episodes_per_spawn"])
    )
    if episodes_per_spawn < 1:
        print("ERROR: --episodes must be >= 1", file=sys.stderr)
        return 1

    if args.full_grid:
        # The full-grid threshold scope (Task 2): BOTH gate eval modes over
        # the frozen grid — D-02: every threshold evaluation runs both.
        if args.mode != "both":
            print(
                "ERROR: --full-grid runs BOTH gate eval_modes (D-02) — pass "
                "--mode both",
                file=sys.stderr,
            )
            return 1
        if args.spawns is not None:
            print(
                "ERROR: --spawns is the targeted-scope flag and cannot "
                "combine with --full-grid",
                file=sys.stderr,
            )
            return 1
        if args.episodes is not None and args.episodes != int(
            gate["episodes_per_spawn"]
        ):
            print(
                f"ERROR: --episodes {args.episodes} != the pinned "
                f"episodes_per_spawn={gate['episodes_per_spawn']} — the "
                "threshold grid is pre-agreed (D-01)",
                file=sys.stderr,
            )
            return 1

        bundle = args.bundle.resolve()
        if not bundle.is_dir():
            print(f"ERROR: bundle dir {bundle} does not exist", file=sys.stderr)
            return 1

        seed_base = int(gate["tuning"]["seed"])
        spawn_indices = ec.full_grid_spawn_indices(gate)
        modes = list(gate["eval_modes"])
        model, chain = ec.load_bundle(bundle, gate)

        records: list[dict] = []
        results_by_mode: dict[str, dict[int, list[dict]]] = {m: {} for m in modes}
        for mode in modes:
            deterministic = mode == "deterministic"
            for idx in spawn_indices:
                # unique per-spawn seed ranges; run_episodes derives +ep
                seed = seed_base + idx * episodes_per_spawn
                recs = ec.run_episodes(
                    model,
                    chain,
                    spawn_index=idx,
                    n_episodes=episodes_per_spawn,
                    deterministic=deterministic,
                    seed=seed,
                )
                results_by_mode[mode][idx] = recs
                records.extend(recs)

        verdict = ec.judge(results_by_mode, gate)
        out_dir = args.out if args.out is not None else RUNS_EVAL_ROOT / utc_stamp()
        results = {
            "scope": "full_grid",
            "bundle": str(bundle),
            "episodes_per_spawn": episodes_per_spawn,
            "modes": modes,
            "seed_base": seed_base,
            "gate": gate,  # the verbatim pin echo proves which pin was consumed
            "gate_echo": gate_echo(gate),  # compact judging pins (04-05 verify)
            "records_count": len(records),
            "verdict": verdict,
            # Top-level mirrors of the verdict headline — the 04-05 plan
            # verify prints d['deterministic_pass'] etc. directly. Derived
            # from the same in-memory verdict block, so they can never
            # disagree with the nested verdict (Rule 3 reconciliation).
            "deterministic_pass": verdict["deterministic_pass"],
            "deterministic_rate": verdict["deterministic_rate"],
            "stochastic_rate": verdict["stochastic_rate"],
        }
        results_path = write_artifacts(out_dir, records, results)

        n_pass = sum(1 for ok in verdict["per_spawn"][gate["primary_bar"]].values() if ok)
        print(
            f"EVAL DONE: {len(records)} episodes over "
            f"{verdict['spawns_evaluated']} spawns, {n_pass} primary-bar "
            f"spawns successful -> deterministic_pass="
            f"{verdict['deterministic_pass']}"
        )
        print(f"OUT_DIR={out_dir}")
        # EXIT-CODE CONTRACT: last stdout line is always the artifact path.
        print(f"RESULTS_JSON={results_path}")
        return 0

    if args.spawns is None:
        print(
            "ERROR: targeted scope needs --spawns <spawn_index> (or wait "
            "for Task 2's --full-grid)",
            file=sys.stderr,
        )
        return 1
    if not 0 <= args.spawns < int(gate["spawns_total"]):
        print(
            f"ERROR: --spawns {args.spawns} outside the frozen grid "
            f"[0, {gate['spawns_total']})",
            file=sys.stderr,
        )
        return 1

    bundle = args.bundle.resolve()
    if not bundle.is_dir():
        print(f"ERROR: bundle dir {bundle} does not exist", file=sys.stderr)
        return 1

    seed_base = int(gate["tuning"]["seed"])
    model, chain = ec.load_bundle(bundle, gate)

    records: list[dict] = []
    for mode in modes:
        deterministic = mode == "deterministic"
        recs = ec.run_episodes(
            model,
            chain,
            spawn_index=args.spawns,
            n_episodes=episodes_per_spawn,
            deterministic=deterministic,
            seed=seed_base,
        )
        records.extend(recs)

    out_dir = args.out if args.out is not None else RUNS_EVAL_ROOT / utc_stamp()
    results = {
        "scope": "targeted",
        "bundle": str(bundle),
        "spawn_index": int(args.spawns),
        "episodes_per_spawn": episodes_per_spawn,
        "modes": modes,
        "seed_base": seed_base,
        "gate": gate,  # the verbatim pin echo proves which pin was consumed
        "gate_echo": gate_echo(gate),  # compact judging pins (04-05 verify)
        "records_count": len(records),
    }
    results_path = write_artifacts(out_dir, records, results)

    n_success = sum(1 for r in records if r["is_success"])
    print(f"EVAL DONE: {len(records)} episodes, {n_success} successes")
    print(f"OUT_DIR={out_dir}")
    # EXIT-CODE CONTRACT: last stdout line is always the artifact path.
    print(f"RESULTS_JSON={results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())