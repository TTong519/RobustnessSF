#!/usr/bin/env python
"""Rendered reward-audit gate (TRAIN-03 tooling, 04-04 Task 3).

Runs a small set of RENDERED episodes through the loaded eval chain and
gates each episode's RAW return against the pre-agreed bounds (T-04-06):

  * a no-success episode must return <= audit_no_success_return_max=2.5
    (no-place arithmetic: the 2.0 dense ceiling — FULL_FLAGS_NO_PLACE_
    FRACTION 0.2 x TERMINAL_PLACE_BONUS 10.0 — minus the 2.0 time-penalty
    ceiling — TIME_PENALTY 0.01 x 200 max control steps — bounds the
    no-place return at 0.0; 2.5 leaves margin);
  * a success must return >= audit_success_return_min=6.0 (the 10.0
    terminal bonus minus the 2.0 dense ceiling minus the 2.0 time
    penalty = 6.0, equality pinned).

The bounds are READ from config/train_gate.json (WR-02), never
literals; the operators (<= / >=) are pinned by tests/test_policy_audit.py.

AUDIT_SPAWN_SET = [12, 0, 6, 18, 24] per the 04-04 plan: the patch-center
spawn first, then the grid spread — five rendered spawns over the frozen
25-spawn grid. All episodes run in the DETERMINISTIC mode (the primary
bar, D-02).

Frames are written as PNG sequences (imageio-ffmpeg is absent from the
pinned env — mp4 is not an option) under
runs/policy_audit/<UTC-timestamp>/frames/spawn<i>/step_%04d.png. Render
verification is PROGRAMMATIC ONLY: per-spawn intensity statistics in the
report (mean/min/max over frames) prove the viewer did not blacken
(gymnasium 1.3.0 GL hazard) — image files are NEVER opened by tooling.

EXIT-CODE CONTRACT: 0 = audit passed (every episode within bounds);
1 = a violating episode (each named by spawn_index + raw_return);
2 = operational error. The last stdout line is always REPORT_JSON=<path>.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # _eval_chain
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so100rob

import imageio.v2 as imageio

import _eval_chain as ec

# The plan's five-spawn audit grid: patch center first, then the spread.
AUDIT_SPAWN_SET = [12, 0, 6, 18, 24]

RUNS_AUDIT_ROOT = Path(__file__).resolve().parents[1] / "runs" / "policy_audit"
REPORT_SCHEMA = "so100rob/policy_audit/1"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rendered reward-audit gate (04-04 Task 3)"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="training bundle dir (the 04-03 four-artifact layout)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="episodes per audited spawn (default: 1)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output dir (default: runs/policy_audit/<UTC-timestamp>)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    gate = ec.load_gate()
    args = parse_args(argv)
    if args.episodes < 1:
        print("ERROR: --episodes must be >= 1", file=sys.stderr)
        return 2

    bundle = args.bundle.resolve()
    if not bundle.is_dir():
        print(f"ERROR: bundle dir {bundle} does not exist", file=sys.stderr)
        return 2
    for spawn in AUDIT_SPAWN_SET:
        if not 0 <= spawn < int(gate["spawns_total"]):
            print(
                f"INTERNAL ERROR: audit spawn {spawn} outside the frozen "
                f"grid [0, {gate['spawns_total']})",
                file=sys.stderr,
            )
            return 2

    seed_base = int(gate["tuning"]["seed"])
    out_dir = args.out if args.out is not None else RUNS_AUDIT_ROOT / utc_stamp()

    # ONE in-process eval chain for the whole audit (D-07 scoping; never
    # closed mid-session, WR-01).
    model, chain = ec.load_bundle(bundle, gate)

    records: list[dict] = []
    frame_summaries: dict[int, dict] = {}
    for spawn in AUDIT_SPAWN_SET:
        frames_dir = out_dir / "frames" / f"spawn{spawn}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_means: list[float] = []

        def frame_hook(step: int, _dir=frames_dir, _means=frame_means):
            # Programmatic render verification (never open the images):
            # the intensity stats prove the offscreen viewer is live.
            frame = np.asarray(ec.raw_env(chain).render())
            if frame.ndim == 3 and frame.shape[-1] == 1:
                frame = frame[..., 0]
            imageio.imwrite(_dir / f"step_{step:04d}.png", frame)
            _means.append(float(frame.mean()))

        recs = ec.run_episodes(
            model,
            chain,
            spawn_index=spawn,
            n_episodes=args.episodes,
            deterministic=True,  # the primary bar (D-02)
            seed=seed_base + spawn,
            frame_hook=frame_hook,
        )
        records.extend(recs)
        frame_summaries[spawn] = {
            "frames": len(frame_means),
            "mean_intensity": (
                float(np.mean(frame_means)) if frame_means else None
            ),
            "min_intensity": float(np.min(frame_means)) if frame_means else None,
            "max_intensity": float(np.max(frame_means)) if frame_means else None,
        }

    verdict = ec.audit_verdict(records, gate)
    report = {
        "schema": REPORT_SCHEMA,
        "bundle": str(bundle),
        "audit_spawn_set": list(AUDIT_SPAWN_SET),
        "episodes_per_spawn": args.episodes,
        "mode": "deterministic",
        "seed_base": seed_base,
        "gate": gate,  # the verbatim pin echo proves which pin was consumed
        "verdict": verdict,
        "frame_summaries": {str(k): v for k, v in frame_summaries.items()},
        "records": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    n_success = sum(1 for r in records if r["is_success"])
    print(
        f"AUDIT DONE: {len(records)} episodes over {len(AUDIT_SPAWN_SET)} "
        f"spawns, {n_success} successes"
    )
    if verdict["violations"]:
        for v in verdict["violations"]:
            print(
                f"AUDIT VIOLATION: spawn_index={v['spawn_index']} "
                f"mode={v['mode']} raw_return={v['raw_return']:.3f} "
                f"is_success={v['is_success']} ({v['reason']})"
            )
        print(f"OUT_DIR={out_dir}")
        # EXIT-CODE CONTRACT: last stdout line is always the artifact path.
        print(f"REPORT_JSON={report_path}")
        return 1
    print(f"OUT_DIR={out_dir}")
    print(f"REPORT_JSON={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())