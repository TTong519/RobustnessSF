#!/usr/bin/env python
"""Physics gate runner (ENV-06, D-06) — the go/no-go evidence for Phase 1.

Runs the spawn-aware scripted grasp-lift-place trajectory on all 25 grid
spawns at the DEFAULT control rate (20 Hz) and evaluates the D-04 success
predicate through so100rob.scripted_policy.resting_at_goal — the pinned
predicate, never a local redefinition (T-01-05). Metric-collection run:
NO render mode (recording perturbs timing).

EXIT-CODE CONTRACT (public interface): exit 0 ONLY when all 25 spawns
pass; exit 1 on any failure or exception — pipelines/CI can never see a
zero on a failed gate.

Output: runs/gate/<UTC-timestamp>/report.json with per-spawn diagnostics
(pass/fail, final XY error, final cube z, cube qvel norm, max lift height,
slip between jaw-close and lift apex, finite-state check, IK residuals)
plus the exact TrajectoryParams used and package versions.

TUNING (A5 resolution): on failure, read the per-spawn diagnostics and
adjust ONE TrajectoryParams knob at a time —
  * slip during lift / cube left behind  -> jaw_closed_ctrl (grip force)
  * cube ejected at descend              -> descend_height / pre_grasp_clearance
  * undershoot at the goal               -> transit_height / ik_iters
  * persistent single-spawn failure       -> scene geometry / wrist grasp
    orientation constants, or DOCUMENT the adversarial spawn in the report
    (D-06: a documented spawn is a finding; an excused one is not).
"""

import datetime as _dt
import json
import platform
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv
from so100rob.scripted_policy import (
    POST_RELEASE_SETTLE_STEPS,
    TrajectoryParams,
    build_trajectory,
    resting_at_goal,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs" / "gate"


def _cube_addressing(env) -> tuple[int, int]:
    """Name-resolved cube freejoint addressing (qposadr, dofadr)."""
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, C.CUBE_JOINT)
    if jid < 0:
        raise ValueError(f"cube joint {C.CUBE_JOINT!r} not found in compiled model")
    return int(env.model.jnt_qposadr[jid]), int(env.model.jnt_dofadr[jid])


class _WarningRecorder:
    """Capture MuJoCo runtime warnings (instability etc.) for the report.

    MuJoCo prints mju_warning to stderr by default; installing a user
    callback turns them into reportable data (the gate's "no MuJoCo
    warnings" bar must be checkable, not assumed).
    """

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self._installed = False
        try:
            mujoco.set_mju_user_warning(self._on_warning)
            self._installed = True
        except AttributeError:  # older binding without the hook
            pass

    def _on_warning(self, message) -> None:
        text = message.decode() if isinstance(message, bytes) else str(message)
        self.warnings.append(text)

    def drain(self) -> list[str]:
        out, self.warnings = self.warnings, []
        return out


def run_episode(env, params: TrajectoryParams, spawn_index: int, rec: _WarningRecorder) -> dict:
    """One gate episode: reset -> read privileged cube pose -> build -> step
    -> settle -> evaluate the D-04 predicate. Returns the diagnostics row."""
    env.reset(seed=0, options={"spawn_index": spawn_index})
    qadr, dadr = _cube_addressing(env)
    data = env.data

    cube_spawn = np.array(data.qpos[qadr : qadr + 3])  # privileged read (D-05)
    debug: dict = {}
    seq = build_trajectory(env.model, data, cube_spawn, C.GOAL_POS, params, debug=debug)

    finite_state = True
    max_cube_z = -np.inf
    lift_apex_z = -np.inf
    lift_apex_xy = None
    xy_at_close = None
    min_cube_z = np.inf
    n_steps = 0

    def observe(during_lift: bool) -> None:
        nonlocal finite_state, max_cube_z, lift_apex_z, lift_apex_xy, min_cube_z
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            finite_state = False
        cube = np.array(data.qpos[qadr : qadr + 3])
        if not np.isfinite(cube).all():
            finite_state = False
            return
        min_cube_z = min(min_cube_z, cube[2])
        if cube[2] > max_cube_z:
            max_cube_z = cube[2]
        if during_lift and cube[2] > lift_apex_z:
            lift_apex_z = cube[2]
            lift_apex_xy = cube[:2].copy()

    for action, label in seq:
        env.step(action)
        n_steps += 1
        observe(during_lift=(label == "lift"))
        if label == "close" and xy_at_close is None:
            xy_at_close = np.array(data.qpos[qadr : qadr + 2])

    # Post-release settle tail: hold the final arm config so the cube comes
    # to rest before the predicate is evaluated.
    last_action = seq[-1][0]
    for _ in range(POST_RELEASE_SETTLE_STEPS):
        env.step(last_action)
        n_steps += 1
        observe(during_lift=False)

    cube_final = np.array(data.qpos[qadr : qadr + 3])
    cube_qvel = np.array(data.qvel[dadr : dadr + 6])
    passed = bool(resting_at_goal(cube_final, cube_qvel)) and finite_state
    # Grasp slip: XY drift of the cube between the start of the close phase
    # and the LIFT-phase apex (cube sliding in the grip during lift).
    if xy_at_close is not None and lift_apex_xy is not None:
        slip = float(np.linalg.norm(lift_apex_xy - xy_at_close))
    else:
        slip = None

    return {
        "spawn_index": spawn_index,
        "grid_x": float(cube_spawn[0]),
        "grid_y": float(cube_spawn[1]),
        "pass": passed,
        "xy_error_m": float(np.linalg.norm(cube_final[:2] - np.asarray(C.GOAL_POS[:2]))),
        "final_cube_z_m": float(cube_final[2]),
        "cube_qvel_norm": float(np.linalg.norm(cube_qvel)),
        "max_cube_z_m": float(max_cube_z) if np.isfinite(max_cube_z) else None,
        "min_cube_z_m": float(min_cube_z) if np.isfinite(min_cube_z) else None,
        "lift_apex_z_m": float(lift_apex_z) if np.isfinite(lift_apex_z) else None,
        "slip_m": slip,
        "finite_state": finite_state,
        "n_control_steps": n_steps,
        "ik_residual_m": debug.get("ik_residual_m", {}),
        "mujoco_warnings": rec.drain(),
    }


def main() -> int:
    params = TrajectoryParams()  # tuned defaults (Task 2 closed-loop result)
    rec = _WarningRecorder()

    # Timing discipline: NO render mode — the gate is metric-collection.
    env = SO100PickPlaceEnv(frame_skip=C.DEFAULT_FRAME_SKIP, spawn_mode="grid")
    try:
        t0 = time.perf_counter()
        spawns = [run_episode(env, params, i, rec) for i in range(C.SPAWN_GRID_SIZE)]
        wall_seconds = time.perf_counter() - t0
    finally:
        env.close()

    leftover_warnings = rec.drain()
    passed_count = sum(1 for s in spawns if s["pass"])
    total = C.SPAWN_GRID_SIZE

    import dataclasses

    report = {
        "gate": "physics",
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "summary": {
            "passed": passed_count,
            "total": total,
            "tolerance_xy_m": C.SUCCESS_TOLERANCE_XY,
            "rest_z_range_m": list(C.CUBE_REST_Z_RANGE),
            "qvel_tol": C.REST_QVEL_TOL,
            "frame_skip": C.DEFAULT_FRAME_SKIP,
            "control_rate_hz": 1.0 / (C.PHYSICS_TIMESTEP * C.DEFAULT_FRAME_SKIP),
            "all_finite": all(s["finite_state"] for s in spawns),
            "mujoco_warnings": len(leftover_warnings) + sum(len(s["mujoco_warnings"]) for s in spawns),
            "wall_seconds": round(wall_seconds, 3),
        },
        "params": dataclasses.asdict(params),
        "versions": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
        "documented_failures": [],
        "spawns": spawns,
    }

    out_dir = RUNS_ROOT / _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"report: {out_dir / 'report.json'}")

    failing = [s for s in spawns if not s["pass"]]
    if passed_count == total:
        print(f"GATE {passed_count}/{total} PASSED")
        return 0
    print(f"GATE {passed_count}/{total} FAILED")
    for s in failing:
        slip = f"{s['slip_m']:.4f}" if s["slip_m"] is not None else "n/a"
        print(
            f"  spawn {s['spawn_index']:2d} (x={s['grid_x']:+.3f}, y={s['grid_y']:+.3f}): "
            f"xy_err={s['xy_error_m']:.4f} final_z={s['final_cube_z_m']:.4f} "
            f"qvel={s['cube_qvel_norm']:.2e} max_z={s['max_cube_z_m']} "
            f"slip={slip} finite={s['finite_state']} "
            f"ik={ {k: round(v, 4) for k, v in s['ik_residual_m'].items()} }"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())