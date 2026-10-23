#!/usr/bin/env python
"""Reward audit runner (D-10, ENV-05 criterion 3) — the Phase 2 evidence trail.

Rolls 28 trajectories through ONE reused env instance — the positive
positive_sequence on all 25 frozen grid spawns plus the three negative
controls on spawn 0 (momentary contact, approach-retreat,
drop-outside-goal, D-09) — capturing the full info contract (stage flags,
stage_index, is_success, reward components, reward_phi) at EVERY step, and
numerically proves the roadmap's exact success-criterion wording:

    "dense reward never accumulates significantly without a stage flag
     advancing (potential-based shaping, terminal reward dominant)"

via three checks per trajectory: the constant-(stage_index, reward_phi)
window dense bound, the full-flags-no-place fraction bound on the
drop-outside episode, and the telescoping closed-form identity.

The one-env reuse doubles as the reset-hygiene regression (Pitfall 6):
the negatives roll AFTER 25 successful positives, so any leaked flag,
grasp-window counter, terminal latch, or prev-Phi snapshot shows up as a
failing negative — exactly the leak scenario the 02-01 consecutive-
episode test pins.

EXIT-CODE CONTRACT (public interface): exit 0 ONLY when all 28
trajectories roll and every assertion passes; exit 1 on any failure or
exception — pipelines/CI can never see a zero on a failed audit. Every
failing check is printed with its trajectory name and step index and
recorded in the report JSON; nothing is swallowed (T-02-05).

Output: runs/reward_audit/<UTC-timestamp>/report.json — a fixed relative
path, no user input reaches it (T-02-06) — with the per-step series,
first-flip indices, dense returns, the criterion-3 numerics, a snapshot of
the pinned stage/reward constants, frame_skip (Pitfall 3: GRASP_WINDOW_STEPS
is pinned in control steps; its sim-time meaning scales with frame_skip),
and package versions. This JSON is the durable evidence trail Phase 4's
TRAIN-03 builds on.

Plain python script (NOT mjpython) — sys.exit signals failure normally
here, unlike the Phase 1 debug-viewer lesson.
"""

import datetime as _dt
import json
import platform
import sys
import time
from pathlib import Path

import gymnasium
import mujoco
import numpy as np

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv
from so100rob.scripted_policy import (
    TrajectoryParams,
    approach_retreat_sequence,
    drop_outside_goal_sequence,
    momentary_contact_sequence,
    positive_sequence,
)

RUNS_ROOT = Path("runs") / "reward_audit"

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


def _cube_addressing(env) -> tuple[int, int]:
    """Name-resolved cube freejoint addressing (qposadr, dofadr)."""
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, C.CUBE_JOINT)
    if jid < 0:
        raise ValueError(f"cube joint {C.CUBE_JOINT!r} not found in compiled model")
    return int(env.model.jnt_qposadr[jid]), int(env.model.jnt_dofadr[jid])


class _WarningRecorder:
    """Capture MuJoCo runtime warnings (instability etc.) for the report.

    MuJoCo prints mju_warning to stderr by default; installing a user
    callback turns them into reportable data (the "no MuJoCo warnings"
    bar must be checkable, not assumed). Verbatim from run_physics_gate.py.
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


class _Audit:
    """Per-trajectory assertion collector (T-02-05: no swallowed failures).

    Every check is recorded (passed + detail) for the report JSON; every
    failure is ALSO printed immediately with the trajectory name so a red
    audit can never look green.
    """

    def __init__(self, name: str, spawn_index: int) -> None:
        self.name = name
        self.spawn_index = spawn_index
        self.assertions: list[dict] = []
        self.failed_checks: list[tuple[str, str]] = []

    def check(self, check_name: str, passed, detail: str = "") -> bool:
        ok = bool(passed)
        self.assertions.append(
            {"name": check_name, "passed": ok, "detail": str(detail)}
        )
        if not ok:
            self.failed_checks.append((check_name, str(detail)))
            print(f"  FAIL [{self.name}] {check_name}: {detail}")
        return ok

    @property
    def passed(self) -> bool:
        return not self.failed_checks


def _roll(env, seq) -> tuple[list[dict], float]:
    """Roll a labeled sequence, capturing (reward, terminated, truncated,
    info) at EVERY step. MUST be called immediately after env.reset — it
    captures the reset-seeded prev-Phi BEFORE any step (the 02-01 lesson:
    after the rollout env._prev_phi holds the last step's Phi, not the
    reset seed; the telescoping identity needs the true phi_0)."""
    phi_0 = float(getattr(env, "_prev_phi"))
    steps: list[dict] = []
    for action, _label in seq:
        _, reward, terminated, truncated, info = env.step(action)
        steps.append(
            {
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": dict(info),
            }
        )
    return steps, phi_0


def _first_flip_indices(steps) -> dict[str, int]:
    flips: dict[str, int] = {}
    for i, s in enumerate(steps):
        for key in _STAGE_FLAG_KEYS:
            if s["info"][key] and key not in flips:
                flips[key] = i
    return flips


def _first_violation(steps, pred) -> int | None:
    """Index of the first step where ``pred(step, i)`` holds, else None."""
    for i, s in enumerate(steps):
        if pred(s, i):
            return i
    return None


def _assert_positive(aud: _Audit, steps, labels, params: TrajectoryParams) -> None:
    """Positive-per-spawn assertions: criterion 1 (strict flag ordering),
    criterion 2's positive half (the window accepting a REAL hold), the
    is_success/terminated gating, the one-shot terminal payment, and the
    finite-state guard."""
    flips = _first_flip_indices(steps)

    aud.check(
        "criterion1_all_flags_flip",
        set(flips) == set(_STAGE_FLAG_KEYS),
        f"expected all four flags to flip, got {dict(flips)}",
    )
    if set(flips) == set(_STAGE_FLAG_KEYS):
        idx = [flips[k] for k in _STAGE_FLAG_KEYS]
        aud.check(
            "criterion1_strict_order_approach_grasp_lift_place",
            idx == sorted(idx) and len(set(idx)) == 4,
            f"first-flip indices must strictly increase approach < grasp < "
            f"lift < place, got {dict(flips)}",
        )

        # Criterion 2, positive half (04-07 re-pin): the grasp flip must land
        # inside the LIFT run — the window COMPLETES as the cube leaves the
        # floor (the off-floor z-clause is the only path to the flag: the
        # counter accumulates during the at-rest close, and the flag sets on
        # the first above-pin z readout). A flip inside the close run means
        # the z-clause is not wired.
        first_close = labels.index("close")
        grasp_i = flips["stage_grasp"]
        aud.check(
            "criterion2_grasp_flip_completes_as_cube_leaves_floor",
            first_close + params.close_dwell_steps
            <= grasp_i
            < first_close + params.close_dwell_steps + params.waypoint_dwell_steps,
            f"grasp flip at step {grasp_i} outside the lift run "
            f"[{first_close + params.close_dwell_steps}, "
            f"{first_close + params.close_dwell_steps + params.waypoint_dwell_steps}) "
            f"— a flip inside the close run means the off-floor z-clause is "
            f"not wired (04-07)",
        )

        place_i = flips["stage_place"]
        bad_success = _first_violation(
            steps, lambda s, i: s["info"]["is_success"] != (i >= place_i)
        )
        aud.check(
            "is_success_gated_by_place_every_step",
            bad_success is None,
            f"step {bad_success}: is_success must be False before the place "
            f"flip ({place_i}) and True from it on (Pitfall 7)",
        )
        bad_terminated = _first_violation(
            steps, lambda s, i: s["terminated"] != (i >= place_i)
        )
        aud.check(
            "terminated_from_place_flip",
            bad_terminated is None,
            f"step {bad_terminated}: terminated must be True on and after "
            f"the place flip ({place_i})",
        )
        terminal_steps = [
            i
            for i, s in enumerate(steps)
            if s["info"]["reward_terminal"] == C.TERMINAL_PLACE_BONUS
        ]
        aud.check(
            "terminal_bonus_paid_exactly_once_at_place",
            terminal_steps == [place_i],
            f"TERMINAL_PLACE_BONUS must be paid exactly once on the place "
            f"step ({place_i}), paid at {terminal_steps}",
        )

    bad_trunc = _first_violation(steps, lambda s, i: s["truncated"])
    aud.check(
        "truncated_never_on_positive_rollout",
        bad_trunc is None,
        f"step {bad_trunc}: the 160-step positive rollout is inside the "
        f"episode cap — truncated must never fire",
    )


def _assert_negative_momentary(aud: _Audit, steps) -> None:
    """D-09a: a 2-step sub-window close then retreat must NEVER set grasp
    (the multi-step verification window is the only path to the flag)."""
    bad_grasp = _first_violation(steps, lambda s, i: s["info"]["stage_grasp"])
    aud.check(
        "momentary_contact_never_sets_grasp",
        bad_grasp is None,
        f"step {bad_grasp}: stage_grasp set on a momentary contact — the "
        f"window is the only path to the flag (D-09a)",
    )
    final = steps[-1]["info"]
    aud.check(
        "momentary_final_no_place_no_success",
        not final["stage_place"] and not final["is_success"],
        f"final place={final['stage_place']} is_success={final['is_success']} "
        "must both be False",
    )


def _assert_negative_approach_retreat(aud: _Audit, steps) -> None:
    """D-09b (Pitfall 4): approach-retreat sets ONLY approach — the
    monotone flags never un-set, so the assertion is on the OTHER flags
    staying False at every step."""
    for key in ("stage_grasp", "stage_lift", "stage_place"):
        bad = _first_violation(steps, lambda s, i, k=key: s["info"][k])
        aud.check(
            f"approach_retreat_{key}_never_true",
            bad is None,
            f"step {bad}: {key} must never set on an approach-retreat (D-09b)",
        )
    bad_success = _first_violation(steps, lambda s, i: s["info"]["is_success"])
    aud.check(
        "approach_retreat_is_success_never_true",
        bad_success is None,
        f"step {bad_success}: is_success must never set (D-09b)",
    )
    final = steps[-1]["info"]
    aud.check(
        "approach_retreat_approach_true_at_end",
        final["stage_approach"],
        "the arm ends parked at the pre-grasp hover above the cube — "
        "approach must be True at episode end",
    )


def _assert_negative_drop_outside(aud: _Audit, steps) -> None:
    """D-09c: grasp-lift-drop-outside-goal sets approach/grasp/lift but
    NEVER place or is_success — the place predicate is goal-anchored, not
    rest-anywhere."""
    final = steps[-1]["info"]
    aud.check(
        "drop_outside_approach_grasp_lift_true_at_end",
        final["stage_approach"] and final["stage_grasp"] and final["stage_lift"],
        f"expected approach/grasp/lift True at the end, got {dict(final)}",
    )
    bad_place = _first_violation(steps, lambda s, i: s["info"]["stage_place"])
    aud.check(
        "drop_outside_place_never_true",
        bad_place is None,
        f"step {bad_place}: place must NEVER set when the cube rests 0.15 m "
        f"from the goal (D-09c)",
    )
    bad_success = _first_violation(steps, lambda s, i: s["info"]["is_success"])
    aud.check(
        "drop_outside_is_success_never_true",
        bad_success is None,
        f"step {bad_success}: is_success must never set (D-09c)",
    )
    bad_term = _first_violation(steps, lambda s, i: s["terminated"])
    aud.check(
        "drop_outside_terminated_never_true",
        bad_term is None,
        f"step {bad_term}: termination follows place only (D-09c)",
    )


def _constant_window_runs(steps, phi_0: float) -> list[dict]:
    """Maximal constant-(stage_index, reward_phi) windows, per trajectory.

    Roadmap criterion 3: "dense reward never accumulates significantly
    without a stage flag advancing". A window is a maximal run of
    consecutive steps k whose (stage_index, reward_phi) pair is UNCHANGED
    from step k-1 (the reset seed (0, phi_0) before step 0) — the exact
    stretches where no flag advanced and Phi did not move. The step where
    the pair CHANGED INTO the window (the flag-advance payment itself) is
    deliberately NOT part of any window: including it would conflate the
    legitimate one-time advance payment with farming. Inside a window each
    step's shaping is exactly (gamma-1)*Phi <= 0 plus the time penalty — a
    correct potential-based reward can never accumulate positively there,
    and any non-potential dense bonus piles up past the bound within a few
    steps.
    """
    runs: list[dict] = []
    prev = (0, phi_0)
    start: int | None = None
    dense = 0.0
    for i, s in enumerate(steps):
        key = (s["info"]["stage_index"], s["info"]["reward_phi"])
        if key == prev:
            if start is None:
                start = i
                dense = 0.0
            dense += s["info"]["reward_shaping"] + s["info"]["reward_time_penalty"]
        elif start is not None:
            runs.append({"start": start, "end": i - 1, "dense": dense})
            start = None
            dense = 0.0
        prev = key
    if start is not None:
        runs.append({"start": start, "end": len(steps) - 1, "dense": dense})
    return runs


def _assert_criterion3(aud: _Audit, steps, phi_0: float, is_drop_outside: bool) -> None:
    """The criterion-3 numerics — roadmap wording quoted per check."""
    # (iii) Telescoping identity: cumulative shaping must match the
    # closed form gamma*Phi_T - Phi_0 + (gamma-1)*sum(Phi_1..Phi_{T-1})
    # computed from the recorded Phi series — proving the dense term is
    # genuinely potential-based (a non-telescoping dense term fails this).
    phis = [phi_0] + [s["info"]["reward_phi"] for s in steps]
    T = len(steps)
    cumulative_shaping = float(sum(s["info"]["reward_shaping"] for s in steps))
    closed_form = (
        C.REWARD_GAMMA * phis[T]
        - phis[0]
        + (C.REWARD_GAMMA - 1) * sum(phis[1:T])
    )
    aud.check(
        "criterion3_telescoping_identity",
        abs(cumulative_shaping - closed_form) <= 1e-6,
        f"cumulative shaping {cumulative_shaping:.9f} != telescoping closed "
        f"form {closed_form:.9f} (tol 1e-6)",
    )

    # (i) Constant-window dense bound: "dense reward never accumulates
    # significantly without a stage flag advancing" — one-sided upper
    # bound; negative accumulation (time penalty) is fine, positive farming
    # is impossible and this proves it.
    runs = _constant_window_runs(steps, phi_0)
    dense_bound = C.AUDIT_DENSE_FRACTION * C.TERMINAL_PLACE_BONUS
    worst = max(runs, key=lambda r: r["dense"], default=None)
    worst_dense = worst["dense"] if worst is not None else 0.0
    aud.check(
        "criterion3_constant_window_dense_bound",
        worst_dense <= dense_bound,
        f"constant-(stage_index, Phi) window [{worst['start']}, {worst['end']}] "
        f"accumulated dense {worst_dense:.6f} > "
        f"AUDIT_DENSE_FRACTION*TERMINAL_PLACE_BONUS = {dense_bound}"
        if worst is not None
        else "no constant window in the trajectory",
    )

    # (ii) Full-flags-no-place bound (drop-outside episode only): the
    # episode reaches every flag but place and must earn a small fraction
    # of a success — "terminal reward dominant".
    dense_return = float(
        sum(
            s["info"]["reward_shaping"] + s["info"]["reward_time_penalty"]
            for s in steps
        )
    )
    if is_drop_outside:
        fraction_bound = C.FULL_FLAGS_NO_PLACE_MAX_FRACTION * C.TERMINAL_PLACE_BONUS
        aud.check(
            "criterion3_full_flags_no_place_bound",
            dense_return <= fraction_bound,
            f"full-flags-no-place dense return {dense_return:.6f} > "
            f"FULL_FLAGS_NO_PLACE_MAX_FRACTION*TERMINAL_PLACE_BONUS = "
            f"{fraction_bound}",
        )


def _assert_finite(aud: _Audit, steps) -> None:
    """T-02-01: NaN/Inf physics must fail loudly — no non-finite captured
    value anywhere in the per-step series."""
    bad = _first_violation(
        steps,
        lambda s, i: (
            not np.isfinite(s["reward"])
            or not all(np.isfinite(v) for v in s["info"].values())
        ),
    )
    aud.check(
        "all_captured_values_finite",
        bad is None,
        f"step {bad}: non-finite captured value (NaN/Inf state must never "
        "silently become flags or reward)",
    )


def run_trajectory(
    env,
    params: TrajectoryParams,
    rec: _WarningRecorder,
    name: str,
    spawn_index: int,
    build_seq,
    kind: str,
) -> dict:
    """One audit trajectory: reset -> read privileged cube pose -> build
    the recipe -> roll -> assert. Returns the report record."""
    env.reset(seed=0, options={"spawn_index": spawn_index})
    qadr, _ = _cube_addressing(env)
    cube_spawn = np.array(env.data.qpos[qadr : qadr + 3])
    seq = build_seq(cube_spawn)
    labels = [label for _, label in seq]

    aud = _Audit(name, spawn_index)
    steps, phi_0 = _roll(env, seq)

    if kind == "positive":
        _assert_positive(aud, steps, labels, params)
    elif kind == "momentary":
        _assert_negative_momentary(aud, steps)
    elif kind == "approach_retreat":
        _assert_negative_approach_retreat(aud, steps)
    elif kind == "drop_outside":
        _assert_negative_drop_outside(aud, steps)
    else:
        raise ValueError(f"unknown trajectory kind: {kind!r}")

    _assert_finite(aud, steps)
    _assert_criterion3(aud, steps, phi_0, is_drop_outside=(kind == "drop_outside"))

    final = steps[-1]["info"]
    runs = _constant_window_runs(steps, phi_0)
    worst = max(runs, key=lambda r: r["dense"], default=None)
    dense_return = float(
        sum(
            s["info"]["reward_shaping"] + s["info"]["reward_time_penalty"]
            for s in steps
        )
    )
    return {
        "name": name,
        "kind": kind,
        "spawn_index": spawn_index,
        "n_steps": len(steps),
        "first_flip_indices": _first_flip_indices(steps),
        "final_flags": {k: bool(final[k]) for k in _STAGE_FLAG_KEYS},
        "is_success": bool(final["is_success"]),
        "terminal_paid_count": sum(
            1
            for s in steps
            if s["info"]["reward_terminal"] == C.TERMINAL_PLACE_BONUS
        ),
        "dense_return": dense_return,
        "phi_0": phi_0,
        "constant_window_count": len(runs),
        "constant_window_max_dense": {
            "start": worst["start"],
            "end": worst["end"],
            "dense": worst["dense"],
        }
        if worst is not None
        else None,
        "mujoco_warnings": rec.drain(),
        "steps": steps,
        "assertions": aud.assertions,
        "pass": aud.passed,
    }


def main() -> int:
    import dataclasses

    params = TrajectoryParams()  # the tuned defaults the gate proved 25/25
    rec = _WarningRecorder()

    # ONE env reused across ALL rollouts (no render_mode — metric
    # collection, timing discipline): the sweep doubles as the Pitfall 6
    # reset-hygiene regression.
    env = SO100PickPlaceEnv(frame_skip=C.DEFAULT_FRAME_SKIP, spawn_mode="grid")
    trajectories: list[dict] = []
    try:
        t0 = time.perf_counter()
        for i in range(C.SPAWN_GRID_SIZE):
            trajectories.append(
                run_trajectory(
                    env, params, rec, f"positive_spawn{i}", i,
                    lambda cube, i=i: positive_sequence(
                        env.model, env.data, cube, C.GOAL_POS, params
                    ),
                    "positive",
                )
            )
        negatives = (
            (
                "momentary_contact",
                lambda cube: momentary_contact_sequence(
                    env.model, env.data, cube, C.GOAL_POS, params
                ),
                "momentary",
            ),
            (
                "approach_retreat",
                lambda cube: approach_retreat_sequence(
                    env.model, env.data, cube, C.GOAL_POS, params
                ),
                "approach_retreat",
            ),
            (
                "drop_outside_goal",
                lambda cube: drop_outside_goal_sequence(
                    env.model, env.data, cube, C.GOAL_POS, params
                ),
                "drop_outside",
            ),
        )
        for name, build, kind in negatives:
            trajectories.append(
                run_trajectory(env, params, rec, name, 0, build, kind)
            )
        wall_seconds = time.perf_counter() - t0
    finally:
        env.close()

    leftover_warnings = rec.drain()
    passed_count = sum(1 for t in trajectories if t["pass"])
    total = len(trajectories)

    report = {
        "audit": "reward",
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "summary": {
            "passed": passed_count,
            "total": total,
            "frame_skip": C.DEFAULT_FRAME_SKIP,
            "control_rate_hz": 1.0 / (C.PHYSICS_TIMESTEP * C.DEFAULT_FRAME_SKIP),
            "wall_seconds": round(wall_seconds, 3),
            "mujoco_warnings": len(leftover_warnings)
            + sum(len(t["mujoco_warnings"]) for t in trajectories),
            "constants": {
                "GRASP_WINDOW_STEPS": C.GRASP_WINDOW_STEPS,
                "APPROACH_DIST_M": C.APPROACH_DIST_M,
                "LIFT_MARGIN_M": C.LIFT_MARGIN_M,
                "GRASP_TRACK_TOL_M": C.GRASP_TRACK_TOL_M,
                "GRASP_JAW_CLOSED_CTRL_MAX": C.GRASP_JAW_CLOSED_CTRL_MAX,
                "GRASP_CUBE_OFF_FLOOR_Z_M": C.GRASP_CUBE_OFF_FLOOR_Z_M,
                "STAGE_POTENTIALS": list(C.STAGE_POTENTIALS),
                "APPROACH_W": C.APPROACH_W,
                "APPROACH_POTENTIAL_SCALE_M": C.APPROACH_POTENTIAL_SCALE_M,
                "TERMINAL_PLACE_BONUS": C.TERMINAL_PLACE_BONUS,
                "TIME_PENALTY": C.TIME_PENALTY,
                "REWARD_GAMMA": C.REWARD_GAMMA,
                "FULL_FLAGS_NO_PLACE_MAX_FRACTION": C.FULL_FLAGS_NO_PLACE_MAX_FRACTION,
                "AUDIT_DENSE_FRACTION": C.AUDIT_DENSE_FRACTION,
                "SUCCESS_TOLERANCE_XY": C.SUCCESS_TOLERANCE_XY,
                "DEFAULT_FRAME_SKIP": C.DEFAULT_FRAME_SKIP,
            },
        },
        "params": dataclasses.asdict(params),
        "versions": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "gymnasium": gymnasium.__version__,
            "numpy": np.__version__,
        },
        "trajectories": trajectories,
    }

    out_dir = RUNS_ROOT / _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"report: {out_dir / 'report.json'}")

    for t in trajectories:
        line = "PASS" if t["pass"] else "FAIL"
        extra = (
            f" flips={t['first_flip_indices']}"
            if t["kind"] == "positive"
            else f" final_flags={t['final_flags']}"
        )
        print(f"  {line}  {t['name']} (spawn {t['spawn_index']}){extra}")

    if passed_count == total:
        print(f"AUDIT PASSED: {passed_count}/{total} trajectories, all assertions green")
        return 0
    failing = [t for t in trajectories if not t["pass"]]
    print(f"AUDIT FAILED: {len(failing)} failing")
    for t in failing:
        failed_names = [a["name"] for a in t["assertions"] if not a["passed"]]
        print(f"  {t['name']}: failed checks: {failed_names}")
    return 1


if __name__ == "__main__":
    sys.exit(main())