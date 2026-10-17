"""Phase 2 integration suite: hand-posed trajectories prove the stage
machine's env wiring on the REAL physics (ENV-05, D-04/D-08).

The happy path is the stock gate_params scripted trajectory plus the
mandatory POST_RELEASE_SETTLE_STEPS settle tail — the exact sequence the
Phase 1 physics gate proved places the cube 25/25. This suite re-rolls it
on spawn 0 capturing the full per-step info/terminated/truncated contract,
then adds the 200-step timeout rollout proving truncation-vs-termination
discipline (Pitfall 2: timeout is truncation, NEVER termination).
"""

import mujoco
import numpy as np
import pytest

from so100rob import constants as C
from so100rob.scripted_policy import (
    POST_RELEASE_SETTLE_STEPS,
    approach_retreat_sequence,
    build_trajectory,
    drop_outside_goal_sequence,
    momentary_contact_sequence,
    positive_sequence,
)

_STAGE_FLAG_KEYS = ("stage_approach", "stage_grasp", "stage_lift", "stage_place")


def _cube_addressing(env) -> tuple[int, int]:
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, C.CUBE_JOINT)
    assert jid >= 0, f"cube joint {C.CUBE_JOINT!r} missing from compiled model"
    return int(env.model.jnt_qposadr[jid]), int(env.model.jnt_dofadr[jid])


def _roll_happy_path(env, params):
    """Reset spawn 0, roll the stock trajectory + settle tail, capture every
    step's (reward, terminated, truncated, info).

    Returns (steps, phi_0): steps is the per-step capture list; phi_0 is the
    potential seeded at reset (present once the reward machine lands —
    Task 2 — None until then).
    """
    env.reset(seed=0, options={"spawn_index": 0})
    # Capture the reset-seeded prev-Phi BEFORE rolling — after the rollout
    # env._prev_phi holds the last step's Phi, not the seed.
    phi_0 = getattr(env, "_prev_phi", None)
    qadr, _ = _cube_addressing(env)
    cube_spawn = np.array(env.data.qpos[qadr : qadr + 3])
    seq = build_trajectory(env.model, env.data, cube_spawn, C.GOAL_POS, params)
    assert len(seq) > 0

    steps = []

    def _capture(action):
        _, reward, terminated, truncated, info = env.step(action)
        steps.append((float(reward), bool(terminated), bool(truncated), dict(info)))

    for action, _label in seq:
        _capture(action)
    # Post-release settle tail (Pitfall 5): the place flag's resting clause
    # is only evaluable after the cube settles — hold the final arm config.
    # The stock trajectory is 120 control steps (5 waypoint dwells x 20 +
    # close 10 + release 10), so the hold runs to the episode CAP (80 steps,
    # a superset of the mandatory 40-step settle) — the rollout is then
    # exactly max_episode_steps and the truncated-never assertion proves
    # terminated preempts truncation AT the cap (Rule 1 fix: the plan's
    # "160 trajectory" arithmetic was off; the 200-step-rollout intent is
    # preserved).
    last_action = seq[-1][0]
    tail_steps = env.max_episode_steps - len(seq)
    assert tail_steps >= POST_RELEASE_SETTLE_STEPS, (
        "the settle tail must fit inside the episode cap"
    )
    for _ in range(tail_steps):
        _capture(last_action)

    return steps, phi_0


def test_happy_path_flips_all_flags_strictly_in_order(gate_env, gate_params):
    """D-04 criterion 1: the gate-params trajectory flips approach, grasp,
    lift, place strictly in order; is_success is False on every step before
    the place flip and True from it on (Pitfall 7 — every step, not
    final-only); terminated follows place; truncated never fires; every
    info value is finite."""
    # Tamper guards (T-01-05 pattern): pinned constants, never literals.
    assert C.GRASP_WINDOW_STEPS == 2
    assert C.SUCCESS_TOLERANCE_XY == 0.02

    steps, phi_0 = _roll_happy_path(gate_env, gate_params)
    assert len(steps) == gate_env.max_episode_steps == 200, (
        "160 trajectory steps + 40 settle = exactly the 200-step episode cap"
    )

    first_flip = {}
    for i, (_r, _term, _trunc, info) in enumerate(steps):
        for key in _STAGE_FLAG_KEYS:
            if info.get(key) and key not in first_flip:
                first_flip[key] = i
        # stage_index equals the count of True flags at every step
        assert info["stage_index"] == sum(1 for k in _STAGE_FLAG_KEYS if info.get(k)), (
            f"step {i}: stage_index {info.get('stage_index')} != count of True flags"
        )
        # every info value is finite (threat T-02-01: NaN state can never
        # silently become flags)
        for k, v in info.items():
            assert np.isfinite(v), f"step {i}: info[{k!r}] = {v!r} is not finite"

    assert set(first_flip) == set(_STAGE_FLAG_KEYS), (
        f"all four flags must flip on the happy path; missing: "
        f"{set(_STAGE_FLAG_KEYS) - set(first_flip)}"
    )
    flips = [first_flip[k] for k in _STAGE_FLAG_KEYS]
    assert flips == sorted(flips) and len(set(flips)) == 4, (
        f"first-flip indices must strictly increase approach < grasp < lift < "
        f"place, got {dict(first_flip)}"
    )

    place_i = first_flip["stage_place"]
    for i, (_r, terminated, truncated, info) in enumerate(steps):
        assert info["is_success"] == (i >= place_i), (
            f"step {i}: is_success {info.get('is_success')} must equal "
            f"(step >= place flip {place_i}) — monotone, every step (Pitfall 7)"
        )
        assert terminated == (i >= place_i), (
            f"step {i}: terminated must be True on and after the place flip"
        )
        assert not truncated, (
            f"step {i}: truncated must never fire on the 200-step success "
            f"rollout (place flips within the settle tail)"
        )

    # --- Reward contract (Task 2, D-05/D-06/D-07) ---
    # phi_0 is the potential seeded at reset (env._prev_phi); the recorded
    # series is phi_1..phi_T.
    assert phi_0 is not None and np.isfinite(phi_0), (
        "reset must seed _prev_phi so the first step's shaping is a true "
        "potential difference, not a jump bonus"
    )
    phis = [phi_0] + [info["reward_phi"] for (_r, _t, _tr, info) in steps]
    cumulative_shaping = 0.0
    terminal_steps = []
    for i, (step_reward, _term, _trunc, info) in enumerate(steps):
        for key in ("reward_shaping", "reward_time_penalty", "reward_terminal", "reward_phi"):
            assert key in info, f"step {i}: info missing reward key {key!r}"
            assert np.isfinite(info[key]), f"step {i}: info[{key!r}] not finite"
        assert np.isfinite(step_reward), f"step {i}: returned reward not finite"
        # exact per-step shaping arithmetic against the recorded Phi series
        assert info["reward_shaping"] == pytest.approx(
            C.REWARD_GAMMA * info["reward_phi"] - phis[i], abs=1e-9
        ), f"step {i}: reward_shaping must be gamma*phi_next - phi_prev (D-05)"
        assert info["reward_time_penalty"] == -C.TIME_PENALTY
        cumulative_shaping += info["reward_shaping"]
        if info["reward_terminal"] == C.TERMINAL_PLACE_BONUS:
            terminal_steps.append(i)
        else:
            assert info["reward_terminal"] == 0.0, (
                f"step {i}: reward_terminal must be 0 or exactly the bonus"
            )
    # the bonus lands EXACTLY once — on the terminated (place-flip) step
    assert terminal_steps == [place_i], (
        f"terminal bonus must be paid exactly once at the place step {place_i}, "
        f"paid at {terminal_steps}"
    )
    # telescoping closed form computed from the recorded Phi series (proof
    # the composed reward is genuinely potential-based — criterion 3)
    T = len(steps)
    closed_form = (
        C.REWARD_GAMMA * phis[T]
        - phis[0]
        + (C.REWARD_GAMMA - 1) * sum(phis[1:T])
    )
    assert cumulative_shaping == pytest.approx(closed_form, abs=1e-6), (
        f"cumulative shaping {cumulative_shaping} != telescoping closed form "
        f"{closed_form} — the dense term must telescope"
    )


def test_timeout_is_truncation_never_termination(gate_env):
    """Pitfall 2: a 200-step home-pose hold (cube untouched at spawn 0, far
    outside SUCCESS_TOLERANCE_XY) sees terminated False throughout and
    truncated True EXACTLY at the step cap."""
    env = gate_env
    assert env.max_episode_steps == 200
    env.reset(seed=0, options={"spawn_index": 0})

    home_action = np.asarray(C.HOME_CTRL, dtype=float)
    for i in range(env.max_episode_steps):
        _, _reward, terminated, truncated, info = env.step(home_action)
        assert not terminated, f"step {i + 1}: timeout rollout must NEVER terminate"
        assert not info["is_success"], "untouched cube must never read success"
        if i < env.max_episode_steps - 1:
            assert not truncated, f"step {i + 1}: truncated only at the cap"
        else:
            assert truncated, "truncated must be True at the step cap (timeout)"
    assert not terminated, "even at the cap, timeout is truncation, never termination"


def _episode_summary(steps):
    """First-flip index per stage flag + the terminal-bonus payment steps."""
    first_flip = {}
    terminal_steps = []
    for i, (_r, _term, _trunc, info) in enumerate(steps):
        for key in _STAGE_FLAG_KEYS:
            if info.get(key) and key not in first_flip:
                first_flip[key] = i
        if info.get("reward_terminal") == C.TERMINAL_PLACE_BONUS:
            terminal_steps.append(i)
    return first_flip, terminal_steps


def test_consecutive_episodes_clean_start(gate_env, gate_params):
    """Pitfall 6 — the exact leak scenario: after a SUCCESSFUL episode, the
    next episode on the SAME env instance must start clean and pay the
    terminal bonus exactly once again. A leaked flag, grasp-window counter,
    step counter, terminal latch, or prev-Phi snapshot fails here — the
    audit runner (02-02) reuses one env across trajectories exactly like
    the physics gate reuses it across spawns, so reset hygiene is
    load-bearing.

    Leak fingerprints this test catches: a leaked _stage would show
    approach already True on the probe steps; a stale _terminal_paid would
    show zero or two payments in run 2; a stale grasp_hold_counter could
    flip grasp early; a stale _prev_phi would distort run 2's first-step
    shaping.
    """
    env = gate_env

    # --- Episode 1: the full happy path, must succeed ---
    steps1, phi0_1 = _roll_happy_path(env, gate_params)
    flips1, terminal1 = _episode_summary(steps1)
    assert set(flips1) == set(_STAGE_FLAG_KEYS), f"episode 1 missing flags: {flips1}"
    idx1 = [flips1[k] for k in _STAGE_FLAG_KEYS]
    assert idx1 == sorted(idx1) and len(set(idx1)) == 4, (
        f"episode 1 flips out of order: {flips1}"
    )
    assert terminal1 == [flips1["stage_place"]], (
        f"episode 1 must pay the terminal bonus exactly once at place: {terminal1}"
    )
    assert steps1[-1][3]["is_success"], "episode 1 must end successful"

    # --- Reset the SAME env instance (same spawn, same seed) and probe ---
    # Measured baseline (Rule 1 deviation from the plan's "all four flags
    # False"): at the home pose the pad midpoint sits ~2.87 cm horizontally
    # from the spawn-0 cube center — BELOW the 5 cm approach threshold — so
    # stage_approach legitimately reads True at episode start. That is
    # physics, not state. The LEAK fingerprint is the other three flags and
    # is_success: episode 1 ended with ALL FOUR flags True, so any leaked
    # _stage or grasp-window latch shows grasp/lift/place True here, and a
    # leaked place flag shows is_success True.
    env.reset(seed=0, options={"spawn_index": 0})
    home_action = np.asarray(C.HOME_CTRL, dtype=float)
    for i in range(5):
        _, _reward, terminated, truncated, info = env.step(home_action)
        for key in ("stage_grasp", "stage_lift", "stage_place"):
            assert not info[key], (
                f"probe step {i}: {key} leaked across the reset — episode 1 "
                "ended with every flag True, so any True here is stale state"
            )
        assert not info["is_success"], "a leaked place flag would read success"
        assert info["stage_index"] == sum(
            1 for k in _STAGE_FLAG_KEYS if info[k]
        ), f"probe step {i}: stage_index must equal the count of True flags"
        assert info["stage_index"] < 4, (
            f"probe step {i}: stage_index {info['stage_index']} — the "
            "end-of-episode-1 value 4 leaking means _stage survived the reset"
        )
        assert not terminated and not truncated

    # --- Episode 2: the full happy path again on the same env ---
    steps2, phi0_2 = _roll_happy_path(env, gate_params)
    flips2, terminal2 = _episode_summary(steps2)
    assert set(flips2) == set(_STAGE_FLAG_KEYS), f"episode 2 missing flags: {flips2}"
    idx2 = [flips2[k] for k in _STAGE_FLAG_KEYS]
    assert idx2 == sorted(idx2) and len(set(idx2)) == 4, (
        f"episode 2 flips out of order: {flips2}"
    )
    assert terminal2 == [flips2["stage_place"]], (
        f"episode 2 must pay the terminal bonus EXACTLY once again at place — "
        f"a stale _terminal_paid latch shows zero, a leaked place flag shows "
        f"two: {terminal2}"
    )
    assert steps2[-1][3]["is_success"], "episode 2 must end successful"
    # Identical reset state (same seed, same spawn) must re-seed the IDENTICAL
    # prev-Phi — a stale _prev_phi from episode 1's terminal step (Phi = 1.0)
    # would differ here and distort episode 2's first-step shaping.
    assert phi0_2 == phi0_1, (
        f"episode 2's reset-seeded prev-Phi {phi0_2} != episode 1's {phi0_1} — "
        "the snapshot leaked across the reset"
    )


# ---------------------------------------------------------------------------
# Negative-control trajectories (D-08/D-09, Plan 02-02 Task 1)
# ---------------------------------------------------------------------------


def _roll_sequence(env, seq):
    """Step a labeled recipe sequence on ``env``, capturing every step's
    (reward, terminated, truncated, info) — the same capture contract the
    happy-path tests use."""
    steps = []
    for action, _label in seq:
        _, reward, terminated, truncated, info = env.step(action)
        steps.append((float(reward), bool(terminated), bool(truncated), dict(info)))
    return steps


def _cube_spawn(env) -> np.ndarray:
    """Reset spawn 0 and return the privileged cube spawn pose (D-05 read)."""
    env.reset(seed=0, options={"spawn_index": 0})
    qadr, _ = _cube_addressing(env)
    return np.array(env.data.qpos[qadr : qadr + 3])


def test_momentary_contact_never_sets_grasp(gate_env, gate_params):
    """D-09a: 1 close step — below GRASP_WINDOW_STEPS=2 — then a pre-grasp
    retreat hold. stage_grasp, stage_lift, stage_place and is_success must
    be False at EVERY step and in the final info (stage_approach may
    legitimately be True — the home-pose pad-to-cube distance is below the
    approach threshold, so approach-at-episode-start is physics, not
    progress)."""
    assert C.GRASP_WINDOW_STEPS == 2, "tamper guard: the pinned window (D-02, 04-06 unblock)"
    env = gate_env
    cube_spawn = _cube_spawn(env)
    seq = momentary_contact_sequence(
        env.model, env.data, cube_spawn, C.GOAL_POS, gate_params,
        close_steps=1,
    )
    # Structural: exactly 1 close-labeled step — below the window.
    close_count = sum(1 for _, label in seq if label == "close")
    assert close_count == 1 < C.GRASP_WINDOW_STEPS, (
        f"momentary contact must close for fewer than GRASP_WINDOW_STEPS "
        f"steps, got {close_count}"
    )

    steps = _roll_sequence(env, seq)
    assert len(steps) > 0
    for i, (_r, _term, _trunc, info) in enumerate(steps):
        for key in ("stage_grasp", "stage_lift", "stage_place"):
            assert not info[key], (
                f"step {i}: {key} must NEVER set on a momentary contact — the "
                f"multi-step verification window is the only path to the grasp "
                f"flag (D-09a)"
            )
        assert not info["is_success"], f"step {i}: is_success must stay False"
    final = steps[-1][3]
    for key in ("stage_grasp", "stage_lift", "stage_place"):
        assert not final[key], f"final info: {key} must be False (D-09a)"


def test_approach_retreat_sets_only_approach(gate_env, gate_params):
    """D-09b (Pitfall 4): pre_grasp + descend (jaw always open), then
    retreat. stage_approach must be True at episode end while stage_grasp,
    stage_lift, stage_place and is_success stay False at EVERY step —
    monotone flags never un-set, so 'only approach sets' is asserted on the
    OTHER flags staying False."""
    env = gate_env
    cube_spawn = _cube_spawn(env)
    seq = approach_retreat_sequence(
        env.model, env.data, cube_spawn, C.GOAL_POS, gate_params
    )
    # Structural: no close-labeled step — the jaw never commands closed.
    assert not any(label == "close" for _, label in seq), (
        "approach-retreat must never enter the close phase"
    )

    steps = _roll_sequence(env, seq)
    for i, (_r, _term, _trunc, info) in enumerate(steps):
        for key in ("stage_grasp", "stage_lift", "stage_place"):
            assert not info[key], (
                f"step {i}: {key} must stay False on an approach-retreat (D-09b)"
            )
        assert not info["is_success"], f"step {i}: is_success must stay False"
    final = steps[-1][3]
    assert final["stage_approach"], (
        "the arm ends the retreat parked at the pre-grasp hover above the cube — "
        "approach must be True at episode end (D-09b)"
    )


def test_drop_outside_goal_no_place(gate_env, gate_params):
    """D-09c: full grasp-lift-place toward GOAL_POS shifted +0.15 m in x,
    plus the settle tail. stage_approach/stage_grasp/stage_lift must be
    True at the end while stage_place and is_success stay False at EVERY
    step and terminated is NEVER True — the place predicate is
    goal-anchored, not rest-anywhere."""
    assert C.SUCCESS_TOLERANCE_XY == 0.02, "tamper guard: the pinned D-04 tolerance"
    env = gate_env
    cube_spawn = _cube_spawn(env)
    seq = drop_outside_goal_sequence(
        env.model, env.data, cube_spawn, C.GOAL_POS, gate_params
    )
    # Structural: the settle tail is included (Pitfall 5 — the place
    # predicate's resting clause is only evaluable after the cube settles).
    assert sum(1 for _, label in seq if label == "settle") >= POST_RELEASE_SETTLE_STEPS

    steps = _roll_sequence(env, seq)
    for i, (_r, terminated, _trunc, info) in enumerate(steps):
        assert not info["stage_place"], (
            f"step {i}: place must NEVER set when the cube rests 0.15 m from "
            f"the goal — the predicate is goal-anchored (D-09c)"
        )
        assert not info["is_success"], f"step {i}: is_success must stay False"
        assert not terminated, f"step {i}: termination follows place only"
    final = steps[-1][3]
    assert final["stage_approach"] and final["stage_grasp"] and final["stage_lift"], (
        "the drop-outside episode grasps and lifts the cube — approach/grasp/"
        f"lift must all be True at the end, got {final}"
    )


def test_positive_sequence_matches_gate_protocol(gate_env, gate_params):
    """D-10: positive_sequence is EXACTLY the gate-proven build_trajectory
    protocol plus the mandatory settle tail — build's length plus
    POST_RELEASE_SETTLE_STEPS steps, all labeled settle, each holding the
    final build action. One definition consumed by tests and the audit
    runner alike."""
    env = gate_env
    cube_spawn = _cube_spawn(env)
    base = build_trajectory(env.model, env.data, cube_spawn, C.GOAL_POS, gate_params)
    seq = positive_sequence(env.model, env.data, cube_spawn, C.GOAL_POS, gate_params)

    assert len(seq) == len(base) + POST_RELEASE_SETTLE_STEPS, (
        f"positive_sequence must be build_trajectory ({len(base)} steps) + "
        f"{POST_RELEASE_SETTLE_STEPS} settle steps, got {len(seq)}"
    )
    # The head is bit-identical to the gate trajectory (deterministic IK warm
    # start from the home keyframe).
    for (a_base, lab_base), (a_seq, lab_seq) in zip(base, seq[: len(base)]):
        assert lab_base == lab_seq
        assert np.array_equal(a_base, a_seq), (
            f"positive_sequence head must equal build_trajectory output at label {lab_base!r}"
        )
    tail = seq[len(base) :]
    last_action = base[-1][0]
    for action, label in tail:
        assert label == "settle", f"tail steps must be labeled settle, got {label!r}"
        assert np.array_equal(action, last_action), (
            "every settle step must hold the final build action"
        )


def test_momentary_contact_sequence_self_validates(gate_env, gate_params):
    """T-02-07: the momentary-contact recipe REFUSES a contradictory
    request — close_steps must be at least 1 and strictly below
    GRASP_WINDOW_STEPS, else the 'negative control' would satisfy the very
    window it exists to probe and silently test nothing."""
    env = gate_env
    cube_spawn = _cube_spawn(env)
    with pytest.raises(ValueError, match="GRASP_WINDOW_STEPS"):
        momentary_contact_sequence(
            env.model, env.data, cube_spawn, C.GOAL_POS, gate_params,
            close_steps=C.GRASP_WINDOW_STEPS,
        )
    with pytest.raises(ValueError, match="close_steps"):
        momentary_contact_sequence(
            env.model, env.data, cube_spawn, C.GOAL_POS, gate_params,
            close_steps=0,
        )


def test_drop_outside_goal_sequence_self_validates(gate_env, gate_params):
    """T-02-07: the drop-outside recipe REFUSES a goal offset that leaves
    the drop point within SUCCESS_TOLERANCE_XY of the real goal — a
    degenerate 'outside' goal would silently test nothing."""
    env = gate_env
    cube_spawn = _cube_spawn(env)
    # A strictly-inside offset (half the tolerance), NOT the exact boundary:
    # in doubles, 0.10 + 0.02 == 0.12000000000000001 lands a hair OUTSIDE the
    # tolerance, so the exact-boundary case is float-ambiguous and the recipe
    # correctly declines to raise there (Rule 1 fix: knife-edge test input).
    for bad_offset in (0.0, C.SUCCESS_TOLERANCE_XY / 2):
        with pytest.raises(ValueError, match="SUCCESS_TOLERANCE_XY"):
            drop_outside_goal_sequence(
                env.model, env.data, cube_spawn, C.GOAL_POS, gate_params,
                goal_offset_m=bad_offset,
            )