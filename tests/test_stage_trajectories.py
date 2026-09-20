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
from so100rob.scripted_policy import POST_RELEASE_SETTLE_STEPS, build_trajectory

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

    phi_0 = getattr(env, "_prev_phi", None)
    return steps, phi_0


def test_happy_path_flips_all_flags_strictly_in_order(gate_env, gate_params):
    """D-04 criterion 1: the gate-params trajectory flips approach, grasp,
    lift, place strictly in order; is_success is False on every step before
    the place flip and True from it on (Pitfall 7 — every step, not
    final-only); terminated follows place; truncated never fires; every
    info value is finite."""
    # Tamper guards (T-01-05 pattern): pinned constants, never literals.
    assert C.GRASP_WINDOW_STEPS == 5
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