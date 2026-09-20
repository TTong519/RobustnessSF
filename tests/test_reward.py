"""Unit suite for the potential-based stage-gated reward (ENV-05 criterion 3,
D-05/D-06/D-07).

Fully env-free: reward.compute is floats-in/floats+dict-out. The criterion-3
assertions are NUMERIC — hold-at-Phi pays strictly negative (identical
repeated states can never farm), the telescoping closed form proves the
implementation is genuinely potential-based, the terminal bonus is one-shot
and 10x dominant (D-06), and every pinned invariant fails loudly when
monkeypatched invalid (threat T-02-02).
"""

import pytest

from so100rob import constants as C
from so100rob import reward


# ---------------------------------------------------------------------------
# Potential ladder (D-05, OQ-3 hybrid)
# ---------------------------------------------------------------------------


def test_potential_ladder_hybrid_bounds():
    """Stage-index 2 and above: exact STAGE_POTENTIALS[k], no sub-term.
    Stage-index 0/1: base plus the approach sub-term bounded in
    [0, APPROACH_W]; the monotone guard potentials[1]+APPROACH_W <=
    potentials[2] holds for every d (Phi can never buy grasp by hovering)."""
    distances = (0.0, 0.01, 0.05, 0.10, 0.19, 0.20, 0.21, 0.50, 2.0)
    for k in (2, 3, 4):
        for d in distances:
            assert reward.potential(k, d) == C.STAGE_POTENTIALS[k], (
                f"potential({k}, {d}) must be the exact base potential"
            )
    for k in (0, 1):
        for d in distances:
            value = reward.potential(k, d)
            clamped = max(0.0, 1.0 - min(max(d / C.APPROACH_POTENTIAL_SCALE_M, 0.0), 1.0))
            expected = C.STAGE_POTENTIALS[k] + C.APPROACH_W * clamped
            assert value == pytest.approx(expected, abs=1e-12), (
                f"potential({k}, {d}): hybrid sub-term mismatch"
            )
            assert C.STAGE_POTENTIALS[k] <= value <= C.STAGE_POTENTIALS[k] + C.APPROACH_W

    # The monotone guard: the best un-grasped Phi stays below the grasp rung
    # for EVERY d, so no approach-hover can out-earn an actual grasp.
    assert C.STAGE_POTENTIALS[1] + C.APPROACH_W <= C.STAGE_POTENTIALS[2]
    for d in distances:
        assert reward.potential(1, d) <= C.STAGE_POTENTIALS[2] + 1e-12


# ---------------------------------------------------------------------------
# compute() exactness (D-05/D-07)
# ---------------------------------------------------------------------------


def test_compute_exact_arithmetic_and_component_keys():
    """compute returns total = gamma*phi_next - phi_prev - TIME_PENALTY (no
    terminal) and a components dict with EXACTLY the three D-07 keys."""
    phi_prev, phi_next = 0.15, 0.35
    total, components = reward.compute(phi_prev, phi_next, False)
    assert set(components) == {"reward_shaping", "reward_time_penalty", "reward_terminal"}
    assert components["reward_shaping"] == pytest.approx(
        C.REWARD_GAMMA * phi_next - phi_prev, abs=1e-12
    )
    assert components["reward_time_penalty"] == -C.TIME_PENALTY
    assert components["reward_terminal"] == 0.0
    assert total == pytest.approx(
        C.REWARD_GAMMA * phi_next - phi_prev - C.TIME_PENALTY, abs=1e-12
    )


def test_hold_at_phi_pays_strictly_negative():
    """Criterion 3 core: compute(phi, phi, False) is strictly negative for
    phi > 0 — holding an identical state PAYS, it never farms (the
    (gamma-1)*phi shaping is non-positive and the time penalty is negative)
    — the so101-nexus hover exploit is impossible by construction."""
    for phi in (0.01, 0.05, 0.15, 0.35, 0.60, 1.0):
        total, components = reward.compute(phi, phi, False)
        assert total < 0, f"hold-at-Phi({phi}) total {total} must be strictly negative"
        assert components["reward_shaping"] <= 0.0


def test_telescoping_closed_form():
    """Proof of potential-basedness: summing compute over a synthetic Phi
    series, cumulative reward_shaping equals
    gamma*Phi_T - Phi_0 + (gamma-1)*sum(Phi_k, k in 1..T-1) within 1e-9 —
    a non-telescoping dense term fails this identity."""
    phis = [0.0, 0.02, 0.06, 0.15, 0.20, 0.35, 0.50, 0.60, 0.90, 1.0, 1.0, 1.0]
    T = len(phis) - 1
    cumulative = 0.0
    for t in range(1, len(phis)):
        _, components = reward.compute(phis[t - 1], phis[t], False)
        cumulative += components["reward_shaping"]
    closed = C.REWARD_GAMMA * phis[T] - phis[0] + (C.REWARD_GAMMA - 1) * sum(phis[1:T])
    assert cumulative == pytest.approx(closed, abs=1e-9), (
        f"cumulative shaping {cumulative} != telescoping closed form {closed}"
    )


def test_terminal_bonus_is_one_shot():
    """Terminal latch contract: compute(..., first_place=True) carries
    reward_terminal == TERMINAL_PLACE_BONUS; a subsequent
    compute(..., first_place=False) at the same Phi carries 0 — the bonus is
    paid exactly once (the env's latch supplies first_place)."""
    total_hit, hit = reward.compute(0.60, 1.0, True)
    assert hit["reward_terminal"] == C.TERMINAL_PLACE_BONUS
    assert total_hit == pytest.approx(
        C.REWARD_GAMMA * 1.0 - 0.60 - C.TIME_PENALTY + C.TERMINAL_PLACE_BONUS, abs=1e-12
    )

    total_after, after = reward.compute(1.0, 1.0, False)
    assert after["reward_terminal"] == 0.0, "the terminal bonus must be one-shot"
    assert total_after == pytest.approx(
        C.REWARD_GAMMA * 1.0 - 1.0 - C.TIME_PENALTY, abs=1e-12
    )


def test_terminal_dominance_bounds():
    """D-06: TERMINAL_PLACE_BONUS >= 10x max STAGE_POTENTIALS, and a
    synthetic full-flags-no-place episode (Phi ceiling 0.6, ~200 steps)
    earns dense return <= FULL_FLAGS_NO_PLACE_MAX_FRACTION x terminal —
    dense reward can never accumulate significantly without place."""
    assert C.TERMINAL_PLACE_BONUS >= 10 * max(C.STAGE_POTENTIALS)

    ceiling = C.STAGE_POTENTIALS[3]  # lift, no place; hybrid inactive there
    dense_total = 0.0
    _, first = reward.compute(0.0, ceiling, False)
    dense_total += first["reward_shaping"] + first["reward_time_penalty"]
    for _ in range(199):
        _, hold = reward.compute(ceiling, ceiling, False)
        dense_total += hold["reward_shaping"] + hold["reward_time_penalty"]
    assert dense_total <= C.FULL_FLAGS_NO_PLACE_MAX_FRACTION * C.TERMINAL_PLACE_BONUS, (
        f"full-flags-no-place dense return {dense_total} exceeds the pinned "
        f"D-06 fraction of the terminal bonus"
    )


# ---------------------------------------------------------------------------
# Fail-loudly validation (threat T-02-02)
# ---------------------------------------------------------------------------


def test_validation_rejects_invalid_constants(monkeypatch):
    """Every pinned invariant monkeypatched invalid and re-run raises
    ValueError NAMING the violated constant."""
    # non-monotone potentials
    monkeypatch.setattr(C, "STAGE_POTENTIALS", (0.0, 0.5, 0.3, 0.6, 1.0))
    with pytest.raises(ValueError, match="STAGE_POTENTIALS"):
        reward._validate_constants()
    monkeypatch.undo()

    # potentials not rooted at 0.0
    monkeypatch.setattr(C, "STAGE_POTENTIALS", (0.1, 0.15, 0.35, 0.60, 1.0))
    with pytest.raises(ValueError, match="STAGE_POTENTIALS"):
        reward._validate_constants()
    monkeypatch.undo()

    # hybrid guard: potentials[1] + APPROACH_W > potentials[2]
    monkeypatch.setattr(C, "APPROACH_W", 0.5)
    with pytest.raises(ValueError, match="APPROACH_W"):
        reward._validate_constants()
    monkeypatch.undo()

    # terminal below 10x max potential
    monkeypatch.setattr(C, "TERMINAL_PLACE_BONUS", 5.0)
    with pytest.raises(ValueError, match="TERMINAL_PLACE_BONUS"):
        reward._validate_constants()
    monkeypatch.undo()

    # gamma outside (0, 1)
    for bad_gamma in (0.0, 1.0, 1.5):
        monkeypatch.setattr(C, "REWARD_GAMMA", bad_gamma)
        with pytest.raises(ValueError, match="REWARD_GAMMA"):
            reward._validate_constants()
        monkeypatch.undo()

    # non-positive time penalty
    for bad_penalty in (0.0, -0.01):
        monkeypatch.setattr(C, "TIME_PENALTY", bad_penalty)
        with pytest.raises(ValueError, match="TIME_PENALTY"):
            reward._validate_constants()
        monkeypatch.undo()

    # audit fractions outside (0, 1)
    for knob in ("FULL_FLAGS_NO_PLACE_MAX_FRACTION", "AUDIT_DENSE_FRACTION"):
        for bad in (0.0, 1.0):
            monkeypatch.setattr(C, knob, bad)
            with pytest.raises(ValueError, match=knob):
                reward._validate_constants()
            monkeypatch.undo()

    # valid constants must pass silently
    reward._validate_constants()