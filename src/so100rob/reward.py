"""Potential-based stage-gated reward (Phase 2, D-05/D-06/D-07).

Rationale: the dense term is potential-based shaping of the exact
gamma*Phi(s') - Phi(s) form, which provably preserves the optimal policy
(Ng, Harada & Russell, "Policy Invariance Under Reward Transformations",
ICML 1999) — any non-potential dense form admits MDPs where the shaped
optimum diverges. A small per-step time penalty encourages brevity, and a
one-time terminal place bonus (10x the total shapable potential, D-06)
dominates so dense reward can never accumulate significantly without a
stage flag advancing (success criterion 3).

Purity contract: floats in, floats + dict out — no mujoco, no env imports.
Constants are read from so100rob.constants at CALL time so the fail-loudly
validation is re-runnable under monkeypatch (threat T-02-02).
"""

from so100rob import constants as C


def _validate_constants() -> None:
    """Fail-loudly check of every pinned reward invariant (threat T-02-02).

    A silently-loosened constant (non-monotone potentials, a terminal bonus
    below 10x, a gamma drifting from the PPO discount) would break the
    policy-invariance guarantee or the criterion-3 bound without any test
    failure — so the module refuses to import until every invariant holds,
    naming the violated constant in the error.
    """
    potentials = tuple(C.STAGE_POTENTIALS)
    if not potentials or potentials[0] != 0.0:
        raise ValueError(
            f"STAGE_POTENTIALS must be non-empty and rooted at 0.0 (Phi(s_0)=0), got {potentials!r}"
        )
    for i in range(len(potentials) - 1):
        if potentials[i + 1] < potentials[i]:
            raise ValueError(
                f"STAGE_POTENTIALS must be monotone non-decreasing, got {potentials!r}"
            )
    if len(potentials) < 3:
        raise ValueError(
            f"STAGE_POTENTIALS needs at least 3 rungs for the hybrid guard, got {potentials!r}"
        )
    if C.APPROACH_W < 0:
        raise ValueError(f"APPROACH_W must be non-negative, got {C.APPROACH_W!r}")
    if C.APPROACH_POTENTIAL_SCALE_M <= 0:
        raise ValueError(
            f"APPROACH_POTENTIAL_SCALE_M must be positive, got {C.APPROACH_POTENTIAL_SCALE_M!r}"
        )
    # Hybrid guard (OQ-3): the best un-grasped Phi (approach rung + full
    # sub-term) must stay at or below the grasp rung — hovering near the
    # cube can never out-earn an actual grasp.
    if potentials[1] + C.APPROACH_W > potentials[2]:
        raise ValueError(
            f"APPROACH_W breaks the hybrid monotone guard: STAGE_POTENTIALS[1] "
            f"+ APPROACH_W ({potentials[1]} + {C.APPROACH_W}) must stay <= "
            f"STAGE_POTENTIALS[2] ({potentials[2]})"
        )
    if C.TERMINAL_PLACE_BONUS < 10 * max(potentials):
        raise ValueError(
            f"TERMINAL_PLACE_BONUS ({C.TERMINAL_PLACE_BONUS}) must be >= 10x the "
            f"max potential ({max(potentials)}) — D-06 terminal dominance"
        )
    if not 0.0 < C.REWARD_GAMMA < 1.0:
        raise ValueError(
            f"REWARD_GAMMA must be in (0, 1) — it MUST equal the Phase 4 PPO "
            f"discount (Pitfall 1), got {C.REWARD_GAMMA!r}"
        )
    if C.TIME_PENALTY <= 0:
        raise ValueError(f"TIME_PENALTY must be positive, got {C.TIME_PENALTY!r}")
    for name in ("FULL_FLAGS_NO_PLACE_MAX_FRACTION", "AUDIT_DENSE_FRACTION"):
        value = getattr(C, name)
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {value!r}")


_validate_constants()  # fail loudly at import (T-02-02)


def potential(stage_index: int, approach_dist_m: float) -> float:
    """Phi(s): the stage-indexed base potential plus the hybrid approach
    sub-term while stage_index < 2 (OQ-3 resolution).

    STAGE_POTENTIALS[stage_index] + APPROACH_W * (1 - clamp(d / scale, 0, 1))
    for un-grasbed stages; the exact base alone once the grasp rung is
    reached. The sub-term is bounded in [0, APPROACH_W] and the hybrid guard
    (validated at import) keeps it below the grasp rung.
    """
    base = float(C.STAGE_POTENTIALS[stage_index])
    if stage_index < 2:
        clamped = min(max(float(approach_dist_m) / C.APPROACH_POTENTIAL_SCALE_M, 0.0), 1.0)
        return base + C.APPROACH_W * (1.0 - clamped)
    return base


def compute(phi_prev: float, phi_next: float, first_place: bool) -> tuple[float, dict]:
    """One reward step (D-05/D-06/D-07).

    total = gamma*Phi(s') - Phi(s) - TIME_PENALTY, plus TERMINAL_PLACE_BONUS
    iff first_place (the env's one-shot latch supplies first_place — the
    bonus lands exactly once, on the terminated step). Returns
    (total, components) with components keys exactly reward_shaping /
    reward_time_penalty / reward_terminal so the audit and Phase 4's reward
    audit can attribute reward to stages without recomputation (D-07).
    """
    shaping = C.REWARD_GAMMA * float(phi_next) - float(phi_prev)
    terminal = C.TERMINAL_PLACE_BONUS if first_place else 0.0
    components = {
        "reward_shaping": shaping,
        "reward_time_penalty": -C.TIME_PENALTY,
        "reward_terminal": terminal,
    }
    total = shaping - C.TIME_PENALTY + terminal
    return total, components