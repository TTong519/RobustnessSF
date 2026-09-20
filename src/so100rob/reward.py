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

RED-phase stub (Plan 02-01 Task 2): signatures below are final; bodies
raise NotImplementedError so the suite fails on this stub by design, per the
Phase 1 validated RED pattern.
"""

from so100rob import constants as C


def _validate_constants() -> None:
    """Fail-loudly check of every pinned reward invariant (threat T-02-02)."""
    raise NotImplementedError("RED stub (Plan 02-01 Task 2)")


def potential(stage_index: int, approach_dist_m: float) -> float:
    """Phi(s): stage-indexed base potential plus the hybrid approach sub-term
    while stage_index < 2 (OQ-3 resolution)."""
    raise NotImplementedError("RED stub (Plan 02-01 Task 2)")


def compute(phi_prev: float, phi_next: float, first_place: bool) -> tuple[float, dict]:
    """One reward step: gamma*Phi(s') - Phi(s) - TIME_PENALTY, plus the
    one-time TERMINAL_PLACE_BONUS iff first_place."""
    raise NotImplementedError("RED stub (Plan 02-01 Task 2)")