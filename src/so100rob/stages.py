"""Monotone stage state machine for the pick-and-place task (Phase 2, ENV-05).

Purity contract (the payoff of this layout): this module is env-free and
MuJoCo-free. It consumes a ``SimReadout`` — a plain snapshot of privileged
kinematics (numpy arrays and floats) gathered by the env — and mutates a
``StageState`` in place. Nothing here imports the env, steps physics, or
re-derives gripper geometry; unit tests run it with hand-set numbers.

Stage semantics (Phase 2 CONTEXT decisions):
  * approach (D-03): horizontal grasp-point-to-cube-center distance below
    APPROACH_DIST_M.
  * grasp (D-01/D-02): a KINEMATIC HOLD sustained over GRASP_WINDOW_STEPS
    consecutive control steps — jaw closed on the cube (applied jaw ctrl at
    or below GRASP_JAW_CLOSED_CTRL_MAX, keyed on the applied action value
    per the carried A5 note) AND the cube tracking the grasp point within
    GRASP_TRACK_TOL_M (3D). A hold break resets the counter, so momentary
    contact can never satisfy the window (success criterion 2).
  * lift (D-03): cube z above the CUBE_REST_Z_RANGE top plus LIFT_MARGIN_M.
  * place (D-03/T-01-05): the pinned predicate ``resting_at_goal`` from
    so100rob.scripted_policy, IMPORTED — never re-defined locally.

Flags are monotone within an episode (D-04): once True, never un-set.
"""

from dataclasses import dataclass

import numpy as np

from so100rob import constants as C
from so100rob.scripted_policy import resting_at_goal  # T-01-05: import, never re-define


@dataclass
class SimReadout:
    """Privileged per-step simulation snapshot (gathered by the env's step()).

    Every field is finite by contract — the env guards the readout with
    np.isfinite asserts before it reaches the flags (threat T-02-01), so
    NaN/Inf physics state fails loudly instead of poisoning stage flags.
    """

    cube_pos: np.ndarray      # (3,) world position of the cube center
    cube_qvel: np.ndarray     # (6,) cube freejoint [linear; angular] velocity
    grasp_point: np.ndarray   # (3,) world midpoint of the opposing finger pads
    jaw_ctrl_applied: float   # applied jaw actuator ctrl this step (A5)
    approach_dist_m: float    # horizontal |grasp_point - cube_pos| norm [m]


@dataclass
class StageState:
    """Episode-local monotone stage flags + the grasp-window counter (D-04).

    All state is episode-local: the env clears it (a fresh instance) in
    reset_model per Pitfall 6 — never mid-episode.
    """

    approach: bool = False
    grasp: bool = False
    lift: bool = False
    place: bool = False
    grasp_hold_counter: int = 0


def _validate_constants() -> None:
    """Fail-loudly check of the pinned stage constants (threat T-02-02).

    A silently-loosened or malformed constant (window 0, negative tolerance)
    would disable the stage gates without any test failure — so the module
    refuses to import until every pinned value is valid, naming the
    violated constant in the error.
    """
    if not isinstance(C.GRASP_WINDOW_STEPS, int) or C.GRASP_WINDOW_STEPS <= 0:
        raise ValueError(
            f"GRASP_WINDOW_STEPS must be a positive int, got {C.GRASP_WINDOW_STEPS!r}"
        )
    for name in ("APPROACH_DIST_M", "LIFT_MARGIN_M", "GRASP_TRACK_TOL_M"):
        value = getattr(C, name)
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value!r}")
    if not C.GRASP_JAW_CLOSED_CTRL_MAX > 0:
        raise ValueError(
            f"GRASP_JAW_CLOSED_CTRL_MAX must be positive, got {C.GRASP_JAW_CLOSED_CTRL_MAX!r}"
        )


_validate_constants()  # fail loudly at import (T-02-02)


def _hold_pred(sim: SimReadout) -> bool:
    """D-01's kinematic-hold predicate: jaw closed command AND the cube
    tracking the grasp point within GRASP_TRACK_TOL_M (3D)."""
    jaw_closed = sim.jaw_ctrl_applied <= C.GRASP_JAW_CLOSED_CTRL_MAX
    tracks = (
        float(np.linalg.norm(np.asarray(sim.cube_pos) - np.asarray(sim.grasp_point)))
        <= C.GRASP_TRACK_TOL_M
    )
    return jaw_closed and tracks


def update(state: StageState, sim: SimReadout) -> None:
    """Advance the stage machine one control step, mutating ``state`` in place.

    Monotone per D-04: every branch only ever flips a flag False->True; no
    readout can un-set an already-True flag. The hold predicate is evaluated
    only while grasp is still False — after grasp sets, the counter is
    frozen and the window is never re-armed within the episode.
    """
    if not state.approach:
        state.approach = sim.approach_dist_m <= C.APPROACH_DIST_M

    if not state.grasp:
        if _hold_pred(sim):
            state.grasp_hold_counter += 1
            if state.grasp_hold_counter >= C.GRASP_WINDOW_STEPS:
                state.grasp = True
        else:
            state.grasp_hold_counter = 0

    if not state.lift:
        state.lift = float(np.asarray(sim.cube_pos).reshape(-1)[2]) > (
            C.CUBE_REST_Z_RANGE[1] + C.LIFT_MARGIN_M
        )

    if not state.place:
        state.place = resting_at_goal(sim.cube_pos, sim.cube_qvel)


def stage_index(state: StageState) -> int:
    """Number of True flags — 0 (none) .. 4 (place)."""
    return int(state.approach) + int(state.grasp) + int(state.lift) + int(state.place)