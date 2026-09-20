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

RED-phase stub (Plan 02-01 Task 1): signatures below are final; bodies raise
NotImplementedError so the suite fails on this stub by design, per the
Phase 1 validated RED pattern.
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
    """Fail-loudly check of the pinned stage constants (threat T-02-02)."""
    raise NotImplementedError("RED stub (Plan 02-01 Task 1)")


def update(state: StageState, sim: SimReadout) -> None:
    """Advance the stage machine one control step, mutating ``state`` in place."""
    raise NotImplementedError("RED stub (Plan 02-01 Task 1)")


def stage_index(state: StageState) -> int:
    """Number of True flags — 0 (none) .. 4 (place)."""
    raise NotImplementedError("RED stub (Plan 02-01 Task 1)")