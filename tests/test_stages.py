"""Unit suite for the Phase 2 stage state machine (ENV-05, D-01..D-04).

All behaviors verifiable WITHOUT any physics stepping: stages.py is a pure
(StageState, SimReadout) -> in-place-mutation transform over hand-set numpy
numbers — the payoff of the env-free module layout. Synthetic sequences
drive the grasp window through its accept/reject/reset semantics and the
flags through their monotone no-unset contract.
"""

import numpy as np
import pytest

import so100rob.scripted_policy as scripted_policy
from so100rob import constants as C
from so100rob import stages


def _readout(
    cube_pos=(0.0, -0.22, 0.015),
    cube_qvel=None,
    grasp_point=(0.0, -0.22, 0.05),
    jaw_ctrl=0.35,
    approach_dist=0.30,
):
    """Synthetic SimReadout with hand-set numbers (defaults hold nothing).

    Default geometry: cube at rest on the floor, grasp point 3.5 cm above it
    (outside GRASP_TRACK_TOL_M), jaw OPEN — so the default readout triggers
    no stage flag and each test opts into exactly the predicate it probes.
    """
    return stages.SimReadout(
        cube_pos=np.asarray(cube_pos, dtype=float),
        cube_qvel=np.zeros(6) if cube_qvel is None else np.asarray(cube_qvel, dtype=float),
        grasp_point=np.asarray(grasp_point, dtype=float),
        jaw_ctrl_applied=float(jaw_ctrl),
        approach_dist_m=float(approach_dist),
    )


def _holding_readout(jaw_ctrl=None):
    """A kinematic-hold readout: jaw closed command AND cube tracking the
    grasp point exactly (D-01's two clauses both true)."""
    return _readout(
        cube_pos=(0.0, -0.22, 0.05),
        grasp_point=(0.0, -0.22, 0.05),
        jaw_ctrl=C.GRASP_JAW_CLOSED_CTRL_MAX - 0.01 if jaw_ctrl is None else jaw_ctrl,
        approach_dist=0.001,
    )


# ---------------------------------------------------------------------------
# Approach / lift / place threshold semantics (D-03)
# ---------------------------------------------------------------------------


def test_approach_threshold_sets_below_and_at_boundary_not_above():
    """Approach (D-03): horizontal distance below APPROACH_DIST_M sets the
    flag; the boundary (equal) sets; above does not."""
    below = stages.StageState()
    stages.update(below, _readout(approach_dist=C.APPROACH_DIST_M - 1e-6))
    assert below.approach, "distance just below the threshold must set approach"

    boundary = stages.StageState()
    stages.update(boundary, _readout(approach_dist=C.APPROACH_DIST_M))
    assert boundary.approach, "the boundary (equal) must set approach"

    above = stages.StageState()
    stages.update(above, _readout(approach_dist=C.APPROACH_DIST_M + 0.01))
    assert not above.approach, "distance above the threshold must not set approach"


def test_lift_threshold_sets_above_rest_band_margin_not_within_band():
    """Lift (D-03): cube z above CUBE_REST_Z_RANGE top + LIFT_MARGIN_M sets;
    the at-rest band does not."""
    lifted_z = C.CUBE_REST_Z_RANGE[1] + C.LIFT_MARGIN_M
    lifted = stages.StageState()
    stages.update(lifted, _readout(cube_pos=(0.0, -0.22, lifted_z + 1e-6)))
    assert lifted.lift, "cube z just above rest-band top + margin must set lift"

    resting = stages.StageState()
    stages.update(resting, _readout(cube_pos=(0.0, -0.22, C.CUBE_REST_Z_RANGE[1])))
    assert not resting.lift, "cube inside the rest band must not set lift"


def test_place_delegates_to_imported_resting_at_goal():
    """Place (D-03/T-01-05): the flag is the imported pinned predicate —
    identity tamper guard first (any forked predicate fails the suite), then
    resting-at-goal sets, near-goal-but-moving and far-away do not."""
    assert stages.resting_at_goal is scripted_policy.resting_at_goal, (
        "stages.resting_at_goal must BE scripted_policy.resting_at_goal "
        "(T-01-05: single definition of success, never a local fork)"
    )

    resting = stages.StageState()
    stages.update(resting, _readout(cube_pos=C.GOAL_POS, cube_qvel=np.zeros(6)))
    assert resting.place, "cube resting at GOAL_POS with zero qvel must set place"

    moving = stages.StageState()
    stages.update(
        moving, _readout(cube_pos=C.GOAL_POS, cube_qvel=np.full(6, 0.01))
    )
    assert not moving.place, "near-goal but moving (qvel above REST_QVEL_TOL) must not set place"

    far = stages.StageState()
    stages.update(far, _readout(cube_pos=(0.30, 0.10, 0.015)))
    assert not far.place, "cube far from GOAL_POS must not set place"


# ---------------------------------------------------------------------------
# Grasp window semantics (D-01/D-02, success criterion 2)
# ---------------------------------------------------------------------------


def test_grasp_sets_exactly_at_window_step_not_before():
    """Grasp (D-01/D-02): with jaw-closed command and cube tracking, the
    flag sets exactly at the GRASP_WINDOW_STEPS-th consecutive holding step
    — asserted on the step INDEX, not just the end state."""
    window = C.GRASP_WINDOW_STEPS
    state = stages.StageState()
    holding = _holding_readout()

    for step in range(1, window):
        stages.update(state, holding)
        assert not state.grasp, (
            f"grasp must stay False through {window - 1} consecutive holds "
            f"(set at step {step}, window is {window})"
        )
    stages.update(state, holding)  # the window-th consecutive hold
    assert state.grasp, f"grasp must set at the {window}-th consecutive holding step"


def test_grasp_jaw_ctrl_boundary_counts_as_closed():
    """A5: the applied jaw ctrl AT GRASP_JAW_CLOSED_CTRL_MAX is a closed
    command (at or below) — the window accepts it."""
    state = stages.StageState()
    holding = _holding_readout(jaw_ctrl=C.GRASP_JAW_CLOSED_CTRL_MAX)
    for _ in range(C.GRASP_WINDOW_STEPS):
        stages.update(state, holding)
    assert state.grasp, "jaw ctrl exactly at the closed-max boundary must hold"


def test_hold_break_resets_counter_so_window_never_satisfied():
    """A hold broken before the window resets the counter: window-1 holds,
    break, window-1 holds again -> grasp still False."""
    window = C.GRASP_WINDOW_STEPS
    state = stages.StageState()
    holding = _holding_readout()
    breaking = _readout(jaw_ctrl=0.35)  # jaw open -> hold predicate false

    for _ in range(window - 1):
        stages.update(state, holding)
    assert not state.grasp
    stages.update(state, breaking)
    assert state.grasp_hold_counter == 0, "a hold break must reset the counter to 0"

    for step in range(window - 1):
        stages.update(state, holding)
        assert not state.grasp, (
            f"after a break, {window - 1} more holds must not set grasp "
            f"(step {step + 1} of the second stretch)"
        )


def test_momentary_contact_never_sets_grasp():
    """Success criterion 2: sub-window holds (momentary contact) can never
    satisfy the grasp window, no matter how many times they repeat."""
    window = C.GRASP_WINDOW_STEPS
    state = stages.StageState()
    holding = _holding_readout()
    breaking = _readout(jaw_ctrl=0.35)

    for _ in range(3):  # three momentary contacts, each window-1 steps long
        for _ in range(window - 1):
            stages.update(state, holding)
        assert not state.grasp, "momentary contact (sub-window holds) must never set grasp"
        stages.update(state, breaking)
    assert not state.grasp


def test_jaw_open_with_tracking_never_holds():
    """Jaw open (above the closed threshold) with perfect tracking: the hold
    predicate stays false forever — grasp never sets."""
    state = stages.StageState()
    not_closed = _holding_readout(jaw_ctrl=C.GRASP_JAW_CLOSED_CTRL_MAX + 0.05)
    for _ in range(3 * C.GRASP_WINDOW_STEPS):
        stages.update(state, not_closed)
    assert not state.grasp
    assert state.grasp_hold_counter == 0


def test_jaw_closed_without_tracking_never_holds():
    """Jaw closed on empty air (cube NOT tracking the grasp point): the hold
    predicate stays false — closing in empty air must never set grasp."""
    state = stages.StageState()
    air = _readout(
        cube_pos=(0.0, -0.22, 0.015),
        grasp_point=(0.30, -0.22, 0.05),  # 30 cm away: far outside tolerance
        jaw_ctrl=0.0,
    )
    for _ in range(3 * C.GRASP_WINDOW_STEPS):
        stages.update(state, air)
    assert not state.grasp
    assert state.grasp_hold_counter == 0


# ---------------------------------------------------------------------------
# Monotonicity + fresh-state + stage-index (D-04)
# ---------------------------------------------------------------------------


def test_flags_monotone_never_unset_on_violating_readouts():
    """Pitfall 4 / D-04: once any flag is True, subsequent violating readouts
    never un-set it — flags only flip False->True within an episode."""
    state = stages.StageState()
    # Drive all four flags True in order.
    stages.update(state, _readout(approach_dist=C.APPROACH_DIST_M))          # approach
    for _ in range(C.GRASP_WINDOW_STEPS):                                   # grasp
        stages.update(state, _holding_readout())
    stages.update(state, _readout(cube_pos=(0.0, -0.22, 0.10)))             # lift
    stages.update(state, _readout(cube_pos=C.GOAL_POS, cube_qvel=np.zeros(6)))  # place
    assert state.approach and state.grasp and state.lift and state.place

    # Maximally violating readout: far, open-jaw, at-rest-band, far-from-goal.
    violating = _readout(cube_pos=(0.30, 0.10, 0.015), approach_dist=0.50, jaw_ctrl=0.35)
    for _ in range(3):
        stages.update(state, violating)
    assert state.approach and state.grasp and state.lift and state.place, (
        "a violating readout must never un-set an already-True flag (D-04 monotone)"
    )


def test_fresh_stage_state_defaults():
    """A fresh StageState has all flags False, stage_index 0, counter 0."""
    state = stages.StageState()
    assert not state.approach
    assert not state.grasp
    assert not state.lift
    assert not state.place
    assert state.grasp_hold_counter == 0
    assert stages.stage_index(state) == 0


def test_stage_index_counts_true_flags():
    """stage_index equals the number of True flags (0..4)."""
    state = stages.StageState()
    assert stages.stage_index(state) == 0
    state.approach = True
    assert stages.stage_index(state) == 1
    state.grasp = True
    assert stages.stage_index(state) == 2
    state.lift = True
    assert stages.stage_index(state) == 3
    state.place = True
    assert stages.stage_index(state) == 4


# ---------------------------------------------------------------------------
# Tamper guards + fail-loudly validation (T-01-05 pattern, threat T-02-02)
# ---------------------------------------------------------------------------


def test_pinned_constants_tamper_guards():
    """The stage-machine thresholds must be the pinned constants, referenced
    from constants — never local literals (T-01-05 tamper-guard pattern)."""
    assert C.GRASP_WINDOW_STEPS == 5, "D-02 pins the grasp window at 5 control steps"
    assert C.SUCCESS_TOLERANCE_XY == 0.02, "D-04 predicate tolerance must stay pinned"


def test_validation_rejects_invalid_constants(monkeypatch):
    """Threat T-02-02: re-running module validation with any pinned constant
    monkeypatched invalid raises ValueError NAMING the violated constant."""
    # window <= 0
    monkeypatch.setattr(C, "GRASP_WINDOW_STEPS", 0)
    with pytest.raises(ValueError, match="GRASP_WINDOW_STEPS"):
        stages._validate_constants()
    monkeypatch.undo()

    # non-positive thresholds
    for knob, bad in (
        ("APPROACH_DIST_M", 0.0),
        ("LIFT_MARGIN_M", -0.01),
        ("GRASP_TRACK_TOL_M", 0.0),
    ):
        monkeypatch.setattr(C, knob, bad)
        with pytest.raises(ValueError, match=knob):
            stages._validate_constants()
        monkeypatch.undo()

    # valid constants must pass silently
    stages._validate_constants()