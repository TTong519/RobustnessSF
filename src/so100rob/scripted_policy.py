"""Scripted open-loop grasp-lift-place trajectory (D-05/D-08, ENV-06).

Phase 1's physics-gate policy: a spawn-aware, open-loop (in the RL sense —
no feedback control, NOT state-blind) trajectory builder that reads the
privileged cube pose and emits a fixed sequence of 6-dim joint-target
actions. Reused by Phase 2's hand-posed trajectory verification, so it is
a real module, not a throwaway gate script.

RED STUB: build_trajectory raises NotImplementedError by design so the
test suite fails on the missing behavior, not on collection
(Plan 01-03 Task 1 action).
"""

import numpy as np

from so100rob import constants as C


class TrajectoryParams:
    """Placeholder for the planned TrajectoryParams dataclass (RED stub)."""

    def __getattr__(self, name):  # pragma: no cover - RED stub only
        raise NotImplementedError(f"TrajectoryParams.{name} (RED stub)")


def build_trajectory(model, data, cube_pose, goal_pos, params=None):
    """Emit the labeled grasp-lift-place action sequence (RED stub)."""
    raise NotImplementedError("build_trajectory is not implemented (RED stub)")