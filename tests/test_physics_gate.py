"""ENV-06 integration suite: the physics gate as one re-runnable command.

Encodes the achieved 25/25 bar (Plan 01-03 Task 2) as a parametrized
pytest integration test — every grid spawn runs the same reset ->
privileged-pose read -> build_trajectory -> env.step path as the gate
script (driven directly through the env, never shelled out), then the D-04
predicate (so100rob.scripted_policy.resting_at_goal, the pinned definition)
decides pass/fail. This is a regression guard around behavior that
already exists, so no RED cycle — the whole Phase 1 suite must stay green
with it.

Timing: 25 spawns x ~160 control steps at ~11.4k physics-steps/s verified
throughput — the module runs in a few seconds.
"""

import mujoco
import numpy as np
import pytest

from so100rob import constants as C
from so100rob.scripted_policy import (
    POST_RELEASE_SETTLE_STEPS,
    build_trajectory,
    resting_at_goal,
)

# No-ejection floor (Pitfall 1's warning sign): the cube center must stay
# above this height for the whole episode — a smash-through or a floor
# interpenetration both violate it.
_CUBE_Z_FLOOR = 0.01


def _cube_addressing(env) -> tuple[int, int]:
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, C.CUBE_JOINT)
    assert jid >= 0, f"cube joint {C.CUBE_JOINT!r} missing from compiled model"
    return int(env.model.jnt_qposadr[jid]), int(env.model.jnt_dofadr[jid])


@pytest.mark.parametrize("spawn_index", range(C.SPAWN_GRID_SIZE))
def test_scripted_trajectory_places_cube_on_every_grid_spawn(
    gate_env, gate_params, spawn_index
):
    """All 25 frozen grid spawns pass the D-04 predicate, NaN-free, no
    ejection, at the default 20 Hz control rate (D-06 bar)."""
    env = gate_env
    # Metric-collection discipline: the gate env is never a recorded env.
    assert env.render_mode is None
    # Tamper guard (T-01-05): the predicate tolerance must be the pinned
    # D-04 constant, referenced from constants — never a local literal.
    assert C.SUCCESS_TOLERANCE_XY == 0.02

    env.reset(seed=0, options={"spawn_index": spawn_index})
    qadr, dadr = _cube_addressing(env)

    # Privileged cube pose read (D-05: open-loop, not state-blind).
    cube_spawn = np.array(env.data.qpos[qadr : qadr + 3])
    seq = build_trajectory(env.model, env.data, cube_spawn, C.GOAL_POS, gate_params)
    assert len(seq) > 0

    finite_state = True
    for action, label in seq:
        env.step(action)
        finite_state &= bool(
            np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()
        )
        cube_z = float(env.data.qpos[qadr + 2])
        assert cube_z > _CUBE_Z_FLOOR, (
            f"spawn {spawn_index}: cube z {cube_z:.4f} below {_CUBE_Z_FLOOR} "
            f"during phase {label!r} (ejection/interpenetration)"
        )

    # Post-release settle tail: hold the final arm config so the cube
    # rests before the predicate is evaluated.
    last_action = seq[-1][0]
    for _ in range(POST_RELEASE_SETTLE_STEPS):
        env.step(last_action)
        finite_state &= bool(
            np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()
        )
        cube_z = float(env.data.qpos[qadr + 2])
        assert cube_z > _CUBE_Z_FLOOR

    assert finite_state, f"spawn {spawn_index}: non-finite qpos/qvel during episode"

    cube_final = np.array(env.data.qpos[qadr : qadr + 3])
    cube_qvel = np.array(env.data.qvel[dadr : dadr + 6])
    assert resting_at_goal(cube_final, cube_qvel), (
        f"spawn {spawn_index} failed the D-04 predicate: final "
        f"xy_err={np.linalg.norm(cube_final[:2] - np.asarray(C.GOAL_POS[:2])):.4f} "
        f"z={cube_final[2]:.4f} qvel_norm={np.linalg.norm(cube_qvel):.2e}"
    )