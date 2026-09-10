"""ENV-02: spawn reproducibility — frozen seed grid + grid/continuous modes.

The 25-spawn 5x5 grid is frozen as checked-in data
(config/spawn_seeds.json, D-11): reproducibility is proven by data-file
equality, not RNG discipline. Grid resets take options={"spawn_index": i};
continuous mode (baseline training, D-12) samples uniformly over the patch
from the env's seeded generator.
"""

import json

import mujoco
import numpy as np
import pytest

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv

# Settle drift tolerance for cube XY vs the file entry (research Pattern 2:
# the settle phase micro-drifts the resting cube sub-micron amounts).
XY_TOL = 1e-4


def load_seed_entries():
    """Read config/spawn_seeds.json and return the raw 25-entry list."""
    with open(C.SPAWN_SEEDS_PATH) as f:
        raw = json.load(f)
    entries = raw["spawns"] if isinstance(raw, dict) and "spawns" in raw else raw
    return entries


def cube_xy(env):
    """Current cube spawn XY, read via the named freejoint."""
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, C.CUBE_JOINT)
    adr = env.model.jnt_qposadr[jid]
    x, y = env.data.qpos[adr], env.data.qpos[adr + 1]
    return float(x), float(y)


def cube_qpos(env):
    """Full 7-dof cube freejoint qpos copy (x, y, z, quat w-first)."""
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, C.CUBE_JOINT)
    adr = env.model.jnt_qposadr[jid]
    return np.asarray(env.data.qpos[adr : adr + 7]).copy()


# ---------------------------------------------------------------------------
# Test 1: the checked-in file IS the frozen protocol
# ---------------------------------------------------------------------------


def test_seed_file_has_25_valid_entries():
    entries = load_seed_entries()
    assert len(entries) == 25
    for entry in entries:
        assert len(entry) == 3
        x, y, z = (float(v) for v in entry)
        assert np.isfinite(x) and np.isfinite(y) and np.isfinite(z)
        assert C.SPAWN_PATCH_X[0] <= x <= C.SPAWN_PATCH_X[1]
        assert C.SPAWN_PATCH_Y[0] <= y <= C.SPAWN_PATCH_Y[1]
        assert z == C.SPAWN_Z


# ---------------------------------------------------------------------------
# Test 2: grid reset maps spawn_index i to file entry i (order-stable)
# ---------------------------------------------------------------------------


def test_grid_reset_matches_file_entry_per_index(grid_env):
    entries = load_seed_entries()
    for i in range(25):
        grid_env.reset(options={"spawn_index": i})
        x, y = cube_xy(grid_env)
        assert x == pytest.approx(entries[i][0], abs=XY_TOL), f"index {i}: x"
        assert y == pytest.approx(entries[i][1], abs=XY_TOL), f"index {i}: y"


# ---------------------------------------------------------------------------
# Test 3: identical spawns across independently constructed instances
# ---------------------------------------------------------------------------


def test_identical_spawns_across_instances():
    entries = load_seed_entries()
    env1 = SO100PickPlaceEnv(spawn_mode="grid")
    env2 = SO100PickPlaceEnv(spawn_mode="grid")
    try:
        env1.reset(seed=0, options={"spawn_index": 7})
        env2.reset(seed=0, options={"spawn_index": 7})
        x1, y1 = cube_xy(env1)
        x2, y2 = cube_xy(env2)
        # byte-identical across instances and runs (deterministic settle)
        assert x1 == x2
        assert y1 == y2
        # and the shared position is entry 7 of the file
        assert x1 == pytest.approx(entries[7][0], abs=XY_TOL)
        assert y1 == pytest.approx(entries[7][1], abs=XY_TOL)
    finally:
        env1.close()
        env2.close()


# ---------------------------------------------------------------------------
# Test 4: reset() without options deterministically uses spawn_index 0
# ---------------------------------------------------------------------------


def test_grid_reset_defaults_to_index_zero(grid_env):
    entries = load_seed_entries()
    grid_env.reset(seed=0)
    x, y = cube_xy(grid_env)
    assert x == pytest.approx(entries[0][0], abs=XY_TOL)
    assert y == pytest.approx(entries[0][1], abs=XY_TOL)


# ---------------------------------------------------------------------------
# Test 5: malformed seed data fails loudly (T-01-03)
# ---------------------------------------------------------------------------


def _make_env(spawn_seeds_path):
    return SO100PickPlaceEnv(spawn_mode="grid", spawn_seeds_path=spawn_seeds_path)


def test_malformed_seed_file_raises(tmp_path):
    # (a) fewer than 25 entries -> ValueError citing the expected count
    p24 = tmp_path / "spawn_seeds_24.json"
    p24.write_text(json.dumps({"spawns": [[0.0, -0.227, C.SPAWN_Z]] * 24}))
    with pytest.raises(ValueError, match="25"):
        _make_env(p24)

    # (b) an out-of-patch coordinate -> ValueError naming the patch violation
    bad = [[0.0, -0.227, C.SPAWN_Z]] * 24 + [[0.5, -0.227, C.SPAWN_Z]]
    p_oob = tmp_path / "spawn_seeds_oob.json"
    p_oob.write_text(json.dumps({"spawns": bad}))
    with pytest.raises(ValueError, match="patch"):
        _make_env(p_oob)

    # (c) missing file -> ValueError naming the path (empty/degenerate input)
    with pytest.raises(ValueError, match="spawn_seeds_missing.json"):
        _make_env(tmp_path / "spawn_seeds_missing.json")


# ---------------------------------------------------------------------------
# Test 6: continuous mode is seeded-uniform over the patch (D-12)
# ---------------------------------------------------------------------------


def test_continuous_mode_seeded_uniform(continuous_env):
    continuous_env.reset(seed=123)
    x1, y1 = cube_xy(continuous_env)
    continuous_env.reset(seed=123)
    x2, y2 = cube_xy(continuous_env)

    # same seed -> same spawn sample (fresh sample per reset, reproducible)
    assert x1 == x2
    assert y1 == y2

    # inside the patch rectangle
    assert C.SPAWN_PATCH_X[0] <= x1 <= C.SPAWN_PATCH_X[1]
    assert C.SPAWN_PATCH_Y[0] <= y1 <= C.SPAWN_PATCH_Y[1]

    # different seed -> different spawn (overwhelmingly probable)
    continuous_env.reset(seed=456)
    x3, y3 = cube_xy(continuous_env)
    assert (x3, y3) != (x1, y1)


# ---------------------------------------------------------------------------
# Test 7: idempotent reset — same index twice, identical cube state
# ---------------------------------------------------------------------------


def test_grid_reset_idempotent(grid_env):
    grid_env.reset(seed=0, options={"spawn_index": 11})
    q1 = cube_qpos(grid_env)
    grid_env.reset(options={"spawn_index": 11})
    q2 = cube_qpos(grid_env)
    np.testing.assert_allclose(q1, q2, atol=1e-9)