"""SO100PickPlaceEnv — Gymnasium environment for the SO-ARM100 pick-and-place.

Gymnasium MujocoEnv subclass over the L1 task scene
(src/so100rob/assets/scene_pickplace.xml, which includes the read-only
upstream SO-ARM100 model). Scope after Phase 2:

  * keyframe ("home") reset + explicit cube spawn + settle (verified recipe,
    research Pattern 2 — the keyframe zero-pads the cube to world origin,
    so the overwrite is MANDATORY)
  * step() composes the Phase 2 stage machine + potential-based reward:
    stage flags / stage_index / is_success / reward components / reward_phi
    in info every step (flat scalar keys — the cross-phase contract for
    Phases 3-5), terminated=True on the step the place flag first sets,
    truncated=True at the step cap without success (timeout is truncation,
    NEVER termination — PPO bootstrapping depends on the distinction)
  * observation is a minimal flat state vector — the full obs pipeline
    (images) is Phase 3 scope

The control update rate is the constructor-time frame_skip (ENV-04): the
physics timestep is pinned in the scene (0.002 s) and a "rate change" is
exactly a decimation change, never a model change.
"""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

from so100rob import constants as C
from so100rob import stages
from so100rob.scripted_policy import grasp_point

# Minimal Phase 1 observation: qpos (13) + qvel (12) + 2D XY offset of the
# cube center from the goal position.
_OBS_DIM = 13 + 12 + 2


def load_spawn_grid(path: Path | str) -> np.ndarray:
    """Load and schema-validate the frozen 25-spawn seed grid (D-11, T-01-03).

    The evaluation spawn protocol is checked-in DATA: this function reads it
    verbatim and never re-derives positions. A missing, empty, or malformed
    file raises a clear ValueError citing the path and the reason — the env
    must fail loudly rather than silently proceed with a broken protocol.
    """
    path = Path(path)
    try:
        with open(path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"spawn seed file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed spawn seed file {path}: not valid JSON ({exc})") from None

    entries = raw.get("spawns") if isinstance(raw, dict) else raw
    expected = C.SPAWN_GRID_SIZE
    if not isinstance(entries, list) or len(entries) != expected:
        found = len(entries) if isinstance(entries, list) else type(entries).__name__
        raise ValueError(f"{path}: expected exactly {expected} spawn entries, found {found}")

    grid = np.zeros((expected, 3), dtype=np.float64)
    for i, entry in enumerate(entries):
        try:
            coords = [float(v) for v in entry]
        except (TypeError, ValueError):
            raise ValueError(f"{path}: spawn entry {i} is not a sequence of numbers: {entry!r}") from None
        if len(coords) != 3:
            raise ValueError(f"{path}: spawn entry {i} must have exactly 3 coordinates, got {len(coords)}")
        if not all(math.isfinite(v) for v in coords):
            raise ValueError(f"{path}: spawn entry {i} has non-finite coordinates: {entry!r}")
        x, y, _ = coords
        if not C.SPAWN_PATCH_X[0] <= x <= C.SPAWN_PATCH_X[1]:
            raise ValueError(
                f"{path}: spawn entry {i} x={x} outside the patch x-bounds {C.SPAWN_PATCH_X}"
            )
        if not C.SPAWN_PATCH_Y[0] <= y <= C.SPAWN_PATCH_Y[1]:
            raise ValueError(
                f"{path}: spawn entry {i} y={y} outside the patch y-bounds {C.SPAWN_PATCH_Y}"
            )
        grid[i] = coords
    return grid


class SO100PickPlaceEnv(MujocoEnv):
    """SO-ARM100 pick-and-place env (Phase 1 walking skeleton)."""

    # No class-level render_fps: MujocoEnv asserts it against 1/dt inside
    # __init__ (Pitfall 4). It is set per-instance below, after frame_skip
    # is known, so every control rate carries a coherent fps.
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        frame_skip: int = C.DEFAULT_FRAME_SKIP,
        render_mode: str | None = None,
        width: int = 64,
        height: int = 64,
        spawn_mode: str = "grid",
        spawn_seeds_path: Path | str | None = None,
        episode_sim_seconds: float = C.EPISODE_SIM_SECONDS,
    ) -> None:
        # Spawn mode + seed-file validation BEFORE the (slow) model compile:
        # a malformed frozen-data file fails fast and loudly (T-01-03).
        if spawn_mode not in ("grid", "continuous"):
            raise ValueError(
                f"unsupported spawn_mode: {spawn_mode!r} (expected 'grid' or 'continuous')"
            )
        self.spawn_mode = spawn_mode
        if spawn_mode == "grid":
            self.spawn_seeds_path = (
                Path(spawn_seeds_path) if spawn_seeds_path is not None else C.SPAWN_SEEDS_PATH
            )
            self._spawn_grid = load_spawn_grid(self.spawn_seeds_path)

        super().__init__(
            model_path=str(C.SCENE_XML_PATH),
            frame_skip=frame_skip,
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32
            ),
            render_mode=render_mode,
            width=width,
            height=height,
            camera_name=C.CAMERA_NAME,
        )

        # Per-instance render_fps (Pitfall 4): a fresh dict so non-default
        # frame_skip values never assert-fail against a stale class-level fps.
        self.metadata = {**SO100PickPlaceEnv.metadata, "render_fps": int(round(1 / self.dt))}

        # Sim-second episode cap (ENV-04): episodes are defined in sim
        # seconds, derived from dt so different frame_skip values yield
        # identical sim-time episodes (D-07). frame_skip itself is
        # construction-only — a different control rate is a different env
        # instance, never a runtime mutation.
        self.episode_sim_seconds = episode_sim_seconds
        self.max_episode_steps = int(np.ceil(episode_sim_seconds / self.dt))

        # Reset options stash: gymnasium 1.3.0's MujocoEnv.reset() accepts
        # options but does NOT forward them into reset_model() — we capture
        # them here so reset_model can honor options={"spawn_index": i}.
        self._reset_options: dict | None = None

        # Cached cube joint addressing (constant for the compiled model).
        self._cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, C.CUBE_JOINT
        )
        self._cube_qposadr = self.model.jnt_qposadr[self._cube_joint_id]
        self._cube_dofadr = self.model.jnt_dofadr[self._cube_joint_id]

        # Cached jaw actuator index, resolved BY NAME and cached once —
        # never index the action vector by a guessed offset (the same
        # fail-loudly name-resolution pattern as the cube addressing).
        self._jaw_act_idx = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Jaw"
        )
        if self._jaw_act_idx < 0:
            raise ValueError("actuator 'Jaw' not found in compiled model")

        self._goal_xy = np.asarray(C.GOAL_POS[:2], dtype=np.float64)

        # Episode-local stage/reward state placeholders — reset_model owns
        # the authoritative clearing (Pitfall 6); these exist so the
        # attributes are defined even before the first reset.
        self._stage = stages.StageState()
        self._elapsed_steps = 0

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        cube_xy = np.asarray(self.data.qpos[self._cube_qposadr : self._cube_qposadr + 2])
        obs = np.concatenate(
            [
                self.data.qpos,
                self.data.qvel,
                cube_xy - self._goal_xy,
            ]
        )
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def _gather_readout(self, action) -> stages.SimReadout:
        """Snapshot the privileged kinematics the stage machine consumes.

        Read from the cached name-resolved cube addresses plus
        grasp_point(model, data) and the applied action's jaw ctrl. The
        readout is guarded with np.isfinite asserts (threat T-02-01) —
        NaN/Inf physics state fails loudly here instead of silently
        becoming stage flags or reward.
        """
        cube_pos = np.array(self.data.qpos[self._cube_qposadr : self._cube_qposadr + 3])
        cube_qvel = np.array(self.data.qvel[self._cube_dofadr : self._cube_dofadr + 6])
        pad_midpoint = np.asarray(grasp_point(self.model, self.data), dtype=float)
        jaw_ctrl_applied = float(np.asarray(action).reshape(-1)[self._jaw_act_idx])
        approach_dist_m = float(np.linalg.norm(pad_midpoint[:2] - cube_pos[:2]))

        assert all(
            np.isfinite(v).all() if isinstance(v, np.ndarray) else np.isfinite(v)
            for v in (cube_pos, cube_qvel, pad_midpoint)
        ) and math.isfinite(jaw_ctrl_applied) and math.isfinite(approach_dist_m), (
            "non-finite simulation state in the stage readout — NaN/Inf "
            "physics must fail loudly, never poison stage flags/reward (T-02-01)"
        )
        return stages.SimReadout(
            cube_pos=cube_pos,
            cube_qvel=cube_qvel,
            grasp_point=pad_midpoint,
            jaw_ctrl_applied=jaw_ctrl_applied,
            approach_dist_m=approach_dist_m,
        )

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.do_simulation(action, self.frame_skip)
        self._elapsed_steps += 1

        readout = self._gather_readout(action)
        stages.update(self._stage, readout)

        # Timeout is truncation, NEVER termination (Pitfall 2 — PPO
        # bootstrapping depends on the distinction); success terminates on
        # the very step the place flag first sets.
        terminated = bool(self._stage.place)
        truncated = (not terminated) and (self._elapsed_steps >= self.max_episode_steps)

        # Flat scalar info contract (cross-phase API: Phase 3 wrappers and
        # Phase 5 funnel metrics consume these keys BY NAME). is_success
        # mirrors the place flag EVERY step (Pitfall 7 — monotone episode
        # view, never final-step-only).
        info: dict = {
            "stage_approach": bool(self._stage.approach),
            "stage_grasp": bool(self._stage.grasp),
            "stage_lift": bool(self._stage.lift),
            "stage_place": bool(self._stage.place),
            "stage_index": stages.stage_index(self._stage),
            "is_success": bool(self._stage.place),
        }
        return self._get_obs(), 0.0, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """Capture reset options, then defer to MujocoEnv.reset.

        gymnasium 1.3.0's MujocoEnv.reset() swallows ``options`` (it never
        reaches reset_model), so grid-mode spawn selection needs this stash.
        """
        self._reset_options = options
        return super().reset(seed=seed, options=options)

    def _sample_spawn(self) -> tuple[float, float, float]:
        """Select the cube spawn position for this reset (grid or continuous).

        Grid mode (evaluation + physics gate): spawn_index i maps verbatim to
        file entry i of the frozen seed grid (D-11) — order-stable, no
        runtime re-derivation. Continuous mode (baseline training, D-12):
        a fresh uniform sample over the patch from the env's seeded
        generator.
        """
        options = self._reset_options or {}
        if self.spawn_mode == "grid":
            idx = options.get("spawn_index", 0)
            if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)):
                raise ValueError(f"spawn_index must be an int, got {idx!r}")
            if not 0 <= idx < C.SPAWN_GRID_SIZE:
                raise ValueError(f"spawn_index {idx} out of range [0, {C.SPAWN_GRID_SIZE})")
            x, y, z = (float(v) for v in self._spawn_grid[idx])
            return x, y, z
        x = float(self.np_random.uniform(C.SPAWN_PATCH_X[0], C.SPAWN_PATCH_X[1]))
        y = float(self.np_random.uniform(C.SPAWN_PATCH_Y[0], C.SPAWN_PATCH_Y[1]))
        return x, y, C.SPAWN_Z

    def reset_model(self) -> np.ndarray:
        """Keyframe reset + explicit cube spawn + settle.

        Order matters (research Pattern 2 / Pitfall 1):
          1. mj_resetDataKeyframe(home) — sets BOTH qpos and ctrl;
             the cube freejoint pose comes back zero-padded to world origin
             (half inside the floor).
          2. Overwrite cube qpos with the selected spawn (grid entry or
             continuous sample; w-first quaternion, fixed yaw per D-03) and
             zero the cube qvel.
          3. mj_forward, then SETTLE_STEPS raw physics steps — settle is not
             a control step, so raw mj_step, not do_simulation.
          4. Assert the cube is at rest before returning (Pitfall 7 — never
             return an unsettled spawn).
        """
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, C.HOME_KEY)
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)

        x, y, z = self._sample_spawn()
        self.data.qpos[self._cube_qposadr : self._cube_qposadr + 3] = (x, y, z)
        self.data.qpos[self._cube_qposadr + 3 : self._cube_qposadr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[self._cube_dofadr : self._cube_dofadr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        for _ in range(C.SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)

        cube_qvel = np.asarray(self.data.qvel[self._cube_dofadr : self._cube_dofadr + 6])
        cube_qvel_norm = float(np.linalg.norm(cube_qvel))
        assert cube_qvel_norm < C.REST_QVEL_TOL, (
            "unsettled cube spawn: qvel norm "
            f"{cube_qvel_norm:.3e} >= {C.REST_QVEL_TOL:.3e} — never return a "
            "still-falling/bouncing spawn (Pitfall 7)"
        )

        # Episode-local stage state: a FRESH machine on every reset (Pitfall
        # 6 — flags, grasp-window counter, and the step counter must never
        # leak across episodes; the env is reused across trajectories by the
        # audit runner exactly as the physics gate reuses it across spawns).
        self._stage = stages.StageState()
        self._elapsed_steps = 0

        return self._get_obs()