"""SO100PickPlaceEnv — Phase 1 walking-skeleton Gymnasium environment.

Gymnasium MujocoEnv subclass over the L1 task scene
(src/so100rob/assets/scene_pickplace.xml, which includes the read-only
upstream SO-ARM100 model). Phase 1 scope is deliberately minimal:

  * keyframe ("home") reset + explicit cube spawn + settle (verified recipe,
    research Pattern 2 — the keyframe zero-pads the cube to world origin,
    so the overwrite is MANDATORY)
  * step() is a pure physics delegate — no reward, no stage flags, no
    termination logic (Phase 2 scope)
  * observation is a minimal flat state vector — the full obs pipeline
    (images) is Phase 3 scope

The control update rate is the constructor-time frame_skip (ENV-04): the
physics timestep is pinned in the scene (0.002 s) and a "rate change" is
exactly a decimation change, never a model change.
"""

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

from so100rob import constants as C

# Minimal Phase 1 observation: qpos (13) + qvel (12) + 2D XY offset of the
# cube center from the goal position.
_OBS_DIM = 13 + 12 + 2


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
        episode_sim_seconds: float = C.EPISODE_SIM_SECONDS,
    ) -> None:
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

        # Sim-second episode cap (ENV-04): derived from dt so different
        # frame_skip values yield identical sim-time episodes (D-07).
        self.max_episode_steps = int(np.ceil(episode_sim_seconds / self.dt))

        # Phase 1: grid mode always spawns at the patch center (the checked-in
        # 25-spawn seed file lands in Plan 02; continuous mode in a later plan).
        if spawn_mode not in ("grid",):
            raise ValueError(f"unsupported spawn_mode: {spawn_mode!r}")
        self.spawn_mode = spawn_mode

        # Cached cube joint addressing (constant for the compiled model).
        self._cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, C.CUBE_JOINT
        )
        self._cube_qposadr = self.model.jnt_qposadr[self._cube_joint_id]
        self._cube_dofadr = self.model.jnt_dofadr[self._cube_joint_id]

        self._goal_xy = np.asarray(C.GOAL_POS[:2], dtype=np.float64)

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

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.do_simulation(action, self.frame_skip)
        return self._get_obs(), 0.0, False, False, {}

    def reset_model(self) -> np.ndarray:
        """Keyframe reset + explicit cube spawn + settle.

        Order matters (research Pattern 2 / Pitfall 1):
          1. mj_resetDataKeyframe(home) — sets BOTH qpos and ctrl;
             the cube freejoint pose comes back zero-padded to world origin
             (half inside the floor).
          2. Overwrite cube qpos (w-first quaternion, fixed yaw per D-03) and
             zero the cube qvel.
          3. mj_forward, then SETTLE_STEPS raw physics steps — settle is not
             a control step, so raw mj_step, not do_simulation.
          4. Assert the cube is at rest before returning (Pitfall 7 — never
             return an unsettled spawn).
        """
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, C.HOME_KEY)
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)

        self.data.qpos[self._cube_qposadr : self._cube_qposadr + 3] = C.SPAWN_CENTER
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

        return self._get_obs()