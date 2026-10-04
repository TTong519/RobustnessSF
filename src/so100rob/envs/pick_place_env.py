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
  * observation is obs_mode-parameterized (ENV-03, Phase 3): default
    (D-12) state mode emits a flat float32 Box with the pinned 28-dim
    D-01/D-02/D-03 state vector; opt-in obs_mode="vision" (requires
    render_mode="rgb_array") emits a spaces.Dict{state, image} with a
    raw uint8 channel-last 64x64 image key (grayscale optional)

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
from so100rob import reward
from so100rob import stages
from so100rob.scripted_policy import grasp_point


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
        x, y, z = coords
        if not C.SPAWN_PATCH_X[0] <= x <= C.SPAWN_PATCH_X[1]:
            raise ValueError(
                f"{path}: spawn entry {i} x={x} outside the patch x-bounds {C.SPAWN_PATCH_X}"
            )
        if not C.SPAWN_PATCH_Y[0] <= y <= C.SPAWN_PATCH_Y[1]:
            raise ValueError(
                f"{path}: spawn entry {i} y={y} outside the patch y-bounds {C.SPAWN_PATCH_Y}"
            )
        # z is validated too (T-01-03 full-schema): a corrupted z would
        # otherwise flow straight into reset_model's cube overwrite — a
        # grossly wrong z fails only later at the settle guard far from
        # the root cause, and a mildly wrong z can settle to rest and
        # silently pass. The pinned spawn z is SPAWN_Z (~0.5 mm above the
        # floor); the accepted band runs from the rest-band floor to it.
        if not C.CUBE_REST_Z_RANGE[0] <= z <= C.SPAWN_Z + 1e-9:
            raise ValueError(
                f"{path}: spawn entry {i} z={z} outside the accepted spawn z band "
                f"[{C.CUBE_REST_Z_RANGE[0]}, {C.SPAWN_Z}]"
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
        obs_mode: str = C.OBS_MODE_STATE,
        grayscale: bool = False,
    ) -> None:
        # Spawn mode + obs mode validation BEFORE the (slow) model compile:
        # a malformed frozen-data file or a bad obs_mode fails fast and
        # loudly (T-01-03; the spawn_mode fail-fast pattern).
        if spawn_mode not in ("grid", "continuous"):
            raise ValueError(
                f"unsupported spawn_mode: {spawn_mode!r} (expected 'grid' or 'continuous')"
            )
        if obs_mode not in (C.OBS_MODE_STATE, C.OBS_MODE_VISION):
            raise ValueError(
                f"unsupported obs_mode: {obs_mode!r} "
                f"(expected {C.OBS_MODE_STATE!r} or {C.OBS_MODE_VISION!r})"
            )
        if obs_mode == C.OBS_MODE_VISION and render_mode != "rgb_array":
            raise ValueError(
                "obs_mode='vision' requires render_mode='rgb_array' "
                f"(got render_mode={render_mode!r}) — the image observation "
                "is rendered through the env's offscreen renderer"
            )
        self.spawn_mode = spawn_mode
        self._obs_mode = obs_mode
        self._grayscale = grayscale
        if spawn_mode == "grid":
            self.spawn_seeds_path = (
                Path(spawn_seeds_path) if spawn_seeds_path is not None else C.SPAWN_SEEDS_PATH
            )
            self._spawn_grid = load_spawn_grid(self.spawn_seeds_path)

        # Observation space per obs_mode (ENV-03), built in a local BEFORE
        # super().__init__ (MujocoEnv simply assigns the space we pass).
        # The DEFAULT stays flat-state (D-12): scripts/benchmark.py,
        # run_reward_audit.py, and run_physics_gate.py construct the env
        # with no obs_mode, so their Phase 1/2 measured numbers stay
        # reproducible; vision is opt-in.
        _state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(C.OBS_STATE_DIM,), dtype=np.float32
        )
        if obs_mode == C.OBS_MODE_VISION:
            _n_channels = 1 if grayscale else 3
            observation_space: spaces.Space = spaces.Dict(
                {
                    C.OBS_STATE_KEY: _state_space,
                    C.OBS_IMAGE_KEY: spaces.Box(
                        low=0, high=255, shape=(height, width, _n_channels), dtype=np.uint8
                    ),
                }
            )
        else:
            observation_space = _state_space

        super().__init__(
            model_path=str(C.SCENE_XML_PATH),
            frame_skip=frame_skip,
            observation_space=observation_space,
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
        # Fail loudly on a missing cube joint (the same name-resolution
        # pattern as the Jaw and arm-joint checks below): mj_name2id
        # returns -1 for an unknown name, and a -1 index would silently
        # wrap to the LAST joint's qpos/dof address via numpy negative
        # indexing — every cube block would read the wrong physics with
        # no crash and no failed finite guard (the "COSTLY" silent-
        # corruption class).
        self._cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, C.CUBE_JOINT
        )
        if self._cube_joint_id < 0:
            raise ValueError(f"cube joint {C.CUBE_JOINT!r} not found in compiled model")
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

        # Fail-loudly arm-joint verification (the same name-resolution
        # pattern as the cube and Jaw lookups): _get_obs reads the arm
        # blocks as the positional slices qpos[:6]/qvel[:6], so
        # construction must PROVE the compiled model's first six
        # qpos/qvel rows are exactly C.JOINT_NAMES in order. If the
        # read-only upstream XML ever gained or reordered joints, the env
        # must refuse to construct — otherwise the observation would
        # silently read the wrong physics (no crash; the finite guard
        # passes) and invalidate every downstream VecNormalize/PPO
        # checkpoint, the class the constants file labels "COSTLY".
        arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in C.JOINT_NAMES
        ]
        if any(jid < 0 for jid in arm_joint_ids):
            missing = [n for n, jid in zip(C.JOINT_NAMES, arm_joint_ids) if jid < 0]
            raise ValueError(f"arm joints not found in compiled model: {missing}")
        arm_qposadr = [int(self.model.jnt_qposadr[jid]) for jid in arm_joint_ids]
        arm_dofadr = [int(self.model.jnt_dofadr[jid]) for jid in arm_joint_ids]
        if arm_qposadr != list(range(len(C.JOINT_NAMES))) or arm_dofadr != list(
            range(len(C.JOINT_NAMES))
        ):
            raise ValueError(
                f"arm joints {C.JOINT_NAMES} are not the compiled model's first "
                f"{len(C.JOINT_NAMES)} qpos/qvel rows (qpos addresses "
                f"{arm_qposadr}, dof addresses {arm_dofadr}) — the "
                f"qpos[:{len(C.JOINT_NAMES)}]/qvel[:{len(C.JOINT_NAMES)}] "
                f"observation blocks would read the wrong physics"
            )

        # Episode-local stage/reward state placeholders — reset_model owns
        # the authoritative clearing (Pitfall 6); these exist so the
        # attributes are defined even before the first reset.
        # Cached home keyframe id, resolved BY NAME and cached once (the
        # same fail-loud name-resolution pattern as the cube, Jaw, and
        # arm-joint checks above). reset_model feeds this id to
        # mj_resetDataKeyframe on every reset: a -1 keyframe id makes it
        # silently degrade to a plain mj_resetData (verified against the
        # pinned mujoco 3.12.0), zeroing all qpos AND ctrl, so every
        # episode would start from a corrupt all-zero arm pose with no
        # crash and no failed settle guard (the settle check reads only
        # the cube qvel).
        self._home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, C.HOME_KEY
        )
        if self._home_key_id < 0:
            raise ValueError(f"keyframe {C.HOME_KEY!r} not found in compiled model")

        self._stage = stages.StageState()
        self._elapsed_steps = 0
        self._terminal_paid = False
        self._prev_phi = 0.0

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self):
        """Assemble the observation for the env's obs_mode (ENV-03, Phase 3).

        The 28-dim state vector (D-01/D-02/D-03) is mode-independent: state
        mode returns it flat, vision mode returns it as the Dict's "state"
        entry with the rendered "image" key purely additive. Assembly draws
        NO RNG (Pitfall 10 — check_env's seeded-reset determinism depends on
        it) and raises ValueError on any non-finite block so NaN/Inf physics
        fails loudly instead of poisoning the observation (T-03-01).
        """
        qpos = self.data.qpos
        qvel = self.data.qvel
        adr = self._cube_qposadr

        arm_qpos = np.asarray(qpos[:6], dtype=np.float64)
        arm_qvel = np.asarray(qvel[:6], dtype=np.float64)
        cube_pos = np.asarray(qpos[adr : adr + 3], dtype=np.float64)
        # Cube yaw from the w-first (scalar-first) freejoint quaternion
        # (D-01): the 4-dim quaternion itself NEVER enters the vector.
        w, x, y, z = (float(v) for v in qpos[adr + 3 : adr + 7])
        cube_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cube_qvel = np.asarray(qvel[self._cube_dofadr : self._cube_dofadr + 6], dtype=np.float64)
        pad_midpoint = np.asarray(grasp_point(self.model, self.data), dtype=np.float64)
        gripper_to_cube = pad_midpoint - cube_pos
        cube_to_goal = cube_pos - np.asarray(C.GOAL_POS, dtype=np.float64)

        # Fail-loudly finite guard (the _gather_readout style, threat
        # T-03-01): non-finite physics must never become an observation.
        # An explicit raise, NOT a bare assert — python -O strips asserts,
        # which would silently disable this guard exactly where Phase 4/5
        # run the env.
        if not (
            all(
                np.isfinite(v).all()
                for v in (
                    arm_qpos,
                    arm_qvel,
                    cube_pos,
                    cube_qvel,
                    gripper_to_cube,
                    cube_to_goal,
                )
            )
            and math.isfinite(cube_yaw)
        ):
            raise ValueError(
                "non-finite simulation state in the observation — NaN/Inf "
                "physics must fail loudly, never poison the observation (T-03-01)"
            )

        state = np.concatenate(
            [
                arm_qpos,          # [0:6]   arm qpos (incl. Jaw aperture)
                arm_qvel,          # [6:12]  arm qvel
                cube_pos,          # [12:15] cube position (D-01)
                [cube_yaw],        # [15]    cube yaw (D-01)
                cube_qvel,         # [16:22] cube qvel, 6D (D-03)
                gripper_to_cube,   # [22:25] gripper->cube 3D offset (D-02)
                cube_to_goal,      # [25:28] cube->goal 3D offset (D-02)
            ]
        ).astype(np.float32)

        if self._obs_mode == C.OBS_MODE_STATE:
            return state

        # Vision branch (ENV-03): the gymnasium offscreen renderer returns a
        # vertically-flipped NEGATIVE-STRIDE view that torch.from_numpy
        # rejects — one np.ascontiguousarray copy at the env boundary
        # (Pitfall 2). Frames stay raw uint8 [0, 255]: SB3 divides image
        # observations by 255 internally, so pre-normalizing here would
        # double-normalize and silently ruin learning (Pitfall 9).
        frame = np.ascontiguousarray(self.render())
        if self._grayscale:
            # Rec.601 luma (same weights as scripts/benchmark.py's timed
            # conversion). The trailing channel is KEPT — a squeezed (H, W)
            # frame is a 2-dim Box that fails SB3's is_image_space.
            gray = np.rint(frame.astype(np.float32) @ np.asarray(C.OBS_LUMA, dtype=np.float32))
            frame = np.ascontiguousarray(
                gray.clip(0, 255).astype(np.uint8)[:, :, None]
            )
        return {C.OBS_STATE_KEY: state, C.OBS_IMAGE_KEY: frame}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def _gather_readout(self, action) -> stages.SimReadout:
        """Snapshot the privileged kinematics the stage machine consumes.

        Read from the cached name-resolved cube addresses plus
        grasp_point(model, data) and the applied action's jaw ctrl. The
        readout is guarded with np.isfinite raises (threat T-02-01) —
        NaN/Inf physics state fails loudly here instead of silently
        becoming stage flags or reward.
        """
        cube_pos = np.array(self.data.qpos[self._cube_qposadr : self._cube_qposadr + 3])
        cube_qvel = np.array(self.data.qvel[self._cube_dofadr : self._cube_dofadr + 6])
        pad_midpoint = np.asarray(grasp_point(self.model, self.data), dtype=float)
        jaw_ctrl_applied = float(np.asarray(action).reshape(-1)[self._jaw_act_idx])
        approach_dist_m = float(np.linalg.norm(pad_midpoint[:2] - cube_pos[:2]))

        # Fail-loudly finite guard (threat T-02-01): an explicit raise,
        # NOT a bare assert — python -O strips asserts and would silently
        # let NaN/Inf physics become stage flags and reward.
        if not (
            all(
                np.isfinite(v).all() if isinstance(v, np.ndarray) else np.isfinite(v)
                for v in (cube_pos, cube_qvel, pad_midpoint)
            )
            and math.isfinite(jaw_ctrl_applied)
            and math.isfinite(approach_dist_m)
        ):
            raise ValueError(
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

    def step(self, action) -> tuple[np.ndarray | dict, float, bool, bool, dict]:
        self.do_simulation(action, self.frame_skip)
        self._elapsed_steps += 1

        readout = self._gather_readout(action)
        stages.update(self._stage, readout)

        # Timeout is truncation, NEVER termination (Pitfall 2 — PPO
        # bootstrapping depends on the distinction); success terminates on
        # the very step the place flag first sets.
        terminated = bool(self._stage.place)
        truncated = (not terminated) and (self._elapsed_steps >= self.max_episode_steps)

        # Potential-based stage-gated reward (D-05/D-06): one-shot terminal
        # latch — first_place is True only on the step place first sets, so
        # TERMINAL_PLACE_BONUS lands exactly once, on the terminated step.
        first_place = self._stage.place and not self._terminal_paid
        self._terminal_paid = bool(self._stage.place)
        phi_now = reward.potential(stages.stage_index(self._stage), readout.approach_dist_m)
        step_reward, components = reward.compute(self._prev_phi, phi_now, first_place)
        self._prev_phi = phi_now

        # Flat scalar info contract (cross-phase API: Phase 3 wrappers and
        # Phase 5 funnel metrics consume these keys BY NAME). is_success
        # mirrors the place flag EVERY step (Pitfall 7 — monotone episode
        # view, never final-step-only); reward components + Phi mirror into
        # info per D-07 so audits attribute reward without recomputation.
        info: dict = {
            "stage_approach": bool(self._stage.approach),
            "stage_grasp": bool(self._stage.grasp),
            "stage_lift": bool(self._stage.lift),
            "stage_place": bool(self._stage.place),
            "stage_index": stages.stage_index(self._stage),
            "is_success": bool(self._stage.place),
        }
        info.update(components)
        info["reward_phi"] = phi_now
        return self._get_obs(), float(step_reward), terminated, truncated, info

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

    def reset_model(self) -> np.ndarray | dict:
        """Keyframe reset + explicit cube spawn + settle.

        Order matters (research Pattern 2 / Pitfall 1):
          1. mj_resetDataKeyframe(home) — sets BOTH qpos and ctrl;
             the cube freejoint pose comes back zero-padded to world origin
             (half inside the floor). The keyframe id was resolved and
             checked once in the constructor (_home_key_id).
          2. Overwrite cube qpos with the selected spawn (grid entry or
             continuous sample; w-first quaternion, fixed yaw per D-03) and
             zero the cube qvel.
          3. mj_forward, then SETTLE_STEPS raw physics steps — settle is not
             a control step, so raw mj_step, not do_simulation.
          4. Verify the cube is at rest before returning (Pitfall 7 — never
             return an unsettled spawn; raises if not).
        """
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        x, y, z = self._sample_spawn()
        self.data.qpos[self._cube_qposadr : self._cube_qposadr + 3] = (x, y, z)
        self.data.qpos[self._cube_qposadr + 3 : self._cube_qposadr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[self._cube_dofadr : self._cube_dofadr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        for _ in range(C.SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)

        cube_qvel = np.asarray(self.data.qvel[self._cube_dofadr : self._cube_dofadr + 6])
        cube_qvel_norm = float(np.linalg.norm(cube_qvel))
        # Pitfall 7 guard as an explicit raise, NOT a bare assert (python
        # -O strips asserts — the env is the primary guard for Phase 4/5).
        if not cube_qvel_norm < C.REST_QVEL_TOL:
            raise ValueError(
                "unsettled cube spawn: qvel norm "
                f"{cube_qvel_norm:.3e} >= {C.REST_QVEL_TOL:.3e} — never return a "
                "still-falling/bouncing spawn (Pitfall 7)"
            )

        # Episode-local stage/reward state: a FRESH machine on every reset
        # (Pitfall 6 — flags, grasp-window counter, step counter, terminal
        # latch, and the prev-Phi snapshot must never leak across episodes;
        # the env is reused across trajectories by the audit runner exactly
        # as the physics gate reuses it across spawns).
        self._stage = stages.StageState()
        self._elapsed_steps = 0
        self._terminal_paid = False
        # Seed prev-Phi at the home-pose approach distance so the FIRST
        # step's shaping is a true potential difference, not a jump bonus.
        # After the settle the arm sits at the home keyframe with home ctrl.
        seed_readout = self._gather_readout(np.asarray(C.HOME_CTRL, dtype=float))
        self._prev_phi = reward.potential(
            stages.stage_index(self._stage), seed_readout.approach_dist_m
        )

        return self._get_obs()