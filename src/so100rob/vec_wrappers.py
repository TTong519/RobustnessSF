"""Vectorized-env wrappers for the SB3 training pipeline (Phase 3, TRAIN-02).

Decision contract implemented here (see
.planning/phases/03-observations-sb3-pipeline/03-CONTEXT.md):

  * D-11 — ImageOnlyFrameStack stacks ONLY the image key of the Dict
    observation: the raw uint8 channel-last frame (64, 64, C) becomes
    (64, 64, n_stack * C) while the state key passes through untouched.
    SB3's plain VecFrameStack stacks EVERY Dict key (state 28 -> 112
    dims, RESEARCH Pitfall 3) — redundant dimensions plus normalization
    statistics tracking a partially zero-padded vector — so the
    image-only variant is genuinely required. The stacking mechanics
    themselves are SB3's StackedObservations (channels_order=None ->
    channels-last detection), never hand-rolled: the ring buffer,
    done-reset semantics, and the low/high np.repeat space math are
    subtle and library-tested.
  * D-12 — the state-only fallback pipeline does NOT use this module:
    it wraps the flat (28,) Box in SB3's plain VecFrameStack(n_stack)
    (proven shape (112,) in tests/test_sb3_pipeline.py), so the
    fallback is a drop-in configuration change, not a re-engineered
    pipeline.

Constants discipline (Pitfall 11): stack depth and key names read from
so100rob.constants.OBS_* — never re-literalized here.
"""

from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper

from so100rob import constants as C


class ImageOnlyFrameStack(VecEnvWrapper):
    """D-11: stack ONLY the image key; every other key passes through.

    Reuses SB3's StackedObservations for the image key; never stacks the
    state key (its qvel blocks already carry motion history — stacking
    state is redundant dimensions). channels_order=None auto-detects
    channels-last for the HWC uint8 image space: (64, 64, 3) ->
    (64, 64, 12), grayscale (64, 64, 1) -> (64, 64, 4) at the pinned
    n_stack=C.OBS_N_STACK.
    """

    def __init__(
        self,
        venv,
        n_stack: int = C.OBS_N_STACK,
        image_key: str = C.OBS_IMAGE_KEY,
    ) -> None:
        # Fail-loudly contract validation BEFORE any machinery: this
        # wrapper is defined only over Dict observation spaces that
        # carry the image key. Anything else is a pipeline-construction
        # bug (wrong obs_mode, wrong wrapper order) and must not
        # silently proceed.
        if not isinstance(venv.observation_space, spaces.Dict):
            raise ValueError(
                "ImageOnlyFrameStack expects a Dict observation space with an "
                f"image key ({image_key!r}); got "
                f"{type(venv.observation_space).__name__}: {venv.observation_space}. "
                "The D-12 state-only fallback uses plain SB3 VecFrameStack over "
                "the flat Box instead — never this wrapper."
            )
        if image_key not in venv.observation_space.spaces:
            raise ValueError(
                f"image key {image_key!r} not found in the Dict observation "
                f"space; expected one of {sorted(venv.observation_space.spaces)} "
                f"(keys are pinned in so100rob.constants: "
                f"state={C.OBS_STATE_KEY!r}, image={C.OBS_IMAGE_KEY!r})"
            )
        super().__init__(venv)
        self.n_stack = n_stack
        self.image_key = image_key
        raise NotImplementedError(
            "ImageOnlyFrameStack stacking machinery not implemented yet — "
            "TDD RED stub (Plan 03-02 Task 1); the GREEN commit reuses SB3's "
            "StackedObservations for the image key only (D-11)."
        )

    # VecEnvWrapper is an ABC: reset/step_wait must exist for the stub to be
    # instantiable-into-failure. Both raise the RED NotImplementedError.
    def reset(self):  # noqa: D102 — stub, replaced by the GREEN commit
        raise NotImplementedError("RED stub: stacking machinery not implemented (D-11)")

    def step_wait(self):  # noqa: D102 — stub, replaced by the GREEN commit
        raise NotImplementedError("RED stub: stacking machinery not implemented (D-11)")