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
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations

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
        # SB3's own stacking machinery, one instance for the image key only
        # (D-11): ring buffer, done-reset semantics, and the np.repeat space
        # math are library-tested — never hand-rolled. channels_order=None
        # auto-detects channels-last for the HWC uint8 image space.
        self.stacked = StackedObservations(
            venv.num_envs, n_stack, venv.observation_space[image_key], channels_order=None
        )
        # The wrapper's space is the venv's Dict with ONLY the image key
        # replaced by the stacked image space.
        new_spaces = dict(venv.observation_space.spaces)
        new_spaces[image_key] = self.stacked.stacked_observation_space
        self.observation_space = spaces.Dict(new_spaces)

    def reset(self):
        """Reset the underlying venv, then the image stack with the first frame.

        SB3's stacker zero-fills the older slots and places the reset
        frame in the NEWEST slot — identical to plain VecFrameStack, so
        the D-12 fallback and this wrapper share one reset semantics. A
        fresh episode NEVER inherits frames from the previous one
        (Pitfall 3): the stack is zeroed before the new frame lands.
        """
        obs = self.venv.reset()
        stacked_img = self.stacked.reset(obs[self.image_key])
        obs = {**obs, self.image_key: stacked_img}
        return obs

    def step_wait(self):
        """Step the venv; delegate ONLY the image key to the stacker.

        StackedObservations.update consumes the dones itself: for each
        done env it zeroes that env's stack and places the auto-reset
        frame in the newest slot, so the post-done obs carries zero
        contribution from the previous episode (Pitfall 3). Every other
        key passes through untouched.

        The stacker is constructed over the image Box subspace, so the
        per-env ``terminal_observation`` (a Dict) must be split into the
        image key before it enters the stacker and the stacked terminal
        image written back afterwards — exactly the sub-info plumbing
        SB3's own Dict branch performs, applied to the single key
        (skeleton fix: passing the full Dict infos into the Box-space
        stacker crashes on the first episode termination).
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        sub_infos = [
            {"terminal_observation": info["terminal_observation"][self.image_key]}
            if "terminal_observation" in info
            else {}
            for info in infos
        ]
        stacked_img, sub_infos = self.stacked.update(obs[self.image_key], dones, sub_infos)
        for env_idx, info in enumerate(infos):
            if "terminal_observation" in info:
                info["terminal_observation"][self.image_key] = sub_infos[env_idx][
                    "terminal_observation"
                ]
        obs = {**obs, self.image_key: stacked_img}
        return obs, rewards, dones, infos