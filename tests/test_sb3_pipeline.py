"""SB3 vectorized-pipeline contract tests (Plan 03-02, TRAIN-02 pipeline half).

Permanent in-process gate over the exact wrapper chain Phase 4's baseline
and Phases 6-7's evaluation/fine-tuning loops reuse (decision contract in
.planning/phases/03-observations-sb3-pipeline/03-CONTEXT.md):

  * D-11 — ImageOnlyFrameStack stacks ONLY the image key: (64, 64, 3) ->
    (64, 64, 12) and grayscale (64, 64, 1) -> (64, 64, 4) at
    n_stack=C.OBS_N_STACK=4, state (28,) untouched. SB3's plain
    VecFrameStack would stack EVERY key (RESEARCH Pitfall 3) — the D-12
    state-only fallback shape (28 * 4,) == (112,) is proven separately.
  * D-08 — VecNormalize(norm_obs_keys=[C.OBS_STATE_KEY]) normalizes the
    state key only; the uint8 image key stays out of obs_rms (SB3 divides
    images by 255 internally — double normalization silently ruins
    learning) and the running statistics round-trip exactly through
    save/load.
  * D-09 — checkpoint bundles travel as (model file + VecNormalize pkl)
    and reload in the STRICT order VecNormalize.load -> rebuild stack ->
    rebuild transpose -> PPO.load().set_env, reproducing actions bitwise
    (a.tobytes() == b.tobytes(), deterministic=True).
  * T-03-SC — the toolchain is pinned (stable-baselines3==2.9.0,
    tensorboard==2.21.0) and test_pinned_sb3_versions fails the suite on
    any version drift.

GL note (empirical, this phase): the FIRST render after an offscreen
viewer is created differs slightly from every later same-state render
(one-time GL warm-up); renders from the second onward are fully
deterministic — including across other live viewers. Every test below
resets its envs right after construction (the conftest fixture
convention), so all COMPARED frames come from the deterministic regime.
Vision envs are never closed mid-run (gymnasium 1.3.0's OffScreenViewer
close() terminates the process-wide GLFW state); they are left to GC.
"""

import importlib.metadata

import numpy as np
import pytest
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, VecTransposeImage
from torch.utils.tensorboard import SummaryWriter  # noqa: F401  (T-03-SC import proof)

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv
from so100rob.vec_wrappers import ImageOnlyFrameStack

# A strongly-moving arm action (position actuators head from the home
# keyframe toward these ctrl targets): guaranteed to change the rendered
# frame within a few steps, so stale-stack detection is never vacuous.
_ACTION = np.array([0.5, -1.0, 1.0, 1.0, -1.0, 0.13], dtype=np.float32)


def _vision_venv(n_envs: int = 2, grayscale: bool = False, **extra_kwargs):
    """DummyVecEnv of vision-mode envs, seeded, with the pinned 64x64 render."""
    return make_vec_env(
        SO100PickPlaceEnv,
        n_envs=n_envs,
        seed=0,
        env_kwargs=dict(
            obs_mode=C.OBS_MODE_VISION,
            render_mode="rgb_array",
            width=C.OBS_IMAGE_SIZE,
            height=C.OBS_IMAGE_SIZE,
            grayscale=grayscale,
            **extra_kwargs,
        ),
    )


def test_pinned_sb3_versions():
    """T-03-SC: the training toolchain is pinned and importable — any drift fails the suite."""
    assert importlib.metadata.version("stable-baselines3") == "2.9.0"
    assert importlib.metadata.version("tensorboard") == "2.21.0"
    assert stable_baselines3.__version__ == "2.9.0"
    assert SummaryWriter is not None  # tensorboard importable (Pitfall 6)


def test_stack_image_only_rgb():
    """D-11: vision (64,64,3) stacks to (64,64,12) at n_stack=4; state (2,28) unchanged; newest channel block == current frame."""
    raw = _vision_venv()
    stacked = ImageOnlyFrameStack(raw)
    # The wrapper rebuilds the Dict space with ONLY the image key replaced.
    assert stacked.observation_space[C.OBS_STATE_KEY].shape == (C.OBS_STATE_DIM,)
    assert stacked.observation_space[C.OBS_STATE_KEY].dtype == np.float32
    n_ch = 3
    assert (
        stacked.observation_space[C.OBS_IMAGE_KEY].shape
        == (C.OBS_IMAGE_SIZE, C.OBS_IMAGE_SIZE, n_ch * C.OBS_N_STACK)
    )
    obs = stacked.reset()
    assert set(obs) == {C.OBS_STATE_KEY, C.OBS_IMAGE_KEY}
    assert obs[C.OBS_STATE_KEY].shape == (2, C.OBS_STATE_DIM)
    img = obs[C.OBS_IMAGE_KEY]
    assert img.shape == (2, C.OBS_IMAGE_SIZE, C.OBS_IMAGE_SIZE, n_ch * C.OBS_N_STACK)
    assert img.dtype == np.uint8
    # SB3 reset semantics (verified against StackedObservations source):
    # older slots are ZERO-filled, the newest slot holds the first frame —
    # identical to plain VecFrameStack, so the D-12 fallback and this
    # wrapper share one reset contract.
    for k in range(C.OBS_N_STACK - 1):
        np.testing.assert_array_equal(
            img[0, :, :, k * n_ch : (k + 1) * n_ch],
            np.zeros((C.OBS_IMAGE_SIZE, C.OBS_IMAGE_SIZE, n_ch), dtype=np.uint8),
        )
    # After one step the NEWEST block equals the env's current frame
    # (fresh same-state render, deterministic regime).
    obs, _, dones, _ = stacked.step(np.tile(_ACTION, (2, 1)))
    assert not dones.any()
    current = raw.envs[0].unwrapped._get_obs()[C.OBS_IMAGE_KEY]
    np.testing.assert_array_equal(
        obs[C.OBS_IMAGE_KEY][0, :, :, -n_ch:], current,
        "newest channel block must be the current frame",
    )


def test_stack_image_only_grayscale():
    """D-11/A7: grayscale (64,64,1) stacks to (64,64,4); state (2,28) unchanged; the stacked space still satisfies SB3's image detection."""
    raw = _vision_venv(grayscale=True)
    stacked = ImageOnlyFrameStack(raw)
    obs = stacked.reset()
    assert obs[C.OBS_STATE_KEY].shape == (2, C.OBS_STATE_DIM)
    img = obs[C.OBS_IMAGE_KEY]
    assert img.shape == (2, C.OBS_IMAGE_SIZE, C.OBS_IMAGE_SIZE, 1 * C.OBS_N_STACK)
    assert img.dtype == np.uint8
    # A7: the 4-channel stacked grayscale space is still detected as an
    # image space — a mis-detected space would silently fall to a flat
    # MLP path instead of the CNN.
    assert is_image_space(stacked.observation_space[C.OBS_IMAGE_KEY])


def test_stack_state_passthrough():
    """Non-image keys pass through bit-exact: no stacking, no normalization, no dtype change."""
    raw_a = _vision_venv()
    raw_b = _vision_venv()  # lockstep reference
    stacked = ImageOnlyFrameStack(raw_a)
    obs_w = stacked.reset()
    obs_ref = raw_b.reset()
    assert obs_w[C.OBS_STATE_KEY].dtype == np.float32
    np.testing.assert_array_equal(obs_w[C.OBS_STATE_KEY], obs_ref[C.OBS_STATE_KEY])
    for _ in range(4):
        obs_w, _, dones_w, _ = stacked.step(np.tile(_ACTION, (2, 1)))
        obs_ref, _, dones_ref, _ = raw_b.step(np.tile(_ACTION, (2, 1)))
        assert not dones_w.any() and not dones_ref.any()
        assert obs_w[C.OBS_STATE_KEY].dtype == np.float32
        np.testing.assert_array_equal(
            obs_w[C.OBS_STATE_KEY], obs_ref[C.OBS_STATE_KEY],
            "state entries must equal the unwrapped env's observations",
        )


def test_no_cross_episode_bleed():
    """Pitfall 3: after a truncation the stack resets clean — older slots ZERO, newest slot the NEW episode's first frame; zero contribution from the previous episode."""
    # Short episodes (0.5 s / 0.05 s dt -> 10 control steps) so the run
    # truncates quickly.
    raw = _vision_venv(episode_sim_seconds=0.5)
    stacked = ImageOnlyFrameStack(raw)
    stacked.reset()
    last_frame = None
    dones = np.zeros(2, dtype=bool)
    steps = 0
    while not dones.any():
        obs, _, dones, _ = stacked.step(np.tile(_ACTION, (2, 1)))
        if not dones.any():
            # The stacker returns a shared rolling buffer — copy what we
            # keep, and only from a PRE-done step (the done step's obs is
            # already the auto-reset stack).
            last_frame = obs[C.OBS_IMAGE_KEY][0, :, :, -3:].copy()
        steps += 1
        assert steps < 50, "episode did not truncate within the expected cap"
    assert dones.all()  # both envs truncate at the same 10-step cap
    # The step's obs is the AUTO-RESET stack (DummyVecEnv resets on done).
    post = obs[C.OBS_IMAGE_KEY][0].copy()
    n_ch = post.shape[-1] // C.OBS_N_STACK
    frame = post[:, :, -n_ch:]
    # SB3 done-reset semantics: the older slots are zeroed, the newest
    # slot is the new episode's first frame — ZERO contribution from the
    # previous episode (the plan's Pitfall-3 intent, enforced literally).
    zeros = np.zeros_like(frame)
    for k in range(C.OBS_N_STACK - 1):
        np.testing.assert_array_equal(post[:, :, k * n_ch : (k + 1) * n_ch], zeros)
    assert frame.max() > 0, "the new first frame must be non-degenerate"
    # ... and that frame is exactly the underlying env's current,
    # freshly-reset state (same-state re-render, deterministic regime).
    current = raw.envs[0].unwrapped._get_obs()[C.OBS_IMAGE_KEY]
    np.testing.assert_array_equal(frame, current)
    # Non-vacuousness: the pre-truncation frame (moved-pose arm, non-zero)
    # differs from the fresh-reset frame (home pose), so a stale stack
    # could not have passed the zeroed-older-slots assertion by accident.
    assert last_frame.max() > 0
    assert not np.array_equal(last_frame, frame)
    # Explicit reset: the stack starts the next episode clean as well.
    obs = stacked.reset()
    post = obs[C.OBS_IMAGE_KEY][0].copy()
    frame = post[:, :, -n_ch:]
    for k in range(C.OBS_N_STACK - 1):
        np.testing.assert_array_equal(post[:, :, k * n_ch : (k + 1) * n_ch], zeros)
    np.testing.assert_array_equal(frame, current)


def test_state_mode_flat_fallback_shape():
    """D-12: the state-only fallback pipeline is plain SB3 VecFrameStack over the flat Box — shape (28*4,) == (112,)."""
    venv = make_vec_env(SO100PickPlaceEnv, n_envs=2, seed=0)
    stacked = VecFrameStack(venv, n_stack=C.OBS_N_STACK)
    assert stacked.observation_space.shape == (C.OBS_STATE_DIM * C.OBS_N_STACK,)
    assert stacked.observation_space.shape == (112,)
    obs = stacked.reset()
    assert obs.shape == (2, C.OBS_STATE_DIM * C.OBS_N_STACK)


def test_wrapper_rejects_bad_spaces():
    """Fail-loudly constructor contract: a non-Dict space or a missing image key raises ValueError naming the expected space."""
    flat_venv = make_vec_env(SO100PickPlaceEnv, n_envs=1, seed=0)
    with pytest.raises(ValueError, match="Dict"):
        ImageOnlyFrameStack(flat_venv)
    dict_venv = _vision_venv(n_envs=1)
    with pytest.raises(ValueError, match="image key"):
        ImageOnlyFrameStack(dict_venv, image_key="nope")


def _pipeline_chain():
    """The exact wrapper chain Phase 4 reuses (RESEARCH Pattern 3):

    make_vec_env -> VecNormalize(norm_obs_keys=[state]) ->
    ImageOnlyFrameStack -> VecTransposeImage. Returns (chain, vec_normalize).
    """
    venv = _vision_venv()
    # Pitfall 1: SB3 2.9.0 renamed the parameter to norm_obs_keys — the
    # default normalizes ALL keys, which would float-normalize the uint8
    # image key and corrupt the ENV-03 contract.
    norm = VecNormalize(
        venv,
        norm_obs_keys=[C.OBS_STATE_KEY],
        norm_obs=True,
        norm_reward=True,
        gamma=C.REWARD_GAMMA,  # Pitfall 11: both gamma sites read the constant
    )
    return VecTransposeImage(ImageOnlyFrameStack(norm)), norm


def test_train_save_reload_bitwise_and_stats_roundtrip(tmp_path):
    """D-08/D-09: tiny real PPO learn, checkpoint bundle, strict-order reload, bitwise-identical actions, exact VecNormalize stats round-trip (Pitfalls 4/5/7/9)."""
    chain, norm = _pipeline_chain()

    # Tiny budget with pinned divisibility (Pitfall 7): 768 total steps =
    # 3 rollout buffers of n_steps 128 x n_envs 2; batch 64 divides 256.
    model = PPO(
        "MultiInputPolicy",
        chain,
        n_steps=128,
        batch_size=64,
        n_epochs=2,
        gamma=C.REWARD_GAMMA,  # Pitfall 11: PPO's gamma reads the constant
        seed=0,
        device="cpu",
        policy_kwargs=dict(net_arch=[16]),
        verbose=0,  # no tensorboard_log here: the suite stays filesystem-clean (03-03 owns TB)
    )
    model.learn(total_timesteps=768)

    # D-08: the checkpoint bundle is (model file + VecNormalize pkl) —
    # the two ALWAYS travel together.
    model_path = str(tmp_path / "ppo_tiny")
    stats_path = str(tmp_path / "vecnormalize.pkl")
    model.save(model_path)
    norm.save(stats_path)

    # Stats snapshot BEFORE save is what must round-trip exactly.
    state_mean = norm.obs_rms[C.OBS_STATE_KEY].mean.copy()
    state_var = norm.obs_rms[C.OBS_STATE_KEY].var.copy()
    ret_mean = norm.ret_rms.mean.copy()
    ret_var = norm.ret_rms.var.copy()
    # Pitfall 9: the image key is NEVER in obs_rms — uint8 images are
    # normalized inside SB3 (preprocess_obs /255), never by VecNormalize.
    assert C.OBS_IMAGE_KEY not in norm.obs_rms

    # RELOAD in the strict order (Pitfall 4): fresh raw venv ->
    # VecNormalize.load -> rebuild stack -> rebuild transpose ->
    # PPO.load().set_env. Any other order is the classic silent break
    # (space mismatch) this test exists to catch.
    fresh_raw = _vision_venv()  # identical raw construction to training
    loaded_norm = VecNormalize.load(stats_path, fresh_raw)
    loaded_chain = VecTransposeImage(ImageOnlyFrameStack(loaded_norm))
    loaded_model = PPO.load(model_path, device="cpu")
    loaded_model.set_env(loaded_chain)

    # The loaded model's space keys are exactly the Dict contract's.
    assert set(loaded_model.observation_space.spaces.keys()) == {
        C.OBS_STATE_KEY,
        C.OBS_IMAGE_KEY,
    }

    # D-09 bitwise equality: fixed obs built in the LOADED model's own
    # post-stack post-transpose shapes — read from the space, never
    # hand-counted (Pitfall 5).
    fixed_obs = {
        C.OBS_STATE_KEY: np.zeros(
            (1, *loaded_model.observation_space[C.OBS_STATE_KEY].shape), dtype=np.float32
        ),
        C.OBS_IMAGE_KEY: np.zeros(
            (1, *loaded_model.observation_space[C.OBS_IMAGE_KEY].shape), dtype=np.uint8
        ),
    }
    a, _ = model.predict(fixed_obs, deterministic=True)
    b, _ = loaded_model.predict(fixed_obs, deterministic=True)
    assert np.asarray(a).tobytes() == np.asarray(b).tobytes()

    # D-08 stats round-trip: exact (np.array_equal, not allclose) on the
    # state key's obs_rms mean/var and on ret_rms; image still absent.
    assert np.array_equal(loaded_norm.obs_rms[C.OBS_STATE_KEY].mean, state_mean)
    assert np.array_equal(loaded_norm.obs_rms[C.OBS_STATE_KEY].var, state_var)
    assert np.array_equal(loaded_norm.ret_rms.mean, ret_mean)
    assert np.array_equal(loaded_norm.ret_rms.var, ret_var)
    assert C.OBS_IMAGE_KEY not in loaded_norm.obs_rms