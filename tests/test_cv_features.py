"""Classical-CV color-blob centroid feature tests (Plan 04-08, Task 1).

Pin the 04-08 CV observation contract — the user's option-2 D-03 decision
(made 2026-10-23, ranked option 2 in runs/investigation-04-zclause.md
section 5; mechanism per runs/investigation-04-cve-groundwork.md):

  * the pure perception functions (so100rob.cv_features) — synthetic-frame
    exactness (the endpoint contract u = mean_col/(W-1), v =
    mean_row/(H-1); the pinned neutral for an empty mask; exact closed
    uint8 threshold semantics) and the PURITY contract (T-04-23: a pure
    function of the rendered RGB frame — the module can import neither the
    physics bindings nor any env module, so the perception path cannot read
    simulator state even by accident; the state vector already carries the
    TRUE cube pose, so a privileged centroid would test nothing);
  * the env contract (SO100PickPlaceEnv(cv_features=True) requires vision
    mode, grows the state key 28 -> 32 with the 4-dim centroid tail, and
    leaves the legacy 28-dim default/state/vision contracts untouched);
  * the cross-mode prefix law (the first 28 dims are bit-identical to a
    state-mode reset with the same seed — the tail is purely additive);
  * the chain shape (VecNormalize(norm_obs_keys=['state']) -> stack ->
    transpose passes the 32-dim float32 state through; obs_rms tracks 32;
    the uint8 image key stays out of obs_rms — the D-08 discipline);
  * the blast-radius honesty (32-dim VecNormalize stats FAIL LOUDLY on a
    legacy 28-dim venv — the 04-05/04-06/04-07 failed-baseline bundles can
    never silently load on the CV chain);
  * the tamper guards (the identity split OBS_STATE_DIM=28 legacy /
    OBS_STATE_CV_DIM=32 training contract, the probe-measured bounds, the
    neutral, and the untouched 04-07 predicate pins);
  * the judged chains match (train.resolve_env_kwargs AND
    _eval_chain.build_eval_venv both construct cv_features=True — T-04-25:
    the two chains that judge the baseline share one contract).

RED contract (#3770, the established missing-artifact pattern): every test
that consumes a not-yet-existing artifact converts its absence into an
in-body NAMED failure — the module import, the constants read (getattr
with a failing default), and the env kwarg construction (try/except
TypeError) all pytest.fail citing the 04-08 contract; nothing fails as a
collection or import error.

GL discipline (WR-01, suite-wide): vision envs are NEVER closed
mid-process (gymnasium 1.3.0's OffScreenViewer.close() terminates
process-global GLFW state); only renderer-less state-mode envs are closed.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecTransposeImage

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv
from so100rob.vec_wrappers import ImageOnlyFrameStack

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CV_SRC = REPO_ROOT / "src" / "so100rob" / "cv_features.py"


# ---------------------------------------------------------------------------
# RED-safe artifact loaders (named failures, never collection errors)
# ---------------------------------------------------------------------------


def _cv_features():
    """Import so100rob.cv_features — the 04-08 pure-perception module."""
    try:
        import so100rob.cv_features as cvf

        return cvf
    except ImportError as exc:
        pytest.fail(
            "so100rob.cv_features does not exist yet — RED: the pure "
            "color-blob centroid module is the missing 04-08 artifact "
            f"(ImportError: {exc})"
        )


def _pin(name: str):
    """Read a 04-08 observation constant — getattr with a failing default."""
    value = getattr(C, name, None)
    if value is None:
        pytest.fail(
            f"constants.{name} is missing — RED: the 04-08 CV observation "
            "contract is not pinned in the observation block"
        )
    return value


def _cv_env_kwargs(**overrides) -> dict:
    """The CV-mode vision construction kwargs (the OBS_DECISION format + cv)."""
    kwargs = dict(
        obs_mode=C.OBS_MODE_VISION,
        render_mode="rgb_array",
        width=C.OBS_IMAGE_SIZE,
        height=C.OBS_IMAGE_SIZE,
        grayscale=True,
        cv_features=True,
    )
    kwargs.update(overrides)
    return kwargs


def _make_cv_env(**overrides):
    """Construct the CV-mode env — TypeError converts to a named failure."""
    try:
        return SO100PickPlaceEnv(**_cv_env_kwargs(**overrides))
    except TypeError as exc:
        pytest.fail(
            "SO100PickPlaceEnv has no cv_features kwarg — RED: the 04-08 "
            "CV observation contract does not exist in the env constructor "
            f"(TypeError: {exc})"
        )


def _make_cv_venv(n_envs: int = 2):
    """A CV-mode vision vec-env (the hardened construction's env_kwargs)."""
    try:
        return make_vec_env(
            SO100PickPlaceEnv, n_envs=n_envs, seed=0, env_kwargs=_cv_env_kwargs()
        )
    except TypeError as exc:
        pytest.fail(
            "SO100PickPlaceEnv has no cv_features kwarg — RED: the 04-08 "
            "CV observation contract does not exist (the vec-env factory "
            f"raised TypeError: {exc})"
        )


def _load_train():
    path = SCRIPTS_DIR / "train.py"
    if not path.exists():
        pytest.fail("scripts/train.py does not exist")
    spec = importlib.util.spec_from_file_location("so100rob_train_runner_0408", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_eval_chain():
    if not (SCRIPTS_DIR / "_eval_chain.py").exists():
        pytest.fail("scripts/_eval_chain.py does not exist")
    import sys

    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import _eval_chain as ec

    return ec


# ---------------------------------------------------------------------------
# Test 1 — synthetic-frame correctness (the pure functions)
# ---------------------------------------------------------------------------


def test_extract_centroids_synthetic_frames():
    """Planted in-bounds blobs at known coordinates give exactly
    u = mean_col/(W-1), v = mean_row/(H-1); the endpoints are exact; an
    empty mask gives the pinned neutral CV_BLOB_MISSING_UV (never NaN,
    never a raise); a pixel one step OUTSIDE a pinned bound is excluded
    (exact closed uint8 thresholds, no tolerance)."""
    cvf = _cv_features()
    H = W = 16
    # Colors INSIDE the probe-measured pinned bounds (probe artifact
    # runs/cv_probe/<ts>/probe.json: cube (80,66,14)-(255,217,55),
    # goal (69,17,22)-(224,38,51)) on a zero background (outside both).
    cube_color = (200, 150, 40)
    goal_color = (220, 30, 40)

    frame = np.zeros((H, W, 3), dtype=np.uint8)
    cube_pixels = [(4, 3), (4, 4), (5, 4)]  # (row, col)
    for r, c in cube_pixels:
        frame[r, c] = cube_color
    frame[12, 10] = goal_color

    out = cvf.extract_centroids(frame)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)
    assert out.dtype == np.float32
    exp_cube_u = float(np.mean([3, 4, 4])) / (W - 1)
    exp_cube_v = float(np.mean([4, 4, 5])) / (H - 1)
    assert out[0] == pytest.approx(exp_cube_u, abs=1e-12)
    assert out[1] == pytest.approx(exp_cube_v, abs=1e-12)
    assert out[2] == pytest.approx(10 / (W - 1), abs=1e-12)
    assert out[3] == pytest.approx(12 / (H - 1), abs=1e-12)

    # The exact endpoint contract: a single cube pixel at (0, 0) gives
    # (0.0, 0.0) and at (W-1, H-1) gives (1.0, 1.0).
    corner = np.zeros((H, W, 3), dtype=np.uint8)
    corner[0, 0] = cube_color
    o = cvf.extract_centroids(corner)
    assert o[0] == 0.0 and o[1] == 0.0
    corner = np.zeros((H, W, 3), dtype=np.uint8)
    corner[H - 1, W - 1] = cube_color
    o = cvf.extract_centroids(corner)
    assert o[0] == 1.0 and o[1] == 1.0

    # An empty cube region gives the pinned neutral for the cube channels
    # — no NaN, no raise (the finite guard covers the tail downstream).
    missing = _pin("CV_BLOB_MISSING_UV")
    assert missing == (0.5, 0.5)
    only_goal = np.zeros((H, W, 3), dtype=np.uint8)
    only_goal[8, 8] = goal_color
    o = cvf.extract_centroids(only_goal)
    assert o[0] == missing[0] and o[1] == missing[1]
    assert o[2] == pytest.approx(8 / (W - 1), abs=1e-12)
    assert np.isfinite(o).all()

    # Exact uint8 threshold semantics: one step outside a pinned bound is
    # EXCLUDED — no tolerance, anti-aliased edges are deterministic.
    cube_hi = _pin("CV_CUBE_RGB_HI")
    just_out_hi = list(cube_color)
    just_out_hi[2] = int(cube_hi[2]) + 1  # B one above the pinned hi
    frame_out = np.zeros((H, W, 3), dtype=np.uint8)
    frame_out[4, 4] = tuple(just_out_hi)
    o = cvf.extract_centroids(frame_out)
    assert o[0] == missing[0] and o[1] == missing[1], (
        "a pixel one step outside the pinned cube hi bound must fall "
        "outside the mask (exact closed thresholds)"
    )
    cube_lo = _pin("CV_CUBE_RGB_LO")
    just_below_lo = list(cube_color)
    just_below_lo[0] = int(cube_lo[0]) - 1  # R one below the pinned lo
    frame_out = np.zeros((H, W, 3), dtype=np.uint8)
    frame_out[4, 4] = tuple(just_below_lo)
    o = cvf.extract_centroids(frame_out)
    assert o[0] == missing[0] and o[1] == missing[1]


def test_blob_centroid_and_masks_pure_contracts():
    """cube_mask/goal_mask are per-channel boolean tests of the pinned
    bounds; blob_centroid returns float64 (u, v) normalized by (W-1)/(H-1),
    with the pinned neutral on an empty mask."""
    cvf = _cv_features()
    H = W = 8
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[2, 5] = (200, 150, 40)  # inside the cube bounds only
    cm = cvf.cube_mask(frame)
    gm = cvf.goal_mask(frame)
    assert cm.dtype == np.bool_ and cm.shape == (H, W)
    assert cm.sum() == 1 and cm[2, 5]
    assert gm.sum() == 0

    u, v = cvf.blob_centroid(cm)
    assert isinstance(u, float) and isinstance(v, float)
    assert u == pytest.approx(5 / (W - 1), abs=1e-12)
    assert v == pytest.approx(2 / (H - 1), abs=1e-12)

    empty = np.zeros((H, W), dtype=bool)
    u, v = cvf.blob_centroid(empty)
    assert (u, v) == C.CV_BLOB_MISSING_UV


# ---------------------------------------------------------------------------
# Test 2 — purity (T-04-23: never privileged simulator state)
# ---------------------------------------------------------------------------


def test_cv_features_purity_source_scan():
    """T-04-23: cv_features is a pure function of the rendered RGB frame —
    a structural source scan asserts the module imports neither the
    physics bindings, any env module, nor the scripted policy; a privileged
    centroid would silently turn the perception experiment into a tautology."""
    _cv_features()  # RED: the module must exist
    if not CV_SRC.exists():
        pytest.fail(f"{CV_SRC} does not exist — RED: the 04-08 module is missing")
    src = CV_SRC.read_text()
    assert "import mujoco" not in src, "cv_features must not import the physics bindings"
    assert "mujoco" not in src.replace("import mujoco", ""), (
        "cv_features must not reference the physics bindings at all"
    )
    assert "so100rob.envs" not in src, "cv_features must not import any env module"
    assert "scripted_policy" not in src, "cv_features must not import the scripted policy"
    # numpy-only (T-04-SC): no new perception dependency was added.
    assert "cv2" not in src
    assert "opencv" not in src.lower()


def test_extract_centroids_needs_no_env():
    """Purity, exercised: extract_centroids runs on a bare numpy frame with
    no env in scope and computes from the frame alone — the probe's affine
    bar is the evidence the mask tracks the RENDERED cube, and this test is
    the evidence the value cannot come from anywhere but the frame."""
    cvf = _cv_features()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[3, 7] = (220, 30, 40)  # goal-bounds color only
    out = cvf.extract_centroids(frame)
    # cube channels: neutral (empty cube mask); goal channels: the pixel.
    assert out[0] == 0.5 and out[1] == 0.5
    assert out[2] == pytest.approx(7 / 15, abs=1e-12)
    assert out[3] == pytest.approx(3 / 15, abs=1e-12)


# ---------------------------------------------------------------------------
# Test 3 — the env CV contract (and the legacy contracts untouched)
# ---------------------------------------------------------------------------


def test_env_cv_contract():
    """SO100PickPlaceEnv(vision, rgb_array, grayscale, cv_features=True):
    Dict{state: Box(32,) float32 == (OBS_STATE_CV_DIM,), image: uint8
    (64, 64, 1) unchanged}; reset() and one step() stay in space. The
    DEFAULT construction still emits the flat (28,) Box and the legacy
    vision mode still emits the 28-dim state key (the legacy contract is
    untouched)."""
    cv_dim = _pin("OBS_STATE_CV_DIM")
    assert cv_dim == 32 == C.OBS_STATE_DIM + _pin("OBS_CV_FEATURE_DIM")

    env = _make_cv_env()
    space = env.observation_space
    assert isinstance(space, spaces.Dict)
    assert set(space.keys()) == {C.OBS_STATE_KEY, C.OBS_IMAGE_KEY}
    state_space = space[C.OBS_STATE_KEY]
    assert isinstance(state_space, spaces.Box)
    assert state_space.dtype == np.float32
    assert state_space.shape == (cv_dim,)
    image_space = space[C.OBS_IMAGE_KEY]
    assert image_space.dtype == np.uint8
    assert image_space.shape == (C.OBS_IMAGE_SIZE, C.OBS_IMAGE_SIZE, 1)

    obs, _info = env.reset(seed=0)
    assert space.contains(obs)
    assert obs[C.OBS_STATE_KEY].shape == (cv_dim,)
    assert obs[C.OBS_STATE_KEY].dtype == np.float32
    obs, _rew, _term, _trunc, _info = env.step(
        np.asarray(C.HOME_CTRL, dtype=np.float64)
    )
    assert space.contains(obs)
    # The tail is at indices [28:32], purely additive on the legacy prefix.
    assert obs[C.OBS_STATE_KEY].shape[0] == C.OBS_STATE_DIM + 4

    # The DEFAULT construction is untouched: flat (28,) Box.
    legacy = SO100PickPlaceEnv()
    try:
        assert legacy.observation_space.shape == (C.OBS_STATE_DIM,) == (28,)
        assert legacy.observation_space.dtype == np.float32
    finally:
        legacy.close()

    # The legacy VISION mode is untouched: 28-dim state key, image unchanged.
    legacy_v = SO100PickPlaceEnv(
        obs_mode=C.OBS_MODE_VISION,
        render_mode="rgb_array",
        width=C.OBS_IMAGE_SIZE,
        height=C.OBS_IMAGE_SIZE,
        grayscale=True,
    )
    assert legacy_v.observation_space[C.OBS_STATE_KEY].shape == (C.OBS_STATE_DIM,)
    assert legacy_v.observation_space[C.OBS_IMAGE_KEY].shape == (
        C.OBS_IMAGE_SIZE,
        C.OBS_IMAGE_SIZE,
        1,
    )
    # WR-01: neither vision env is closed mid-process (glfw.terminate()).
    obs, _ = legacy_v.reset(seed=0)
    assert legacy_v.observation_space.contains(obs)


def test_cv_tail_lives_on_the_state_key():
    """The CV feature concatenates onto the EXISTING state key — no new
    Dict key: the obs Dict has exactly {state, image} and the tail occupies
    indices [OBS_STATE_DIM:OBS_STATE_CV_DIM]."""
    env = _make_cv_env()
    obs, _ = env.reset(seed=0)
    assert set(obs.keys()) == {C.OBS_STATE_KEY, C.OBS_IMAGE_KEY}
    state = obs[C.OBS_STATE_KEY]
    assert state.shape == (_pin("OBS_STATE_CV_DIM"),)
    # The tail channels are the measured centroids in [0, 1] (or the
    # neutral (0.5, 0.5) on an empty mask) — never NaN/Inf.
    tail = state[C.OBS_STATE_DIM :]
    assert tail.shape == (4,)
    assert np.isfinite(tail).all()
    assert tail.min() >= 0.0 and tail.max() <= 1.0


# ---------------------------------------------------------------------------
# Test 4 — the cross-mode prefix law (the tail is purely additive)
# ---------------------------------------------------------------------------


def test_cv_prefix_identity():
    """A state-mode env and a vision+cv env reset with the SAME seed produce
    BIT-IDENTICAL first-28-dim vectors — the CV tail is purely additive at
    indices [28:32] and the legacy assembly is untouched."""
    state_env = SO100PickPlaceEnv()
    cv_env = _make_cv_env()
    try:
        s_obs, _ = state_env.reset(seed=123)
        cv_obs, _ = cv_env.reset(seed=123)
        assert s_obs.shape == (C.OBS_STATE_DIM,)
        state_key = cv_obs[C.OBS_STATE_KEY]
        assert state_key.shape == (_pin("OBS_STATE_CV_DIM"),)
        np.testing.assert_array_equal(
            s_obs, state_key[: C.OBS_STATE_DIM],
            err_msg="the first 28 dims must be bit-identical across the "
            "state mode and the CV vision mode (the tail is purely additive)",
        )
    finally:
        # Close ONLY the renderer-less state env (WR-01).
        state_env.close()


# ---------------------------------------------------------------------------
# Test 5 — the fail-loudly kwarg (the spawn_mode/obs_mode pattern)
# ---------------------------------------------------------------------------


def test_cv_features_requires_vision_mode():
    """cv_features=True with obs_mode='state' (the default) fails loudly:
    ValueError naming vision — the existing fail-fast construction
    pattern; a silent cv_features=True on the flat state Box would emit a
    28-dim obs under a 32-dim contract."""
    raised = None
    try:
        SO100PickPlaceEnv(cv_features=True)
    except TypeError as exc:
        pytest.fail(
            "SO100PickPlaceEnv has no cv_features kwarg — RED: the 04-08 "
            f"CV observation contract does not exist (TypeError: {exc})"
        )
    except ValueError as exc:
        raised = exc
    assert raised is not None, (
        "cv_features=True with obs_mode='state' must raise ValueError "
        "naming vision (the 04-08 fail-loudly contract)"
    )
    assert "vision" in str(raised)


# ---------------------------------------------------------------------------
# Test 6 — the chain shape (the D-08 discipline on the new contract)
# ---------------------------------------------------------------------------


def test_cv_chain_shape():
    """The CV chain make_vec_env(vision+cv) -> VecNormalize(norm_obs_keys=
    ['state']) -> ImageOnlyFrameStack(4) -> VecTransposeImage passes the
    state key through at (n_envs, 32) float32, tracks a 32-dim state in
    obs_rms, and keeps the image key out of obs_rms."""
    venv = _make_cv_venv(n_envs=2)
    norm = VecNormalize(
        venv,
        norm_obs_keys=[C.OBS_STATE_KEY],
        norm_obs=True,
        norm_reward=True,
        gamma=C.REWARD_GAMMA,
    )
    chain = VecTransposeImage(ImageOnlyFrameStack(norm, n_stack=C.OBS_N_STACK))
    cv_dim = _pin("OBS_STATE_CV_DIM")

    obs = chain.reset()
    assert obs[C.OBS_STATE_KEY].shape == (2, cv_dim)
    assert obs[C.OBS_STATE_KEY].dtype == np.float32
    obs, _rew, _dones, _infos = chain.step(
        np.zeros((2, 6), dtype=np.float32)
    )
    assert obs[C.OBS_STATE_KEY].shape == (2, cv_dim)
    assert obs[C.OBS_STATE_KEY].dtype == np.float32

    # D-08 on the new contract: the state key's running statistics are
    # 32-dim; the uint8 image key is NEVER in obs_rms.
    assert norm.obs_rms[C.OBS_STATE_KEY].mean.shape == (cv_dim,)
    assert norm.obs_rms[C.OBS_STATE_KEY].var.shape == (cv_dim,)
    assert C.OBS_IMAGE_KEY not in norm.obs_rms
    # The stacked+transposed image space is the unchanged 4x1-channel 64x64.
    assert chain.observation_space[C.OBS_IMAGE_KEY].shape == (
        C.OBS_N_STACK,
        C.OBS_IMAGE_SIZE,
        C.OBS_IMAGE_SIZE,
    )


# ---------------------------------------------------------------------------
# Test 7 — blast-radius honesty (cross-layout loads fail loudly)
# ---------------------------------------------------------------------------


def test_cross_layout_bundle_load_fails_loudly(tmp_path):
    """VecNormalize statistics saved over a 32-dim state key FAIL LOUDLY
    when loaded onto a legacy 28-dim venv — the 04-05/04-06/04-07
    failed-baseline bundles can never silently load on the CV chain
    (SB3 2.9.0 VecNormalize.load -> set_venv -> check_shape_equal)."""
    cv_venv = _make_cv_venv(n_envs=1)
    cv_norm = VecNormalize(
        cv_venv,
        norm_obs_keys=[C.OBS_STATE_KEY],
        norm_obs=True,
        norm_reward=True,
        gamma=C.REWARD_GAMMA,
    )
    cv_norm.reset()
    cv_norm.step(np.zeros((1, 6), dtype=np.float32))
    stats_path = tmp_path / "cv_vecnormalize.pkl"
    cv_norm.save(str(stats_path))

    legacy_venv = make_vec_env(
        SO100PickPlaceEnv,
        n_envs=1,
        seed=0,
        env_kwargs=dict(
            obs_mode=C.OBS_MODE_VISION,
            render_mode="rgb_array",
            width=C.OBS_IMAGE_SIZE,
            height=C.OBS_IMAGE_SIZE,
            grayscale=True,
        ),
    )
    assert legacy_venv.observation_space[C.OBS_STATE_KEY].shape == (C.OBS_STATE_DIM,)
    raised = None
    try:
        VecNormalize.load(str(stats_path), legacy_venv)
    except (ValueError, AssertionError) as exc:
        raised = exc
    assert raised is not None, (
        "a 32-dim VecNormalize load onto a legacy 28-dim venv must fail "
        "loudly (shape mismatch) — no silent cross-layout measurement"
    )


# ---------------------------------------------------------------------------
# Test 8 — the tamper guards (the identity split + the frozen 04-07 pins)
# ---------------------------------------------------------------------------


def test_cv_constants_tamper_guards():
    """OBS_STATE_DIM stays 28 (the legacy/default pin); OBS_CV_FEATURE_DIM
    == 4 and OBS_STATE_CV_DIM == 32 == the sum; the CV bounds are
    well-formed uint8 bounds (lo <= hi per channel, all in [0, 255]);
    CV_BLOB_MISSING_UV == (0.5, 0.5); and the 04-07 grasp predicate pins
    are untouched (the CV track is an OBSERVATION change only)."""
    assert C.OBS_STATE_DIM == 28  # the legacy/default contract, unchanged
    assert _pin("OBS_CV_FEATURE_DIM") == 4
    cv_dim = _pin("OBS_STATE_CV_DIM")
    assert cv_dim == 32 == C.OBS_STATE_DIM + _pin("OBS_CV_FEATURE_DIM")

    for lo_name, hi_name in (
        ("CV_CUBE_RGB_LO", "CV_CUBE_RGB_HI"),
        ("CV_GOAL_RGB_LO", "CV_GOAL_RGB_HI"),
    ):
        lo, hi = _pin(lo_name), _pin(hi_name)
        assert len(lo) == 3 and len(hi) == 3
        assert all(isinstance(v, int) for v in list(lo) + list(hi))
        assert all(l <= h for l, h in zip(lo, hi)), (lo_name, hi_name)
        assert all(0 <= v <= 255 for v in list(lo) + list(hi))
    assert _pin("CV_BLOB_MISSING_UV") == (0.5, 0.5)

    # The frozen 04-07 predicate (the CV track changes observation only).
    assert C.GRASP_CUBE_OFF_FLOOR_Z_M == 0.017
    assert C.GRASP_WINDOW_STEPS == 2
    assert C.GRASP_TRACK_TOL_M == 0.05


def test_cv_bounds_match_probe_artifact():
    """The pinned bounds are the probe's measured bounds (probe-before-pin:
    the constants must equal the runs/cv_probe/<ts>/probe.json values this
    plan measured — a post-hoc edit invalidates the pin)."""
    probe_paths = sorted((REPO_ROOT / "runs" / "cv_probe").glob("*/probe.json"))
    if not probe_paths:
        pytest.fail(
            "runs/cv_probe/<ts>/probe.json is missing — the mask bounds "
            "must be probe-measured, never guessed (04-08 probe-before-pin)"
        )
    probe = json.loads(probe_paths[-1].read_text())
    assert list(_pin("CV_CUBE_RGB_LO")) == probe["cube_rgb_lo"]
    assert list(_pin("CV_CUBE_RGB_HI")) == probe["cube_rgb_hi"]
    assert list(_pin("CV_GOAL_RGB_LO")) == probe["goal_rgb_lo"]
    assert list(_pin("CV_GOAL_RGB_HI")) == probe["goal_rgb_hi"]


# ---------------------------------------------------------------------------
# Test 9 — the judged chains match (T-04-25)
# ---------------------------------------------------------------------------


def test_judged_chains_share_the_cv_contract():
    """train.resolve_env_kwargs (the training chain) AND
    _eval_chain.build_eval_venv (the eval/audit chain) BOTH construct
    cv_features=True — the two chains that judge the baseline share ONE
    32-dim contract (a judged chain built on the wrong layout would
    measure the wrong contract)."""
    train = _load_train()
    gate = train.load_gate()
    kwargs = train.resolve_env_kwargs(gate)
    if kwargs.get("cv_features") is not True:
        pytest.fail(
            "train.resolve_env_kwargs does not construct cv_features=True "
            "— the TRAINING chain is off the 04-08 CV observation contract"
        )

    ec = _load_eval_chain()
    eval_venv = ec.build_eval_venv(gate)
    raw = eval_venv.envs[0].unwrapped
    if getattr(raw, "_cv_features", None) is not True:
        pytest.fail(
            "build_eval_venv does not construct cv_features=True — the "
            "EVAL chain is off the 04-08 CV observation contract"
        )
    assert raw.observation_space[C.OBS_STATE_KEY].shape == (
        _pin("OBS_STATE_CV_DIM"),
    )
    # WR-01: the eval vision env is never closed mid-process.