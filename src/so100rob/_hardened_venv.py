"""Hardened spawn-topology vec-env construction (D-07 — the STATE.md
Phase 4 contract).

EVERY spawn-topology construction MUST go through this module (D-07):
the full hardened construction — watchdogged readiness barrier + per-worker
first-render warm-up reset inside the construction window + watched
untimed reset/warm-up with bounded kill/rebuild/retry + terminate ->
bounded join -> SIGKILL worker reaping. Timed regions are NEVER armed
(the SIGALRM watchdog covers untimed setup exchanges only), pinned by
tests/test_smoke_watchdog.py. The macOS GL-on-Metal first-render wedge is
BOUNDED, not eliminated (G-03-2 diagnosis with C-stack evidence:
.planning/debug/subproc-topology-benchmark-hang.md) — the failure is
non-deterministic, so any "simpler" construction can pass one run and
hang the overnight one. scripts/run_sb3_smoke.py is a thin consumer of
this module; scripts/train.py (04-03) imports it for its own construction
and both tripwire windows.

Two-window faulthandler scheme (04-01, replacing the old 25-min hardcode
flagged by 03-REVIEW IN-09):

  * ``SETUP_TRIPWIRE_S`` — a COMPUTED worst-case bound over every
    bounded-retry untimed-setup window in a full run (all construction
    attempts + bench-setup attempts + kill/join bounds + readiness
    delay, times a safety factor). It strictly dominates the ~38-min
    worst-case bounded-retry path, so a healthy retry-laden run is never
    killed by its own backstop. Arm it over the setup phase
    (``faulthandler.dump_traceback_later(SETUP_TRIPWIRE_S, exit=True)``)
    and cancel it once construction completes.
  * ``train_window_s()`` / ``arm_train_tripwire()`` — the TRAIN window:
    ``window_hours x 3600`` seconds READ from config/obs_gate.json
    (never a re-hardcoded 12 h — the D-04/WR-02 pin discipline). train.py
    arms it AFTER the setup tripwire is cancelled, immediately before
    ``model.learn()``.

The faulthandler tripwire is a process-global backstop of last resort
(it dumps all thread tracebacks and exits nonzero on a silent hang — the
EXIT-CODE CONTRACT forbids hanging); it does not perturb timed
measurements, unlike the SIGALRM watchdog which never runs over a timed
region.

Extracted verbatim (names and semantics identical) from
scripts/run_sb3_smoke.py (03-04, commit 4bd9da6) by 04-01 Task 2.
"""

from __future__ import annotations

import faulthandler
import json
import signal
import time
from pathlib import Path

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/so100rob/ -> repo root
GATE_PATH = REPO_ROOT / "config" / "obs_gate.json"

# ---------------------------------------------------------------------------
# Pinned construction constants (D-06/D-07). Budgets are module constants
# so a runaway run fails loudly rather than silently expanding (T-03-04).
# Pitfall 7 divisibility: batch_size 64 divides n_steps x n_envs =
# 128 x 8 = 1024 (the smoke/Phase-4 rollout buffer).
# ---------------------------------------------------------------------------

N_ENVS = 8                 # D-07: the mid candidate from 8-12 envs
SEED = 0                   # identical across formats/runners (comparability)
TOPO_REPEATS = 3           # median of >= 3 repeats (the benchmark.py protocol)
TOPO_WARMUP_STEPS = 200    # untimed: worker spawn + renderer creation + GL warm-up

# --- macOS GL-on-Metal first-render wedge hardening (03-04, G-03-2) -------
# Root cause (diagnosed with per-worker faulthandler dumps + macOS
# `sample` C-stacks — .planning/debug/subproc-topology-benchmark-hang.md):
# each spawn worker lazily creates its offscreen GL context and performs
# its first mjr_readPixels INSIDE env.reset() — i.e. AFTER the readiness
# barrier — and on macOS those GL calls ride the deprecated
# OpenGL-on-Metal layer; under 8-way concurrent first use Apple's stack
# can block PERMANENTLY (observed stuck sites: glfwCreateWindow ->
# CGLFlushDrawable -> GLDContextRec::beginRenderPass, and mjr_readPixels
# -> MTLCompilerFSCache::openSync -> flock on the shared on-disk Metal
# compiler cache). The parent then blocks forever in SB3's in-order
# remote.recv. The failure is non-deterministic (the same code passed
# fully on Sept 23) — it is BOUNDED here, not eliminated. The probe
# numbers, re-interpreted honestly: the first reset/step batch hung 4/12
# times WITHOUT a readiness delay and 0/12 WITH a 5 s delay + an
# all-worker get_attr barrier round-trip — the barrier spaces worker
# imports away from the first GL use, REDUCING (not eliminating) the
# contention wedge; a post-barrier command into a connected socketpair
# cannot be lost. Coverage contract (untimed setup ONLY; timed regions
# are NEVER armed, so measurements are untouched):
#   1. construction + worker-ready delay + get_attr readiness barrier +
#      the vec env's real reset() (the per-worker FIRST-RENDER WARM-UP)
#      all under one SIGALRM watchdog, bounded retry + full worker
#      cleanup,
#   2. the benchmark repeat's untimed reset + warm-up under the same
#      watchdog (_watched_reset_warmup) — on timeout: kill, rebuild,
#      retry the WHOLE repeat, bounded count, loud RuntimeError on
#      exhaustion; a wedged venv is KILLED, never closed (close() joins
#      workers with no timeout),
#   3. a wedged worker is provably dead before its replacement renders
#      (terminate -> bounded join -> SIGKILL in _kill_venv — the Metal
#      compiler-cache flock is shared state),
#   4. a global faulthandler tripwire (SETUP_TRIPWIRE_S) as the backstop
#      of last resort.
SPAWN_WORKER_READY_DELAY_S = 5.0  # spaces worker imports away from the first GL use (reduces the contention wedge)
CONSTRUCTION_TIMEOUT_S = 60.0     # a healthy construction + barrier + first-render warm-up is ~5-10 s
CONSTRUCTION_ATTEMPTS = 3
BENCH_SETUP_TIMEOUT_S = 60.0      # SIGALRM window for one benchmark repeat's untimed reset + warm-up (subproc only; healthy ~2-6 s = 10x margin; untimed setup, NEVER the timed region)
BENCH_SETUP_ATTEMPTS = 3         # bounded kill/rebuild/retry-the-repeat count per benchmark repeat; exhaustion raises RuntimeError -> exit 1 per the EXIT-CODE CONTRACT, never a silent hang
_KILL_JOIN_S = 5.0               # per-phase join bound in _kill_venv

# --- SETUP tripwire: computed worst-case bound (IN-09, two-window scheme) --
# The old 25-min hardcode was SMALLER than the run's own bounded-retry
# worst path (~38 min): a healthy run deep in its retry ladder could be
# killed by its own backstop. SETUP_TRIPWIRE_S is therefore COMPUTED from
# the construction constants as the worst-case path — every watchdogged
# window at its own bound — times a safety factor:
#   * one construction ladder: CONSTRUCTION_ATTEMPTS x (timeout + ready
#     delay + kill bound), where the kill bound is the two bounded joins
#     in _kill_venv (after terminate and after SIGKILL);
#   * one benchmark repeat ladder: BENCH_SETUP_ATTEMPTS x (construction
#     ladder + one bench-setup window + its kill bound);
#   * the full worst path: both benchmark topologies x TOPO_REPEATS
#     repeat ladders, plus the smoke's 4 remaining construction windows
#     (2 formats x train + reload chains).
# The result (~2.6 h) dominates both the ~38-min worst path and the
# full-smoke bounded-retry worst case, so it can only ever fire on a
# genuine silent hang — never on a healthy (or maximally retry-laden)
# run. Never replace this with a hardcoded number (T-04-02).
BENCH_TOPOLOGIES = 2             # the D-07 benchmark measures dummy + subproc
SMOKE_CONSTRUCTION_WINDOWS = 4   # 2 formats x (train chain + reload-proof chain)
SETUP_TRIPWIRE_SAFETY = 1.5      # generous: bigger only ever fires on a true hang
_KILL_BOUND_S = 2.0 * _KILL_JOIN_S  # bounded join after terminate + after SIGKILL
WORST_CONSTRUCTION_S = CONSTRUCTION_ATTEMPTS * (
    CONSTRUCTION_TIMEOUT_S + SPAWN_WORKER_READY_DELAY_S + _KILL_BOUND_S
)
WORST_BENCH_WINDOW_S = BENCH_SETUP_TIMEOUT_S + _KILL_BOUND_S
WORST_BENCH_REPEAT_S = BENCH_SETUP_ATTEMPTS * (
    WORST_CONSTRUCTION_S + WORST_BENCH_WINDOW_S
)
SETUP_TRIPWIRE_S = SETUP_TRIPWIRE_SAFETY * (
    BENCH_TOPOLOGIES * TOPO_REPEATS * WORST_BENCH_REPEAT_S
    + SMOKE_CONSTRUCTION_WINDOWS * WORST_CONSTRUCTION_S
)


# ---------------------------------------------------------------------------
# TRAIN window (read from the committed pin — never re-hardcoded, D-04)
# ---------------------------------------------------------------------------


def train_window_s() -> float:
    """The TRAIN-window tripwire length: window_hours x 3600 seconds
    READ from config/obs_gate.json — the committed pin, never a
    re-hardcoded 12 h (WR-02 discipline). train.py (04-03) may instead
    pass its own gate's train_window_hours explicitly to
    arm_train_tripwire(seconds=...)."""
    gate = json.loads(GATE_PATH.read_text())
    return float(gate["window_hours"]) * 3600.0


def arm_train_tripwire(seconds: float | None = None) -> float:
    """Arm the TRAIN-window faulthandler backstop and return the armed
    window (seconds). Call AFTER the hardened construction completes and
    the SETUP tripwire is cancelled (train.py: immediately before
    model.learn(); cancel after). seconds=None resolves through
    train_window_s() (window_hours from config/obs_gate.json). Any
    still-armed setup window is cancelled first so the two windows never
    overlap."""
    if seconds is None:
        seconds = train_window_s()
    faulthandler.cancel_dump_traceback_later()  # the setup window must not leak into the train window
    faulthandler.dump_traceback_later(seconds, exit=True)
    return seconds


# ---------------------------------------------------------------------------
# Construction helpers (D-07: the start method travels through
# vec_env_kwargs, never as a make_vec_env kwarg — no such parameter exists)
# ---------------------------------------------------------------------------


def _vision_env_kwargs(grayscale: bool) -> dict:
    """Vision-mode env kwargs per D-05 format; default frame_skip."""
    return dict(
        obs_mode=C.OBS_MODE_VISION,
        render_mode="rgb_array",
        width=C.OBS_IMAGE_SIZE,
        height=C.OBS_IMAGE_SIZE,
        grayscale=grayscale,
    )


def _make_topology(
    kind: str,
    grayscale: bool = False,
    env_kwargs: dict | None = None,
    wrapper_class=None,
    wrapper_kwargs: dict | None = None,
):
    """8-env vision vec-env on the given topology ('dummy' | 'subproc').

    env_kwargs (None = unchanged smoke behavior) merges OVER the vision
    kwargs so 04-03's training construction can add per-worker options
    (spawn_mode="continuous" — Pitfall 1) through the SAME hardened path
    instead of a second builder. wrapper_class/wrapper_kwargs (04-03,
    additive like env_kwargs) pass through to make_vec_env's per-env
    factory — gymnasium RescaleAction(gate rescale_min/max) per the
    pinned action_space_mode rides INSIDE each worker env (below
    VecNormalize, above the raw env), so the hardened construction
    remains the only spawn-topology builder (D-07)."""
    kwargs = _vision_env_kwargs(grayscale)
    if env_kwargs:
        kwargs.update(env_kwargs)
    if kind == "dummy":
        return make_vec_env(
            SO100PickPlaceEnv,
            n_envs=N_ENVS,
            seed=SEED,
            env_kwargs=kwargs,
            wrapper_class=wrapper_class,
            wrapper_kwargs=wrapper_kwargs,
        )
    if kind == "subproc":
        return make_vec_env(
            SO100PickPlaceEnv,
            n_envs=N_ENVS,
            seed=SEED,
            env_kwargs=kwargs,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "spawn"},  # Pitfall 8: via vec_env_kwargs
            wrapper_class=wrapper_class,
            wrapper_kwargs=wrapper_kwargs,
        )
    raise ValueError(f"unknown topology kind: {kind!r}")


def _close_if_safe(venv, kind: str) -> None:
    """Close subproc workers (their own processes; GL teardown is contained).

    Dummy vision envs are NEVER closed mid-process: gymnasium 1.3.0's
    OffScreenViewer.close() terminates the process-global GLFW state
    (03-01 decision) — in-process renderers are left to GC, the pattern
    the 03-02 suite established.
    """
    if kind == "subproc":
        venv.close()


class _ConstructionTimeout(Exception):
    """A subproc untimed-setup exchange desynced (macOS GL-on-Metal
    first-render wedge): construction/readiness, the per-worker first
    render, or a benchmark reset/warm-up."""


def _on_construction_alarm(signum, frame) -> None:
    raise _ConstructionTimeout()


def _kill_venv(venv) -> None:
    """Terminate hung spawn workers and close their pipes so a retry
    starts from a clean worker set.

    A worker still alive after the bounded join gets SIGKILL: a wedged
    worker can be stuck on the on-disk Metal compiler-cache flock
    (shared state), so it must be provably dead before a retry's fresh
    workers render, or the retry can wedge on the same lock. (SB3's
    _worker installs no SIGTERM handler, so terminate normally
    suffices; the escalation is the guarantee.)"""
    for p in venv.processes:
        p.terminate()
    for p in venv.processes:
        p.join(_KILL_JOIN_S)
        if p.is_alive():
            p.kill()  # SIGKILL escalation for a SIGTERM-immune wedge
            p.join(_KILL_JOIN_S)
    for r in venv.remotes:
        try:
            r.close()
        except Exception:
            pass


def _make_topology_robust(
    kind: str,
    grayscale: bool = False,
    env_kwargs: dict | None = None,
    wrapper_class=None,
    wrapper_kwargs: dict | None = None,
):
    """Construct a vec-env with every untimed setup exchange bounded.

    DummyVecEnv is in-process (no pipes, no concurrent fresh GL contexts)
    and constructed directly. A spawn SubprocVecEnv gets: construction +
    worker-ready delay + an all-worker get_attr readiness barrier + a
    per-worker FIRST-RENDER WARM-UP (the vec env's real reset(), which is
    where each worker lazily creates its offscreen GL context and
    performs its first mjr_readPixels — the non-deterministic macOS
    GL-on-Metal first-render wedge window), all under a SIGALRM watchdog
    with bounded retry and full worker cleanup — a wedged attempt is
    killed and rebuilt instead of hanging the run. Because every
    construction through this path ends with a completed first reset,
    later consumers inherit the coverage: model.learn()'s internal reset
    and every later render run on already-created GL contexts. Timed
    regions are never armed.
    """
    if kind != "subproc":
        return _make_topology(kind, grayscale, env_kwargs)
    signal.signal(signal.SIGALRM, _on_construction_alarm)
    last: Exception | None = None
    for attempt in range(1, CONSTRUCTION_ATTEMPTS + 1):
        venv = None
        signal.setitimer(signal.ITIMER_REAL, CONSTRUCTION_TIMEOUT_S)
        try:
            venv = _make_topology(kind, grayscale, env_kwargs, wrapper_class, wrapper_kwargs)
            # Workers need ~2 s to finish importing __main__ and reach
            # their command loop; a batch sent before that can desync.
            time.sleep(SPAWN_WORKER_READY_DELAY_S)
            # Readiness barrier: a harmless round-trip to ALL 8 workers.
            # Only when this returns is every worker provably in its
            # command loop.
            venv.get_attr("spec")
            # First-render warm-up (route 2): the FIRST reset is where
            # each worker lazily creates its offscreen GL context and
            # performs its first mjr_readPixels (vision obs -> render
            # inside reset_model -> _get_obs) — on macOS those GL calls
            # ride the OpenGL-on-Metal layer and can wedge
            # non-deterministically under 8-way concurrent first use.
            # Running it HERE keeps the wedge inside the watchdoged
            # construction window, so a wedged worker is killed/rebuilt/
            # retried below instead of hanging the run, and every LATER
            # reset/render (including model.learn()'s internal reset in
            # the smoke trains / Phase 4 training) runs on
            # already-created GL contexts.
            venv.reset()
            signal.setitimer(signal.ITIMER_REAL, 0)
            if attempt > 1:
                print(
                    f"[smoke] subproc construction succeeded on attempt "
                    f"{attempt}/{CONSTRUCTION_ATTEMPTS} "
                    f"(macOS GL-on-Metal first-render wedge)",
                    flush=True,
                )
            return venv
        except _ConstructionTimeout as exc:
            last = exc
            print(
                f"[smoke] subproc construction attempt "
                f"{attempt}/{CONSTRUCTION_ATTEMPTS} desynced (> "
                f"{CONSTRUCTION_TIMEOUT_S:.0f} s — macOS GL-on-Metal "
                f"first-render wedge); killing workers and retrying",
                flush=True,
            )
            if venv is not None:
                _kill_venv(venv)
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if venv is not None:
                _kill_venv(venv)
            raise
    raise RuntimeError(
        f"subproc vec-env construction desynced after "
        f"{CONSTRUCTION_ATTEMPTS} attempts (macOS GL-on-Metal "
        f"first-render wedge — see "
        f".planning/debug/subproc-topology-benchmark-hang.md)"
    ) from last


def _watched_reset_warmup(venv, kind: str, action) -> None:
    """Untimed benchmark setup — reset + warm-up steps (route 1).

    For non-subproc topologies this runs the reset and warm-up directly
    (current behavior: the contention wedge needs concurrent fresh
    processes, and kill/rebuild is meaningless in-process). For subproc it
    runs under the SAME SIGALRM watchdog as construction: a worker whose
    first or later GL exchange wedges in Apple's OpenGL-on-Metal layer
    stalls this exchange, so _ConstructionTimeout (the same failure class
    — an untimed-setup exchange desynced) propagates to the caller, which
    kills and retries. The watchdog disarms in a finally — BEFORE the
    caller's timed region, which is never armed."""
    if kind != "subproc":
        venv.reset()
        for _ in range(TOPO_WARMUP_STEPS):  # untimed warm-up
            venv.step(action)
        return
    signal.signal(signal.SIGALRM, _on_construction_alarm)
    signal.setitimer(signal.ITIMER_REAL, BENCH_SETUP_TIMEOUT_S)
    try:
        venv.reset()
        for _ in range(TOPO_WARMUP_STEPS):  # untimed warm-up
            venv.step(action)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)