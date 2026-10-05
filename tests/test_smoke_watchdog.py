"""Watchdog-coverage regression tests for the SB3 smoke runner (Plan 03-04,
G-03-2 / TRAIN-02).

Pin the BOUND on the non-deterministic macOS GL-on-Metal first-render wedge
diagnosed in .planning/debug/subproc-topology-benchmark-hang.md: a spawn
worker's FIRST render — lazy offscreen GL context creation plus the first
mjr_readPixels, both inside env.reset() AFTER the readiness barrier — can
wedge permanently under 8-way concurrent first use. The hardening contract
these tests pin (all untimed setup, never a timed region):

  * Route 2 — the per-worker first-render warm-up (a real venv.reset())
    happens INSIDE the watchdogged construction window, after the
    get_attr readiness barrier and before the watchdog disarm, so GL
    context creation is covered by the existing kill/rebuild/retry path.
  * Route 1 — the benchmark repeat's untimed reset + warm-up steps run
    under the same SIGALRM watchdog (_watched_reset_warmup); a timeout
    raises the SAME _ConstructionTimeout and retries the WHOLE repeat
    from construction, bounded by BENCH_SETUP_ATTEMPTS.
  * _kill_venv escalates terminate -> bounded join -> SIGKILL, so a
    worker wedged on the on-disk Metal compiler-cache flock (shared
    state) is provably dead before a retry's fresh workers render.
  * The watchdog is DISARMED before every perf_counter timed region —
    D-07 medians and D-04 gate numbers stay comparable.
  * Exhaustion fails LOUDLY (RuntimeError, exit 1 per the EXIT-CODE
    CONTRACT) — never a silent infinite hang.

GL note (empirical, this phase — see test_sb3_pipeline.py): the FIRST
render after a viewer is created is the non-deterministic one; that is
exactly the wedge window these tests force the runner to bound. All stubs
here are tiny local classes with just the surface each code path touches
(reset/step/get_attr/processes/remotes) — no real envs, no GL, no subproc
workers, so this file runs in seconds.
"""

import multiprocessing as _mp
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# The runner is a script, not a package module: import it by putting
# scripts/ on sys.path (importing is side-effect-free — the __main__
# guard holds and faulthandler is only armed inside main()).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_sb3_smoke  # noqa: E402

# Since 04-01 Task 2 the hardened construction lives in the package
# module so100rob._hardened_venv (D-07), and the runner is a thin
# consumer that imports (re-exports) it. monkeypatch targets and call
# sites are therefore NAMESPACED BY WHO CONSUMES THEM:
#   * names consumed INSIDE a _hardened_venv function (watchdog windows,
#     warm-up steps, ready delay, kill bound, _make_topology,
#     _ConstructionTimeout) are patched/called on the MODULE (hv.*) —
#     patching the runner's re-export would never reach them;
#   * names consumed by the RUNNER's own _bench_topology (repeat/timed
#     counts, attempt counts, and the _make_topology_robust/_kill_venv/
#     _close_if_safe seams it calls) are patched on run_sb3_smoke, the
#     same as before the extraction.
from so100rob import _hardened_venv as hv


class _StubVenv:
    """Minimal SubprocVecEnv surface the hardening paths touch."""

    def __init__(self, reset_delay_s: float = 0.0, step_delay_s: float = 0.0):
        self.reset_delay_s = reset_delay_s
        self.step_delay_s = step_delay_s
        self.calls: list[str] = []
        self.processes = []
        self.remotes = []

    def get_attr(self, name):
        self.calls.append("get_attr")
        return [None]

    def reset(self):
        self.calls.append("reset")
        if self.reset_delay_s:
            time.sleep(self.reset_delay_s)

    def step(self, action):
        self.calls.append("step")
        time.sleep(self.step_delay_s)


def _itimer_disarmed() -> bool:
    """The SIGALRM itimer must be fully off: getitimer returns the
    2-tuple (seconds, interval) — disarmed is exactly (0.0, 0.0)."""
    return signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_hung_reset_is_bounded(monkeypatch):
    """A hung reset fires the watchdog in ~BENCH_SETUP_TIMEOUT_S (not the hang), and the finally-disarm holds on the timeout path."""
    monkeypatch.setattr(hv, "BENCH_SETUP_TIMEOUT_S", 0.5)
    stub = _StubVenv(reset_delay_s=30.0)
    action = np.zeros((8, 6), dtype=np.float32)
    t0 = time.perf_counter()
    with pytest.raises(hv._ConstructionTimeout):
        hv._watched_reset_warmup(stub, "subproc", action)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"watchdog fired too late: {elapsed:.2f} s"
    # The finally-disarm holds on the TIMEOUT path, so the timed region
    # that follows can never be armed.
    assert _itimer_disarmed()


def test_timeout_kills_rebuilds_and_retries_the_repeat(monkeypatch, capsys):
    """A wedged repeat is killed and rebuilt; the retried repeat completes and is timed strictly positively; the healthy venv (only) is closed."""
    monkeypatch.setattr(run_sb3_smoke, "TOPO_REPEATS", 1)
    monkeypatch.setattr(hv, "TOPO_WARMUP_STEPS", 2)
    monkeypatch.setattr(run_sb3_smoke, "TOPO_TIMED_STEPS", 3)
    monkeypatch.setattr(hv, "BENCH_SETUP_TIMEOUT_S", 0.5)
    hung = _StubVenv(reset_delay_s=30.0)
    healthy = _StubVenv(step_delay_s=0.001)  # timed region elapsed > 0
    stubs = [hung, healthy]
    monkeypatch.setattr(
        run_sb3_smoke,
        "_make_topology_robust",
        lambda kind, grayscale=False: stubs.pop(0),
    )
    kills: list = []
    closes: list = []
    monkeypatch.setattr(run_sb3_smoke, "_kill_venv", lambda v: kills.append(v))
    monkeypatch.setattr(
        run_sb3_smoke, "_close_if_safe", lambda v, kind: closes.append(v)
    )

    median, repeats = run_sb3_smoke._bench_topology("subproc")

    assert len(repeats) == 1, "the retried repeat must complete exactly once"
    assert repeats[0] > 0 and median > 0
    assert kills == [hung], "exactly the hung stub must be killed"
    assert closes == [healthy], "only the healthy stub must be closed"
    assert "retrying the repeat" in capsys.readouterr().out
    assert _itimer_disarmed()


def test_exhausted_attempts_fail_loudly(monkeypatch):
    """Bounded attempts: every stub wedges -> both killed, RuntimeError matching the exhaustion message, no infinite hang (~2x timeout, not TOPO x hang)."""
    monkeypatch.setattr(run_sb3_smoke, "TOPO_REPEATS", 1)
    monkeypatch.setattr(hv, "TOPO_WARMUP_STEPS", 1)
    monkeypatch.setattr(run_sb3_smoke, "TOPO_TIMED_STEPS", 2)
    monkeypatch.setattr(hv, "BENCH_SETUP_TIMEOUT_S", 0.5)
    monkeypatch.setattr(run_sb3_smoke, "BENCH_SETUP_ATTEMPTS", 2)
    hung_stubs = [_StubVenv(reset_delay_s=30.0), _StubVenv(reset_delay_s=30.0)]
    monkeypatch.setattr(
        run_sb3_smoke,
        "_make_topology_robust",
        lambda kind, grayscale=False: hung_stubs.pop(0),
    )
    kills: list = []
    monkeypatch.setattr(run_sb3_smoke, "_kill_venv", lambda v: kills.append(v))

    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="setup desynced after 2 attempts"):
        run_sb3_smoke._bench_topology("subproc")
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"exhaustion must be bounded, took {elapsed:.2f} s"
    assert len(kills) == 2, "both hung stubs must have been killed"
    assert _itimer_disarmed()


def test_first_render_warmup_inside_construction_watchdog(monkeypatch):
    """REAL _make_topology_robust('subproc'): venv.reset() (the per-worker first-render warm-up) happens AFTER the get_attr barrier, inside the watchdoged window, and the itimer is disarmed afterwards."""
    stub = _StubVenv()
    monkeypatch.setattr(
        hv, "_make_topology", lambda kind, grayscale=False, env_kwargs=None: stub
    )
    monkeypatch.setattr(hv, "SPAWN_WORKER_READY_DELAY_S", 0.0)

    returned = hv._make_topology_robust("subproc")

    assert returned is stub
    assert stub.calls == ["get_attr", "reset"], (
        "the first-render warm-up reset must happen after the readiness "
        "barrier, before the watchdog disarm"
    )
    assert _itimer_disarmed()


def _sigterm_ignoring_child():
    """Module-level fork target: ignore SIGTERM, then sleep.

    A fork child is required because a spawn child would have to import
    this pytest test module, which it cannot.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(300)


def test_kill_venv_reaps_sigterm_ignoring_worker(monkeypatch):
    """_kill_venv escalation: a worker wedged on shared state ignores SIGTERM, so terminate -> bounded join -> SIGKILL must reap it within the bound."""
    monkeypatch.setattr(hv, "_KILL_JOIN_S", 0.25)
    child = _mp.get_context("fork").Process(target=_sigterm_ignoring_child)
    child.start()
    stub = _StubVenv()
    stub.processes = [child]

    t0 = time.perf_counter()
    hv._kill_venv(stub)
    elapsed = time.perf_counter() - t0

    assert not child.is_alive(), (
        "a SIGTERM-ignoring worker must be SIGKILL-reaped, not left behind"
    )
    assert elapsed < 2.0, f"kill path took {elapsed:.2f} s — not bounded"
    child.join()