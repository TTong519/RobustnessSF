#!/usr/bin/env python
"""SB3 smoke runner (D-06/D-07/D-09) — the TRAIN-02 decision-half measurement.

Executes the full SB3 training pipeline end-to-end on the real env, twice
(one 16384-step PPO smoke train per image format), and evaluates the
pre-pinned D-04 throughput gate from the measured numbers:

  * Phase A — D-07 topology benchmark: DummyVecEnv vs SubprocVecEnv
    (spawn start method, passed through vec_env_kwargs — never a
    make_vec_env kwarg), 8 vision envs, 200 untimed warm-up steps (worker
    spawn + lazy renderer creation + first-frame GL warm-up excluded from
    the timed region), then 1000 timed vec steps with time.perf_counter,
    median of >= 3 repeats (the scripts/benchmark.py protocol). The
    higher median wins and is the topology both smoke trains use.
  * Phase B — per format ("rgb", "grayscale"): the full chain on the
    winning topology — VecNormalize(norm_obs_keys=[C.OBS_STATE_KEY],
    norm_obs=True, norm_reward=True, gamma=C.REWARD_GAMMA) ->
    ImageOnlyFrameStack(C.OBS_N_STACK) -> VecTransposeImage ->
    PPO("MultiInputPolicy", ...) with the default CNN — timed
    learn(total_timesteps=16384), then the D-08 checkpoint bundle
    (model.save + vecnormalize.save).
  * Phase C — D-09 reload proof per format: fresh identical vec-env ->
    VecNormalize.load -> ImageOnlyFrameStack -> VecTransposeImage ->
    PPO.load().set_env (the strict order), then fixed_obs built in the
    LOADED model's observation-space shapes (read from the space, never
    hand-counted) and a bitwise action-equality check
    (a.tobytes() == b.tobytes(), deterministic=True).
  * Phase D — gate evaluation: the floor is READ from
    config/obs_gate.json (never re-hardcoded); D-05 picks the format
    (higher steps/sec wins; within a relative 10% band -> grayscale);
    the chosen format's measured steps/sec is evaluated against the floor
    -> decision is exactly one of "vision-64x64-gray",
    "vision-64x64-rgb", "state-only-fallback". The DECISION, either way,
    is an output, never a failure (D-06's bar is pipeline integrity; the
    fallback is the gate working as designed per D-04/D-12).

EXIT-CODE CONTRACT (public interface): exit 0 ONLY when both topology
candidates were measured, both format smoke trains completed, both
reload+bitwise checks passed, the gate was evaluated, and the JSON was
written — with "SMOKE PASSED: <decision>" printed and "RESULTS_JSON=<path>"
as the LAST stdout line; exit 1 on any failed collector check or exception
— a pipeline can never see a zero on a failed smoke. Every failing check
is printed immediately with its detail and lands in the results JSON;
nothing is swallowed.

Output: runs/obs_decision/<UTC-timestamp>/results.json (gitignored
runtime artifact) with per-format steps/sec, mean episodic reward,
bitwise_equal, elapsed time, checkpoint paths, TensorBoard dir, both
topology medians and the winner, the gate fields, machine/version
context, and the full collector record — the machine-readable source
OBS_DECISION.md points to.

All runs must go through the pinned 3.12 venv (established project
discipline) — use ``uv run python scripts/run_sb3_smoke.py``. The
``if __name__ == "__main__":`` guard is MANDATORY: spawn-method
SubprocVecEnv workers re-import this module, and without the guard each
worker would recursively spawn workers (Pitfall 8).

macOS spawn-race hardening (measured on this machine, see the SPAWN_*
constants block): the first command batch sent to all spawn workers can
desync against their bootstrap (~30-50% without a readiness delay, 0/12
with one), so every subproc construction goes through a watchdogged
readiness barrier with bounded retry, and a global faulthandler
tripwire converts any later wedged exchange into a traceback dump +
nonzero exit rather than a silent infinite hang.
"""

from __future__ import annotations

import faulthandler
import json
import math
import signal
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecTransposeImage

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv
from so100rob.vec_wrappers import ImageOnlyFrameStack

# Reuse the Phase 1 benchmark's machine/version helpers so the runner and
# BENCHMARK.md/OBS_DECISION.md share one measurement-context convention.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import machine_context, package_versions  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs" / "obs_decision"
GATE_PATH = REPO_ROOT / "config" / "obs_gate.json"

# ---------------------------------------------------------------------------
# Pinned runner constants (D-06/D-07). Budgets are module constants so a
# runaway run fails the collector rather than silently expanding (T-03-04).
# Pitfall 7 divisibility: batch_size 64 divides n_steps x n_envs = 128 x 8 =
# 1024, and 16384 = 16 rollout buffers of 1024 steps.
# ---------------------------------------------------------------------------

SMOKE_TOTAL_TIMESTEPS = 16384  # D-06: pipeline-integrity smoke, no slope bar
N_ENVS = 8                     # D-07: the mid candidate from 8-12 envs
N_STEPS = 128                  # PPO rollout length per env
BATCH_SIZE = 64                # divides the 1024-step rollout buffer (Pitfall 7)
N_EPOCHS = 2                   # PPO epochs per update
TOPO_WARMUP_STEPS = 200        # untimed: worker spawn + renderer creation + GL warm-up
TOPO_TIMED_STEPS = 1000        # timed vec steps per repeat (x N_ENVS env steps)
TOPO_REPEATS = 3               # median of >= 3 (the benchmark.py protocol)
SEED = 0                       # identical for both formats (comparability)
FORMATS = ("rgb", "grayscale")  # D-05 candidates, measured identically

# D-05 tie band, verbatim from the pin (config/obs_gate.json format_rule):
# "higher measured steps/sec wins; within a relative 10% band -> grayscale".
TIE_BAND_REL = 0.10

# --- macOS spawn robustness (Rule 3 hardening, empirically isolated) -----
# On this machine (macOS arm64, Python 3.12, spawn start method) a command
# batch sent to ALL 8 workers before they finish their spawn bootstrap can
# deadlock permanently: the Connection args travel through macOS's
# fd-reduction machinery (no SCM_RIGHTS on this platform -> the
# multiprocessing resource_sharer DupFd dance), and a first-batch command
# racing that handoff is silently lost — main blocks in recv forever
# while every worker sits idle at its own recv. Measured with throwaway
# probes (12 constructions, fresh process): make() itself never hangs
# (12/12 OK — the get_spaces handshake is single-worker), but the first
# reset/step batch hung 4/12 times WITHOUT a readiness delay and 0/12
# times WITH a 4 s delay + an all-worker barrier round-trip. Hence:
#   1. after construction, wait for workers to finish importing,
#   2. run a get_attr round-trip to ALL 8 workers as a readiness barrier,
#   3. both under a SIGALRM watchdog with bounded retry + full worker
#      cleanup — a hung attempt can never wedge the run.
# Timed regions are NEVER armed: the watchdog covers construction +
# readiness only, so measurements are untouched.
SPAWN_WORKER_READY_DELAY_S = 5.0  # workers need ~2 s to import + reach the command loop
CONSTRUCTION_TIMEOUT_S = 60.0     # a healthy construction + barrier is ~5-8 s
CONSTRUCTION_ATTEMPTS = 3
# Global tripwire: if ANY later exchange desyncs mid-run (never observed
# once past the first batch, but the failure mode is a silent infinite
# hang), dump all Python tracebacks and exit nonzero — the EXIT-CODE
# CONTRACT forbids hanging silently.
FAULTHANDLER_TRIPWIRE_S = 25 * 60


# ---------------------------------------------------------------------------
# Assertion collector (T-02-05 pattern from run_reward_audit.py: every check
# recorded, failures printed immediately, never swallowed)
# ---------------------------------------------------------------------------


class _Collector:
    """Records every check for the results JSON; failures print immediately."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.assertions: list[dict] = []

    def check(self, check_name: str, passed, detail: str = "") -> bool:
        ok = bool(passed)
        self.assertions.append(
            {"name": check_name, "passed": ok, "detail": str(detail)}
        )
        if not ok:
            print(f"  FAIL [{self.name}] {check_name}: {detail}", flush=True)
        return ok

    @property
    def passed(self) -> bool:
        return all(a["passed"] for a in self.assertions)

    @property
    def failed(self) -> list[dict]:
        return [a for a in self.assertions if not a["passed"]]


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


def _make_topology(kind: str, grayscale: bool = False):
    """8-env vision vec-env on the given topology ('dummy' | 'subproc')."""
    if kind == "dummy":
        return make_vec_env(
            SO100PickPlaceEnv,
            n_envs=N_ENVS,
            seed=SEED,
            env_kwargs=_vision_env_kwargs(grayscale),
        )
    if kind == "subproc":
        return make_vec_env(
            SO100PickPlaceEnv,
            n_envs=N_ENVS,
            seed=SEED,
            env_kwargs=_vision_env_kwargs(grayscale),
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "spawn"},  # Pitfall 8: via vec_env_kwargs
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
    """A subproc construction/readiness exchange desynced (macOS spawn race)."""


def _on_construction_alarm(signum, frame) -> None:
    raise _ConstructionTimeout()


def _kill_venv(venv) -> None:
    """Terminate hung spawn workers and close their pipes so a retry
    starts from a clean worker set."""
    for p in venv.processes:
        p.terminate()
    for p in venv.processes:
        p.join(5)
    for r in venv.remotes:
        try:
            r.close()
        except Exception:
            pass


def _make_topology_robust(kind: str, grayscale: bool = False):
    """Construct a vec-env, hardened against the macOS spawn bootstrap race.

    DummyVecEnv is in-process (no pipes) and constructed directly. A spawn
    SubprocVecEnv gets: construction + worker-ready delay + an all-worker
    get_attr readiness barrier, all under a SIGALRM watchdog with bounded
    retry and full worker cleanup — a desynced attempt is killed and
    rebuilt instead of wedging the run (see the SPAWN_* constants block
    for the measured evidence). Once the barrier round-trip completes,
    every worker is provably in its command loop, so the first real
    command batch (the protocol's reset/warm-up) cannot race the spawn
    bootstrap. Timed regions are never armed.
    """
    if kind != "subproc":
        return _make_topology(kind, grayscale)
    signal.signal(signal.SIGALRM, _on_construction_alarm)
    last: Exception | None = None
    for attempt in range(1, CONSTRUCTION_ATTEMPTS + 1):
        venv = None
        signal.setitimer(signal.ITIMER_REAL, CONSTRUCTION_TIMEOUT_S)
        try:
            venv = _make_topology(kind, grayscale)
            # Workers need ~2 s to finish importing __main__ and reach
            # their command loop; a batch sent before that can desync.
            time.sleep(SPAWN_WORKER_READY_DELAY_S)
            # Readiness barrier: a harmless round-trip to ALL 8 workers.
            # Only when this returns is every worker provably in its
            # command loop.
            venv.get_attr("spec")
            signal.setitimer(signal.ITIMER_REAL, 0)
            if attempt > 1:
                print(
                    f"[smoke] subproc construction succeeded on attempt "
                    f"{attempt}/{CONSTRUCTION_ATTEMPTS} (macOS spawn race)",
                    flush=True,
                )
            return venv
        except _ConstructionTimeout as exc:
            last = exc
            print(
                f"[smoke] subproc construction attempt "
                f"{attempt}/{CONSTRUCTION_ATTEMPTS} desynced (> "
                f"{CONSTRUCTION_TIMEOUT_S:.0f} s — macOS spawn race); "
                f"killing workers and retrying",
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
        f"{CONSTRUCTION_ATTEMPTS} attempts (macOS spawn race)"
    ) from last


# ---------------------------------------------------------------------------
# Phase A — D-07 topology benchmark (benchmark.py protocol)
# ---------------------------------------------------------------------------


def _bench_topology(kind: str) -> tuple[float, list[float]]:
    """Median env-steps/sec over TOPO_REPEATS repeats of warm-up-then-time."""
    repeats: list[float] = []
    action = np.tile(np.asarray(C.HOME_CTRL, dtype=np.float32), (N_ENVS, 1))
    for _ in range(TOPO_REPEATS):
        venv = _make_topology_robust(kind)  # constructed OUTSIDE the timed region
        try:
            venv.reset()
            for _ in range(TOPO_WARMUP_STEPS):  # untimed warm-up
                venv.step(action)
            t0 = time.perf_counter()
            for _ in range(TOPO_TIMED_STEPS):  # timed region
                venv.step(action)
            elapsed = time.perf_counter() - t0
        finally:
            _close_if_safe(venv, kind)
        repeats.append(TOPO_TIMED_STEPS * N_ENVS / elapsed)
        print(
            f"[smoke] topology {kind}: repeat {len(repeats)}/{TOPO_REPEATS} "
            f"{repeats[-1]:.1f} env steps/s ({elapsed:.2f} s for "
            f"{TOPO_TIMED_STEPS * N_ENVS} env steps)",
            flush=True,
        )
    return statistics.median(repeats), repeats


# ---------------------------------------------------------------------------
# Phase B — per-format timed PPO smoke train + D-08 checkpoint bundle
# ---------------------------------------------------------------------------


def _build_chain(kind: str, grayscale: bool):
    """The proven 03-02 chain on the winning topology (RESEARCH Pattern 3).

    VecNormalize(norm_obs_keys=[C.OBS_STATE_KEY]) — the SB3 2.9.0 parameter
    name (Pitfall 1) — normalizes the state key only; the uint8 image key
    stays out of obs_rms (SB3 divides images by 255 internally, Pitfall 9).
    Returns (transposed_chain, vec_normalize) so the stats object can be
    saved alongside the model (D-08).
    """
    venv = _make_topology_robust(kind, grayscale)
    norm = VecNormalize(
        venv,
        norm_obs_keys=[C.OBS_STATE_KEY],
        norm_obs=True,
        norm_reward=True,
        gamma=C.REWARD_GAMMA,  # Pitfall 11: reads the constant
    )
    stacked = ImageOnlyFrameStack(norm, n_stack=C.OBS_N_STACK)
    return VecTransposeImage(stacked), norm


def _smoke_train(kind: str, fmt: str, run_dir: Path, coll: _Collector) -> dict:
    """One timed 16384-step PPO train + checkpoint bundle save (D-06/D-08)."""
    grayscale = fmt == "grayscale"
    chain, norm = _build_chain(kind, grayscale)
    fmt_dir = run_dir / fmt
    fmt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tb" / fmt

    model = PPO(
        "MultiInputPolicy",
        chain,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=C.REWARD_GAMMA,  # Pitfall 11: reads the constant
        seed=SEED,
        device="cpu",
        tensorboard_log=str(tb_dir),
        verbose=0,  # the runner prints its own progress
    )
    print(f"[smoke] {fmt}: training {SMOKE_TOTAL_TIMESTEPS} steps ...", flush=True)
    t0 = time.perf_counter()
    model.learn(total_timesteps=SMOKE_TOTAL_TIMESTEPS)
    elapsed = time.perf_counter() - t0
    steps_per_sec = SMOKE_TOTAL_TIMESTEPS / elapsed

    ep_rewards = [float(e["r"]) for e in model.ep_info_buffer]
    mean_ep_reward = float(np.mean(ep_rewards)) if ep_rewards else None

    model_path = fmt_dir / "ppo_smoke"
    stats_path = fmt_dir / "vecnormalize.pkl"
    model.save(str(model_path))
    norm.save(str(stats_path))

    coll.check(
        f"{fmt}_train_completed",
        steps_per_sec > 0,
        f"{steps_per_sec:.1f} steps/s over {SMOKE_TOTAL_TIMESTEPS} steps "
        f"({elapsed:.1f} s)",
    )
    # D-06 soft check: recorded for the trend, NO slope threshold at 16k steps.
    coll.check(
        f"{fmt}_episodic_reward_recorded",
        bool(ep_rewards),
        f"{len(ep_rewards)} episodes in ep_info_buffer, mean r = "
        f"{mean_ep_reward if mean_ep_reward is not None else 'n/a'}",
    )
    print(
        f"[smoke] {fmt}: {steps_per_sec:.1f} steps/s ({elapsed:.1f} s), "
        f"mean episodic reward {mean_ep_reward} over {len(ep_rewards)} episodes",
        flush=True,
    )
    return {
        "steps_per_sec": steps_per_sec,
        "mean_ep_reward": mean_ep_reward,
        "n_episodes_in_buffer": len(ep_rewards),
        "elapsed_s": elapsed,
        "tb_dir": str(tb_dir),
        "model_path": str(model_path) + ".zip",
        "vecnormalize_path": str(stats_path),
        "_model": model,  # in-memory handle for the reload proof
        "_norm": norm,
        "_kind": kind,
    }


# ---------------------------------------------------------------------------
# Phase C — D-09 strict-order reload + bitwise action equality
# ---------------------------------------------------------------------------


def _reload_proof(kind: str, fmt: str, run_dir: Path, trained, coll: _Collector) -> bool:
    """Reload the bundle in the strict order and prove bitwise equality."""
    stats_path = run_dir / fmt / "vecnormalize.pkl"
    model_path = run_dir / fmt / "ppo_smoke"

    # Strict order (Pitfall 4): fresh identical vec-env -> VecNormalize.load
    # -> rebuild stack -> rebuild transpose -> PPO.load().set_env. set_env on
    # the bare loaded VecNormalize raises a space mismatch (verified failure).
    fresh_venv = _make_topology_robust(kind, grayscale=(fmt == "grayscale"))
    loaded_norm = VecNormalize.load(str(stats_path), fresh_venv)
    loaded_chain = VecTransposeImage(ImageOnlyFrameStack(loaded_norm, n_stack=C.OBS_N_STACK))
    loaded_model = PPO.load(str(model_path), device="cpu")
    loaded_model.set_env(loaded_chain)

    # fixed_obs built in the LOADED model's observation-space shapes — read
    # from the space, never hand-counted (Pitfall 5).
    fixed_obs = {
        C.OBS_STATE_KEY: np.zeros(
            (1, *loaded_model.observation_space[C.OBS_STATE_KEY].shape),
            dtype=np.float32,
        ),
        C.OBS_IMAGE_KEY: np.full(
            (1, *loaded_model.observation_space[C.OBS_IMAGE_KEY].shape),
            128,
            dtype=np.uint8,
        ),
    }
    coll.check(
        f"{fmt}_state_shape_from_space",
        loaded_model.observation_space[C.OBS_STATE_KEY].shape == (C.OBS_STATE_DIM,),
        f"loaded state space shape "
        f"{loaded_model.observation_space[C.OBS_STATE_KEY].shape} != "
        f"({C.OBS_STATE_DIM},)",
    )
    img_shape = loaded_model.observation_space[C.OBS_IMAGE_KEY].shape
    coll.check(
        f"{fmt}_image_shape_transposed_from_space",
        len(img_shape) == 3 and img_shape[-2:] == (C.OBS_IMAGE_SIZE, C.OBS_IMAGE_SIZE),
        f"loaded image space shape {img_shape} is not a transposed "
        f"stacked (*C, {C.OBS_IMAGE_SIZE}, {C.OBS_IMAGE_SIZE}) frame",
    )

    a, _ = trained.predict(fixed_obs, deterministic=True)
    b, _ = loaded_model.predict(fixed_obs, deterministic=True)
    bitwise = np.asarray(a).tobytes() == np.asarray(b).tobytes()
    coll.check(
        f"{fmt}_reload_bitwise_equal",
        bitwise,
        "trained.predict vs reloaded.predict differ on identical fixed_obs "
        "(deterministic=True)",
    )
    print(f"[smoke] {fmt}: reload bitwise {'EQUAL' if bitwise else 'MISMATCH'}", flush=True)
    _close_if_safe(loaded_chain, kind)
    return bitwise


# ---------------------------------------------------------------------------
# Phase D — D-05 format selection + D-04 gate evaluation + results JSON
# ---------------------------------------------------------------------------


def _evaluate_gate(gate: dict, formats: dict, coll: _Collector) -> dict:
    """Apply the pinned rules; return the gate block for the results JSON."""
    # The floor comes from the pin — never re-hardcoded. The pin's own
    # arithmetic is re-checked so a tampered/inconsistent pin fails loudly.
    required = math.ceil(gate["phase4_budget_steps"] / (gate["window_hours"] * 3600))
    coll.check(
        "gate_pin_arithmetic",
        required == gate["required_throughput_steps_per_sec"],
        f"ceil(budget/window) = {required} != pinned floor "
        f"{gate['required_throughput_steps_per_sec']}",
    )

    rgb_sps = formats["rgb"]["steps_per_sec"]
    gray_sps = formats["grayscale"]["steps_per_sec"]
    # D-05, verbatim from the pin: "higher measured steps/sec wins; within a
    # relative 10% band -> grayscale".
    rel_diff = abs(rgb_sps - gray_sps) / max(rgb_sps, gray_sps)
    if rel_diff <= TIE_BAND_REL:
        chosen_fmt = "grayscale"
        d05_applied = (
            f"tie band: rgb {rgb_sps:.1f} vs grayscale {gray_sps:.1f} steps/s "
            f"(relative difference {rel_diff * 100:.1f}% <= {TIE_BAND_REL * 100:.0f}%) "
            f"-> grayscale"
        )
    else:
        chosen_fmt = "rgb" if rgb_sps > gray_sps else "grayscale"
        d05_applied = (
            f"higher wins: rgb {rgb_sps:.1f} vs grayscale {gray_sps:.1f} steps/s "
            f"(relative difference {rel_diff * 100:.1f}% > {TIE_BAND_REL * 100:.0f}%) "
            f"-> {chosen_fmt}"
        )

    measured = formats[chosen_fmt]["steps_per_sec"]
    floor = gate["required_throughput_steps_per_sec"]
    margin_pct = (measured - floor) / floor * 100.0
    if measured >= floor:
        decision = f"vision-64x64-{'gray' if chosen_fmt == 'grayscale' else 'rgb'}"
    else:
        decision = "state-only-fallback"
    coll.check(
        "gate_evaluated",
        decision
        in ("vision-64x64-gray", "vision-64x64-rgb", "state-only-fallback"),
        f"decision {decision!r} is not one of the three allowed values",
    )
    return {
        "phase4_budget_steps": gate["phase4_budget_steps"],
        "window_hours": gate["window_hours"],
        "required_throughput_steps_per_sec": floor,
        "measured_steps_per_sec": measured,
        "margin_pct": margin_pct,
        "decision": decision,
        "chosen_format": chosen_fmt,
        "d05_applied": d05_applied,
    }


def main() -> int:
    # Global tripwire (see FAULTHANDLER_TRIPWIRE_S): if any exchange wedges
    # mid-run, dump every thread's Python traceback and exit nonzero —
    # the EXIT-CODE CONTRACT forbids a silent infinite hang.
    faulthandler.dump_traceback_later(FAULTHANDLER_TRIPWIRE_S, exit=True)
    gate = json.loads(GATE_PATH.read_text())
    coll = _Collector("sb3_smoke")

    run_dir = RUNS_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase A: D-07 topology benchmark ---------------------------------
    topology: dict = {}
    for kind in ("dummy", "subproc"):
        median, repeats = _bench_topology(kind)
        topology[kind] = {
            "steps_per_sec": median,
            "repeats": repeats,
            "warmup_vec_steps": TOPO_WARMUP_STEPS,
            "timed_vec_steps": TOPO_TIMED_STEPS,
            "timed_env_steps": TOPO_TIMED_STEPS * N_ENVS,
            "n_envs": N_ENVS,
            "repeats_count": len(repeats),
        }
        print(f"[smoke] topology {kind}: median {median:.1f} env steps/s", flush=True)
    coll.check(
        "topology_both_measured",
        topology["dummy"]["steps_per_sec"] > 0 and topology["subproc"]["steps_per_sec"] > 0,
        f"dummy {topology['dummy']['steps_per_sec']}, "
        f"subproc {topology['subproc']['steps_per_sec']}",
    )
    winner = (
        "dummy"
        if topology["dummy"]["steps_per_sec"] >= topology["subproc"]["steps_per_sec"]
        else "subproc"
    )
    topology["winner"] = winner
    print(f"[smoke] topology winner: {winner}", flush=True)

    # --- Phases B + C: per-format timed trains + reload proofs ------------
    formats: dict = {}
    for fmt in FORMATS:
        record = _smoke_train(winner, fmt, run_dir, coll)
        model = record.pop("_model")
        record.pop("_norm")
        kind = record.pop("_kind")
        record["bitwise_equal"] = _reload_proof(winner, fmt, run_dir, model, coll)
        formats[fmt] = record

    # --- Phase D: gate evaluation + results JSON --------------------------
    gate_block = _evaluate_gate(gate, formats, coll)

    versions = package_versions()
    for p in ("stable-baselines3", "tensorboard"):
        versions[p] = pkg_version(p)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine": machine_context(),
        "versions": versions,
        "smoke_config": {
            "total_timesteps": SMOKE_TOTAL_TIMESTEPS,
            "n_envs": N_ENVS,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "seed": SEED,
            "device": "cpu",
            "gamma": C.REWARD_GAMMA,
            "n_stack": C.OBS_N_STACK,
            "norm_obs_keys": [C.OBS_STATE_KEY],
            "norm_reward": True,
        },
        "topology": topology,
        "formats": formats,
        "gate": gate_block,
        "checks": coll.assertions,
    }
    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    if coll.passed:
        print(
            f"[smoke] all {len(coll.assertions)} collector checks passed "
            f"(rgb {formats['rgb']['steps_per_sec']:.1f} / "
            f"grayscale {formats['grayscale']['steps_per_sec']:.1f} steps/s; "
            f"measured {gate_block['measured_steps_per_sec']:.1f} vs floor "
            f"{gate_block['required_throughput_steps_per_sec']} steps/s, "
            f"margin {gate_block['margin_pct']:+.1f}%)",
            flush=True,
        )
        print(f"SMOKE PASSED: {gate_block['decision']}", flush=True)
        print(f"RESULTS_JSON={out_path}", flush=True)
        faulthandler.cancel_dump_traceback_later()
        return 0
    print("SMOKE FAILED: failing checks:", flush=True)
    for a in coll.failed:
        print(f"  {a['name']}: {a['detail']}", flush=True)
    print(f"RESULTS_JSON={out_path}", flush=True)
    faulthandler.cancel_dump_traceback_later()
    return 1


if __name__ == "__main__":
    sys.exit(main())