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

macOS GL-wedge hardening (diagnosed with C-stack evidence — see
.planning/debug/subproc-topology-benchmark-hang.md): a spawn worker's
FIRST render — lazy offscreen GL context creation plus the first
mjr_readPixels, both inside env.reset() AFTER the readiness barrier —
can wedge non-deterministically in Apple's OpenGL-on-Metal layer under
8-way concurrent first use; the parent then blocks in SB3's in-order
remote.recv. The hardened construction lives EXACTLY ONCE, in the
package module so100rob._hardened_venv (D-07 — the STATE.md Phase 4
contract, extracted by 04-01 Task 2): it watches EVERY untimed setup
exchange (construction + readiness barrier + per-worker first-render
warm-up + the benchmark's reset/warm-up) with bounded
kill/rebuild/retry, never arms a timed region, and this runner arms its
computed SETUP_TRIPWIRE_S as the backstop of last resort (the old
25-min hardcode is gone — IN-09).
"""

from __future__ import annotations

import faulthandler
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecTransposeImage

from so100rob import constants as C
from so100rob._hardened_venv import (  # D-07: the hardened construction, imported — never re-defined
    BENCH_SETUP_ATTEMPTS,
    BENCH_SETUP_TIMEOUT_S,
    N_ENVS,
    SEED,
    SETUP_TRIPWIRE_S,
    TOPO_REPEATS,
    TOPO_WARMUP_STEPS,
    _close_if_safe,
    _ConstructionTimeout,
    _kill_venv,
    _make_topology_robust,
    _watched_reset_warmup,
)
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
#
# The construction constants (N_ENVS, SEED, TOPO_REPEATS,
# TOPO_WARMUP_STEPS, the SPAWN_*/BENCH_*/CONSTRUCTION_* hardening budgets,
# SETUP_TRIPWIRE_S, and the hardened construction functions themselves)
# live in so100rob._hardened_venv (D-07) and are imported above — the
# hardened construction exists EXACTLY ONCE (04-01 Task 2), so the Phase 4
# training run (04-03) reuses the proven path instead of a copy. Only the
# runner-specific smoke budgets stay here.
# ---------------------------------------------------------------------------

SMOKE_TOTAL_TIMESTEPS = 16384  # D-06: pipeline-integrity smoke, no slope bar
N_STEPS = 128                  # PPO rollout length per env
BATCH_SIZE = 64                # divides the 1024-step rollout buffer (Pitfall 7)
N_EPOCHS = 2                   # PPO epochs per update
TOPO_TIMED_STEPS = 1000        # timed vec steps per repeat (x N_ENVS env steps) — untimed/timed split stays at the benchmark layer
FORMATS = ("rgb", "grayscale")  # D-05 candidates, measured identically

# D-05 tie band: READ from the pin's machine-readable
# format_rule_tie_band_rel field at gate evaluation (config/obs_gate.json —
# WR-02), never re-hardcoded here. The pin's format_rule prose ("higher
# measured steps/sec wins; within a relative 10% band -> grayscale") is
# cross-checked against the numeric field in _evaluate_gate so a re-pin
# that changes either side fails loudly instead of silently disagreeing
# with the committed pin.


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
# The hardened construction helpers (_vision_env_kwargs, _make_topology,
# _close_if_safe, _ConstructionTimeout, _on_construction_alarm, _kill_venv,
# _make_topology_robust, _watched_reset_warmup) live in
# so100rob._hardened_venv (D-07) and are imported at the top — the
# hardened construction exists EXACTLY ONCE (04-01 Task 2). The two
# functions below are the runner-specific layers: the timed benchmark
# protocol (_bench_topology) and the PPO chains, all consuming the
# imported hardened path.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase A — D-07 topology benchmark (benchmark.py protocol)
# ---------------------------------------------------------------------------


def _bench_topology(kind: str) -> tuple[float, list[float]]:
    """Median env-steps/sec over TOPO_REPEATS repeats of warm-up-then-time.

    Coverage contract (G-03-2): every UNTIMED setup exchange is bounded
    for subproc — construction + readiness barrier + per-worker
    first-render warm-up (inside _make_topology_robust) and this
    benchmark's reset + warm-up steps (inside _watched_reset_warmup) —
    with bounded kill/rebuild/retry-the-repeat on timeout. The timed
    region (the TOPO_TIMED_STEPS perf_counter loop) is NEVER armed."""
    repeats: list[float] = []
    action = np.tile(np.asarray(C.HOME_CTRL, dtype=np.float32), (N_ENVS, 1))
    for _ in range(TOPO_REPEATS):
        elapsed: float | None = None
        venv = None
        for attempt in range(1, BENCH_SETUP_ATTEMPTS + 1):
            venv = _make_topology_robust(kind)  # constructed OUTSIDE the timed region
            try:
                _watched_reset_warmup(venv, kind, action)
                t0 = time.perf_counter()
                for _ in range(TOPO_TIMED_STEPS):  # timed region
                    venv.step(action)
                elapsed = time.perf_counter() - t0
                break
            except _ConstructionTimeout:
                print(
                    f"[smoke] topology {kind}: setup attempt "
                    f"{attempt}/{BENCH_SETUP_ATTEMPTS} desynced (> "
                    f"{BENCH_SETUP_TIMEOUT_S:.0f} s — macOS GL-on-Metal "
                    f"first-render wedge); killing workers and retrying "
                    f"the repeat",
                    flush=True,
                )
                # Kill, NEVER close: SubprocVecEnv.close() joins workers
                # with NO timeout, so closing a wedged venv re-hangs the
                # run; _kill_venv already closes the remotes. Only the
                # healthy path closes (below, via _close_if_safe).
                _kill_venv(venv)
        if elapsed is None:
            raise RuntimeError(
                f"topology {kind}: untimed setup desynced after "
                f"{BENCH_SETUP_ATTEMPTS} attempts (macOS GL-on-Metal "
                f"first-render wedge — see "
                f".planning/debug/subproc-topology-benchmark-hang.md)"
            )
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

    # The D-05 tie band ALSO comes from the pin (WR-02): the
    # machine-readable format_rule_tie_band_rel field, cross-checked
    # against the pin's format_rule prose. A re-pin that changes either
    # side (or drops the field) fails loudly here instead of the runner
    # silently applying a different selection rule than the committed pin.
    if "format_rule_tie_band_rel" not in gate:
        raise ValueError(
            "config/obs_gate.json is missing the D-05 tie band field "
            "'format_rule_tie_band_rel' — the runner never re-hardcodes it"
        )
    tie_band_rel = float(gate["format_rule_tie_band_rel"])
    prose_band = f"{tie_band_rel * 100:.0f}% band"
    coll.check(
        "gate_tie_band_pin_consistent",
        0.0 < tie_band_rel < 1.0 and prose_band in str(gate.get("format_rule", "")),
        f"pin field format_rule_tie_band_rel={tie_band_rel} is inconsistent "
        f"with the format_rule prose (expected {prose_band!r} in "
        f"{gate.get('format_rule')!r})",
    )

    rgb_sps = formats["rgb"]["steps_per_sec"]
    gray_sps = formats["grayscale"]["steps_per_sec"]
    # D-05, verbatim from the pin: "higher measured steps/sec wins; within a
    # relative 10% band -> grayscale".
    rel_diff = abs(rgb_sps - gray_sps) / max(rgb_sps, gray_sps)
    if rel_diff <= tie_band_rel:
        chosen_fmt = "grayscale"
        d05_applied = (
            f"tie band: rgb {rgb_sps:.1f} vs grayscale {gray_sps:.1f} steps/s "
            f"(relative difference {rel_diff * 100:.1f}% <= {tie_band_rel * 100:.0f}%) "
            f"-> grayscale"
        )
    else:
        chosen_fmt = "rgb" if rgb_sps > gray_sps else "grayscale"
        d05_applied = (
            f"higher wins: rgb {rgb_sps:.1f} vs grayscale {gray_sps:.1f} steps/s "
            f"(relative difference {rel_diff * 100:.1f}% > {tie_band_rel * 100:.0f}%) "
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
    # Global SETUP tripwire (SETUP_TRIPWIRE_S from so100rob._hardened_venv —
    # the COMPUTED worst-case bound over every watchdogged untimed-setup
    # window, IN-09): if an exchange still wedges through some future
    # watchdog coverage gap, dump every thread's Python traceback and exit
    # nonzero — the EXIT-CODE CONTRACT forbids a silent infinite hang.
    faulthandler.dump_traceback_later(SETUP_TRIPWIRE_S, exit=True)
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
        # Close the trained chain AFTER the reload proof (the trained
        # model's predict() never touches its vec env): on the subproc
        # winner this kills the format's 8 spawn workers instead of
        # leaving them idle with live GL contexts through the second
        # train, both reloads, the gate evaluation, and the JSON write —
        # stale render-capable workers are exactly the GL-contention
        # exposure the hardening exists to remove. No-op for dummy.
        _close_if_safe(model.env, kind)
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