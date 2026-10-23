#!/usr/bin/env python
"""Production training runner (04-03, TRAIN-01 mechanical half).

One proven end-to-end path — a reorganization/extension of the smoke
runner's train -> bundle -> strict-order-reload pipeline, NOT a
from-scratch build (04-CONTEXT.md code insights). Every hardening and
pin from 04-01/04-02 is consumed through so100rob._hardened_venv and
config/train_gate.json, none re-implemented:

  * construction — the 8-env subproc spawn chain through the hardened
    path ONLY (D-07): make_vec_env(SubprocVecEnv-spawn) -> VecNormalize
    (norm_obs_keys=["state"], norm_obs=True, norm_reward=True,
    gamma=C.REWARD_GAMMA) -> ImageOnlyFrameStack(4) -> VecTransposeImage
    -> PPO(MultiInputPolicy, device="cpu"), per-worker env kwargs
    SO100PickPlaceEnv(obs_mode="vision", grayscale=True,
    spawn_mode="continuous") — the gate's spawn_mode (Pitfall 1: a
    grid-default training env silently trains on one fixed cube
    position), each env wrapped in gymnasium RescaleAction(-1, 1) per
    the pinned action_space_mode (04-02 Task 1) INSIDE the env factory,
    below VecNormalize, above the raw env.
  * two-window faulthandler scheme (D-04/D-07): the SETUP tripwire
    (SETUP_TRIPWIRE_S, computed in so100rob._hardened_venv) is armed
    around every hardened construction and cancelled once it succeeds;
    the train-window tripwire (train_window_hours x 3600 READ from the
    gate) is armed immediately before learn() and cancelled after —
    never armed over any timed region.
  * progress — a custom callback (no tqdm/rich, no progress_bar=True —
    Pitfall 8) logging every episode's RAW return (Monitor's
    info["episode"]["r"] — raw below VecNormalize), is_success, and
    stage flags to stdout every STDOUT_EPISODE_INTERVAL episodes and to
    TensorBoard (tb/<run-name>/).
  * bundle — the save shape is contract for 04-04/04-05/Phase 5+:
    exactly four artifacts in the run dir (ppo_final.zip,
    vecnormalize.pkl, config.json with the seed/lr/budget/gate echo,
    gate.json = the verbatim committed pin). The final step's bundle is
    the ONLY selection candidate (D-05); periodic checkpoints, if any,
    are crash-recovery only.
  * reload proof — the D-09 strict order (fresh hardened venv ->
    VecNormalize.load -> stack -> transpose -> PPO.load().set_env) with
    a bitwise action-equality check and an in-space predict check, in
    TRAINING mode (the loaded normalizer keeps its saved flags; the
    WR-03 eval-mode recipe lives in reload_bundle(eval_mode=True) for
    the 04-04 eval/audit runners).

ZERO pinned numbers are hardcoded here: gamma resolves through the
gate's gamma_source to so100rob.constants.REWARD_GAMMA, n_steps /
batch_size / n_envs / device come from fixed_overrides, the budget,
window, seed, spawn_mode, action-space mode, and tuning protocol come
from config/train_gate.json (WR-02/T-04-07 — the resolved-config gate
echo in every bundle proves what ran).

EXIT-CODE CONTRACT (the smoke runner's public interface): exit 0 ONLY
when the run trained through the full hardened chain, the bundle was
saved complete, and the reload proof passed — with "TRAIN PASSED:
<detail>" printed and "BUNDLE_DIR=<path>" as the LAST stdout line;
exit 1 on any failure, with the exception and a remediation hint on
stderr — never a silent partial bundle.

All runs must go through the pinned 3.12 venv (established project
discipline): ``uv run python scripts/train.py``. The ``if __name__ ==
"__main__":`` guard is MANDATORY: spawn-method SubprocVecEnv workers
re-import this module, and without the guard each worker would
recursively spawn workers (Pitfall 8).
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib
import importlib.metadata
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, VecTransposeImage

from so100rob import constants as C
from so100rob._hardened_venv import (  # D-07: the hardened construction, imported — never re-defined
    SEED,
    SETUP_TRIPWIRE_S,
    _close_if_safe,
    _make_topology_robust,
    arm_train_tripwire,
)
from so100rob.vec_wrappers import ImageOnlyFrameStack

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "config" / "train_gate.json"
RUNS_ROOT = REPO_ROOT / "runs" / "train"
TUNE_ROOT = RUNS_ROOT / "tune"  # the tuning pass: <variant>/ bundles + selection.json
TB_ROOT = REPO_ROOT / "tb"
GATE_SCHEMA = "so100rob/train_gate/1"
CONFIG_SCHEMA = "so100rob/train_run_config/1"

# ---------------------------------------------------------------------------
# Runner-specific operational constants (the smoke-runner pattern: module
# constants with decision-ID docstrings). Every number the gate pins is
# READ from the gate/constants at run time — never restated here (WR-02).
# ---------------------------------------------------------------------------

GRAYSCALE = True  # OBS_DECISION D-05: vision-64x64-gray (the Phase 3 format decision)
N_EPOCHS = 2  # the smoke/D-06 epoch count (a speed choice, not a tuned one)
STDOUT_EPISODE_INTERVAL = 20  # progress print cadence (operational, not a gate pin)
LAST_N_EPISODES = 100  # the gate's selection_rule "last-100-episode mean raw return" window
CHECKPOINT_INTERVAL_STEPS = 250_000  # D-05: periodic-checkpoint cadence — CRASH RECOVERY ONLY, never a selection candidate
VERSION_PACKAGES = (  # provenance block in every bundle config (operational)
    "mujoco",
    "gymnasium",
    "stable-baselines3",
    "torch",
    "tensorboard",
    "numpy",
)


# ---------------------------------------------------------------------------
# Gate reading + config resolution (WR-02/T-04-07: read, never re-hardcode)
# ---------------------------------------------------------------------------


def load_gate() -> dict:
    """Read config/train_gate.json and fail loudly on a schema mismatch —
    a drifted or foreign pin must never train."""
    gate = json.loads(GATE_PATH.read_text())
    if gate.get("schema") != GATE_SCHEMA:
        raise ValueError(
            f"config/train_gate.json schema {gate.get('schema')!r} != "
            f"{GATE_SCHEMA!r} — refusing to train from a foreign/unknown pin"
        )
    return gate


def resolve_gamma(gate: dict) -> float:
    """PPO's discount, resolved through the gate's gamma_source path to the
    pinned constant (Pitfall 2: a divergent discount silently invalidates
    the Phase 2 reward design — never a literal)."""
    source = str(gate["fixed_overrides"]["gamma_source"])
    module_name, attr = source.rsplit(".", 1)
    value = getattr(importlib.import_module(module_name), attr)
    return float(value)


def resolve_env_kwargs(gate: dict) -> dict:
    """Per-worker training env kwargs: the pinned 64x64-gray vision
    observation, with spawn_mode from the gate (Pitfall 1: training envs
    are continuous — the grid default would fix the cube at spawn 0)."""
    return dict(
        obs_mode=C.OBS_MODE_VISION,
        render_mode="rgb_array",
        width=C.OBS_IMAGE_SIZE,
        height=C.OBS_IMAGE_SIZE,
        grayscale=GRAYSCALE,
        spawn_mode=gate["spawn_mode"],
    )


def resolve_action_wrapper(gate: dict):
    """The pinned action-space mode (04-02 Task 1) as a
    make_vec_env(wrapper_class=..., wrapper_kwargs=...) pair.

    rescale_minus_one_one -> gymnasium RescaleAction(gate rescale_min/max)
    around each raw env, applied by the env factory INSIDE each worker —
    below VecNormalize, above the raw env (the order is the contract).
    raw_ctrlrange -> no wrapper (the smoke-proven raw ctrlrange Box).
    Anything else fails loudly instead of silently training raw.
    """
    mode = gate["action_space_mode"]
    if mode == "rescale_minus_one_one":
        return RescaleAction, dict(
            min_action=gate["rescale_min"], max_action=gate["rescale_max"]
        )
    if mode == "raw_ctrlrange":
        return None, None
    raise ValueError(
        f"unknown action_space_mode {mode!r} in config/train_gate.json — "
        "expected 'rescale_minus_one_one' or 'raw_ctrlrange'"
    )


def _default_steps(gate: dict, mode: str) -> int:
    """Per-mode budget, every branch READ from the gate (D-04/D-03)."""
    if mode == "retry":
        return int(gate["retry"]["retry_budget_steps"])
    if mode == "tune":
        return int(gate["tuning"]["steps_per_variant"])
    return int(gate["training_budget_steps"])


def resolve_config(
    gate: dict,
    mode: str = "run",
    steps: int | None = None,
    lr: float | None = None,
    run_name: str = "train",
) -> dict:
    """Resolve EVERY pinned number from the gate/constants into one
    JSON-serializable config object (the provenance record saved into the
    bundle). Divisibility is re-checked fail-loudly so an inconsistent
    gate can never train (Pitfall 7)."""
    fo = gate["fixed_overrides"]
    n_steps = int(fo["n_steps"])
    batch_size = int(fo["batch_size"])
    n_envs = int(fo["n_envs"])
    if (n_envs * n_steps) % batch_size != 0:
        raise ValueError(
            f"gate divisibility violation: n_envs x n_steps = "
            f"{n_envs} x {n_steps} = {n_envs * n_steps} is not divisible "
            f"by batch_size={batch_size} (Pitfall 7)"
        )
    return {
        "schema": CONFIG_SCHEMA,
        "mode": mode,
        "run_name": run_name,
        "steps": int(steps) if steps is not None else _default_steps(gate, mode),
        "learning_rate": (
            float(lr)
            if lr is not None
            else float(gate["tuning"]["tie_fallback_lr"])  # pinned middle grid entry / SB3 default
        ),
        "seed": int(gate["tuning"]["seed"]),
        "gamma": resolve_gamma(gate),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_envs": n_envs,
        "n_epochs": N_EPOCHS,
        "device": str(fo["device"]),
        "action_space_mode": gate["action_space_mode"],
        "spawn_mode": gate["spawn_mode"],
        "checkpoint_selection": gate["checkpoint_selection"],
        "grayscale": GRAYSCALE,
        "n_stack": C.OBS_N_STACK,
        "norm_obs_keys": [C.OBS_STATE_KEY],
        "env_kwargs": resolve_env_kwargs(gate),
        "gate_echo": {
            "training_budget_steps": gate["training_budget_steps"],
            "train_window_hours": gate["train_window_hours"],
            "action_space_mode": gate["action_space_mode"],
            "spawn_mode": gate["spawn_mode"],
            "checkpoint_selection": gate["checkpoint_selection"],
            "lr_grid": gate["tuning"]["lr_grid"],
            "seed": gate["tuning"]["seed"],
        },
    }


# ---------------------------------------------------------------------------
# Chain construction (the proven 03-02 chain, D-07 hardened path)
# ---------------------------------------------------------------------------


def build_chain(venv, gamma: float):
    """VecNormalize(norm_obs_keys=[state]) -> ImageOnlyFrameStack(4) ->
    VecTransposeImage over ANY already-constructed venv. Returns
    (transposed_chain, vec_normalize) so the stats object can be saved
    alongside the model (the bundle travels as a pair)."""
    norm = VecNormalize(
        venv,
        norm_obs_keys=[C.OBS_STATE_KEY],
        norm_obs=True,
        norm_reward=True,
        gamma=gamma,  # Pitfall 2: both gamma sites read the resolved constant
    )
    stacked = ImageOnlyFrameStack(norm, n_stack=C.OBS_N_STACK)
    return VecTransposeImage(stacked), norm


def _construct(gate: dict, cfg: dict):
    """Hardened training construction (D-07) under the SETUP tripwire:
    the tripwire is armed BEFORE the construction and cancelled once the
    chain is up — never armed over learn() (that is the train window) or
    any timed region."""
    wrapper_class, wrapper_kwargs = resolve_action_wrapper(gate)
    faulthandler.dump_traceback_later(SETUP_TRIPWIRE_S, exit=True)
    try:
        venv = _make_topology_robust(
            "subproc",
            grayscale=cfg["grayscale"],
            env_kwargs=cfg["env_kwargs"],
            wrapper_class=wrapper_class,
            wrapper_kwargs=wrapper_kwargs,
        )
        chain, norm = build_chain(venv, cfg["gamma"])
    except BaseException:
        faulthandler.cancel_dump_traceback_later()
        raise
    faulthandler.cancel_dump_traceback_later()
    return chain, norm


# ---------------------------------------------------------------------------
# Bundle save + reload (the save shape is contract for Phase 5+)
# ---------------------------------------------------------------------------


def save_bundle(model, norm, run_dir: Path, cfg: dict, gate: dict) -> Path:
    """Write the complete bundle: EXACTLY the four artifacts (ppo_final.zip,
    vecnormalize.pkl, config.json, gate.json). config.json carries the
    resolved config + gate echo + versions + timestamp — what actually
    ran is provable from the bundle alone (T-04-07)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "ppo_final"
    stats_path = run_dir / "vecnormalize.pkl"
    model.save(str(model_path))
    norm.save(str(stats_path))  # stats at the FINAL step (D-05)

    payload = dict(cfg)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["versions"] = {
        p: importlib.metadata.version(p) for p in VERSION_PACKAGES
    }
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2) + "\n")
    # The gate echo is the verbatim committed pin (git history proves the
    # pin precedes the run; the copy proves which pin the run consumed).
    (run_dir / "gate.json").write_text(GATE_PATH.read_text())
    return run_dir


def reload_bundle(run_dir: Path, venv, eval_mode: bool = False):
    """Reload a bundle in the STRICT order (D-09): VecNormalize.load ->
    ImageOnlyFrameStack -> VecTransposeImage -> PPO.load().set_env.

    Training-mode reload (eval_mode=False, the runner's own proof) keeps
    the normalizer's SAVED flags — the bundle reloads as a
    training-capable chain. eval_mode=True applies the WR-03 recipe
    (norm.training=False AND norm.norm_reward=False — the SB3 2.9.0
    attribute names) for the 04-04 eval/audit runners.
    """
    loaded_norm = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), venv)
    chain = VecTransposeImage(
        ImageOnlyFrameStack(loaded_norm, n_stack=C.OBS_N_STACK)
    )
    loaded_model = PPO.load(str(run_dir / "ppo_final"), device="cpu")
    loaded_model.set_env(chain)
    if eval_mode:
        loaded_norm.training = False
        loaded_norm.norm_reward = False
    return loaded_model, loaded_norm


def _fixed_obs(model) -> dict:
    """fixed_obs built in the LOADED model's observation-space shapes —
    read from the space, never hand-counted (Pitfall 5)."""
    return {
        C.OBS_STATE_KEY: np.zeros(
            (1, *model.observation_space[C.OBS_STATE_KEY].shape),
            dtype=np.float32,
        ),
        C.OBS_IMAGE_KEY: np.full(
            (1, *model.observation_space[C.OBS_IMAGE_KEY].shape),
            128,
            dtype=np.uint8,
        ),
    }


def reload_proof(trained_model, run_dir: Path, gate: dict, cfg: dict) -> bool:
    """Prove the saved bundle reloads as the SAME policy: a fresh hardened
    construction (SETUP tripwindow armed around it), strict-order reload
    in TRAINING mode (flags stay True), then a bitwise action-equality
    check plus an in-space predict check against the trained model."""
    fresh_chain, fresh_norm = _construct(gate, cfg)
    # The strict order re-stacks around the loaded normalizer, so the raw
    # venv (below VecNormalize) is what reload_bundle must attach to.
    loaded_model, _loaded_norm = reload_bundle(run_dir, fresh_norm.venv)
    fixed_obs = _fixed_obs(loaded_model)
    a, _ = trained_model.predict(fixed_obs, deterministic=True)
    b, _ = loaded_model.predict(fixed_obs, deterministic=True)
    bitwise = np.asarray(a).tobytes() == np.asarray(b).tobytes()
    space = loaded_model.action_space
    within = bool(np.all(np.asarray(b) >= space.low) and np.all(np.asarray(b) <= space.high))
    _close_if_safe(loaded_model.env, "subproc")
    run_name = cfg["run_name"]
    if not bitwise:
        print(
            f"[train:{run_name}] reload proof FAILED: trained.predict vs "
            "reloaded.predict differ on identical fixed_obs (deterministic=True)",
            file=sys.stderr,
            flush=True,
        )
        return False
    if not within:
        print(
            f"[train:{run_name}] reload proof FAILED: reloaded predict is "
            "outside the action space",
            file=sys.stderr,
            flush=True,
        )
        return False
    print(f"[train:{run_name}] reload proof: bitwise EQUAL, predict within space", flush=True)
    return True


# ---------------------------------------------------------------------------
# Tuning selection rule (D-06: a pure, unit-tested function)
# ---------------------------------------------------------------------------


def apply_selection_rule(gate: dict, episode_returns: dict) -> tuple[str, dict]:
    """The pinned tuning selection rule (gate tuning.selection_rule), as a
    pure function over per-variant episode logs: the metric is each
    variant's mean RAW episodic return over its LAST LAST_N_EPISODES
    completed episodes; the highest metric wins; a relative tie within
    tie_band_rel between the best two falls back to tie_fallback_lr (the
    pinned middle grid entry). Returns (chosen_lr_key, record) with the
    record JSON-serializable (it IS runs/train/tune/selection.json).

    Fail-loud pin cross-checks (the WR-02 prose pattern): the rule text
    and the numeric tie band must agree with what this function applies.
    """
    tuning = gate["tuning"]
    rule_prose = str(tuning["selection_rule"])
    tie_band_rel = float(tuning["tie_band_rel"])
    fallback = str(float(tuning["tie_fallback_lr"]))
    if f"last-{LAST_N_EPISODES}-episode" not in rule_prose:
        raise ValueError(
            f"gate tuning.selection_rule no longer names the "
            f"last-{LAST_N_EPISODES}-episode window this runner applies — "
            "re-pin or update the runner (WR-02 prose cross-check)"
        )
    if f"{tie_band_rel * 100:.0f}%" not in rule_prose:
        raise ValueError(
            f"gate tuning.tie_band_rel={tie_band_rel} disagrees with the "
            f"selection_rule prose (WR-02 prose cross-check)"
        )
    if not episode_returns:
        raise ValueError("selection rule needs at least one variant's episode log")

    metrics = {
        str(lr): float(np.mean(list(returns)[-LAST_N_EPISODES:]))
        for lr, returns in episode_returns.items()
    }
    if fallback not in metrics:
        raise ValueError(
            f"tie_fallback_lr {fallback!r} is not among the measured "
            f"variants {sorted(metrics)} — the fallback must be a grid entry"
        )
    ranked = sorted(metrics.items(), key=lambda kv: kv[1], reverse=True)
    best_lr, best_v = ranked[0]
    chosen = best_lr
    applied = (
        f"highest last-{LAST_N_EPISODES}-episode mean raw return: {best_lr} "
        f"({best_v:.4f})"
    )
    if len(ranked) >= 2:
        second_lr, second_v = ranked[1]
        if best_v == 0.0:
            # A zero best metric has no relative scale: only an exactly
            # equal second is a tie (0.0 vs -1.0 is a clear win, not a tie).
            rel = 0.0 if second_v == 0.0 else float("inf")
        else:
            rel = abs(best_v - second_v) / abs(best_v)
        if rel <= tie_band_rel:
            chosen = fallback
            applied = (
                f"relative tie within the {tie_band_rel * 100:.0f}% band "
                f"between the best two ({best_lr} {best_v:.4f} vs {second_lr} "
                f"{second_v:.4f}, relative difference {rel * 100:.2f}%) -> "
                f"tie_fallback_lr {fallback}"
            )
    record = {
        "metrics": metrics,
        "chosen_lr": chosen,
        "rule": rule_prose,
        "tie_band_rel": tie_band_rel,
        "tie_fallback_lr": fallback,
        "last_n_episodes": LAST_N_EPISODES,
        "applied": applied,
    }
    return chosen, record


# ---------------------------------------------------------------------------
# Retry escalation (D-03: one retry at 2x budget, then stop-and-investigate)
# ---------------------------------------------------------------------------


def check_retry_allowed(gate: dict, confirmed: bool = False) -> bool:
    """D-03 gate: retry mode refuses to run unless (1) the invocation is an
    explicitly confirmed D-03 response (the runner never loops or
    self-escalates), (2) the gate's retry.max_retries allows it, and (3)
    the retry budget is exactly 2x the training budget — every number read
    from the gate, never a literal."""
    if not confirmed:
        raise ValueError(
            "retry mode requires --confirm-retry: D-03 allows exactly ONE "
            "retry at 2x budget as a deliberate escalation response — the "
            "runner never loops or self-escalates"
        )
    max_retries = int(gate["retry"]["max_retries"])
    if max_retries < 1:
        raise ValueError(
            f"gate retry.max_retries={max_retries} forbids a retry "
            f"(D-03 on_second_miss={gate['retry']['on_second_miss']!r}: "
            "stop-and-investigate)"
        )
    budget = int(gate["training_budget_steps"])
    retry_budget = int(gate["retry"]["retry_budget_steps"])
    if retry_budget != 2 * budget:
        raise ValueError(
            f"gate retry.retry_budget_steps={retry_budget} != 2 x "
            f"training_budget_steps={2 * budget} — the D-03 retry is 2x "
            "the original budget (fail-loud on a tampered pin)"
        )
    return True


# ---------------------------------------------------------------------------
# Callback assembly (D-05: periodic checkpoints are crash-recovery only)
# ---------------------------------------------------------------------------


def build_callbacks(cfg: dict, run_dir: Path, progress: TrainProgressCallback) -> list:
    """Progress callback always; a periodic CheckpointCallback for the full
    run/retry modes ONLY, writing under <run_dir>/checkpoints/ with the
    matching VecNormalize snapshot (save_vecnormalize=True — a checkpoint
    without its stats is unusable, Pitfall 7). D-05: the final step's
    bundle is the sole selection candidate; NOTHING reads the checkpoints
    dir — it exists purely so a crashed run can restart, not resume."""
    callbacks: list = [progress]
    if cfg["mode"] in ("run", "retry"):
        if CHECKPOINT_INTERVAL_STEPS % cfg["n_envs"] != 0:
            raise ValueError(
                f"checkpoint interval {CHECKPOINT_INTERVAL_STEPS} is not "
                f"divisible by n_envs={cfg['n_envs']}"
            )
        callbacks.append(
            CheckpointCallback(
                save_freq=CHECKPOINT_INTERVAL_STEPS // cfg["n_envs"],
                save_path=str(run_dir / "checkpoints"),
                name_prefix="ppo",
                save_vecnormalize=True,
            )
        )
    return callbacks


# ---------------------------------------------------------------------------
# Progress callback (Pitfall 8: no tqdm/rich, no progress_bar=True)
# ---------------------------------------------------------------------------


class TrainProgressCallback(BaseCallback):
    """Log every completed episode's RAW return (Monitor's episode record
    is raw below VecNormalize), is_success, and stage flags to stdout
    every STDOUT_EPISODE_INTERVAL episodes and to TensorBoard via the SB3
    logger. Keeps the full episode list (tune mode writes it out) plus the
    rolling LAST_N_EPISODES window (the gate's selection metric)."""

    def __init__(self, run_name: str):
        super().__init__(verbose=0)
        self.run_name = run_name
        self.episodes: list[dict] = []
        self.window: deque = deque(maxlen=LAST_N_EPISODES)
        self._t0: float | None = None

    def _on_training_start(self) -> None:
        self._t0 = time.perf_counter()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            rec = {
                "episode_index": len(self.episodes),
                "num_timesteps": int(self.model.num_timesteps),
                "r": float(info["episode"]["r"]),  # RAW task return (Monitor, below VecNormalize)
                "l": int(info["episode"]["l"]),
                "is_success": bool(info.get("is_success", False)),
                "stage_approach": bool(info.get("stage_approach", False)),
                "stage_grasp": bool(info.get("stage_grasp", False)),
                "stage_lift": bool(info.get("stage_lift", False)),
                "stage_place": bool(info.get("stage_place", False)),
            }
            self.episodes.append(rec)
            self.window.append(rec)
            self._log_tb()
            if len(self.episodes) % STDOUT_EPISODE_INTERVAL == 0:
                self._print_progress()
        return True

    def _log_tb(self) -> None:
        w = list(self.window)
        self.logger.record("train/raw_ep_return_mean", float(np.mean([e["r"] for e in w])))
        self.logger.record("train/success_rate", float(np.mean([e["is_success"] for e in w])))
        self.logger.record("train/stage_grasp_rate", float(np.mean([e["stage_grasp"] for e in w])))
        self.logger.record("train/stage_lift_rate", float(np.mean([e["stage_lift"] for e in w])))
        self.logger.record("train/stage_place_rate", float(np.mean([e["stage_place"] for e in w])))

    def _print_progress(self) -> None:
        w = list(self.window)
        elapsed = time.perf_counter() - (self._t0 or time.perf_counter())
        sps = self.model.num_timesteps / elapsed if elapsed > 0 else 0.0
        print(
            f"[train:{self.run_name}] episode {len(self.episodes)}: "
            f"last-{len(w)} mean raw return {np.mean([e['r'] for e in w]):.3f}, "
            f"success {np.mean([e['is_success'] for e in w]):.2f}, "
            f"steps {self.model.num_timesteps}, {sps:.1f} steps/s",
            flush=True,
        )

    def last_window_mean_raw_return(self) -> float:
        """The gate's selection metric: mean raw episodic return over the
        last LAST_N_EPISODES completed episodes."""
        if not self.window:
            return float("nan")
        return float(np.mean([e["r"] for e in self.window]))


# ---------------------------------------------------------------------------
# The one training path (tracer slice and 1.5M run differ only in steps)
# ---------------------------------------------------------------------------


def run_train(gate: dict, cfg: dict, run_dir: Path) -> dict:
    """Construct the hardened chain, train under the train-window
    tripwire, save the final bundle, run the reload proof, and tear the
    workers down through _close_if_safe (Pitfall 9)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if SEED != cfg["seed"]:
        raise ValueError(
            f"gate tuning seed {cfg['seed']} != hardened-construction SEED "
            f"{SEED} — the env seeding and the PPO seed must agree "
            "(comparability contract)"
        )
    progress = TrainProgressCallback(cfg["run_name"])

    print(
        f"[train:{cfg['run_name']}] constructing the 8-env hardened chain "
        f"(spawn_mode={cfg['spawn_mode']}, action_space_mode="
        f"{cfg['action_space_mode']}) ...",
        flush=True,
    )
    chain, norm = _construct(gate, cfg)

    tb_dir = TB_ROOT / cfg["run_name"]
    model = PPO(
        "MultiInputPolicy",
        chain,
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],  # Pitfall 2: reads the resolved constant, never a literal
        learning_rate=cfg["learning_rate"],
        seed=cfg["seed"],
        device=cfg["device"],
        tensorboard_log=str(tb_dir),
        verbose=0,  # the runner prints its own progress
    )

    # Train window: READ from the gate (D-04), armed around learn() ONLY —
    # never over the construction (SETUP window) or any timed region.
    window_s = arm_train_tripwire(seconds=float(gate["train_window_hours"]) * 3600.0)
    print(
        f"[train:{cfg['run_name']}] learn({cfg['steps']} steps) at lr "
        f"{cfg['learning_rate']} with a {window_s / 3600.0:.0f} h "
        "train-window tripwire armed",
        flush=True,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=cfg["steps"], callback=build_callbacks(cfg, run_dir, progress))
    elapsed = time.perf_counter() - t0
    faulthandler.cancel_dump_traceback_later()

    save_bundle(model, norm, run_dir, cfg, gate)
    reload_ok = reload_proof(model, run_dir, gate, cfg)
    # The trained chain closes AFTER the reload proof (predict needs no
    # env; stale render-capable workers are the GL-contention exposure
    # the hardening exists to remove — WR-04/Pitfall 9).
    _close_if_safe(model.env, "subproc")

    return {
        "steps_per_sec": cfg["steps"] / elapsed if elapsed > 0 else 0.0,
        "elapsed_s": elapsed,
        "n_episodes": len(progress.episodes),
        "last_window_mean_raw_return": progress.last_window_mean_raw_return(),
        "episodes": progress.episodes,
        "reload_ok": reload_ok,
    }


# ---------------------------------------------------------------------------
# Tuning pass (04-03 Task 2, D-06) — sweep from the gate, selection from logs
# ---------------------------------------------------------------------------


def _chosen_lr_from_selection() -> float | None:
    """The lr chosen by the last tuning pass, read from
    runs/train/tune/selection.json. This is the resume path: the full run
    consumes the tuning decision without re-running the sweep. Returns
    None when no pass has run (the caller then uses the gate default)."""
    path = TUNE_ROOT / "selection.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["chosen_lr"])


def _selection_json_line() -> str:
    """The tune pass's machine-readable selection-record pointer.

    Repo-relative spelling (04-07 Rule 3 reconciliation, the 04-05
    tooling-to-plan-spelling pattern): the plan/VALIDATION verify greps
    the literal "SELECTION_JSON=runs/train/tune/selection.json" to prove
    the selection record's freshness — an absolute path would never
    match it. The path resolves against the repo root (the runner's
    cwd contract).
    """
    return f"SELECTION_JSON={(TUNE_ROOT / 'selection.json').relative_to(REPO_ROOT)}"


def run_tune(gate: dict) -> dict:
    """The gate's LR tuning pass: run EVERY tuning.lr_grid entry once at
    tuning.steps_per_variant steps (no --lr/--steps override — the sweep
    comes from the gate), write each variant's four-artifact bundle plus
    its full episode log, then apply the pure selection rule (D-06) over
    the per-variant episode returns and record the decision in
    runs/train/tune/selection.json. D-05: the tune pass exists to pick
    ONE lr — the final bundle of the full run stays the sole selection
    candidate; a variant that fails its reload proof aborts the pass (no
    selection from an unproven bundle)."""
    tuning = gate["tuning"]
    episode_returns: dict[str, list[float]] = {}
    variants: dict[str, dict] = {}
    for lr in tuning["lr_grid"]:
        variant = f"lr-{lr}"
        run_dir = TUNE_ROOT / variant
        cfg = resolve_config(
            gate, mode="tune", steps=None, lr=float(lr), run_name=f"tune-{variant}"
        )
        print(
            f"[tune] variant {variant}: {cfg['steps']} steps at "
            f"lr={cfg['learning_rate']} -> {run_dir}",
            flush=True,
        )
        result = run_train(gate, cfg, run_dir)
        if not result["reload_ok"]:
            raise RuntimeError(
                f"tune variant {variant} failed its reload proof — the "
                "tuning pass aborts (selection never reads an unproven "
                "bundle)"
            )
        with (run_dir / "episodes.jsonl").open("w") as fh:
            for ep in result["episodes"]:
                fh.write(json.dumps(ep) + "\n")
        key = str(float(lr))
        episode_returns[key] = [e["r"] for e in result["episodes"]]
        variants[key] = {
            "run_dir": str(run_dir),
            "n_episodes": result["n_episodes"],
            "steps_per_sec": result["steps_per_sec"],
            "last_window_mean_raw_return": result["last_window_mean_raw_return"],
        }
        print(
            f"[tune] variant {variant}: {result['n_episodes']} episodes, "
            f"last-{LAST_N_EPISODES} mean raw return "
            f"{result['last_window_mean_raw_return']:.3f} "
            f"({result['steps_per_sec']:.1f} steps/s)",
            flush=True,
        )
    chosen, record = apply_selection_rule(gate, episode_returns)
    record["seed"] = int(tuning["seed"])
    record["steps_per_variant"] = int(tuning["steps_per_variant"])
    record["variants"] = variants
    TUNE_ROOT.mkdir(parents=True, exist_ok=True)
    selection_path = TUNE_ROOT / "selection.json"
    selection_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


# ---------------------------------------------------------------------------
# Entry point (spawn-method discipline: the __main__ guard is MANDATORY)
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train.py",
        description=(
            "Production training runner (04-03): the hardened 8-env "
            "vision chain, PPO through the pinned train gate, and the "
            "four-artifact reload-proven bundle."
        ),
    )
    p.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=["run", "tune", "retry"],
        help=(
            "run = the default full run (budget from the gate); "
            "tune = the LR tuning pass (04-03 Task 2); "
            "retry = the D-03 one-retry escalation (04-05)"
        ),
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override the gate's per-mode budget (tracer slices use this)",
    )
    p.add_argument("--lr", type=float, default=None, help="override the learning rate")
    p.add_argument(
        "--confirm-retry",
        action="store_true",
        help=(
            "the explicit D-03 confirmation for retry mode — without it "
            "retry mode refuses to run (the runner never loops)"
        ),
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "run directory (default runs/train/<UTC-ts>/; the tracer must "
            "pass a non-2* name like runs/train/tracer-<ts> so it never "
            "satisfies the final-bundle glob)"
        ),
    )
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()
    try:
        gate = load_gate()

        if args.mode == "tune":
            # The sweep is gate data (lr_grid x steps_per_variant), not a
            # CLI choice — an --lr/--steps override would de-pin the
            # tuning pass (WR-02).
            if args.lr is not None or args.steps is not None:
                raise ValueError(
                    "tune mode takes no --lr/--steps override: the sweep "
                    "comes from the gate (config/train_gate.json "
                    "tuning.lr_grid / tuning.steps_per_variant)"
                )
            record = run_tune(gate)
            print(f"[tune] selection: {record['applied']}", flush=True)
            print(
                f"[tune] chosen lr {record['chosen_lr']} from metrics "
                f"{ {k: round(v, 3) for k, v in record['metrics'].items()} }",
                flush=True,
            )
            print(_selection_json_line(), flush=True)
            return 0

        if args.mode == "retry":
            # D-03 gate FIRST, before any heavy construction: retry mode
            # refuses without the explicit confirmation and the gate's
            # consent (the runner never loops or self-escalates).
            check_retry_allowed(gate, confirmed=args.confirm_retry)

        if args.out:
            out_dir = Path(args.out)
        else:
            out_dir = RUNS_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        # LR resolution order: explicit --lr, else the tuning pass's chosen
        # lr (the resume path: `run` consumes selection.json), else the
        # gate's pinned default. Provenance is printed either way.
        lr = args.lr
        if lr is not None:
            lr_source = "--lr"
        else:
            lr = _chosen_lr_from_selection()
            lr_source = (
                "runs/train/tune/selection.json (tuning-pass choice)"
                if lr is not None
                else "gate tuning.tie_fallback_lr (no tuning pass found)"
            )

        cfg = resolve_config(
            gate, mode=args.mode, steps=args.steps, lr=lr, run_name=out_dir.name
        )
        print(
            f"[train:{cfg['run_name']}] mode={cfg['mode']} steps={cfg['steps']} "
            f"lr={cfg['learning_rate']} (from {lr_source}) seed={cfg['seed']} "
            f"gamma={cfg['gamma']} n_envs={cfg['n_envs']} "
            f"n_steps={cfg['n_steps']} batch_size={cfg['batch_size']} "
            f"device={cfg['device']}",
            flush=True,
        )
        result = run_train(gate, cfg, out_dir)
        if not result["reload_ok"]:
            print(
                f"TRAIN FAILED: reload proof on the bundle at {out_dir} did "
                "not pass — the bundle is NOT a valid reference",
                file=sys.stderr,
                flush=True,
            )
            return 1
        print(
            f"[train:{cfg['run_name']}] {result['steps_per_sec']:.1f} steps/s "
            f"({result['elapsed_s']:.1f} s), {result['n_episodes']} episodes, "
            f"last-{min(len(result['episodes']), LAST_N_EPISODES)} mean raw "
            f"return {result['last_window_mean_raw_return']:.3f}",
            flush=True,
        )
        print(
            f"TRAIN PASSED: mode={cfg['mode']}, {cfg['steps']} steps, "
            f"lr={cfg['learning_rate']}, {result['n_episodes']} episodes -> "
            f"{out_dir}",
            flush=True,
        )
        print(f"BUNDLE_DIR={out_dir}", flush=True)
        return 0
    except Exception as exc:  # EXIT-CODE CONTRACT: nothing swallowed
        print(
            f"TRAIN FAILED: {type(exc).__name__}: {exc}\n"
            "Remediation: the gate pin (config/train_gate.json), the env, "
            "and the hardened construction (so100rob._hardened_venv) must "
            "all be consistent; if a spawn construction desynced, just "
            "re-run — the wedge is bounded, not eliminated.",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())