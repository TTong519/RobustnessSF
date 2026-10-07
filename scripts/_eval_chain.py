"""Shared eval-mode chain builder for the 04-04 evaluation/audit runners.

SCOPING (D-07): eval is NOT a spawn topology. Evaluation and the reward
audit run against ONE in-process DummyVecEnv (single worker, n_envs=1) —
never SubprocVecEnv, never the hardened multi-worker construction. The
in-process vision env is NEVER closed mid-session (WR-01 / 04-RESEARCH
Pitfall 9: gymnasium 1.3.0's OffScreenViewer.close() calls process-global
glfw.terminate()); the process exit reclaims it.

WR-03 RECIPE (pinned by tests/test_eval_mode.py): the moment a bundle is
loaded, the reloaded VecNormalize gets BOTH eval flags set —

    loaded_norm.training = False
    loaded_norm.norm_reward = False

``.training`` alone does NOT stop reward normalization (SB3 2.9.0's
step_wait applies ``normalize_reward`` whenever ``norm_reward`` is True,
independently of ``.training``), and there is NO ``set_deterministic`` —
determinism is the per-call ``model.predict(obs, deterministic=...)``
flag (D-02: both modes run in every evaluation).

The frozen spawn grid is READ from so100rob.constants.SPAWN_SEEDS_PATH
(the Phase 1 D-11 data file, moved into the package by the Phase 1 WR-03
fix — config/spawn_seeds.json is the stale pre-move path) and never
re-derived at runtime.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
    VecTransposeImage,
)

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv
from so100rob.vec_wrappers import ImageOnlyFrameStack

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "config" / "train_gate.json"
GATE_SCHEMA = "so100rob/train_gate/1"
RUNS_ROOT = REPO_ROOT / "runs"

# OBS_DECISION D-05: vision-64x64-gray (the Phase 3 format decision).
GRAYSCALE = True

# Eval/audit envs are grid mode + set_options per episode over the frozen
# 25-spawn grid (gate _doc.pitfall_1_spawn_mode: training envs are the
# continuous ones; grid-default resets are the training hazard, and here
# the grid IS the point).
EVAL_SPAWN_MODE = "grid"

# Info-contract keys captured in every episode record (Phase 2 reward
# ladder + stage machine; consumed by 04-05 judgment and Phase 5).
STAGE_FLAG_KEYS = ("stage_approach", "stage_grasp", "stage_lift", "stage_place")
REWARD_COMPONENT_KEYS = (
    "reward_shaping",
    "reward_time_penalty",
    "reward_terminal",
    "reward_phi",
)

VERSION_PACKAGES = (  # provenance block (operational, mirrors train.py)
    "stable-baselines3",
    "gymnasium",
    "mujoco",
    "torch",
    "numpy",
)


# ---------------------------------------------------------------------------
# Gate + frozen spawn grid (WR-02: read, never re-hardcode)
# ---------------------------------------------------------------------------


def load_gate() -> dict:
    """Read config/train_gate.json and fail loudly on a schema mismatch —
    a drifted or foreign pin must never evaluate."""
    gate = json.loads(GATE_PATH.read_text())
    if gate.get("schema") != GATE_SCHEMA:
        raise ValueError(
            f"config/train_gate.json schema {gate.get('schema')!r} != "
            f"{GATE_SCHEMA!r} — refusing to evaluate from a foreign/unknown pin"
        )
    return gate


def resolve_gamma(gate: dict) -> float:
    """The discount through the gate's gamma_source path (never a literal)."""
    source = str(gate["fixed_overrides"]["gamma_source"])
    module_name, attr = source.rsplit(".", 1)
    return float(getattr(importlib.import_module(module_name), attr))


def load_frozen_spawns(gate: dict | None = None) -> list[list[float]]:
    """The frozen 25-spawn 5x5 grid, READ from the packaged data file
    (D-11: immutable once baseline training began; no runtime re-derivation).

    Integrity-checked against the gate's spawns_total and the
    SPAWN_GRID_SIZE constant — a grid/pin mismatch fails loudly."""
    grid = json.loads(C.SPAWN_SEEDS_PATH.read_text())["spawns"]
    if gate is None:
        gate = load_gate()
    if len(grid) != int(gate["spawns_total"]):
        raise ValueError(
            f"frozen spawn grid has {len(grid)} entries but the gate pins "
            f"spawns_total={gate['spawns_total']} — refusing to evaluate "
            "over a mismatched grid"
        )
    if len(grid) != C.SPAWN_GRID_SIZE:
        raise ValueError(
            f"frozen spawn grid has {len(grid)} entries but "
            f"SPAWN_GRID_SIZE={C.SPAWN_GRID_SIZE}"
        )
    return grid


# ---------------------------------------------------------------------------
# The ONE in-process eval chain (D-07 scoping)
# ---------------------------------------------------------------------------


def build_eval_venv(gate: dict, n_envs: int = 1):
    """ONE in-process env, wrapped in the TRAINING order: raw env ->
    Monitor -> RescaleAction -> DummyVecEnv (make_vec_env's own order:
    Monitor first, wrapper_class outside — mirrored here so eval measures
    exactly what training built).

    D-07 scoping: n_envs=1 only — a spawn topology here would be a
    silent measurement change, not a performance choice.
    """
    if n_envs != 1:
        raise ValueError(
            f"eval/audit run on ONE in-process env (n_envs=1), got {n_envs} "
            "— D-07 scoping: eval is NOT a spawn topology"
        )

    def _make():
        env = SO100PickPlaceEnv(
            frame_skip=C.DEFAULT_FRAME_SKIP,
            render_mode="rgb_array",
            width=C.OBS_IMAGE_SIZE,
            height=C.OBS_IMAGE_SIZE,
            obs_mode=C.OBS_MODE_VISION,
            grayscale=GRAYSCALE,
            spawn_mode=EVAL_SPAWN_MODE,
        )
        # Monitor FIRST (training order), logging the RAW task return
        # below VecNormalize (04-RESEARCH Pattern 3).
        env = Monitor(env)
        mode = gate["action_space_mode"]
        if mode != "rescale_minus_one_one":
            raise ValueError(
                f"unknown action_space_mode {mode!r} in config/train_gate.json "
                "— expected 'rescale_minus_one_one' (the 04-02 Task 1 "
                "resolved mode)"
            )
        return RescaleAction(
            env,
            min_action=gate["rescale_min"],
            max_action=gate["rescale_max"],
        )

    return DummyVecEnv([_make])


def load_bundle(bundle_dir, gate: dict | None = None, n_envs: int = 1):
    """Reload a training bundle in eval mode (the WR-03 recipe) over ONE
    in-process env. STRICT ORDER (D-09): fresh venv -> VecNormalize.load ->
    ImageOnlyFrameStack -> VecTransposeImage -> PPO.load(env=chain) — the
    model is attached AT load, never via a bare load + ``.set_env(chain)``
    (SB3 2.9.0's ``set_env`` asserts ``num_envs == n_envs``, and the real
    04-03 bundle was trained on n_envs=8 while the eval chain is ONE env
    (D-07 scoping) — ``PPO.load(env=...)`` adopts the chain's spaces and
    n_envs instead; found as a Rule 1 bug in 04-05 Task 1).

    The WR-03 flags are pinned AT LOAD — the returned chain's normalizer
    has training=False AND norm_reward=False before any predict call.
    Returns (model, chain).
    """
    bundle_dir = Path(bundle_dir)
    if gate is None:
        gate = load_gate()
    for artifact in ("ppo_final.zip", "vecnormalize.pkl", "config.json", "gate.json"):
        if not (bundle_dir / artifact).exists():
            raise FileNotFoundError(
                f"bundle {bundle_dir} is missing {artifact} — the 04-03 "
                "four-artifact layout is contract"
            )
    venv = build_eval_venv(gate, n_envs=n_envs)
    loaded_norm = VecNormalize.load(str(bundle_dir / "vecnormalize.pkl"), venv)
    # WR-03 recipe, AT LOAD — pinned by tests/test_eval_mode.py.
    loaded_norm.training = False
    loaded_norm.norm_reward = False
    chain = VecTransposeImage(ImageOnlyFrameStack(loaded_norm, n_stack=C.OBS_N_STACK))
    model = PPO.load(str(bundle_dir / "ppo_final"), env=chain, device="cpu")
    return model, chain


def unwrap_vec_normalize(chain) -> VecNormalize:
    """The VecNormalize inside a loaded eval chain (WR-03 introspection)."""
    node = chain
    while not isinstance(node, VecNormalize):
        node = node.venv
    return node


def raw_env(chain):
    """The unwrapped SO100PickPlaceEnv inside a loaded eval chain."""
    node = chain
    while not isinstance(node, DummyVecEnv):
        node = node.venv
    return node.envs[0].unwrapped


def cube_position(chain) -> np.ndarray:
    """Current cube position from the live in-process env (spawn-targeting
    introspection for tests/audit)."""
    env = raw_env(chain)
    return np.asarray(
        env.data.qpos[env._cube_qposadr : env._cube_qposadr + 3], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# make_synthetic_bundle — the tiny untrained test bundle (tests + tracer)
# ---------------------------------------------------------------------------


def make_synthetic_bundle(path, gate: dict | None = None):
    """Write a TINY SYNTHETIC bundle (fresh untrained PPO + fresh
    VecNormalize) in EXACTLY the 04-03 four-artifact layout — the tests
    and the Task 1 tracer prove the eval tooling end-to-end without the
    1.5M-step artifact (04-03 runs in parallel; the real bundle is
    judged in 04-05).

    Saves as a TRAINING-capable bundle (flags untouched in the save);
    load_bundle applies the WR-03 flags on the way in, so the round trip
    exercises the real recipe. Returns the chain (callers keep it
    referenced; per WR-01 it is NEVER closed mid-session).
    """
    if gate is None:
        gate = load_gate()
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    venv = build_eval_venv(gate)
    norm = VecNormalize(
        venv,
        norm_obs_keys=[C.OBS_STATE_KEY],
        norm_obs=True,
        norm_reward=True,
        gamma=resolve_gamma(gate),
    )
    chain = VecTransposeImage(ImageOnlyFrameStack(norm, n_stack=C.OBS_N_STACK))
    model = PPO(
        "MultiInputPolicy",
        chain,
        n_steps=128,
        batch_size=64,
        n_epochs=2,
        gamma=resolve_gamma(gate),  # Pitfall 2: read, never a literal
        seed=int(gate["tuning"]["seed"]),
        device=str(gate["fixed_overrides"]["device"]),
    )
    model.save(str(out_dir / "ppo_final"))
    norm.save(str(out_dir / "vecnormalize.pkl"))
    payload = {
        "mode": "synthetic",
        "purpose": "04-04 eval-tooling test/tracer bundle (untrained)",
        "seed": int(gate["tuning"]["seed"]),
        "gamma": resolve_gamma(gate),
        "action_space_mode": gate["action_space_mode"],
        "spawn_mode": EVAL_SPAWN_MODE,
        "n_envs": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "versions": {p: importlib.metadata.version(p) for p in VERSION_PACKAGES},
    }
    (out_dir / "config.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "gate.json").write_text(GATE_PATH.read_text())
    return chain


# ---------------------------------------------------------------------------
# Pure judgment layer (TRAIN-03) — no env, no I/O; the 04-05 judgment and
# every test consume these over records + the gate alone
# ---------------------------------------------------------------------------


def per_spawn_success(episodes, majority: int) -> bool:
    """D-01 majority rule: a spawn counts successful on a majority of its
    episodes (>= majority successes of episodes_per_spawn)."""
    successes = sum(1 for ep in episodes if ep.get("is_success"))
    return successes >= int(majority)


def full_grid_spawn_indices(gate: dict) -> list[int]:
    """The full frozen grid as spawn indices, in the file's entry order —
    integrity-checked against gate spawns_total AND SPAWN_GRID_SIZE inside
    load_frozen_spawns (a doctored pin fails loudly, T-04-04)."""
    grid = load_frozen_spawns(gate)
    return list(range(len(grid)))


def judge(results: dict, gate: dict) -> dict:
    """The pre-agreed TRAIN-03 verdict over {mode: {spawn_index: records}}.

    The bar is the PRIMARY mode's (gate primary_bar, D-02): deterministic_pass
    = successful primary spawns >= gate min_spawns_successful, judged over
    EXACTLY the gate spawns_total grid (an off-grid primary result set
    fails loudly — never a partial sweep). The stochastic rate is a
    reported reference number; it can never flip the deterministic verdict.
    """
    total = int(gate["spawns_total"])
    majority = int(gate["per_spawn_majority"])
    primary = str(gate["primary_bar"])

    if primary not in results:
        raise ValueError(
            f"the primary bar {primary!r} is missing from results — the "
            "TRAIN-03 verdict cannot be computed"
        )

    per_spawn: dict[str, dict[int, bool]] = {}
    for mode, mode_results in results.items():
        if set(mode_results.keys()) != set(range(total)):
            raise ValueError(
                f"mode {mode!r} evaluated {sorted(mode_results)} but the "
                f"frozen grid is spawns_total={total} — refusing to judge "
                "a partial/off-grid sweep"
            )
        per_spawn[mode] = {
            i: per_spawn_success(mode_results[i], majority) for i in range(total)
        }

    n_primary = sum(1 for ok in per_spawn[primary].values() if ok)
    stochastic = per_spawn.get("stochastic")
    n_stochastic = (
        sum(1 for ok in stochastic.values() if ok) if stochastic is not None else None
    )
    return {
        "deterministic_pass": n_primary >= int(gate["min_spawns_successful"]),
        "deterministic_rate": n_primary / total,
        "stochastic_rate": (
            n_stochastic / total if n_stochastic is not None else None
        ),
        "per_spawn": per_spawn,
        "spawns_evaluated": total,
    }


# ---------------------------------------------------------------------------
# Reward-audit bounds (T-04-06) — the rendered audit gate's judgment
# ---------------------------------------------------------------------------


def audit_episode_ok(gate: dict, raw_return: float, is_success: bool) -> bool:
    """The pre-agreed audit bound for ONE episode (T-04-06), READ from
    the gate (WR-02 — never the literals):

    * no-success episodes: raw_return <= audit_no_success_return_max
      (2.5 — the no-place arithmetic: the 2.0 dense ceiling
      (FULL_FLAGS_NO_PLACE_MAX_FRACTION x TERMINAL_PLACE_BONUS) minus the
      2.0 time-penalty ceiling (TIME_PENALTY x 200 max steps) bounds the
      no-place return at 0.0, so 2.5 leaves margin);
    * successes: raw_return >= audit_success_return_min (6.0 — the 10.0
      terminal bonus minus the 2.0 dense ceiling minus the 2.0 time
      penalty, equality pinned).

    The operators are the pin: a record AT the bound is compliant;
    2.6 (no success) or 5.9 (success) is a reward-hacking violation.
    """
    raw = float(raw_return)
    if is_success:
        return raw >= float(gate["audit_success_return_min"])
    return raw <= float(gate["audit_no_success_return_max"])


def audit_verdict(records: list[dict], gate: dict) -> dict:
    """The audit conjunction over episode records: every episode must
    satisfy audit_episode_ok; every violating episode is named by its
    spawn_index so the runner's nonzero exit can name it."""
    violations = []
    for rec in records:
        is_success = bool(rec.get("is_success", False))
        raw = float(rec.get("raw_return", 0.0))
        if not audit_episode_ok(gate, raw, is_success):
            violations.append(
                {
                    "spawn_index": int(rec["spawn_index"]),
                    "mode": rec.get("mode"),
                    "raw_return": raw,
                    "is_success": is_success,
                    "reason": (
                        "success return below audit_success_return_min"
                        if is_success
                        else "no-success return above "
                        "audit_no_success_return_max"
                    ),
                }
            )
    return {"pass": len(violations) == 0, "violations": violations}


# ---------------------------------------------------------------------------
# Episode execution + record contract
# ---------------------------------------------------------------------------


def episode_record(mode: str, spawn_index: int, final_info: dict, seed=None) -> dict:
    """One JSON-serializable episode row (the EVAL-02-anticipating shape).

    raw_return is Monitor's episode record — the RAW task return, logged
    below VecNormalize (whose norm_reward is False anyway under WR-03, but
    Monitor is the source of truth for the raw number)."""
    episode = final_info.get("episode", {})
    return {
        "mode": mode,
        "spawn_index": int(spawn_index),
        "seed": None if seed is None else int(seed),
        "is_success": bool(final_info.get("is_success", False)),
        "raw_return": float(episode.get("r", 0.0)),
        "steps": int(episode.get("l", 0)),
        "stage_flags": {k: bool(final_info.get(k, False)) for k in STAGE_FLAG_KEYS},
        "reward_components": {
            k: float(final_info.get(k, 0.0)) for k in REWARD_COMPONENT_KEYS
        },
    }


def run_episodes(
    model,
    chain,
    spawn_index: int,
    n_episodes: int,
    deterministic: bool,
    seed=None,
    frame_hook=None,
) -> list[dict]:
    """Run n episodes at ONE spawn through the loaded eval chain.

    Per episode (Pitfall 6 reset hygiene): an EXPLICIT targeted reset —
    set_options({'spawn_index': i, 'seed': s}) -> seed -> reset — so the
    env consumes the requested spawn at THIS reset and never inherits a
    previous episode's. Determinism is the per-call predict flag (D-02).
    frame_hook(step), when given, is called after every step with the
    1-based step index (the policy-audit frame capture).
    """
    mode = "deterministic" if deterministic else "stochastic"
    records = []
    for ep in range(int(n_episodes)):
        ep_seed = None if seed is None else int(seed) + ep
        options = {"spawn_index": int(spawn_index)}
        if ep_seed is not None:
            options["seed"] = ep_seed
        chain.set_options(options)
        if ep_seed is not None:
            chain.seed(ep_seed)
        obs = chain.reset()
        step = 0
        while True:
            action, _ = model.predict(obs, deterministic=bool(deterministic))
            obs, _rewards, dones, infos = chain.step(action)
            step += 1
            if frame_hook is not None:
                frame_hook(step)
            if dones[0]:
                # VecEnv done semantics (Pitfall 5): the final info carries
                # is_success, the stage flags, the reward components, AND
                # Monitor's raw episode record.
                records.append(
                    episode_record(mode, spawn_index, infos[0], seed=ep_seed)
                )
                break
    return records