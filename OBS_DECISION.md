# Observation-Space Decision — TRAIN-02 (Phase 3)

**Status: DECIDED — vision, 64x64 grayscale, 4-frame stack.** The pre-agreed
gate and selection rule above were pinned as committed data (commit
`cc1945b`) and confirmed unchanged (option-a) at the blocking decision
checkpoint BEFORE the smoke run executed; the Results, Gate evaluation, and
Decision sections below were appended strictly from the smoke run's measured
numbers. A pre-measurement doc containing results would have defeated the
entire D-04 point.

This document records the Phase 3 obs-space decision for the SO-ARM100
pick-and-place experiment: **vision (64x64 stacked frames, grayscale or RGB)
or the state-only fallback**. The decision consumes the smoke run's **real
measured PPO training throughput** — never the Phase 1 mock-CNN column, which
is a forward-only, batch-1 lower bound (backward passes, minibatch
reprocessing, and rollout buffering all push real training throughput well
below it). Once documented, the decision is one-way: Phase 4's policy
architecture (CNN on stacked 64x64 frames vs MLP on the 28-dim state), every
downstream checkpoint, and every fine-tuning comparison in Phases 6-7 derive
from it; reversing after the Phase 4 baseline trains would invalidate the
baseline.

## Pre-agreed decision rule (D-04)

The gate is a wall-clock budget, pinned as committed data in
[`config/obs_gate.json`](config/obs_gate.json) — the machine-readable source of
these numbers; this doc quotes it, tooling reads the JSON, and the pin commit
provably precedes the smoke-run results in git history.

- **Phase 4 baseline training budget:** 1,500,000 env steps
- **Acceptable wall-clock training window:** 12 h (an overnight run) = 43,200 s
- **Required real PPO training throughput (the floor):**

  required_throughput = ceil(1.5e6 / (12 x 3600)) = ceil(34.72) = **35 steps/s**

**The rule, verbatim from the pin:**

> measured full-smoke PPO training throughput of the chosen image format >=
> required_throughput_steps_per_sec (35) -> vision (64x64, format per D-05);
> below -> state-only fallback (D-12).

**Why pre-agreement matters (D-04):** a floor chosen after seeing the measured
number is post-hoc rationalization — the exact failure mode this rule exists
to prevent. The numbers above were proposed (research assumption A1: the 1.5M
budget is the roadmap's planned Phase 4 baseline sizing; 12 h fits an overnight
run with rerun slack) and confirmed at the blocking decision checkpoint
BEFORE the smoke run executed. Neither this document nor the runner may adjust
them after measurement; the smoke runner reads the floor from
`config/obs_gate.json` and never re-hardcodes it.

## Pre-agreed format-selection rule (D-05)

The image format (64x64 grayscale vs 64x64 RGB) is decided from measured
numbers, not a priori. The rule, verbatim from the pin:

> higher measured steps/sec wins; within a relative 10% band -> grayscale

Grayscale wins ties because: its first conv layer is 1->32 instead of 3->32
input channels (smaller CNN input cost); the fixed-lit simulation scene makes
color near-redundant for this task; and Phase 1 already measured raw stepping
near-parity (gray64 286.6 vs rgb64 293.0 steps/s), so raw throughput alone
cannot separate the formats. Both candidates are measured by the same
protocol below and compared strictly by the rule above.

## Measurement protocol (D-07)

### Topology benchmark (D-07 — chosen by benchmark, not assumption)

- Candidates: `DummyVecEnv` vs `SubprocVecEnv` via
  `make_vec_env(..., vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "spawn"})`
  — the start method is passed through `vec_env_kwargs`, never as a
  `make_vec_env` kwarg (no such parameter exists in SB3 2.9.0).
- 8 envs, real vision Dict obs (`obs_mode="vision"`, `render_mode="rgb_array"`,
  default frame_skip).
- Per repeat: construct, reset, run **200 untimed warm-up steps** (excludes
  worker spawn, model compile, lazy renderer creation, and first-frame GL
  warm-up from the timed region), then time **1000 steps** with
  `time.perf_counter()`.
- **Median of at least 3 repeats** per candidate — the `scripts/benchmark.py`
  protocol. The higher median wins and is the topology both smoke trains use.

### Per-format PPO smoke train (identical for both formats)

For each format ("rgb", "grayscale"), on the winning topology, with the
format's env kwargs (`grayscale=True` or `False`):

- `VecNormalize(venv, norm_obs_keys=["state"], norm_obs=True, norm_reward=True, gamma=C.REWARD_GAMMA)`
  — then `ImageOnlyFrameStack(C.OBS_N_STACK)` — then `VecTransposeImage`.
- `PPO("MultiInputPolicy", env, n_steps=128, batch_size=64, n_epochs=2,
  gamma=C.REWARD_GAMMA, seed=0, device="cpu", tensorboard_log=<run_dir>/tb/<format>)`
  with the default CNN; total 16384 timesteps (batch_size 64 divides the
  n_steps x n_envs = 1024 rollout buffer — Pitfall 7).
- `learn(total_timesteps=16384)` timed with `time.perf_counter()`; steps/sec
  recorded per format.
- Checkpoint bundle saved per format: `model.save(...)` +
  `vecnormalize.save(...)` (D-08), then the strict-order reload proof and the
  bitwise action-equality check (D-09).

**VecNormalize configuration, documented (D-08):** both trains share
`VecNormalize(norm_obs_keys=["state"], norm_obs=True, norm_reward=True)` — the
state key only is running-mean normalized; the image key stays uint8 [0,255]
for SB3's built-in normalization, never pre-normalized. `norm_reward=True` is
SB3's default and is what the research probe verified; D-08's stat save/load
is proven under it by the 03-02 suite test
(`tests/test_sb3_pipeline.py::test_train_save_reload_bitwise_and_stats_roundtrip`).
Phase 4 may revisit reward normalization separately when it owns the training
config.

### Reload proof (D-09)

Fresh identical vec-env construction, then `VecNormalize.load(pkl, fresh_venv)`,
then `ImageOnlyFrameStack`, then `VecTransposeImage`, then
`PPO.load(model_path, device="cpu")` with `set_env` on the rebuilt chain (the
strict order — `set_env` on the bare loaded VecNormalize raises a space
mismatch). `fixed_obs` built in the LOADED model's observation-space shapes
(state `(1, C.OBS_STATE_DIM)` float32; the image in the space's own stacked,
transposed shape — read from the space, never hand-counted), then
`trained.predict(fixed_obs, deterministic=True)` vs
`reloaded.predict(fixed_obs, deterministic=True)` must satisfy
`a.tobytes() == b.tobytes()` (bitwise).

## Context the gate is judged against (PRIOR-PHASE context — not the decision input)

The decision input is the smoke run's measured steps/s above. The Phase 1
benchmark (`BENCHMARK.md`, machine source
`runs/benchmark/20260913T164431Z/results.json`) is quoted here as context
only — medians of 3 repeats, 2000 timed steps, 200 warm-up steps excluded,
Apple M4 Pro:

| Config | Raw stepping (steps/s) | + mock CNN forward (steps/s) |
| --- | --- | --- |
| `state_only` (no render) | 7886.4 | n/a |
| `rgb64` (64x64 RGB) | 293.0 | 261.7 |
| `gray64` (64x64 grayscale) | 286.6 | 257.9 |
| `rgb128` (128x128 RGB) | 276.3 | 224.3 |

- **The mock-CNN column is a forward-only, batch-1 lower bound.** Real PPO
  training adds backward passes, minibatch reprocessing, and rollout
  buffering, so real training throughput is substantially below 224-262
  steps/s. The gate must NOT consume the mock column — it consumes the smoke
  run's measured training throughput.
- **Honest expected-range note (assumption A2 — assumed, not promised):** the
  vision path is EXPECTED to land well above the 35 steps/s floor — roughly
  150-600 real steps/s is the assumed band, i.e. the floor sits 4-20x below
  it, so the gate decides vision unless something is genuinely wrong. **The
  measurement decides, not the expectation.** If the measured value lands
  near the floor, that fact is stated verbatim in the Gate evaluation section
  and the fallback fires exactly as designed — the D-04 gate working, never a
  failure, a surprise, or a problem to route around (decision-honesty
  prohibition, TRAIN-02).

## Results

Measured by `scripts/run_sb3_smoke.py` (exit 0, all 13 collector checks
passed) on 2026-09-23. Machine: Apple M4 Pro (macOS 26.3.1, arm64),
Python 3.12.10, mujoco 3.12.0, gymnasium 1.3.0, torch 2.14.0, numpy 2.5.2,
stable-baselines3 2.9.0, tensorboard 2.21.0. Full payload:
`runs/obs_decision/20260923T024037Z/results.json`.

### D-07 topology benchmark (8 envs, 200 warm-up + 1000 timed vec steps per repeat, median of 3)

| Topology | Median env steps/s | Repeats (env steps/s) |
| --- | --- | --- |
| `DummyVecEnv` | 235.6 | 235.6, 235.3, 236.3 |
| `SubprocVecEnv` (spawn) | **742.2** | 745.5, 742.2, 706.3 |

**Winner: SubprocVecEnv (spawn)** — 3.1x DummyVecEnv. Rendering (the cost
that matters for vision obs) parallelizes across worker processes; at 8 envs
the spawn worker set pays for itself easily on this machine. Both smoke
trains ran on the winning topology. (Runner note: spawn constructions go
through a watchdogged readiness barrier with bounded retry — a measured
macOS spawn bootstrap race, documented in the runner and in this phase's
SUMMARY; timed regions are never touched by it.)

### Per-format PPO smoke trains (identical: 16384 steps, 8 envs, n_steps=128, batch_size=64, n_epochs=2, seed=0, device=cpu, default CNN, both reloads bitwise-equal)

| Format | Measured steps/s | Mean episodic reward (80 episodes) | Reload bitwise |
| --- | --- | --- | --- |
| rgb | 233.7 | -2.1968 | EQUAL |
| grayscale | **252.8** | -2.1946 | EQUAL |

The D-06 soft check is satisfied: mean episodic reward is logged per format
(-2.20 vs -2.19 — statistically indistinguishable at 16k steps, exactly as
expected for a smoke run whose bar is pipeline integrity, not learning).

## Gate evaluation

The pinned gate, quoted verbatim from
[`config/obs_gate.json`](config/obs_gate.json): required throughput =
**35 steps/s** (= ceil(1.5e6 / (12 x 3600)) = ceil(34.72)); the pinned
numbers were confirmed unchanged (option-a) at the blocking decision
checkpoint on 2026-09-23, before any measurement existed.

- **D-05 format selection, applied:** rgb measured 233.7 steps/s, grayscale
  measured 252.8 steps/s — relative difference **7.6%**, inside the
  pre-agreed relative 10% band -> **grayscale** (the tie-band rule fired,
  not raw superiority: both numbers are quoted above and the band is the
  pinned 10%).
- **D-04 gate, evaluated on grayscale's measured 252.8 steps/s against the
  pre-agreed floor of 35 steps/s:**

  measured 252.8 >= floor 35 -> **vision**; margin
  **+622.4%** (measured is 7.2x the floor).

## The Decision

**Vision: 64x64 grayscale, 4-frame stack (obs_mode="vision",
grayscale=True), SubprocVecEnv-spawn topology, 8 envs.**

Phase 4 trains PPO with a `MultiInputPolicy` (default CNN on the stacked
4-channel 64x64 image key + MLP on the 28-dim normalized state key) through
the proven chain:
`make_vec_env(SO100PickPlaceEnv, n_envs=8, spawn) ->
VecNormalize(norm_obs_keys=["state"], norm_obs=True, norm_reward=True,
gamma=C.REWARD_GAMMA) -> ImageOnlyFrameStack(4) -> VecTransposeImage ->
PPO("MultiInputPolicy", ...)`. The state-only fallback (D-12: plain
VecFrameStack(4) over the flat 28-dim Box, MLP policy) was NOT selected —
the gate did not fire it. With measured 252.8 steps/s, the 1.5M-step Phase 4
baseline budget takes ~1.6 h of wall clock — ~13% of the 12 h window
(1.5e6 / 252.8 = 5931 s ≈ 1.65 h), leaving substantial rerun slack.

## Honest framing

The margin is not small: 252.8 steps/s sits **7.2x above** the 35 steps/s
floor — far from near-threshold, so the decision-honesty prohibition's
near-floor clause does not engage. The measured value lands at the low end
of the assumed A2 band (150-600 steps/s), i.e. the expectation was honest
and the measurement confirms it. Two facts are worth stating plainly so
they are not over-read later:

1. **Grayscale won on the tie-band, not on dominance.** The measured
   throughputs are near-parity (233.7 vs 252.8 steps/s, 7.6% relative
   difference), which is exactly the situation the pre-agreed D-05 rule
   anticipated: throughput alone cannot separate the formats, so the
   tie-break (smaller first-conv input, color near-redundant in the
   fixed-lit scene) decides — per the rule, not per taste.
2. **The measured steps/s is full-pipeline PPO training throughput on the
   winning topology**, not the Phase 1 mock-CNN forward-only lower bound
   (261.7 rgb64) — the two are numerically similar by coincidence; only
   the smoke number feeds the gate.

## TensorBoard

Per-format training curves (rollout/ep_rew_mean, losses — the D-06 soft
trend) are logged for both smoke trains:

- rgb: `runs/obs_decision/20260923T024037Z/tb/rgb/`
- grayscale: `runs/obs_decision/20260923T024037Z/tb/grayscale/`

View with `tensorboard --logdir runs/obs_decision/20260923T024037Z/tb`.

## Security note

Checkpoint bundles (`ppo_smoke.zip`, `vecnormalize.pkl`) are
pickle-backed deserializations (T-03-05, accepted): they are this project's
own local outputs written inside the repo's gitignored `runs/` tree and
are trusted-local-only. Never load a model or stats file from an untrusted
source into any later phase that reloads checkpoints.

## Machine-readable source

The machine-readable source of every number quoted above is
`runs/obs_decision/20260923T024037Z/results.json` (gitignored runtime
artifact, the `runs/` convention). Tooling and later phases read the JSON —
do not re-parse this markdown.