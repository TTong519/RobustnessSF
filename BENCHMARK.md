# Throughput Benchmark — Phase 3 Observation-Space Candidates (TRAIN-02, measured half)

Median real control steps/sec on this Mac for the four Phase 3 obs-space
candidates, each render config timed both raw and with a mock no-grad
~NatureCNN forward pass per control step. This is the **measured input** to
the Phase 3 obs-space decision — the decision itself is made and documented
in Phase 3, not here.

## Results (median of 3 repeats, 2000 timed steps each, 200 warm-up steps excluded)

| Config                          | Raw stepping (steps/s) | + mock CNN forward (steps/s) |
| ------------------------------- | ---------------------- | ---------------------------- |
| `state_only` (no render)        | **7886.4**             | — (n/a, nothing to forward)  |
| `rgb64` (64x64 RGB)             | 293.0                  | 261.7                        |
| `gray64` (64x64 grayscale)      | 286.6                  | 257.9                        |
| `rgb128` (128x128 RGB)          | 276.3                  | 224.3                        |

Sanity gate (Pitfall 8): state-only raw (7886.4) is **28.5x** the slowest
render config raw (276.3) — the state-only loop is genuinely not rendering.

## Machine and version context

- **Machine:** Apple M4 Pro, arm64 (macOS 26.3.1), Python 3.12.10
- **Packages (pinned venv):** mujoco 3.12.0, gymnasium 1.3.0, torch 2.14.0, numpy 2.5.2
- **Env:** `SO100PickPlaceEnv` default constructor — frame_skip 25 (20 Hz
  control), physics timestep 0.002 s, home ctrl held constant (this measures
  stepping cost, not policy variety)
- **Mock CNN:** ~NatureCNN size (32f 8x8 s4 → 64f 4x4 s2 → 64f 3x3 s1 →
  flatten → linear 512), `eval()` under `torch.no_grad()`, one forward per
  control step at batch size 1

## Machine-readable source

Full per-repeat values, protocol parameters, and context:
[`runs/benchmark/20260913T164431Z/results.json`](runs/benchmark/20260913T164431Z/results.json)
(gitignored output dir; this timestamp is the official measurement). Phase 3
reads that JSON — do not re-parse this markdown table.

Re-run: `uv run python scripts/benchmark.py` (add `--steps N --repeats R`,
or `--quick` for the seconds-long structural smoke mode).

## Interpretation notes (no decision here — Phase 3 decides)

- **Research expectation-setters** (single-run, research-grade, on-machine):
  state-only ~11,400 steps/s and 64x64 render ~332 steps/s were measured at
  **frame_skip 10**; this official benchmark runs the **default frame_skip 25**
  (20 Hz control), which does 2.5x more physics substeps per control step —
  hence the state-only figure here (~7.9k steps/s) is lower than the research
  number while the render configs (~290 steps/s, render-dominated) match it
  closely. Both sets tell the same story.
- **Render cost dominates and is nearly resolution-independent** in this
  range: 64x64 and 128x128 raw throughput differ by only ~6% (293 vs 276
  steps/s) — fixed per-frame GL overhead dominates, as the research session
  predicted. The 64-vs-128 choice will hinge on CNN cost, not render cost.
- **The mock-CNN overhead measured here is a forward-only, batch-1 lower
  bound.** Real PPO training adds backward passes, minibatch reprocessing,
  and rollout buffering, so effective training throughput will be
  substantially below the "with mock CNN" column. The research literature's
  ~91% pixel-RL cost in CNN updates is consistent with that.
- Per the plan's prohibition, every figure above is the median of >= 3
  repeats after warm-up — no un-warmed single-run number is presented as
  official.
- **What this does NOT say:** whether vision-based PPO fits the compute
  budget, and which obs space Phase 3 adopts. That decision (TRAIN-02) is
  Phase 3's success criterion and is documented from
  `runs/benchmark/20260913T164431Z/results.json`, not from these numbers or
  the research baselines.