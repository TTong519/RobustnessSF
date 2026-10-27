# BASELINE — the frozen reference point (Phase 4, TRAIN-01 + TRAIN-03)

**Status: FROZEN (D-03 outcome (b), 2026-10-27).** This is the single
trusted reference point every Phase 5-7 degradation and recovery number
is measured against. The bundle below is the sole baseline candidate
(D-05 final-step selection): re-picking it after seeing Phase 5 results
invalidates the comparison.

## The baseline bundle

- **Bundle:** `runs/train/20261027T000117Z` (the 3M D-03 retry run —
  outcome (b): the 1.5M run missed at 9/25, the one pinned retry passed
  at 20/25; the RETRY bundle is the frozen reference). Four artifacts:
  `ppo_final.zip`, `vecnormalize.pkl`, `config.json`, `gate.json`; the
  strict-order reload proof passed **bitwise EQUAL**.
- **Resolved config** (from the bundle's `config.json`, which carries the
  gate echo):
  - mode `retry`, steps `3,000,000` (= 2 x the gate's 1.5M
    `training_budget_steps`, the D-03 escalation, gate-checked by the
    runner), 24,063 episodes, 346.3 steps/s.
  - lr `3e-4`, provenance `runs/train/tune/selection.json` (the 04-09
    re-tune on the new reward contract: last-100 mean raw returns 1e-4
    −2.073 / 3e-4 −1.614 / 1e-3 −1.623; 0.58% relative tie between the
    best two within the 1% band -> `tie_fallback_lr` 3e-4; seed 0).
  - gamma `0.99`, read from `so100rob.constants.REWARD_GAMMA` through the
    gate's `gamma_source` (Pitfall 1 — the shaping's policy-invariance
    guarantee requires the learner's discount to match).
  - seed `0`, n_envs `8`, n_steps `128`, batch_size `64`, device `cpu`,
    action_space_mode `rescale_minus_one_one` (gymnasium RescaleAction
    around the raw asymmetric ctrlrange), spawn_mode `continuous`,
    checkpoint_selection `final-step`.
- **Gate pin:** `config/train_gate.json` (schema `so100rob/train_gate/1`,
  copied verbatim into the bundle as `gate.json`).

## Observation contract (what the policy sees)

- **The 04-07 z-clause CNN-only 28-dim contract** (the 04-08
  classical-CV centroid tail is RETIRED from every judged chain by the
  04-09 pre-agreement — evidence: parity at the 3M retry (12/25 vs
  CNN-only 13/25) at a worse first-run cost (3/25); the CV family stays
  DORMANT opt-in in the codebase, preserved and test-guarded; a 32-dim
  CV bundle load on this contract fails loudly at VecNormalize.load).
- `SO100PickPlaceEnv(obs_mode="vision", render_mode="rgb_array",
  width=64, height=64, grayscale=True)` — Dict{state: Box(28,) float32,
  image: uint8 (64, 64, 1)}; the state layout is the pinned ENV-03
  28-dim vector (6 arm qpos + 6 arm qvel + 3 cube pos + 1 cube yaw + 6
  cube qvel + 3 gripper-to-cube + 3 cube-to-goal).
- Pipeline: 8-env `SubprocVecEnv` (spawn start method, the hardened
  construction) -> `VecNormalize(norm_obs_keys=["state"])` (the uint8
  image key is NEVER mean-normalized — SB3 divides by 255 internally) ->
  `ImageOnlyFrameStack(4)` -> `VecTransposeImage` -> PPO
  `MultiInputPolicy` (CnnPolicy-family extractor), device cpu.

## Reward contract (what the policy optimizes)

- **Potential-based stage-gated shaping** (`src/so100rob/reward.py`):
  Phi = `STAGE_POTENTIALS[stage_index]` + the hybrid approach sub-term
  while un-grasped; dense reward = `gamma*Phi(s') - Phi(s)` (Ng/Harada/
  Russell invariance, gamma = REWARD_GAMMA = 0.99) − `TIME_PENALTY`
  0.01/step; `TERMINAL_PLACE_BONUS` 10.0 paid once on the place step.
- **The 04-09 pins (the ONLY reward-structure change):**
  `APPROACH_W = 0.15`, `APPROACH_POTENTIAL_SCALE_M = 0.10` — the
  approach gradient is `0.15 / 0.10 = 1.5 Phi/m` (6x the pre-04-09 0.25
  Phi/m), so a representative 1-cm-per-control-step approach move nets
  `0.015 - 0.01 = +0.005` per step against the time penalty (previously
  −0.0075: sustained reaching was locally penalized until a discrete
  flag flipped). The ramp concentrates on the 0-10 cm home-to-cube band;
  far spawns expose more reachable gradient than near ones — the signal
  discriminates spawn distance.
- **Hybrid guard:** `STAGE_POTENTIALS[1] + APPROACH_W = 0.30 <=
  STAGE_POTENTIALS[2] = 0.35` (margin 0.05) — hovering can never
  out-earn a grasp; validated at import and re-pinned by tests
  (`tests/test_reward.py::test_04_09_hover_guard_margin`).
- **Decision record (honesty):** the reward-structure change was the
  user's fourth D-03 decision (2026-10-24, ranked option 4 in
  `runs/investigation-04-cvtrack.md` §5) made DESPITE that addendum
  ranking it "still least indicated" — quoted verbatim: "the shaping
  demonstrably teaches carrying under both contracts (audit returns
  +9.6..+10.0); the frozen constants and the 2.5/6.0 audit-bound
  derivation should not be re-opened on this evidence" — the user
  overrode that ranking with it in full view. The change is recorded as
  a deliberate override, never as evidence-indicated. (The outcome —
  9/25 then 20/25 vs the held bar — is the first pass in the phase's
  history.)
- **Audit bounds (RE-DERIVED, not assumed):**
  `audit_no_success_return_max = 2.5`, `audit_success_return_min = 6.0`
  — re-derived in-test from TERMINAL_PLACE_BONUS 10.0 / TIME_PENALTY
  0.01 / FULL_FLAGS_NO_PLACE_MAX_FRACTION 0.2 / the 200-step horizon,
  PLUS the 04-09 sub-term-ceiling row: `max(STAGE_POTENTIALS) +
  APPROACH_W = 1.15 < 2.0` dense ceiling
  (`tests/test_train_gate.py::test_audit_bounds_derivation`).
- **Grasp predicate (04-07, frozen):** the off-floor z-clause — window 2
  consecutive control steps, 3D track tolerance 0.05 m, applied jaw ctrl
  <= 0.20, AND the cube center strictly above
  `GRASP_CUBE_OFF_FLOOR_Z_M = 0.017` at window completion — a
  closed-jaw push can never set the grasp flag.

## Measured capability (the TRAIN-03 judgment)

- **Deterministic (the primary bar): 20/25 spawns (0.80) >= the HELD
  18/25 bar** — per-spawn majority of 3 episodes over the frozen
  25-spawn grid, 75 episodes, seed base 0
  (`runs/eval/20261027T032735Z/results.json`, gate_echo
  `min_spawns_successful: 18`, records_count 150). Solved spawns:
  {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21,
  23}; missed: {3, 11, 15, 22, 24}.
- **Stochastic (reported alongside, D-02): 21/25 (0.84).**
- **Rendered audit (no reward hacking):**
  `runs/policy_audit/20261027T032838Z/verdict.json` —
  `no_reward_hacking: true`, violations [] — 4/5 rendered carries
  (spawns 12/0/6/18 at +9.62…+10.02, short success GIFs of 37-66
  frames; spawn 24 a −2.31 no-progress failure), bounds 2.5/6.0 read
  from the gate. 4x upscaled GIFs under `frames/`.
- **Reward-structure audit:** 28/28 trajectories green on the new
  constants (`runs/reward_audit/20261026T223409Z/report.json` — the
  telescoping identity, the constant-window dense bound, the
  full-flags-no-place bound, and the approach-retreat negative control
  all hold under the 6x stronger sub-term).

## Human visual confirmation

**Carried pending** — the rendered audit GIFs (the two new 04-09 sets
`runs/policy_audit/20261027T000047Z/frames/` and
`runs/policy_audit/20261027T032838Z/frames/`, plus the six carried
04-06/04-07/04-08 sets) await the end-of-phase human review; the record
lives in
`.planning/phases/04-baseline-training/04-VALIDATION.md` (the
`04-09 observation record:` line). The NUMERIC gates above are green
regardless; the visual review confirms the stage flags visibly match the
reward narrative (real carries, no pushing) on the judged bundle.

## Reproduction

```bash
# 1. the LR re-tune (the sweep is gate data; writes
#    runs/train/tune/selection.json — chosen 3e-4 via the 1% tie band)
uv run python scripts/train.py tune

# 2. the 1.5M run (consumes selection.json automatically)
uv run python scripts/train.py run

# 3. on a nonzero-but-sub-bar miss, the ONE pinned D-03 retry
#    (this produced the frozen baseline)
uv run python scripts/train.py retry --confirm-retry

# 4. the TRAIN-03 judgment + the rendered audit
uv run python scripts/evaluate_baseline.py --bundle runs/train/20261027T000117Z --full-grid --mode both
uv run python scripts/run_policy_audit.py --bundle runs/train/20261027T000117Z
```

**Flagged assumption:** bit-exact reproducibility is NOT claimed —
subproc worker timing perturbs trajectories; what IS claimed is the
reload-proven bundle (bitwise action equality on the saved artifacts),
the pinned seed/gate constants in `config.json`/`gate.json`, and the
frozen spawn protocol (`src/so100rob/data/spawn_seeds.json`).

## Downstream consumers

Phase 5's sensor-noise wrapper and EVERY Phase 5-7 consumer load THIS
28-dim CNN-only observation contract and THIS reward (the strengthened
approach gradient), against THIS bundle. Any future reward-constant,
predicate, or observation-contract change is a NEW pre-agreement and
invalidates this reference point.

---

**TRAIN-01** — PPO baseline trained reproducibly via stable-baselines3:
the four-artifact bundle (checkpoint + normalization stats + config +
seed) with the bitwise reload proof, as required.
**TRAIN-03** — baseline reached the pre-agreed success threshold
(20/25 deterministic >= 18/25 held), verified by a rendered reward audit
that rules out reward hacking (4/5 rendered carries at +9.62…+10.02,
`no_reward_hacking: true`).

*Frozen by 04-09 Task 5 (D-03 outcome (b)) on 2026-10-27. History:
04-05 4/25 + 8/25 (miss) -> 04-06 predicate loosened, 2/25 + 1/25
(miss) -> 04-07 off-floor z-clause, 10/25 + 13/25 (miss) -> 04-08
classical-CV features, 3/25 + 12/25 (miss) -> 04-09 approach-gradient
strengthening + CV retirement, 9/25 + 20/25 (PASS).*