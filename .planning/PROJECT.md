# SO-ARM100 RL Robustness Experiment

## What This Is

A small simulation experiment testing how a reinforcement learning policy trained on one task adapts to performing that same task under changed conditions. A deep RL policy (stable-baselines3) is trained in MuJoCo — using the existing `trs_so_arm100` model — to do pick-and-place with the object randomized within a small area. The trained policy is then evaluated under perturbed conditions (control update rate, action noise, action latency, sensor noise), fine-tuned under each condition, and the results are written up in a report.

## Core Value

A clear, quantified picture of how an RL policy's pick-and-place performance degrades under changed conditions — and how quickly fine-tuning recovers it.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Train a baseline deep RL policy (PPO or SAC via stable-baselines3) for pick-and-place on the SO-ARM100 in MuJoCo, with object position randomized within a small area
- [ ] Evaluate the baseline policy under four perturbation conditions: changed control update rate, action noise, action latency, and sensor noise
- [ ] Fine-tune the policy under each perturbed condition and measure recovery speed
- [ ] Measure task success with a stage breakdown (approach / grasp / lift / place) so degradation can be localized
- [ ] Produce a results report with plots: success rate and stage metrics vs. each perturbation, before and after fine-tuning

### Out of Scope

- Real hardware — this is simulation-only; validating on the physical arm is a separate future effort
- Domain randomization during baseline training — perturbations are applied only at evaluation/fine-tuning time so the "same task, different conditions" comparison stays clean
- Camera/lighting/visual perturbations — the sweep covers update rate, action noise, action latency, and sensor noise only
- Reusable general-purpose eval harness beyond this experiment — the deliverable is the report, tooling is a byproduct
- Multi-task or curriculum learning — one task, one policy

## Context

- **Existing model:** `trs_so_arm100/` contains the MuJoCo SO-ARM100 model (`so_arm100.xml`, `scene.xml`), README, and assets. This is the robot definition the experiment builds on.
- **Why this exists:** the user wants to understand whether/how an RL policy trained for one robotic task can adapt to the same task under different operating conditions — a robustness question, not a capability question.
- **Prior art informs the design:** perturbation-style robustness benchmarks (e.g. LIBERO-plus, SO-101 VLA failure analysis) and domain-randomization practices (control jitter, action latency, observation noise injection) are well established; this project applies that lens to deep RL on the SO-ARM100.
- **Observations:** the policy observes joint state plus vision (low-resolution camera images). Kept deliberately cheap so training is tractable.

## Constraints

- **Compute**: Local Mac only (Apple Silicon) — all training and evaluation runs locally. Vision RL is slow, so image observations must be small (low-res, optionally grayscale) and the experiment sized so phases complete in reasonable wall-clock time. This is the primary reason to keep everything modest.
- **Simplicity**: Keep the project reasonably simple — this is a small test project, not a research platform. Prefer the boring, standard choice over the clever one; cut anything that doesn't directly feed the report.
- **Tech stack**: MuJoCo + Gymnasium custom env + stable-baselines3 (PPO or SAC). Python.
- **Simulation-only**: No real-hardware code paths; no sim-to-real work in this milestone.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Deep RL (SB3 PPO/SAC) rather than imitation learning (LeRobot) | The experiment is about RL adaptation; SB3 is the standard, well-documented deep RL toolchain | — Pending |
| State + vision observations | Matches realistic robot setups; acknowledged cost: vision deep RL is slow locally | — Pending |
| Four perturbation conditions (update rate, action noise, action latency, sensor noise) | Covers the timing/noise dimensions the user cares about without a visual-perturbation rabbit hole | — Pending |
| Measure + fine-tune ("both") | Degradation alone shows fragility; recovery speed shows adaptability — together they answer the question | — Pending |
| Stage-broken success metric (approach/grasp/lift/place) | Success-rate saturation can hide degradation; stage breakdown localizes where failures appear | — Pending |
| Object randomized within a small area during baseline training | Standard practice; fairer robustness comparison than a fully fixed spawn | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-09-03 after initialization*