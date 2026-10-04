#!/usr/bin/env python
"""Wall-clock throughput benchmark for the Phase 3 obs-space candidates (TRAIN-02).

Implements RESEARCH.md Pattern 6 exactly (D-13/D-14/D-15): one raw
SO100PickPlaceEnv per run, four observation configs — state_only (no render),
rgb64, gray64 (rgb64 render + per-step numpy luminance conversion), rgb128 —
with each render config timed in two modes: raw stepping, and stepping plus a
mock no-grad ~NatureCNN forward per control step.

Timing protocol (Pitfall 8):
  * env AND CNN are constructed OUTSIDE the timed region;
  * 200 warm-up control steps untimed (includes renderer creation — lazy on
    first render — and first-frame GL warm-up);
  * a fixed N control steps timed with time.perf_counter();
  * >= 3 repeats per (config, mode) pair, median reported;
  * state_only genuinely never renders (render_mode=None, no render call in
    its loop — an accidental render would ruin the baseline, render is ~30x
    physics cost).

Output: runs/benchmark/<UTC-timestamp>/results.json (gitignored). The last
stdout line is ``RESULTS_JSON=<path>`` so tooling (the smoke test) can locate
it. The committed BENCHMARK.md at the repo root is the human-readable summary.

All runs must go through the pinned 3.12 venv (Pitfall 5: system-Homebrew
torch import crashes with libomp Error #15) — use ``uv run python``.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import gymnasium
import mujoco
import numpy as np
import torch
from torch import nn

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs" / "benchmark"

# Protocol constants (Pattern 6 / Pitfall 8).
WARMUP_STEPS = 200
DEFAULT_STEPS = 2000
DEFAULT_REPEATS = 3
QUICK_STEPS = 50
QUICK_REPEATS = 1

# Timing modes: state_only is raw-only; each render config is timed in both.
MODE_RAW = "raw"
MODE_MOCK_CNN = "mock_cnn"

# Config table (D-13): name -> (render_size, grayscale). state_only is special.
RENDER_CONFIGS: dict[str, tuple[int, bool]] = {
    "rgb64": (64, False),
    "gray64": (64, True),
    "rgb128": (128, False),
}
STATE_ONLY = "state_only"

# Rec. 601 luminance weights for the gray64 per-step conversion.
_LUMA = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)


# ---------------------------------------------------------------------------
# Mock CNN (D-14) — roughly NatureCNN size, built OUTSIDE the timed loop
# ---------------------------------------------------------------------------


def build_mock_cnn(in_channels: int, size: int) -> nn.Sequential:
    """3 conv layers (NatureCNN geometry) + flatten + linear to 512.

    32 filters 8x8 stride 4 -> 64 filters 4x4 stride 2 -> 64 filters 3x3
    stride 1, then a 512-unit linear head. eval() mode, used only under
    torch.no_grad() — it exists to price the CNN forward, not to learn.
    """
    h = w = size
    h = (h - 8) // 4 + 1
    w = (w - 8) // 4 + 1
    h = (h - 4) // 2 + 1
    w = (w - 4) // 2 + 1
    h = (h - 3) // 1 + 1
    w = (w - 3) // 1 + 1
    flat = 64 * h * w
    model = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(flat, 512),
    )
    model.eval()
    return model


def frame_to_input(frame: np.ndarray, gray: bool) -> torch.Tensor:
    """uint8 HWC frame -> float32 (1, C, H, W) tensor in [0, 1].

    For gray64 the per-step numpy luminance conversion is part of the config's
    measured cost (D-13: 'rgb64 render followed by a numpy luminance
    conversion'), so it belongs in this pipeline, not the raw mode's loop.
    """
    if gray:
        # Mirrors the env-side luma -> rint -> clip transform (the env's
        # grayscale _get_obs), plus the /255 mock-CNN input rescale.
        arr = np.rint(frame.astype(np.float32) @ _LUMA).clip(0, 255)  # (H, W)
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        t = t.to(torch.float32).div_(255.0)
    else:
        # ascontiguousarray: mujoco render returns a vertically-flipped
        # (negative-stride) array that torch.from_numpy rejects — one copy
        # is unavoidable and is part of the measured inference pipeline.
        t = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)  # (3,H,W)
        t = t.unsqueeze(0).to(torch.float32).div_(255.0)  # (1, 3, H, W)
    return t


# ---------------------------------------------------------------------------
# Single timed runs — env and CNN constructed before, timing after warm-up
# ---------------------------------------------------------------------------


def time_state_only(steps: int, warmup: int) -> float:
    """State-only config: genuinely NO renderer, NO render call (Pitfall 8)."""
    env = SO100PickPlaceEnv(render_mode=None)
    env.reset(seed=0)
    action = np.asarray(C.HOME_CTRL, dtype=np.float32)
    try:
        for _ in range(warmup):
            env.step(action)
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(action)
        elapsed = time.perf_counter() - t0
    finally:
        env.close()
    return steps / elapsed


def time_render(
    size: int, gray: bool, with_cnn: bool, steps: int, warmup: int
) -> float:
    """A render config (rgb64 / gray64 / rgb128), raw or with mock CNN.

    The env's single renderer instance is reused via env.render() (gymnasium
    caches one OffScreenViewer per mode, created lazily on the FIRST render —
    so the warm-up loop below owns renderer creation and GL warm-up, never
    the timed loop).
    """
    env = SO100PickPlaceEnv(render_mode="rgb_array", width=size, height=size)
    env.reset(seed=0)
    action = np.asarray(C.HOME_CTRL, dtype=np.float32)
    cnn = build_mock_cnn(1 if gray else 3, size) if with_cnn else None
    try:
        with torch.no_grad():
            for _ in range(warmup):
                env.step(action)
                frame = env.render()
                if cnn is not None:
                    cnn(frame_to_input(frame, gray))
            t0 = time.perf_counter()
            for _ in range(steps):
                env.step(action)
                frame = env.render()
                if cnn is not None:
                    cnn(frame_to_input(frame, gray))
            elapsed = time.perf_counter() - t0
    finally:
        env.close()
    return steps / elapsed


def single_run(config: str, mode: str, steps: int, warmup: int) -> float:
    if config == STATE_ONLY:
        if mode != MODE_RAW:
            raise ValueError(f"{STATE_ONLY} has only the raw mode, got {mode!r}")
        return time_state_only(steps, warmup)
    size, gray = RENDER_CONFIGS[config]
    return time_render(size, gray, mode == MODE_MOCK_CNN, steps, warmup)


# ---------------------------------------------------------------------------
# Machine / version context
# ---------------------------------------------------------------------------


def machine_context() -> dict:
    ctx = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        ctx["chip"] = chip.stdout.strip() or platform.machine()
    except (OSError, subprocess.SubprocessError):
        ctx["chip"] = platform.machine()
    return ctx


def package_versions() -> dict:
    return {p: pkg_version(p) for p in ("mujoco", "gymnasium", "torch", "numpy")}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_benchmark(steps: int, repeats: int, warmup: int, quick: bool) -> Path:
    entries = []
    configs = [STATE_ONLY, *RENDER_CONFIGS]
    for config in configs:
        modes = [MODE_RAW] if config == STATE_ONLY else [MODE_RAW, MODE_MOCK_CNN]
        for mode in modes:
            values = [single_run(config, mode, steps, warmup) for _ in range(repeats)]
            entries.append(
                {
                    "config": config,
                    "mode": mode,
                    "steps_per_sec": statistics.median(values),
                    "repeats": values,
                    "n_steps": steps,
                    "warmup_steps": warmup,
                }
            )
            print(
                f"[benchmark] {config}/{mode}: median "
                f"{entries[-1]['steps_per_sec']:.1f} steps/s "
                f"(repeats: {', '.join(f'{v:.1f}' for v in values)})",
                file=sys.stderr,
            )

    # Pitfall 8 sanity gate: state-only must strictly exceed every render
    # config's raw throughput by a wide margin (render is ~30x physics). A
    # near-parity result means the state-only loop is accidentally rendering.
    state_raw = next(
        e["steps_per_sec"] for e in entries if e["config"] == STATE_ONLY
    )
    render_raws = {
        e["config"]: e["steps_per_sec"]
        for e in entries
        if e["config"] != STATE_ONLY and e["mode"] == MODE_RAW
    }
    slowest_render = min(render_raws.values())
    ratio = state_raw / slowest_render
    sanity = "OK" if state_raw > slowest_render and ratio > 2.0 else "FAIL"
    print(
        f"[benchmark] sanity gate (Pitfall 8): state_only={state_raw:.1f} vs "
        f"slowest render raw={slowest_render:.1f} (ratio {ratio:.1f}x) -> {sanity}",
        file=sys.stderr,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quick": quick,
        "n_steps": steps,
        "warmup_steps": warmup,
        "repeats": repeats,
        "machine": machine_context(),
        "versions": package_versions(),
        "sanity_gate": {
            "state_only_raw": state_raw,
            "slowest_render_raw": slowest_render,
            "ratio": ratio,
            "status": sanity,
        },
        "results": entries,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    # The sanity gate must actually GATE (WR-02): the failing results.json is
    # written first (evidence for debugging), then the run fails loudly — a
    # FAIL status must never reach a downstream consumer via exit 0.
    if sanity != "OK":
        raise RuntimeError(
            f"Pitfall 8 sanity gate FAILED (state_only={state_raw:.1f} vs "
            f"slowest render raw={slowest_render:.1f}, ratio {ratio:.1f}x) — "
            "state_only may be rendering"
        )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Throughput benchmark: 4 obs configs x 2 timing modes (TRAIN-02)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"timed control steps per run (default {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"repeats per (config, mode); median is reported (default {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"smoke mode: steps={QUICK_STEPS}, repeats={QUICK_REPEATS}",
    )
    args = parser.parse_args()

    quick = args.quick
    steps = QUICK_STEPS if quick else args.steps
    repeats = QUICK_REPEATS if quick else args.repeats
    if steps < 1 or repeats < 1:
        parser.error("--steps and --repeats must be >= 1")

    out_path = run_benchmark(steps, repeats, WARMUP_STEPS, quick)
    print(f"RESULTS_JSON={out_path}")


if __name__ == "__main__":
    main()