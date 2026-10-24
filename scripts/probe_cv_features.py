#!/usr/bin/env python
"""The 04-08 CV feature probe (probe-before-pin — the 04-07 discipline).

The color-blob mask bounds pinned in so100rob.constants (CV_CUBE_RGB_LO/HI,
CV_GOAL_RGB_LO/HI) are MEASURED on the real rendered env, never guessed: this
probe renders the vision env at all 25 frozen spawns (arm in view at the home
pose — the worst case for mask separation) and classifies the per-frame color
palettes before any contract is pinned (the 04-08 pre-agreement, the user's
2026-10-23 option-2 decision in runs/investigation-04-zclause.md section 5;
mechanism per runs/investigation-04-cve-groundwork.md).

Palette classification (the plan's rule):
  * CHANGING palette — colors present in SOME but not all 25 frames: the
    moving cube, its anti-aliased boundary, and its contact shadow.
  * STATIC palette — colors present in ALL frames: floor, arm, goal marker.

Bound selection (from pinned scene priors, stated as color-family tests —
the bounding boxes are then the per-channel min/max over each family):
  * CUBE family — changing colors whose channel ORDER matches the cube
    geom's pinned rgba (0.9, 0.6, 0.1): R > G > B, with the channel-ratio
    requirement R >= 2*B (a conservative floor on the true 9:1 ratio). The
    ratio requirement excludes the near-gray cube-floor AA blends that
    would otherwise admit static arm grays into the bounding box.
  * GOAL family — STATIC R-dominant colors (R > G, R > B, R - G > 20): the
    goal site's pinned rgba (1 0 0, semi-transparent) renders as a small
    fixed red cluster blended over the floor.

Acceptance bars (the probe's own gate — a failed bar is the honest probe
outcome; no contract is pinned on a failed probe):
  * per-spawn cube mask count in [1, 20] pixels (dead or contaminated mask);
  * affine fit (world spawn x,y -> cube centroid px) max abs residual < 2.0
    (a false-positive blob would not move with the cube and blows this up);
  * goal mask count >= 1 on every frame (an invisible marker);
  * goal centroid std < 0.5 px across frames (the marker is fixed);
  * mutual exclusion: no static color inside the cube box, no cube-family
    color inside the goal box (contamination would poison either centroid).

Goal-visibility contingency (probe-triggered, visual-only — did NOT fire on
the measured run): if the goal bars fail, scene_pickplace.xml's goal_site
gets size 0.01 -> 0.02 and rgba '1 0 0 0.6' -> '1 0 0 1' (sites never
collide, physics provably unchanged) and the probe is re-run, both runs
recorded.

Output: runs/cv_probe/<UTC-ts>/probe.json with the palettes, the chosen
bounds, the per-spawn mask counts and centroids, and the fit residual; the
LAST stdout line is PROBE_JSON=<repo-relative path> (the plan-verify
spelling). Runs under the pinned venv: uv run python
scripts/probe_cv_features.py. The env is a single in-process vision env,
never closed mid-run (WR-01: OffScreenViewer.close() terminates
process-global GLFW state).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from so100rob import constants as C
from so100rob.envs.pick_place_env import SO100PickPlaceEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs" / "cv_probe"

# Acceptance bars (the 04-08 pre-agreement, stated once here).
CUBE_COUNT_MIN, CUBE_COUNT_MAX = 1, 20
FIT_RESIDUAL_MAX_PX = 2.0
GOAL_COUNT_MIN = 1
GOAL_CENTROID_STD_MAX_PX = 0.5
# Cube-family color test (the pinned cube geom rgba 0.9 0.6 0.1 prior):
# channel order R > G > B with the channel-ratio floor R >= 2*B.
CUBE_RATIO_FLOOR = 2
# Goal-family color test (the pinned goal site rgba (1, 0, 0) prior):
# R-dominant static colors with R - G > 20.
GOAL_RG_MARGIN = 20


def classify_palettes(frames: np.ndarray) -> tuple[list, list]:
    """Per-frame unique colors -> (static, changing) palettes.

    static = present in ALL frames; changing = present in some but not all.
    """
    per_frame = [set(map(tuple, f.reshape(-1, 3).tolist())) for f in frames]
    static = sorted(set.intersection(*per_frame))
    changing = sorted(set().union(*per_frame) - set(static))
    return static, changing


def cube_family(changing) -> list:
    return [c for c in changing if c[0] > c[1] > c[2] and c[0] >= CUBE_RATIO_FLOOR * c[2]]


def goal_family(static) -> list:
    return [c for c in static if c[0] > c[1] and c[0] > c[2] and c[0] - c[1] > GOAL_RG_MARGIN]


def family_bounds(family) -> tuple[list, list]:
    lo = [min(c[k] for c in family) for k in range(3)]
    hi = [max(c[k] for c in family) for k in range(3)]
    return lo, hi


def mask_of(frame: np.ndarray, lo, hi) -> np.ndarray:
    """Closed per-channel uint8 bounds — exact thresholds, no tolerance."""
    return (
        (frame >= np.asarray(lo, dtype=np.uint8))
        & (frame <= np.asarray(hi, dtype=np.uint8))
    ).all(axis=-1)


def centroid_px(mask: np.ndarray):
    """(mean_col, mean_row) of the mask in float64, or None when empty."""
    if not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    return [float(cols.mean()), float(rows.mean())]


def main() -> int:
    grid = json.loads(C.SPAWN_SEEDS_PATH.read_text())["spawns"]
    n = len(grid)

    env = SO100PickPlaceEnv(
        obs_mode=C.OBS_MODE_VISION,
        render_mode="rgb_array",
        width=C.OBS_IMAGE_SIZE,
        height=C.OBS_IMAGE_SIZE,
        grayscale=True,  # the OBS_DECISION format; render() stays raw RGB
        spawn_mode="grid",
    )
    # GL warm-up (the 03-02 finding): the FIRST render after viewer creation
    # differs slightly; discard it so all 25 measured frames come from the
    # deterministic regime. reset() itself renders once inside _get_obs.
    env.reset(seed=0)
    env.render()

    frames = []
    for i in range(n):
        env.reset(seed=0, options={"spawn_index": i})
        frames.append(np.ascontiguousarray(env.render()))
    frames = np.stack(frames)
    # WR-01: the vision env is NEVER closed mid-process (glfw.terminate() is
    # process-global); process exit reclaims the GL context.

    static, changing = classify_palettes(frames)
    print(f"[probe] frames={n} unique-colors: static={len(static)} changing={len(changing)}", flush=True)

    cube_fam = cube_family(changing)
    goal_fam = goal_family(static)
    if not cube_fam:
        print(
            "PROBE FAILED: the changing palette supports no cube-family colors "
            "(R > G > B with R >= 2*B) — the cube is not separable from the "
            "static scene by this mechanism; the feature track's mechanism is "
            "falsified before any code lands (STOP and surface to the user).",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if not goal_fam:
        print(
            "PROBE FAILED: no static R-dominant goal cluster found — the goal "
            "marker is invisible at 64x64 (the probe-triggered visual-only "
            "contingency applies: goal_site size 0.01 -> 0.02, rgba alpha "
            "0.6 -> 1.0, then re-run this probe recording both runs).",
            file=sys.stderr,
            flush=True,
        )
        return 1
    cube_lo, cube_hi = family_bounds(cube_fam)
    goal_lo, goal_hi = family_bounds(goal_fam)
    print(f"[probe] cube bounds lo={cube_lo} hi={cube_hi} (family {len(cube_fam)} colors)", flush=True)
    print(f"[probe] goal bounds lo={goal_lo} hi={goal_hi} (family {len(goal_fam)} colors)", flush=True)

    # Mutual exclusion (contamination honesty checks).
    static_in_cube = [c for c in static if all(cube_lo[k] <= c[k] <= cube_hi[k] for k in range(3))]
    cube_fam_in_goal = [c for c in cube_fam if all(goal_lo[k] <= c[k] <= goal_hi[k] for k in range(3))]
    goal_fam_in_cube = [c for c in goal_fam if all(cube_lo[k] <= c[k] <= cube_hi[k] for k in range(3))]

    cube_counts, goal_counts = [], []
    cube_centroids, goal_centroids = [], []
    for f in frames:
        cm = mask_of(f, cube_lo, cube_hi)
        gm = mask_of(f, goal_lo, goal_hi)
        cube_counts.append(int(cm.sum()))
        goal_counts.append(int(gm.sum()))
        cube_centroids.append(centroid_px(cm))
        goal_centroids.append(centroid_px(gm))

    # Affine fit: world spawn (x, y, 1) -> cube centroid pixel (col, row).
    fit_ok = all(c is not None for c in cube_centroids)
    fit_max_residual = float("inf")
    if fit_ok:
        A = np.array([[g[0], g[1], 1.0] for g in grid])
        cu = np.array([c[0] for c in cube_centroids])
        cv = np.array([c[1] for c in cube_centroids])
        su, _, _, _ = np.linalg.lstsq(A, cu, rcond=None)
        sv, _, _, _ = np.linalg.lstsq(A, cv, rcond=None)
        fit_max_residual = float(
            max(np.abs(A @ su - cu).max(), np.abs(A @ sv - cv).max())
        )

    goal_std_px = None
    if all(c is not None for c in goal_centroids):
        arr = np.asarray(goal_centroids, dtype=np.float64)
        goal_std_px = float(max(arr[:, 0].std(), arr[:, 1].std()))

    print(f"[probe] cube mask counts: min={min(cube_counts)} max={max(cube_counts)}", flush=True)
    print(f"[probe] affine fit max residual: {fit_max_residual:.4f} px", flush=True)
    print(f"[probe] goal mask counts: min={min(goal_counts)}; goal centroid std: {goal_std_px} px", flush=True)

    bars = {
        "cube_count_in_bounds": all(CUBE_COUNT_MIN <= c <= CUBE_COUNT_MAX for c in cube_counts),
        "fit_residual_lt_2px": fit_ok and fit_max_residual < FIT_RESIDUAL_MAX_PX,
        "goal_count_ge_1": all(c >= GOAL_COUNT_MIN for c in goal_counts) and goal_std_px is not None,
        "goal_centroid_std_lt_0_5px": goal_std_px is not None and goal_std_px < GOAL_CENTROID_STD_MAX_PX,
        "no_static_color_in_cube_box": not static_in_cube,
        "no_cube_color_in_goal_box": not cube_fam_in_goal,
        "no_goal_color_in_cube_box": not goal_fam_in_cube,
    }
    for name, ok in bars.items():
        print(f"[probe] bar {name}: {'OK' if ok else 'FAIL'}", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe": "04-08 classical-CV color-blob mask-bound probe",
        "timestamp": ts,
        "n_spawns": n,
        "n_frames": int(frames.shape[0]),
        "frame_shape": list(frames.shape[1:]),
        "cube_rgb_lo": list(cube_lo),
        "cube_rgb_hi": list(cube_hi),
        "goal_rgb_lo": list(goal_lo),
        "goal_rgb_hi": list(goal_hi),
        "cube_mask_counts": cube_counts,
        "goal_mask_counts": goal_counts,
        "cube_centroids_px": cube_centroids,
        "goal_centroids_px": goal_centroids,
        "goal_centroid_std_px": goal_std_px,
        "fit_max_residual_px": fit_max_residual,
        "spawn_world_xy": [[g[0], g[1]] for g in grid],
        "changing_palette": [list(c) for c in changing],
        "static_palette": [list(c) for c in static],
        "cube_family_size": len(cube_fam),
        "goal_family_size": len(goal_fam),
        "bars": bars,
        "bars_spec": {
            "cube_mask_count_range": [CUBE_COUNT_MIN, CUBE_COUNT_MAX],
            "fit_max_residual_px_lt": FIT_RESIDUAL_MAX_PX,
            "goal_mask_count_min": GOAL_COUNT_MIN,
            "goal_centroid_std_px_lt": GOAL_CENTROID_STD_MAX_PX,
            "cube_family_rule": "changing colors with R > G > B and R >= 2*B (the pinned cube rgba 0.9/0.6/0.1 channel prior)",
            "goal_family_rule": "static colors with R > G, R > B, R - G > 20 (the pinned goal site rgba (1,0,0) prior)",
        },
        "static_colors_inside_cube_box": [list(c) for c in static_in_cube],
        "accepted": all(bars.values()),
    }
    probe_path = out_dir / "probe.json"
    probe_path.write_text(json.dumps(payload, indent=2) + "\n")

    if not all(bars.values()):
        failed = [k for k, ok in bars.items() if not ok]
        print(
            f"PROBE FAILED: bars not met: {failed} — no contract is pinned on a "
            "failed probe; the honest probe outcome (surface to the user; the "
            "goal-bar failure path is the visual-only scene contingency).",
            file=sys.stderr,
            flush=True,
        )
        print(f"PROBE_JSON={probe_path.relative_to(REPO_ROOT)}", flush=True)
        return 1

    print(f"PROBE BARS OK: residual {fit_max_residual:.3f} px over {n} spawns", flush=True)
    print(f"PROBE_JSON={probe_path.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())