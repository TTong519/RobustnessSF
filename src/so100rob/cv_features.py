"""Classical-CV color-blob centroids (04-08 pre-agreement).

PURITY CONTRACT (T-04-23): every function in this module is a pure
function of the rendered RGB frame — a numpy-only computation over raw
uint8 pixels. It can never read simulator state: the module imports
neither the physics bindings nor any env module nor the scripted policy
(pinned by the structural source-scan test), and the state key of the
observation already carries the TRUE cube pose — a centroid computed
from privileged state would duplicate that data and silently turn the
perception experiment into a tautology that passes while testing
nothing. The probe's affine fit (see below) is the evidence the mask
tracks the RENDERED cube.

The mask bounds are PROBE-MEASURED, never guessed (probe-before-pin, the
04-07 discipline): scripts/probe_cv_features.py rendered the real env at
all 25 frozen spawns and classified the changing palette (the moving
cube) against the static palette (floor/arm/goal); the bounds are pinned
in so100rob.constants and measured on the 04-08 probe artifact pattern
runs/cv_probe/<ts>/probe.json (cube mask counts 8..13 px/spawn, affine
fit max residual 0.276 px; goal cluster fixed, centroid std 0.0 px).

The centroid layout (consumed by SO100PickPlaceEnv's cv_features mode —
the 32-dim CV observation contract, appended at state indices
[28:32]): [cube_u, cube_v, goal_u, goal_v], with u = mean_col/(W-1) and
v = mean_row/(H-1) in [0, 1]. An empty mask yields the pinned neutral
CV_BLOB_MISSING_UV — never NaN (the env's finite guard covers the tail).
"""

from __future__ import annotations

import numpy as np

from so100rob import constants as C


def cube_mask(frame) -> np.ndarray:
    """Boolean per-pixel test of the probe-measured CUBE bounds.

    Closed per-channel bounds on the raw uint8 frame — exact thresholds,
    no tolerance: a pixel one step outside a pinned bound is excluded
    (anti-aliased edge pixels are deterministically out).
    """
    frame = np.asarray(frame)
    lo = np.asarray(C.CV_CUBE_RGB_LO, dtype=np.uint8)
    hi = np.asarray(C.CV_CUBE_RGB_HI, dtype=np.uint8)
    return ((frame >= lo) & (frame <= hi)).all(axis=-1)


def goal_mask(frame) -> np.ndarray:
    """Boolean per-pixel test of the probe-measured GOAL bounds (the fixed
    R-dominant goal-marker cluster)."""
    frame = np.asarray(frame)
    lo = np.asarray(C.CV_GOAL_RGB_LO, dtype=np.uint8)
    hi = np.asarray(C.CV_GOAL_RGB_HI, dtype=np.uint8)
    return ((frame >= lo) & (frame <= hi)).all(axis=-1)


def blob_centroid(mask) -> tuple[float, float]:
    """(u, v) of a boolean mask: mean column / (W-1), mean row / (H-1).

    Accumulates in float64 over integer pixel coordinates and normalizes
    in float64 (the exact contract; the env casts the assembled tail to
    float32). An EMPTY mask returns the pinned neutral CV_BLOB_MISSING_UV
    — the single blob-undefined case, pinned data, never NaN and never a
    raise (T-04-26).
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(
            f"blob_centroid expects a 2-dim boolean mask, got shape {mask.shape}"
        )
    if not mask.any():
        return (C.CV_BLOB_MISSING_UV[0], C.CV_BLOB_MISSING_UV[1])
    height, width = mask.shape
    if height < 2 or width < 2:
        raise ValueError(
            f"blob_centroid needs W >= 2 and H >= 2 to normalize (got {width}x{height})"
        )
    rows, cols = np.nonzero(mask)
    u = float(cols.mean()) / (width - 1)
    v = float(rows.mean()) / (height - 1)
    return (u, v)


def extract_centroids(frame) -> np.ndarray:
    """The 4-dim CV tail: float32 [cube_u, cube_v, goal_u, goal_v].

    A pure function of the rendered RGB frame ONLY (T-04-23) — the two
    centroids of the probe-measured cube and goal masks. Each channel is
    in [0, 1] (or the pinned 0.5 neutral on an empty mask).
    """
    frame = np.asarray(frame)
    cube_u, cube_v = blob_centroid(cube_mask(frame))
    goal_u, goal_v = blob_centroid(goal_mask(frame))
    return np.asarray([cube_u, cube_v, goal_u, goal_v], dtype=np.float32)