"""Interactive MuJoCo viewer for hand-tuning the task scene (D-05 tool).

MUST be launched via the ``mjpython`` launcher on macOS — ``launch_passive``
needs it. The ONLY blessed invocation, run from the repo root:

    uv run mjpython scripts/debug_viewer.py

(equivalently ``.venv/bin/mjpython scripts/debug_viewer.py`` — the
venv-resolved launcher, so the pinned mujoco 3.12.0 and the editable
``so100rob`` install are used). Do NOT launch with a bare
``mjpython scripts/debug_viewer.py``: without the venv active, PATH
resolution picks /opt/homebrew/bin/mjpython (homebrew python@3.14),
which lacks the project package ``so100rob`` (editable-installed only
into the project venv) and runs mujoco 3.8.0 instead of the pinned
3.12.0 — a different physics version than the gate ran.

NEVER run this from tests or the acceptance gate — it blocks on an
interactive window.

What it does:
  * compiles the L1 task scene (SCENE_XML_PATH — include-and-extend of the
    read-only upstream SO-ARM100 model);
  * applies the same reset protocol as ``SO100PickPlaceEnv.reset_model()``:
    keyframe ("home") reset, explicit cube overwrite at SPAWN_CENTER
    (the keyframe zero-pads the cube to world origin), then the settle loop;
  * opens the passive viewer and steps physics in a sync loop until the
    window closes, so you can eyeball grasp geometry / spawn placement /
    camera angles and iterate on scene_pickplace.xml.

The arm holds the home ctrl throughout (same as the tracer stability test);
edit ``HOME_CTRL`` usage below to drive other poses while tuning.
"""

import os
import sys

import mujoco
import mujoco.viewer
import numpy as np

try:
    from so100rob import constants as C
except ModuleNotFoundError:
    # A bare PATH-resolved mjpython (homebrew python@3.14) has mujoco but not
    # the project venv's editable so100rob — fail fast with the fix instead of
    # a traceback (debug session G-01-1).
    print(
        "so100rob not importable on this interpreter — launch with: "
        "uv run mjpython scripts/debug_viewer.py",
        file=sys.stderr,
    )
    sys.stderr.flush()
    # mjpython's event loop swallows SystemExit and would exit 0, so
    # sys.exit(1) cannot signal failure under the mandatory launcher —
    # os._exit guarantees the nonzero status (debug session G-01-1).
    os._exit(1)


def reset_to_settled_spawn(model, data):
    """Same reset recipe as SO100PickPlaceEnv.reset_model() (research Pattern 2).

    Order matters: keyframe reset first (sets BOTH qpos and ctrl), then the
    MANDATORY cube qpos overwrite (the keyframe zero-pads the cube to world
    origin, half inside the floor), then raw-physics settle to rest.
    """
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, C.HOME_KEY)
    mujoco.mj_resetDataKeyframe(model, data, home_id)

    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, C.CUBE_JOINT)
    qposadr = model.jnt_qposadr[cube_id]
    dofadr = model.jnt_dofadr[cube_id]

    data.qpos[qposadr : qposadr + 3] = C.SPAWN_CENTER
    data.qpos[qposadr + 3 : qposadr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[dofadr : dofadr + 6] = 0.0

    mujoco.mj_forward(model, data)
    for _ in range(C.SETTLE_STEPS):
        mujoco.mj_step(model, data)

    cube_qvel = np.asarray(data.qvel[dofadr : dofadr + 6])
    qvel_norm = float(np.linalg.norm(cube_qvel))
    assert qvel_norm < C.REST_QVEL_TOL, (
        f"unsettled cube spawn: qvel norm {qvel_norm:.3e} >= {C.REST_QVEL_TOL:.3e}"
    )
    return qvel_norm


def main():
    model = mujoco.MjModel.from_xml_path(str(C.SCENE_XML_PATH))
    data = mujoco.MjData(model)

    qvel_norm = reset_to_settled_spawn(model, data)
    print(f"cube settled at SPAWN_CENTER, qvel norm {qvel_norm:.3e}")

    # Hold the home ctrl while the viewer runs (matches the tracer test).
    # The keyframe reset already set ctrl == home qpos; this is just explicit.
    data.ctrl[:] = C.HOME_CTRL

    print("viewer open — close the window to exit")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()