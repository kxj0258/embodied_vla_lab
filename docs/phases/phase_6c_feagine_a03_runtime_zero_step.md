# Phase 6C — Feagine A03 Runtime Zero-Step Gate

Phase 6C installs the supplied runtime into a new isolated Conda environment; it does not modify `evla-core`.
The core extension is built from the supplied source, and only the local MuJoCo wheel is installed.

The runtime gate explicitly loads preset `a03_type_2`, creates `MjModel`/`MjData`, binds `FeagineMjcfRobot`, checks
all expected joints and actuators, runs a core FK/IK consistency target, and writes bounded section/gripper/rotation
commands to `data.ctrl`. It never calls `mj_step`; `data.time` must remain exactly zero before and after the gate.

Passing authorizes only a later short headless simulation design. It does not demonstrate dynamic convergence,
collision safety, target reaching, policy-frame calibration, VLA inference or hardware control.

Runtime v1 installed and built successfully, then returned FAIL at 10/11 because the protocol expected 52 joints.
The static count had included the unnamed `<default><joint>` template; MuJoCo correctly instantiated 51 named joints
and 51 actuators. All other runtime/core/zero-step checks passed. v1 is retained; v2 changes only the corrected named
joint expectation, reuses the installed isolated environment, and uses fresh output/result/log paths.
