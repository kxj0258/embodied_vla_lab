# Phase 4A: viewpoint camera gate

Date: 2026-08-28  
Status: `RUNTIME_PASS`

## Objective

Establish a reproducible paired third-person/eye-in-hand observation contract before claiming or evaluating
viewpoint robustness. The frozen OpenVLA base remains unchanged and still consumes exactly one RGB image. The later
paired experiment changes only whether that image comes from `agentview` or `eye_in_hand`; it does not fuse cameras,
train an adapter or select another checkpoint.

## Gate protocol

- One LIBERO-Goal reset on task 7/init 0/seed 20261101; zero environment steps and zero model predictions.
- Verify native 128x128 `agentview` and `eye_in_hand` observations and direct 640x640 renders.
- Require uint8 RGB, nonconstant images, distinct cross-view pixels, immutable PNG hashes and clean environment close.
- Apply the same later model-side contract to either input: 180-degree rotation followed by OpenVLA's frozen 90%
  center crop. The gate does not load the model and therefore does not measure policy success or reward.

## Preregistered later split

Development task IDs are `[0,3,6,7,8]`; final task IDs are `[1,2,4,5,9]`. Each task uses init states `[0,1,2]`
under paired `agentview` and `eye_in_hand` conditions: 30 development and 30 final episodes. The task split is
deterministic and fixed before observing any eye-in-hand policy result. Development may diagnose only camera,
preprocessing and runtime integration. It cannot train or select a checkpoint. Final runs once only after the
development protocol is frozen.

This experiment can measure robustness to direct viewpoint substitution, not general viewpoint invariance or
real-world robustness. Prior Phase 3/P0 outcomes remain frozen and cannot be used to remove tasks, states or failures.

Config: `configs/eval/phase4a_openvla_viewpoint_camera_gate_v1.yaml`  
Static checker: `scripts/check_phase4a_viewpoint_static.py`  
Runtime probe: `scripts/probe_phase4a_libero_viewpoints.py`

## Static result

The CUDA-free static gate PASSed all 18 checks. It verified the exact frozen base revision, single-reset/zero-step
scope, paired camera names, common preprocessing, fresh no-overwrite targets, disjoint and complete task split, and
30-development/30-final episode arithmetic. Static execution used zero resets, environment steps and predictions.
Canonical record: `results/phase4a_openvla_viewpoint_camera_gate_v1_static.yaml`.

## Runtime v1 failure and corrected v2

The v1 host attempt is classified FAIL, not a policy failure. Reset completed, but direct 640 rendering reused the
observation alias `eye_in_hand` as a MuJoCo camera name; the actual simulator camera is
`robot0_eye_in_hand`. The run exited 1 after 13.27 s with max RSS 2,491,256 KiB, zero steps/predictions, zero
swap/network and clean environment close. Its output and raw log are immutable under
`results/phase4a_openvla_viewpoint_camera_gate_v1_failure.yaml`.

V2 corrects only the simulator camera name, retains the observation alias, frozen checkpoint, split and all
boundaries, and uses new output/log/result paths. Its nine-check static gate PASSed with zero CUDA/reset/step/
prediction. The v2 command is wrapped in a child shell so a nonzero experimental exit cannot terminate the user's
interactive terminal.

## Runtime v2 result

V2 PASSed with host/Python exit 0. It performed one reset, zero environment steps and zero model predictions, then
closed the environment cleanly. Native and direct-render cross-view pixel MAE were 76.4615 and 77.4986, confirming
the views were not duplicate frames. Both 128x128 and 640x640 images were uint8 RGB, nonconstant and hash-verified.
Visual QA found a clear third-person workspace view and a distinct, clear wrist-mounted top-down view.

Internal/host wall time was 7.95/12.17 s; user/system time 19.45/4.48 s, CPU 196%, max RSS 2,486,184 KiB, 13 major
page faults, zero swap and network traffic. Training loss, offline validation, token accuracy, L1, gradients,
checkpoint loading, inference latency, VRAM, policy success and reward are not applicable because no model was
loaded and no action was executed. Canonical result:
`results/phase4a_openvla_viewpoint_camera_gate_v2.yaml`.

Phase 4A camera feasibility is complete. The next gate is a separately implemented and statically frozen Phase 4B
development rollout; this camera result alone makes no viewpoint-robustness claim.
