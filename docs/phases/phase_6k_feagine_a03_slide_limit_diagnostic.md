# Phase 6K — A03 slide-limit per-joint diagnostic

Phase 6K repeats the immutable Phase 6J four targets, seeds, representation, controller and 400-step budget. It
records every limited slide joint's name/range/extrema, first nominal violation step, peak step and penetration,
then compares ±X against ±Y.

Diagnostic PASS only means four complete finite, exact-time records. It cannot revise Phase 6J's 2/4 result or
authorize tolerance relaxation, target/controller changes, VLA, final evaluation or hardware.

## v1 result

Diagnostic execution PASSed. +X violated `feagine_grasper_joint_0`; -X violated the mirrored
`feagine_grasper_joint_1`. Both joints have range `[-0.013,0.005]` m and reached approximately -0.01326307 m at
step 1, producing the 0.26307 mm material violation. The opposite finger penetrated only about 2.45 μm, peaking at
step 33. ±Y penetrations were numerical-scale (≤3.90e-11 m).

The direction-dependent mirrored finger and immediate peak show a transient at the fully-open hard lower bound,
not a PCC proxy limit or sustained divergence. A future development gate may scan a preregistered small gripper
margin away from the hard stop while preserving all arm targets and the original 0.1 mm material threshold.
