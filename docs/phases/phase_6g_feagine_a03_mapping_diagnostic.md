# Phase 6G — A03 target/section/controller mapping diagnostic

Phase 6G repeats the immutable Phase 6F ±X/±Y 35 mm targets and 400-step budget without changing mapping, targets
or thresholds. Its purpose is explanatory, not recovery or success estimation.

For every target it records the raw core IK section angles, wrapper-applied angles after preset clamping, all 48 PCC
actuator targets, core-predicted tool displacement, MJCF actual displacement, pairwise direction cosines and
displacement gains. Execution PASS means only that all four diagnostics completed with finite state, exact time and
complete actuator records; it does not require target success and cannot revise Phase 6F's 0/4 FAIL.

No normalization, sign conversion, angle canonicalization or controller correction is applied in this gate. Any
proposed repair must be derived from this evidence, statically frozen under a new ID and tested without replacing
the original four targets. No VLA, image, reward, final evaluation, rendering or hardware command is included.

## v1 result and diagnosis

The diagnostic completed for 4/4 targets. ±X failed IK at the straight all-zero initial guess and emitted all-zero
section angles and actuator targets. For +Y, core returned second-section theta -0.62636 rad; preset application
clamped it to zero. For -Y, first/third theta -0.42471/-0.20102 rad were clamped to zero. Thus both successful IK
solutions were structurally changed before actuation.

After clamping, core-predicted and MJCF-actual displacements agreed closely (cosine 0.99941/0.99927), while actual
gains were 7.87x/6.80x and the -Y direction was reversed relative to its target. This is strong evidence that the
actuator/dynamics layer follows the applied section configuration; the primary defect is representation handling.
The mathematically equivalent candidate canonicalization is negative theta → positive absolute theta with phi
shifted by π and wrapped to its allowed interval. ±X separately requires frozen nonzero symmetric IK initial seeds
to escape the straight-configuration singularity. Neither repair has been executed here.
