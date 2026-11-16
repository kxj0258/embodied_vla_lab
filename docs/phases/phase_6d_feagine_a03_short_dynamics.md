# Phase 6D — Feagine A03 Short Headless Dynamics

This gate advances the packaged A03 Type-2 MJCF for exactly 200 steps at 0.005 s (one simulated second). It applies
small section bends, closes the gripper and commands a small grasper rotation. It records finite qpos/qvel/ctrl,
joint-limit compliance, terminal proxy motion, contacts, throughput and exact simulated time.

This is an actuator/dynamics smoke, not move-to-target: there is no IK target, task, reward, success criterion,
rendering, policy inference or hardware command. PASS only permits implementing a separately frozen target-reaching
gate.

## v1 result and v2 diagnostic

v1 completed all 200 steps with finite state and exact one-second simulated time. Tip motion was 0.0670 m,
max absolute qvel 0.5921, max control 0.2 and contacts zero. It nevertheless FAILed 6/7 because at least one limited
joint left its nominal range. Since v1 did not record joint identity or penetration magnitude, no safety conclusion
or tolerance relaxation is permitted.

Fresh-ID v2 repeats the identical bounded command solely to record per-joint extrema and penetration. Before seeing
the result, material thresholds are frozen at 1e-3 rad for hinge and 1e-4 m for slide joints. v2 cannot revise v1
and cannot authorize target reaching automatically.

v2 executed all 200 steps and PASSed. Exactly one joint crossed its nominal limit: slide joint
`feagine_grasper_joint_1` reached 0.0050117734 m against a 0.005 m upper limit, a penetration of
1.17734e-5 m (11.77 micrometres). This is 11.77% of the preregistered 1e-4 m material threshold; no hinge joint
penetrated its limit. The decision is `SOFT_LIMIT_PENETRATION_WITHIN_DIAGNOSTIC_THRESHOLD`. This resolves the v1
failure mechanism as a small soft-limit penetration, but preserves the historical v1 FAIL and still does not
establish target reaching, reward, task success, policy transfer or hardware safety.
