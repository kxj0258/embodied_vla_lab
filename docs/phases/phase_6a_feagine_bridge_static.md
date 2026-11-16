# Phase 6A — Feagine Bridge Static Contract

Phase 6A separates a testable VLA-to-continuum software boundary from unavailable Feagine A03 robot assets. The
repository currently contains no A03 MJCF, calibrated geometry, actuator mapping or PCC API, so this phase must not
invent hardware fidelity.

The implemented reference accepts a LIBERO-style 7D delta action, rejects unsupported rotation requests, rate-limits
translation, applies an explicit proper frame transform, checks workspace and gripper bounds, and maps the desired
tip position to a provisional single-section extensible PCC state. Closed-form forward/inverse position kinematics
allow deterministic numerical round-trip testing. Invalid numbers and targets outside workspace, length, curvature
or bend-angle limits fail closed.

All geometry and limits are explicitly provisional. Static PASS can establish schema, numerical consistency and
software fail-safe behavior only. It cannot establish A03 kinematic accuracy, actuator feasibility, collision safety,
MuJoCo stability, VLA transfer, task success or real-robot safety. The next gate requires the actual A03 simulation
asset and actuator/PCC definitions to replace the provisional parameter layer before any simulator execution.

Static v1 completed calculation but raised `yaml.representer.RepresenterError` while serializing one NumPy boolean;
it is preserved as `EXPECTED_EXCEPTION`. Static v2 changes only that packaging type and uses fresh result/config IDs.

Static v2 found a second NumPy boolean at the same serialization boundary and is also preserved as
`EXPECTED_EXCEPTION`. Static v3 normalizes the complete check map to builtin booleans before serialization.

Static v3 then produced a real structured FAIL at 13/14 because its workspace-rejection fixture ignored the prior
translation rate limiter: the requested move was clipped exactly onto the permitted boundary and correctly accepted.
Static v4 changes only that fixture to an unambiguously out-of-workspace target and preserves v3 unchanged.
