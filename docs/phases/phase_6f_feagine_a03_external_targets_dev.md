# Phase 6F — A03 external Cartesian targets development gate

Phase 6F removes Phase 6E's core-generated-target shortcut. Four local Cartesian deltas are written explicitly in
configuration before execution: ±35 mm along X and ±35 mm along Y. Each starts from an independently reset A03
state, preserves the initial tool axis and runs for exactly 400 steps at 5 ms.

Each target must have successful core IK, finite state, exact simulated time, terminal position error ≤15 mm,
progress ≥70% and no material limit violation under the frozen Phase 6D thresholds. Aggregate PASS requires 4/4;
all failures remain recorded. The slightly lower progress threshold than Phase 6E is frozen before observing these
new directions and does not change the same 15 mm absolute accuracy bound.

This development gate has no VLA, camera, reward, obstacle/contact objective, broad workspace sample, final split,
rendering or hardware command. PASS would establish only small symmetric local target reaching.

## v1 result

The aggregate result is FAIL, 0/4. Both ±X targets failed IK after one iteration (IK position error 36.27 mm) and
the simulated tip remained stationary. Both ±Y targets passed the core numerical IK tolerance in 16 iterations,
but the MJCF response moved far beyond and in the wrong effective direction: final errors were 241.63 and
271.86 mm from 35 mm initial errors. All four states remained finite with exact two-second timing, no material
limit violation and zero contacts.

This separates a target/section/controller mapping failure from a dynamics stability failure. The target set,
thresholds, result, output and raw log are immutable. v1 does not justify target replacement, threshold relaxation
or a success claim. A new diagnostic must first record solved section angles, post-clamp applied angles, actuator
targets and predicted-versus-MJCF tip motion without changing these four targets.
