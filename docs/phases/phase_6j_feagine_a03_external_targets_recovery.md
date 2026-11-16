# Phase 6J — reachability-corrected external target recovery

Phase 6J is a development recovery gate derived from, but separate from, immutable Phase 6F. It keeps all four
35 mm lateral directions and the same environment criteria, adds the uniformly measured 1.50 mm axial retraction,
uses the frozen first-reachable seed per direction, and canonicalizes signed theta before applying section angles.

Each independently reset target runs 400 steps at 5 ms. PASS remains strict 4/4 with terminal error ≤15 mm,
progress ≥70%, finite/exact dynamics and no material limit violation. Phase 6F remains 0/4 regardless of outcome.
This is not VLA transfer, a broad workspace benchmark, final evaluation or hardware evidence.

## v1 result

The aggregate result is 2/4 FAIL. All four targets passed IK, canonicalization, finite/exact dynamics, terminal
position error and progress. Final errors were 9.303–9.307 mm and progress 73.434–73.445%, demonstrating that the
reachability correction and signed-theta mapping recovered target motion in every direction.

±Y fully PASSed. ±X failed only the unchanged material-limit gate: maximum slide penetration was
0.26307/0.26307 mm versus the frozen 0.1 mm threshold. Hinges had zero penetration and contacts were zero. Because
v1 did not record the joint identity or time of maximum penetration, no tolerance relaxation or controller change
is justified. A same-target diagnostic must identify the joint and trajectory while preserving this 2/4 result.
