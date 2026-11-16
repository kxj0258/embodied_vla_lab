# Phase 6I — A03 fixed-direction reachability scan

This CPU-only gate preserves the 35 mm lateral magnitude and 0.1 mm IK tolerance, then scans explicit axial
retraction from 0 to 5 mm in 0.25 mm increments. All ±X/±Y directions reuse Phase 6H's eight frozen seeds and
signed-theta canonicalization.

PASS requires all four directions to become reachable somewhere on the frozen grid and their minimum retractions
to agree within 0.5 mm. The scan quantifies the geometric correction needed for a future target contract; it does
not revise Phase 6F, relax tolerance, load MuJoCo or authorize a recovery rollout.

## v1 scan and v2 packaging recovery

v1 completed 672 core IK attempts. All four directions first became reachable at 1.50 mm retraction with zero
directional spread. At 1.25 mm, best position error remained about 0.347 mm; at 1.50 mm it was 0.0982 mm, just
inside the unchanged 0.1 mm tolerance. v1 nevertheless reported 5/6 FAIL solely because its grid assertion compared
decimal YAML floats to multiplication-generated Python floats with exact list equality.

v1 remains immutable. v2 performs no FK/IK call and only validates the same 21 values with 1e-12 m numerical
tolerance, plus source hashes, endpoints, count and reachability evidence. It is a packaging recovery, not a new
scan or a changed scientific threshold.
