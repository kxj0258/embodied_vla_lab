# Phase 6H — signed-theta canonicalization and multi-start IK

This CPU-only gate tests the two repairs suggested by Phase 6G before any MJCF recovery rollout. Negative section
theta is converted to an equivalent nonnegative representation by `theta'=abs(theta)` and
`phi'=wrap(phi+pi)`. FK position/direction equivalence is checked on 256 seeded random configurations plus all four
Phase 6G raw solutions.

The original ±X/±Y 35 mm targets are then solved in core space with eight frozen symmetric nonzero initial seeds.
Solutions are canonicalized and re-evaluated by FK. PASS requires equivalence within 1e-6 and a valid canonical
solution for all four original targets. It does not load MuJoCo, issue actuator commands or revise Phase 6F's 0/4.

## v1 result

The gate returned 4/5 FAIL. Canonicalization is numerically validated across 260 configurations: maximum FK
position and direction differences are 9.22e-8 m and 4.28e-7. However, none of eight frozen seeds produced a
solution within the original 0.1 mm IK tolerance for any target. Best canonical FK residuals cluster tightly at
1.592–1.615 mm while direction errors are only about 2.6–3.5e-5.

This consistent residual across symmetric targets and diverse seeds indicates a geometric target-contract problem:
Phase 6F adds 35 mm lateral displacement while retaining the straight arm's near-maximum axial extension and tip
direction. With fixed section lengths, that pose is likely outside the reachable set. The tolerance is not relaxed
and no approximate solution is promoted. A separate reachability analysis must define targets with explicit axial
retraction before any MJCF recovery gate.
