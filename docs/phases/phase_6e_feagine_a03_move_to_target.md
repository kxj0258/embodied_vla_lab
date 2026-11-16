# Phase 6E — Feagine A03 bounded move-to-target smoke

This development gate asks the packaged A03 Type-2 model to reach one small target generated from the same frozen
three-section PCC core. It is the first closed-loop target-reaching smoke, not a VLA or arbitrary-workspace test.

The target is generated from section angles `[0.06,0,0.04,0,0.02,0]`. Core IK must succeed before applying its
solution. MuJoCo then advances exactly 400 steps at 5 ms (two simulated seconds), with open stationary grasper.
PASS requires a target displacement in `[0.02,0.08]` m, terminal tool-position error at most 15 mm, at least 80%
error reduction, finite state, exact time and no material joint-limit violation under the already frozen Phase 6D
thresholds (1e-3 rad hinge, 1e-4 m slide).

The test has no image, instruction, learned policy, reward function, rendering, obstacle, contact task or hardware
command. Even PASS only authorizes designing a separate externally specified target protocol; it does not establish
VLA transfer, general workspace reachability or robot safety.

## v1 result

Runtime PASSed 9/9. The target was 53.896 mm from the initial tool tip. After 400 steps the terminal error was
9.504 mm, an 82.366% reduction, satisfying all preregistered gates. Core IK converged in four iterations with
1.064 micrometre position error and 7.55e-7 direction error. State and time were finite/exact; maximum hinge
penetration was zero, maximum slide penetration was 1.29e-12 m and contacts remained zero. `success: true` denotes
only this bounded smoke gate; reward was absent and there were zero model predictions or hardware commands.
