# Phase 4 viewpoint robustness closeout

Date: 2026-08-28  
Status: `COMPLETE_NO_GO`

## Research question

Can a frozen third-person OpenVLA policy transfer to a wrist-mounted camera by direct image substitution, and can
a small task-specific wrist-view QLoRA recover the resulting closed-loop failure on one 12 GB GPU?

## Evidence chain

| Gate | Controlled evidence | Result | Decision |
|---|---|---|---|
| 4A camera/runtime | Same reset; native 128 and direct 640 agent/wrist RGB | Valid, distinct views; native/direct cross-view MAE 76.46/77.50 | Camera pipeline PASS |
| 4B direct substitution | Frozen base; 5 dev tasks × 3 states × 2 paired views | Agentview 10/15; wrist 0/15; paired exact p=0.001953 | Direct wrist substitution NO-GO |
| 4C training smoke | Task-7 native wrist demos; rank-8 QLoRA; 20 updates | Engineering PASS; val10 loss improved, token accuracy mixed | Require full validation |
| 4D paired offline | Same 889 wrist frames and 6,223 action tokens | Loss -45.77%; token accuracy +7.73 pp; L1 -32.43%; 10/10 episodes improve | Offline adaptation PASS |
| 4E closed loop | Same task-7 init/seeds as immutable base 0/3 reference | Adapter 0/3; three 600-step timeouts; reward 0 | Closed-loop recovery NO-GO |

## Final conclusion

The camera interface is valid and QLoRA measurably fits wrist-view demonstrations, but neither direct substitution
nor this small task-specific adapter produces wrist closed-loop success. Teacher-forced metrics are insufficient for
viewpoint-transfer checkpoint selection. The step-20 adapter remains a research asset, not a deployable policy.

Final tasks `[1,2,4,5,9]` were never touched. No result authorizes post-outcome tuning, checkpoint switching,
selective reruns or retroactive final execution. A future viewpoint study requires a genuinely new preregistered
mechanism such as multi-view representation alignment or paired observation training, not continuation of this
frozen hypothesis.

## Portfolio-safe claim

Built a preregistered ego-view ablation for OpenVLA with paired camera controls and immutable failures: direct wrist
substitution fell from 10/15 to 0/15; a rank-8 wrist QLoRA improved full-validation loss/token accuracy/L1 but still
scored 0/3 closed loop, demonstrating and measuring an offline-to-control generalization gap.

This claim is development-set evidence, not official LIBERO benchmark performance or general ego-view robustness.
