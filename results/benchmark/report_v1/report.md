# Unified VLA Benchmark Report

Generated: `2026-09-08T16:06:37.459168+00:00`

> These are development results from the unified evaluator; they do not replace the preregistered held-out final results.

| Policy | Success | Steps | Mean inference / episode | Mean wall / episode | Failures |
|---|---:|---:|---:|---:|---|
| act | 5/5 (100.0%) | 394 | 0.182s | 1.831s | `{"success": 5}` |
| openvla_base | 5/5 (100.0%) | 400 | 25.248s | 26.884s | `{"success": 5}` |

## Interpretation

- ACT and OpenVLA are evaluated on the same five development init states.
- OpenVLA inference is isolated in a persistent worker; ACT runs in the LeRobot environment process.
- Success rates are not claims about unseen tasks, perturbations, or the 45-state held-out final protocol.
