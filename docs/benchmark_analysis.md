# VLA benchmark analysis

The current unified evaluation evidence is intentionally separated by scope:

| Protocol | Policy | Result | Meaning |
|---|---|---:|---|
| One LIBERO-Goal task, 45 held-out states | ACT | 45/45 | Single-task held-out reliability |
| One LIBERO-Goal task, 45 held-out states | OpenVLA frozen base | 44/45 | Single-task held-out reliability with one timeout |
| 10 tasks × 5 states | OpenVLA frozen base | 33/50 | Breadth development |
| 10 tasks × 10 states | OpenVLA frozen base | 66/100 | Extended breadth evaluation |

The multi-task result is task-dependent: task 4 and task 7 reached 10/10,
while task 3 reached 2/10 and task 5 reached 4/10. All 34 failures in the
100-episode evaluation were horizon timeouts, not worker crashes or numerical
errors.

The failure analysis is descriptive. It does not support changing the horizon,
removing hard tasks, or selecting only successful states.

Generate the machine-readable and Markdown analysis with:

```bash
python scripts/analyze_benchmark_failures.py \
  --result results/benchmark/benchmark_openvla_libero_heldout45_v1/result.json \
  results/benchmark/benchmark_openvla_libero_breadth50_v1/result.json \
  results/benchmark/benchmark_openvla_libero_breadth100_v1/result.json \
  --output results/benchmark/failure_analysis_v1.json
```
