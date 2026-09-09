# Benchmark Failure Analysis

This report preserves all failures and does not rerun or select episodes.

## openvla_base — `benchmark_openvla_libero_heldout45_v1`

Overall: **44/45** (97.8%)

| Task | Success | Timeout | Mean steps | Timeout states |
|---|---:|---:|---:|---|
| 7: turn_on_the_stove | 44/45 | 1 | 91.3 | [30] |

## openvla_base — `benchmark_openvla_libero_breadth50_v1`

Overall: **33/50** (66.0%)

| Task | Success | Timeout | Mean steps | Timeout states |
|---|---:|---:|---:|---|
| 0: open_the_middle_drawer_of_the_cabinet | 3/5 | 2 | 313.8 | [3, 4] |
| 1: put_the_bowl_on_the_stove | 5/5 | 0 | 82.8 | [] |
| 2: put_the_wine_bottle_on_top_of_the_cabinet | 4/5 | 1 | 276.4 | [2] |
| 3: open_the_top_drawer_and_put_the_bowl_inside | 1/5 | 4 | 523.2 | [1, 2, 3, 4] |
| 4: put_the_bowl_on_top_of_the_cabinet | 5/5 | 0 | 120.6 | [] |
| 5: push_the_plate_to_the_front_of_the_stove | 3/5 | 2 | 319.4 | [1, 2] |
| 6: put_the_cream_cheese_in_the_bowl | 3/5 | 2 | 302.4 | [1, 2] |
| 7: turn_on_the_stove | 5/5 | 0 | 78.8 | [] |
| 8: put_the_bowl_on_the_plate | 2/5 | 3 | 391.0 | [1, 2, 4] |
| 9: put_the_wine_bottle_on_the_rack | 2/5 | 3 | 497.0 | [0, 2, 3] |

## openvla_base — `benchmark_openvla_libero_breadth100_v1`

Overall: **66/100** (66.0%)

| Task | Success | Timeout | Mean steps | Timeout states |
|---|---:|---:|---:|---|
| 0: open_the_middle_drawer_of_the_cabinet | 6/10 | 4 | 322.7 | [1, 3, 4, 6] |
| 1: put_the_bowl_on_the_stove | 8/10 | 2 | 229.2 | [1, 8] |
| 2: put_the_wine_bottle_on_top_of_the_cabinet | 6/10 | 4 | 303.1 | [0, 1, 2, 8] |
| 3: open_the_top_drawer_and_put_the_bowl_inside | 2/10 | 8 | 521.4 | [0, 2, 3, 4, 6, 7, 8, 9] |
| 4: put_the_bowl_on_top_of_the_cabinet | 10/10 | 0 | 102.1 | [] |
| 5: push_the_plate_to_the_front_of_the_stove | 4/10 | 6 | 414.3 | [0, 1, 2, 5, 8, 9] |
| 6: put_the_cream_cheese_in_the_bowl | 6/10 | 4 | 306.4 | [1, 2, 4, 5] |
| 7: turn_on_the_stove | 10/10 | 0 | 78.4 | [] |
| 8: put_the_bowl_on_the_plate | 8/10 | 2 | 184.4 | [3, 4] |
| 9: put_the_wine_bottle_on_the_rack | 6/10 | 4 | 349.6 | [2, 4, 6, 7] |

## Interpretation

- A timeout is an episode that reached the configured horizon without success.
- Single-task held-out and multi-task breadth results are reported separately.
- No failed episode is removed, replaced, or relabeled as success.
