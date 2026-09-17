# Held-out goal-combination recovery

- runs: 128
- zero-shot rule: leave-one-out live heldout return ≥ 0.60

| support | channel | all | seen | heldout |
|---|---|---:|---:|---:|
| `full` | `live` | 0.524 | 0.516 | 0.546 |
| `full` | `silent` | 0.250 | 0.264 | 0.205 |
| `leave_one_out` | `live` | 0.491 | 0.571 | 0.254 |
| `leave_one_out` | `silent` | 0.251 | 0.355 | -0.066 |

| comparison | mean difference | 95% CI |
|---|---:|---:|
| `leave_minus_full|live|heldout` | -0.293 | [-0.344, -0.241] |
| `live_minus_silent|leave_one_out|heldout` | 0.319 | [0.268, 0.371] |
| `live_minus_silent|full|heldout` | 0.341 | [0.272, 0.410] |

| parent stratum | n | heldout mean | passes |
|---|---:|---:|---:|
| `parent_composable` | 9 | 0.263 | 0/9 |
| `parent_noncomposable` | 23 | 0.250 | 0/23 |

The leave-one-out arm removes one target combination from child training while keeping the frozen parent sender able to emit its corresponding two-slot message. Passing the held-out criterion is evidence of behavioral zero-shot recovery, not evidence of open-ended language.
