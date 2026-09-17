# Receiver representation and held-out composition

- runs: 256
- zero-shot rule: leave-one-out live heldout return ≥ 0.60

| representation | support | channel | all | seen | heldout |
|---|---|---|---:|---:|---:|
| `joint_history` | `full` | `live` | 0.524 | 0.516 | 0.546 |
| `joint_history` | `full` | `silent` | 0.250 | 0.262 | 0.214 |
| `joint_history` | `leave_one_out` | `live` | 0.491 | 0.571 | 0.251 |
| `joint_history` | `leave_one_out` | `silent` | 0.251 | 0.352 | -0.056 |
| `slot_local` | `full` | `live` | 0.498 | 0.496 | 0.504 |
| `slot_local` | `full` | `silent` | 0.250 | 0.262 | 0.214 |
| `slot_local` | `leave_one_out` | `live` | 0.497 | 0.553 | 0.328 |
| `slot_local` | `leave_one_out` | `silent` | 0.251 | 0.352 | -0.056 |

| contrast | mean | 95% CI |
|---|---:|---:|
| `slot_local_minus_joint|leave_one_out|live|heldout` | 0.077 | [-0.012, 0.165] |
| `slot_local_minus_joint|full|live|heldout` | -0.043 | [-0.080, -0.006] |
| `leave_minus_full|joint_history|live|heldout` | -0.295 | [-0.345, -0.245] |
| `leave_minus_full|slot_local|live|heldout` | -0.176 | [-0.251, -0.100] |

| parent stratum | representation | full live | leave-one-out live | passes |
|---|---|---:|---:|---:|
| `parent_composable` | `joint_history` | 0.667 | 0.255 | 0/9 |
| `parent_composable` | `slot_local` | 0.667 | 0.667 | 9/9 |
| `parent_noncomposable` | `joint_history` | 0.499 | 0.250 | 0/23 |
| `parent_noncomposable` | `slot_local` | 0.440 | 0.196 | 0/23 |

`slot_local` changes only the receiver's state lookup: at each staged subtask it ignores the other slot. The comparison is therefore a test of representation-induced compositional generalization, not a change to the parent sender or the task payoff.
