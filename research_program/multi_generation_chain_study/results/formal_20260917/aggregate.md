# Alternating multi-generation chain

- chains: 128
- training support: leave-one-out

| generation | role | representation | live held-out | silent held-out | live−silent | semantic live | sender fidelity to parent | composable passes |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `worker` | `joint_history` | 0.250 | 0.169 | +0.081 [+0.043,+0.119] | 0.491 | 1.000 | 0/32 |
| 1 | `worker` | `slot_local` | 0.322 | 0.168 | +0.153 [+0.064,+0.243] | 0.491 | 1.000 | 9/32 |
| 2 | `sender` | `joint_history` | 0.172 | 0.159 | +0.013 [-0.060,+0.086] | 0.471 | 0.871 | 0/32 |
| 2 | `sender` | `slot_local` | 0.185 | 0.157 | +0.028 [-0.088,+0.143] | 0.457 | 0.871 | 6/32 |
| 3 | `worker` | `joint_history` | 0.179 | 0.163 | +0.016 [-0.037,+0.070] | 0.473 | 0.871 | 0/32 |
| 3 | `worker` | `slot_local` | 0.198 | 0.144 | +0.054 [-0.047,+0.155] | 0.462 | 0.871 | 6/32 |

| transition | counts (False→False, False→True, True→False, True→True) |
|---|---|
| `joint_history|g1->g2` | 32, 0, 0, 0 |
| `joint_history|g2->g3` | 32, 0, 0, 0 |
| `slot_local|g1->g2` | 19, 4, 7, 2 |
| `slot_local|g2->g3` | 25, 1, 1, 5 |

The chain is sequential: the final parameters of one replacement event initialize the next event. A pass is held-out live natural return ≥ 0.60 with absolute recombined−natural gap ≤ 0.02.
