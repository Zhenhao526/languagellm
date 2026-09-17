# Population-level perceptual heterogeneity

The sender observes a semantic goal but not partner identity. In the homogeneous population every worker maps visible surface labels identically. In the heterogeneous population workers 1 and 3 swap the two visible labels. The child replaces worker 0, while the sender and the other workers remain frozen.

## Parent population endpoint

| population | form | all-goal natural | all-goal natural−permuted |
|---|---|---:|---:|
| `heterogeneous` | `dual2` | 0.459 | 0.418 |
| `heterogeneous` | `mono4` | 0.458 | 0.417 |
| `homogeneous` | `dual2` | 0.459 | 0.418 |
| `homogeneous` | `mono4` | 0.459 | 0.417 |

The parent contrasts compare the two population modes at matched seeds.

| contrast | factors | mean difference | 95% CI | n |
|---|---|---:|---|---:|
| `heterogeneous_minus_homogeneous_parent_all` | form=mono4, task=factorized | -0.000 | [-0.000, -0.000] | 1 |
| `heterogeneous_minus_homogeneous_parent_message_gap` | form=mono4, task=factorized | -0.000 | [-0.000, -0.000] | 1 |
| `heterogeneous_minus_homogeneous_parent_all` | form=dual2, task=factorized | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous_parent_message_gap` | form=dual2, task=factorized | -0.000 | [-0.000, -0.000] | 1 |

## Held-out live endpoint

| population | representation | form | mapping | support | channel | initial | final | gain | 95% CI | functional |
|---|---|---|---|---|---|---:|---:|---:|---|---:|
| `homogeneous` | `joint_history` | `mono4` | `identity` | `full` | `live` | 0.082 | 0.355 | 0.273 | [0.355, 0.355] | 0/1 |
| `homogeneous` | `joint_history` | `mono4` | `identity` | `full` | `silent` | 0.146 | 0.458 | 0.313 | [0.458, 0.458] | 0/1 |
| `homogeneous` | `joint_history` | `mono4` | `identity` | `leave_one_out` | `live` | 0.082 | 0.250 | 0.168 | [0.250, 0.250] | 0/1 |
| `homogeneous` | `joint_history` | `mono4` | `identity` | `leave_one_out` | `silent` | 0.146 | -0.063 | -0.209 | [-0.063, -0.063] | 0/1 |
| `homogeneous` | `joint_history` | `mono4` | `swap` | `full` | `live` | 0.290 | 0.458 | 0.168 | [0.458, 0.458] | 0/1 |
| `homogeneous` | `joint_history` | `mono4` | `swap` | `full` | `silent` | 0.041 | 0.251 | 0.210 | [0.251, 0.251] | 0/1 |
| `homogeneous` | `joint_history` | `mono4` | `swap` | `leave_one_out` | `live` | 0.290 | 0.250 | -0.040 | [0.250, 0.250] | 0/1 |
| `homogeneous` | `joint_history` | `mono4` | `swap` | `leave_one_out` | `silent` | 0.041 | -0.063 | -0.104 | [-0.063, -0.063] | 0/1 |
| `homogeneous` | `joint_history` | `dual2` | `identity` | `full` | `live` | 0.001 | 0.458 | 0.458 | [0.458, 0.458] | 0/1 |
| `homogeneous` | `joint_history` | `dual2` | `identity` | `full` | `silent` | 0.291 | 0.355 | 0.064 | [0.355, 0.355] | 0/1 |
| `homogeneous` | `joint_history` | `dual2` | `identity` | `leave_one_out` | `live` | 0.001 | 0.250 | 0.249 | [0.250, 0.250] | 0/1 |
| `homogeneous` | `joint_history` | `dual2` | `identity` | `leave_one_out` | `silent` | 0.291 | -0.167 | -0.457 | [-0.167, -0.167] | 0/1 |
| `homogeneous` | `joint_history` | `dual2` | `swap` | `full` | `live` | 0.313 | 0.563 | 0.251 | [0.563, 0.563] | 0/1 |
| `homogeneous` | `joint_history` | `dual2` | `swap` | `full` | `silent` | 0.083 | 0.251 | 0.168 | [0.251, 0.251] | 0/1 |
| `homogeneous` | `joint_history` | `dual2` | `swap` | `leave_one_out` | `live` | 0.313 | 0.250 | -0.063 | [0.250, 0.250] | 0/1 |
| `homogeneous` | `joint_history` | `dual2` | `swap` | `leave_one_out` | `silent` | 0.083 | -0.063 | -0.146 | [-0.063, -0.063] | 0/1 |
| `homogeneous` | `slot_local` | `dual2` | `identity` | `full` | `live` | 0.125 | 0.458 | 0.333 | [0.458, 0.458] | 0/1 |
| `homogeneous` | `slot_local` | `dual2` | `identity` | `full` | `silent` | 0.147 | 0.356 | 0.209 | [0.356, 0.356] | 0/1 |
| `homogeneous` | `slot_local` | `dual2` | `identity` | `leave_one_out` | `live` | 0.125 | 0.250 | 0.125 | [0.250, 0.250] | 0/1 |
| `homogeneous` | `slot_local` | `dual2` | `identity` | `leave_one_out` | `silent` | 0.147 | 0.042 | -0.105 | [0.042, 0.042] | 0/1 |
| `homogeneous` | `slot_local` | `dual2` | `swap` | `full` | `live` | 0.125 | 0.458 | 0.333 | [0.458, 0.458] | 0/1 |
| `homogeneous` | `slot_local` | `dual2` | `swap` | `full` | `silent` | 0.043 | 0.355 | 0.312 | [0.355, 0.355] | 0/1 |
| `homogeneous` | `slot_local` | `dual2` | `swap` | `leave_one_out` | `live` | 0.125 | 0.250 | 0.125 | [0.250, 0.250] | 0/1 |
| `homogeneous` | `slot_local` | `dual2` | `swap` | `leave_one_out` | `silent` | 0.043 | -0.167 | -0.209 | [-0.167, -0.167] | 0/1 |
| `heterogeneous` | `joint_history` | `mono4` | `identity` | `full` | `live` | 0.082 | 0.355 | 0.273 | [0.355, 0.355] | 0/1 |
| `heterogeneous` | `joint_history` | `mono4` | `identity` | `full` | `silent` | 0.146 | 0.458 | 0.313 | [0.458, 0.458] | 0/1 |
| `heterogeneous` | `joint_history` | `mono4` | `identity` | `leave_one_out` | `live` | 0.082 | 0.250 | 0.168 | [0.250, 0.250] | 0/1 |
| `heterogeneous` | `joint_history` | `mono4` | `identity` | `leave_one_out` | `silent` | 0.146 | -0.063 | -0.209 | [-0.063, -0.063] | 0/1 |
| `heterogeneous` | `joint_history` | `mono4` | `swap` | `full` | `live` | 0.290 | 0.458 | 0.168 | [0.458, 0.458] | 0/1 |
| `heterogeneous` | `joint_history` | `mono4` | `swap` | `full` | `silent` | 0.041 | 0.251 | 0.210 | [0.251, 0.251] | 0/1 |
| `heterogeneous` | `joint_history` | `mono4` | `swap` | `leave_one_out` | `live` | 0.290 | 0.250 | -0.040 | [0.250, 0.250] | 0/1 |
| `heterogeneous` | `joint_history` | `mono4` | `swap` | `leave_one_out` | `silent` | 0.041 | -0.063 | -0.104 | [-0.063, -0.063] | 0/1 |
| `heterogeneous` | `joint_history` | `dual2` | `identity` | `full` | `live` | 0.001 | 0.458 | 0.458 | [0.458, 0.458] | 0/1 |
| `heterogeneous` | `joint_history` | `dual2` | `identity` | `full` | `silent` | 0.291 | 0.355 | 0.064 | [0.355, 0.355] | 0/1 |
| `heterogeneous` | `joint_history` | `dual2` | `identity` | `leave_one_out` | `live` | 0.001 | 0.250 | 0.249 | [0.250, 0.250] | 0/1 |
| `heterogeneous` | `joint_history` | `dual2` | `identity` | `leave_one_out` | `silent` | 0.291 | -0.167 | -0.457 | [-0.167, -0.167] | 0/1 |
| `heterogeneous` | `joint_history` | `dual2` | `swap` | `full` | `live` | 0.313 | 0.563 | 0.251 | [0.563, 0.563] | 0/1 |
| `heterogeneous` | `joint_history` | `dual2` | `swap` | `full` | `silent` | 0.083 | 0.251 | 0.168 | [0.251, 0.251] | 0/1 |
| `heterogeneous` | `joint_history` | `dual2` | `swap` | `leave_one_out` | `live` | 0.313 | 0.250 | -0.063 | [0.250, 0.250] | 0/1 |
| `heterogeneous` | `joint_history` | `dual2` | `swap` | `leave_one_out` | `silent` | 0.083 | -0.063 | -0.146 | [-0.063, -0.063] | 0/1 |
| `heterogeneous` | `slot_local` | `dual2` | `identity` | `full` | `live` | 0.125 | 0.458 | 0.333 | [0.458, 0.458] | 0/1 |
| `heterogeneous` | `slot_local` | `dual2` | `identity` | `full` | `silent` | 0.147 | 0.356 | 0.209 | [0.356, 0.356] | 0/1 |
| `heterogeneous` | `slot_local` | `dual2` | `identity` | `leave_one_out` | `live` | 0.125 | 0.250 | 0.125 | [0.250, 0.250] | 0/1 |
| `heterogeneous` | `slot_local` | `dual2` | `identity` | `leave_one_out` | `silent` | 0.147 | 0.042 | -0.105 | [0.042, 0.042] | 0/1 |
| `heterogeneous` | `slot_local` | `dual2` | `swap` | `full` | `live` | 0.125 | 0.458 | 0.333 | [0.458, 0.458] | 0/1 |
| `heterogeneous` | `slot_local` | `dual2` | `swap` | `full` | `silent` | 0.043 | 0.355 | 0.312 | [0.355, 0.355] | 0/1 |
| `heterogeneous` | `slot_local` | `dual2` | `swap` | `leave_one_out` | `live` | 0.125 | 0.250 | 0.125 | [0.250, 0.250] | 0/1 |
| `heterogeneous` | `slot_local` | `dual2` | `swap` | `leave_one_out` | `silent` | 0.043 | -0.167 | -0.209 | [-0.167, -0.167] | 0/1 |

## Communication and heterogeneity checks

`all_natural−permuted` is the causal message check: positive values mean that shuffling messages across a worker's complete goal block changes behavior. `heterogeneous_minus_homogeneous` isolates the effect of population-level label conventions while holding semantic worlds, goals, partners, and random draws fixed.

| contrast | factors | mean difference | 95% CI | n |
|---|---|---:|---|---:|
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=mono4, support=full, channel=live | 0.103 | [0.103, 0.103] | 1 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=mono4, support=full, channel=live | 0.103 | [0.103, 0.103] | 1 |
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=mono4, support=full, channel=silent | -0.207 | [-0.207, -0.207] | 1 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=mono4, support=full, channel=silent | -0.207 | [-0.207, -0.207] | 1 |
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=mono4, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=mono4, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=mono4, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=mono4, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=identity, support=full, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=identity, support=full, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=identity, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=identity, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=swap, support=full, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=swap, support=full, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=swap, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=swap, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=dual2, support=full, channel=live | 0.105 | [0.105, 0.105] | 1 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=dual2, support=full, channel=live | 0.105 | [0.105, 0.105] | 1 |
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=dual2, support=full, channel=silent | -0.103 | [-0.103, -0.103] | 1 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=dual2, support=full, channel=silent | -0.103 | [-0.103, -0.103] | 1 |
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=dual2, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=dual2, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=dual2, support=leave_one_out, channel=silent | 0.103 | [0.103, 0.103] | 1 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=dual2, support=leave_one_out, channel=silent | 0.103 | [0.103, 0.103] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=identity, support=full, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=identity, support=full, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=identity, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=identity, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=swap, support=full, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=swap, support=full, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=swap, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=swap, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `swap_minus_identity` | population_mode=homogeneous, representation=slot_local, form=dual2, support=full, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=slot_local, form=dual2, support=full, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `swap_minus_identity` | population_mode=homogeneous, representation=slot_local, form=dual2, support=full, channel=silent | -0.001 | [-0.001, -0.001] | 1 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=slot_local, form=dual2, support=full, channel=silent | -0.001 | [-0.001, -0.001] | 1 |
| `swap_minus_identity` | population_mode=homogeneous, representation=slot_local, form=dual2, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=slot_local, form=dual2, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `swap_minus_identity` | population_mode=homogeneous, representation=slot_local, form=dual2, support=leave_one_out, channel=silent | -0.208 | [-0.208, -0.208] | 1 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=slot_local, form=dual2, support=leave_one_out, channel=silent | -0.208 | [-0.208, -0.208] | 1 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=identity, support=full, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=identity, support=full, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=identity, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=identity, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=swap, support=full, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=swap, support=full, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=swap, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=swap, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=identity, support=full, channel=live | 0.103 | [0.103, 0.103] | 1 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=identity, support=full, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=identity, support=full, channel=silent | -0.103 | [-0.103, -0.103] | 1 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=identity, support=full, channel=silent | 0.001 | [0.001, 0.001] | 1 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=identity, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=identity, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=identity, support=leave_one_out, channel=silent | -0.103 | [-0.103, -0.103] | 1 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=identity, support=leave_one_out, channel=silent | 0.208 | [0.208, 0.208] | 1 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=swap, support=full, channel=live | 0.105 | [0.105, 0.105] | 1 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=swap, support=full, channel=live | -0.105 | [-0.105, -0.105] | 1 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=swap, support=full, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=swap, support=full, channel=silent | 0.104 | [0.104, 0.104] | 1 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=swap, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=swap, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=swap, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=swap, support=leave_one_out, channel=silent | -0.103 | [-0.103, -0.103] | 1 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=identity, support=full, channel=live | 0.103 | [0.103, 0.103] | 1 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=identity, support=full, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=identity, support=full, channel=silent | -0.103 | [-0.103, -0.103] | 1 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=identity, support=full, channel=silent | 0.001 | [0.001, 0.001] | 1 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=identity, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=identity, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=identity, support=leave_one_out, channel=silent | -0.103 | [-0.103, -0.103] | 1 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=identity, support=leave_one_out, channel=silent | 0.208 | [0.208, 0.208] | 1 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=swap, support=full, channel=live | 0.105 | [0.105, 0.105] | 1 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=swap, support=full, channel=live | -0.105 | [-0.105, -0.105] | 1 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=swap, support=full, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=swap, support=full, channel=silent | 0.104 | [0.104, 0.104] | 1 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=swap, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=swap, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 1 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=swap, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 1 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=swap, support=leave_one_out, channel=silent | -0.103 | [-0.103, -0.103] | 1 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=mono4, mapping=identity, channel=live | -0.105 | [-0.105, -0.105] | 1 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=mono4, mapping=identity, channel=silent | -0.521 | [-0.521, -0.521] | 1 |
| `live_minus_silent_all` | population_mode=homogeneous, representation=joint_history, form=mono4, mapping=identity, support=full | 0.210 | [0.210, 0.210] | 1 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=mono4, mapping=swap, channel=live | -0.208 | [-0.208, -0.208] | 1 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=mono4, mapping=swap, channel=silent | -0.315 | [-0.315, -0.315] | 1 |
| `live_minus_silent_all` | population_mode=homogeneous, representation=joint_history, form=mono4, mapping=swap, support=full | 0.210 | [0.210, 0.210] | 1 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=dual2, mapping=identity, channel=live | -0.208 | [-0.208, -0.208] | 1 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=dual2, mapping=identity, channel=silent | -0.521 | [-0.521, -0.521] | 1 |
| `live_minus_silent_all` | population_mode=homogeneous, representation=joint_history, form=dual2, mapping=identity, support=full | 0.209 | [0.209, 0.209] | 1 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=dual2, mapping=swap, channel=live | -0.313 | [-0.313, -0.313] | 1 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=dual2, mapping=swap, channel=silent | -0.315 | [-0.315, -0.315] | 1 |
| `live_minus_silent_all` | population_mode=homogeneous, representation=joint_history, form=dual2, mapping=swap, support=full | 0.208 | [0.208, 0.208] | 1 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=slot_local, form=dual2, mapping=identity, channel=live | -0.208 | [-0.208, -0.208] | 1 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=slot_local, form=dual2, mapping=identity, channel=silent | -0.315 | [-0.315, -0.315] | 1 |
| `live_minus_silent_all` | population_mode=homogeneous, representation=slot_local, form=dual2, mapping=identity, support=full | 0.209 | [0.209, 0.209] | 1 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=slot_local, form=dual2, mapping=swap, channel=live | -0.208 | [-0.208, -0.208] | 1 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=slot_local, form=dual2, mapping=swap, channel=silent | -0.522 | [-0.522, -0.522] | 1 |
| `live_minus_silent_all` | population_mode=homogeneous, representation=slot_local, form=dual2, mapping=swap, support=full | 0.208 | [0.208, 0.208] | 1 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=mono4, mapping=identity, channel=live | -0.105 | [-0.105, -0.105] | 1 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=mono4, mapping=identity, channel=silent | -0.521 | [-0.521, -0.521] | 1 |
| `live_minus_silent_all` | population_mode=heterogeneous, representation=joint_history, form=mono4, mapping=identity, support=full | 0.210 | [0.210, 0.210] | 1 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=mono4, mapping=swap, channel=live | -0.208 | [-0.208, -0.208] | 1 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=mono4, mapping=swap, channel=silent | -0.315 | [-0.315, -0.315] | 1 |
| `live_minus_silent_all` | population_mode=heterogeneous, representation=joint_history, form=mono4, mapping=swap, support=full | 0.209 | [0.209, 0.209] | 1 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=dual2, mapping=identity, channel=live | -0.208 | [-0.208, -0.208] | 1 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=dual2, mapping=identity, channel=silent | -0.521 | [-0.521, -0.521] | 1 |
| `live_minus_silent_all` | population_mode=heterogeneous, representation=joint_history, form=dual2, mapping=identity, support=full | 0.209 | [0.209, 0.209] | 1 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=dual2, mapping=swap, channel=live | -0.313 | [-0.313, -0.313] | 1 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=dual2, mapping=swap, channel=silent | -0.315 | [-0.315, -0.315] | 1 |
| `live_minus_silent_all` | population_mode=heterogeneous, representation=joint_history, form=dual2, mapping=swap, support=full | 0.209 | [0.209, 0.209] | 1 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=slot_local, form=dual2, mapping=identity, channel=live | -0.208 | [-0.208, -0.208] | 1 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=slot_local, form=dual2, mapping=identity, channel=silent | -0.315 | [-0.315, -0.315] | 1 |
| `live_minus_silent_all` | population_mode=heterogeneous, representation=slot_local, form=dual2, mapping=identity, support=full | 0.209 | [0.209, 0.209] | 1 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=slot_local, form=dual2, mapping=swap, channel=live | -0.208 | [-0.208, -0.208] | 1 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=slot_local, form=dual2, mapping=swap, channel=silent | -0.522 | [-0.522, -0.522] | 1 |
| `live_minus_silent_all` | population_mode=heterogeneous, representation=slot_local, form=dual2, mapping=swap, support=full | 0.208 | [0.208, 0.208] | 1 |

This finite tabular study measures protocol formation and repair under controlled perceptual conventions. It does not claim that the agents possess human language.
