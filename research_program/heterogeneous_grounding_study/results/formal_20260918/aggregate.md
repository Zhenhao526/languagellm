# Population-level perceptual heterogeneity

The sender observes a semantic goal but not partner identity. In the homogeneous population every worker maps visible surface labels identically. In the heterogeneous population workers 1 and 3 swap the two visible labels. The child replaces worker 0, while the sender and the other workers remain frozen.

## Parent population endpoint

| population | form | all-goal natural | all-goal natural−permuted |
|---|---|---:|---:|
| `heterogeneous` | `dual2` | 0.459 | 0.418 |
| `heterogeneous` | `mono4` | 0.458 | 0.417 |
| `homogeneous` | `dual2` | 0.459 | 0.418 |
| `homogeneous` | `mono4` | 0.459 | 0.417 |
| `heterogeneous` | `dual2` | 0.667 | 0.625 |
| `heterogeneous` | `mono4` | 0.562 | 0.417 |
| `homogeneous` | `dual2` | 0.667 | 0.625 |
| `homogeneous` | `mono4` | 0.458 | 0.417 |
| `heterogeneous` | `dual2` | 0.458 | 0.417 |
| `heterogeneous` | `mono4` | 0.458 | 0.417 |
| `homogeneous` | `dual2` | 0.458 | 0.417 |
| `homogeneous` | `mono4` | 0.457 | 0.415 |
| `heterogeneous` | `dual2` | 0.458 | 0.417 |
| `heterogeneous` | `mono4` | 0.458 | 0.416 |
| `homogeneous` | `dual2` | 0.458 | 0.417 |
| `homogeneous` | `mono4` | 0.563 | 0.626 |
| `heterogeneous` | `dual2` | 0.458 | 0.417 |
| `heterogeneous` | `mono4` | 0.458 | 0.417 |
| `homogeneous` | `dual2` | 0.458 | 0.416 |
| `homogeneous` | `mono4` | 0.563 | 0.626 |
| `heterogeneous` | `dual2` | 0.458 | 0.417 |
| `heterogeneous` | `mono4` | 0.458 | 0.417 |
| `homogeneous` | `dual2` | 0.458 | 0.417 |
| `homogeneous` | `mono4` | 0.458 | 0.417 |
| `heterogeneous` | `dual2` | 0.458 | 0.417 |
| `heterogeneous` | `mono4` | 0.458 | 0.417 |
| `homogeneous` | `dual2` | 0.458 | 0.417 |
| `homogeneous` | `mono4` | 0.459 | 0.417 |
| `heterogeneous` | `dual2` | 0.458 | 0.417 |
| `heterogeneous` | `mono4` | 0.458 | 0.417 |
| `homogeneous` | `dual2` | 0.458 | 0.417 |
| `homogeneous` | `mono4` | 0.458 | 0.417 |
| `heterogeneous` | `dual2` | 0.667 | 0.625 |
| `heterogeneous` | `mono4` | 0.458 | 0.417 |
| `homogeneous` | `dual2` | 0.458 | 0.417 |
| `homogeneous` | `mono4` | 0.562 | 0.417 |

The parent contrasts compare the two population modes at matched seeds.

| contrast | factors | mean difference | 95% CI | n |
|---|---|---:|---|---:|
| `heterogeneous_minus_homogeneous_parent_all` | form=mono4, task=factorized | -0.023 | [-0.077, 0.030] | 9 |
| `heterogeneous_minus_homogeneous_parent_message_gap` | form=mono4, task=factorized | -0.046 | [-0.117, 0.025] | 9 |
| `heterogeneous_minus_homogeneous_parent_all` | form=dual2, task=factorized | 0.023 | [-0.030, 0.077] | 9 |
| `heterogeneous_minus_homogeneous_parent_message_gap` | form=dual2, task=factorized | 0.023 | [-0.030, 0.077] | 9 |

## Held-out live endpoint

| population | representation | form | mapping | support | channel | initial | final | gain | 95% CI | functional |
|---|---|---|---|---|---|---:|---:|---:|---|---:|
| `homogeneous` | `joint_history` | `mono4` | `identity` | `full` | `live` | 0.122 | 0.504 | 0.382 | [0.405, 0.603] | 2/9 |
| `homogeneous` | `joint_history` | `mono4` | `identity` | `full` | `silent` | 0.176 | 0.261 | 0.085 | [0.120, 0.403] | 0/9 |
| `homogeneous` | `joint_history` | `mono4` | `identity` | `leave_one_out` | `live` | 0.122 | 0.250 | 0.128 | [0.234, 0.266] | 0/9 |
| `homogeneous` | `joint_history` | `mono4` | `identity` | `leave_one_out` | `silent` | 0.176 | -0.050 | -0.226 | [-0.160, 0.059] | 0/9 |
| `homogeneous` | `joint_history` | `mono4` | `swap` | `full` | `live` | 0.203 | 0.492 | 0.289 | [0.386, 0.598] | 2/9 |
| `homogeneous` | `joint_history` | `mono4` | `swap` | `full` | `silent` | 0.164 | 0.204 | 0.040 | [0.114, 0.295] | 0/9 |
| `homogeneous` | `joint_history` | `mono4` | `swap` | `leave_one_out` | `live` | 0.203 | 0.213 | 0.010 | [0.148, 0.278] | 0/9 |
| `homogeneous` | `joint_history` | `mono4` | `swap` | `leave_one_out` | `silent` | 0.164 | -0.073 | -0.238 | [-0.148, 0.001] | 0/9 |
| `homogeneous` | `joint_history` | `dual2` | `identity` | `full` | `live` | 0.141 | 0.574 | 0.433 | [0.489, 0.658] | 5/9 |
| `homogeneous` | `joint_history` | `dual2` | `identity` | `full` | `silent` | 0.118 | 0.158 | 0.040 | [0.034, 0.282] | 0/9 |
| `homogeneous` | `joint_history` | `dual2` | `identity` | `leave_one_out` | `live` | 0.141 | 0.250 | 0.109 | [0.250, 0.250] | 0/9 |
| `homogeneous` | `joint_history` | `dual2` | `identity` | `leave_one_out` | `silent` | 0.118 | -0.074 | -0.192 | [-0.183, 0.036] | 0/9 |
| `homogeneous` | `joint_history` | `dual2` | `swap` | `full` | `live` | 0.164 | 0.586 | 0.421 | [0.519, 0.652] | 4/9 |
| `homogeneous` | `joint_history` | `dual2` | `swap` | `full` | `silent` | 0.222 | 0.342 | 0.120 | [0.201, 0.484] | 1/9 |
| `homogeneous` | `joint_history` | `dual2` | `swap` | `leave_one_out` | `live` | 0.164 | 0.273 | 0.109 | [0.220, 0.326] | 0/9 |
| `homogeneous` | `joint_history` | `dual2` | `swap` | `leave_one_out` | `silent` | 0.222 | -0.039 | -0.261 | [-0.158, 0.080] | 0/9 |
| `homogeneous` | `slot_local` | `dual2` | `identity` | `full` | `live` | 0.157 | 0.585 | 0.428 | [0.507, 0.663] | 5/9 |
| `homogeneous` | `slot_local` | `dual2` | `identity` | `full` | `silent` | 0.137 | 0.262 | 0.126 | [0.160, 0.364] | 0/9 |
| `homogeneous` | `slot_local` | `dual2` | `identity` | `leave_one_out` | `live` | 0.157 | 0.296 | 0.139 | [0.190, 0.403] | 1/9 |
| `homogeneous` | `slot_local` | `dual2` | `identity` | `leave_one_out` | `silent` | 0.137 | -0.050 | -0.187 | [-0.144, 0.044] | 0/9 |
| `homogeneous` | `slot_local` | `dual2` | `swap` | `full` | `live` | 0.169 | 0.528 | 0.359 | [0.414, 0.641] | 4/9 |
| `homogeneous` | `slot_local` | `dual2` | `swap` | `full` | `silent` | 0.183 | 0.274 | 0.091 | [0.176, 0.371] | 0/9 |
| `homogeneous` | `slot_local` | `dual2` | `swap` | `leave_one_out` | `live` | 0.169 | 0.296 | 0.128 | [0.190, 0.403] | 1/9 |
| `homogeneous` | `slot_local` | `dual2` | `swap` | `leave_one_out` | `silent` | 0.183 | -0.051 | -0.233 | [-0.174, 0.073] | 0/9 |
| `heterogeneous` | `joint_history` | `mono4` | `identity` | `full` | `live` | 0.123 | 0.469 | 0.346 | [0.395, 0.543] | 1/9 |
| `heterogeneous` | `joint_history` | `mono4` | `identity` | `full` | `silent` | 0.176 | 0.261 | 0.085 | [0.120, 0.403] | 0/9 |
| `heterogeneous` | `joint_history` | `mono4` | `identity` | `leave_one_out` | `live` | 0.123 | 0.250 | 0.127 | [0.250, 0.250] | 0/9 |
| `heterogeneous` | `joint_history` | `mono4` | `identity` | `leave_one_out` | `silent` | 0.176 | -0.050 | -0.226 | [-0.160, 0.059] | 0/9 |
| `heterogeneous` | `joint_history` | `mono4` | `swap` | `full` | `live` | 0.204 | 0.458 | 0.254 | [0.417, 0.498] | 0/9 |
| `heterogeneous` | `joint_history` | `mono4` | `swap` | `full` | `silent` | 0.164 | 0.204 | 0.040 | [0.114, 0.295] | 0/9 |
| `heterogeneous` | `joint_history` | `mono4` | `swap` | `leave_one_out` | `live` | 0.204 | 0.250 | 0.046 | [0.250, 0.250] | 0/9 |
| `heterogeneous` | `joint_history` | `mono4` | `swap` | `leave_one_out` | `silent` | 0.164 | -0.073 | -0.238 | [-0.148, 0.001] | 0/9 |
| `heterogeneous` | `joint_history` | `dual2` | `identity` | `full` | `live` | 0.167 | 0.551 | 0.384 | [0.442, 0.660] | 5/9 |
| `heterogeneous` | `joint_history` | `dual2` | `identity` | `full` | `silent` | 0.118 | 0.158 | 0.040 | [0.034, 0.282] | 0/9 |
| `heterogeneous` | `joint_history` | `dual2` | `identity` | `leave_one_out` | `live` | 0.167 | 0.262 | 0.094 | [0.235, 0.288] | 0/9 |
| `heterogeneous` | `joint_history` | `dual2` | `identity` | `leave_one_out` | `silent` | 0.118 | -0.074 | -0.192 | [-0.183, 0.036] | 0/9 |
| `heterogeneous` | `joint_history` | `dual2` | `swap` | `full` | `live` | 0.167 | 0.540 | 0.373 | [0.444, 0.636] | 3/9 |
| `heterogeneous` | `joint_history` | `dual2` | `swap` | `full` | `silent` | 0.222 | 0.342 | 0.120 | [0.201, 0.484] | 1/9 |
| `heterogeneous` | `joint_history` | `dual2` | `swap` | `leave_one_out` | `live` | 0.167 | 0.273 | 0.106 | [0.220, 0.326] | 0/9 |
| `heterogeneous` | `joint_history` | `dual2` | `swap` | `leave_one_out` | `silent` | 0.222 | -0.039 | -0.261 | [-0.158, 0.080] | 0/9 |
| `heterogeneous` | `slot_local` | `dual2` | `identity` | `full` | `live` | 0.189 | 0.574 | 0.385 | [0.489, 0.658] | 5/9 |
| `heterogeneous` | `slot_local` | `dual2` | `identity` | `full` | `silent` | 0.137 | 0.262 | 0.126 | [0.160, 0.364] | 0/9 |
| `heterogeneous` | `slot_local` | `dual2` | `identity` | `leave_one_out` | `live` | 0.189 | 0.296 | 0.107 | [0.104, 0.489] | 2/9 |
| `heterogeneous` | `slot_local` | `dual2` | `identity` | `leave_one_out` | `silent` | 0.137 | -0.050 | -0.187 | [-0.144, 0.044] | 0/9 |
| `heterogeneous` | `slot_local` | `dual2` | `swap` | `full` | `live` | 0.143 | 0.528 | 0.385 | [0.415, 0.641] | 4/9 |
| `heterogeneous` | `slot_local` | `dual2` | `swap` | `full` | `silent` | 0.183 | 0.274 | 0.091 | [0.176, 0.371] | 0/9 |
| `heterogeneous` | `slot_local` | `dual2` | `swap` | `leave_one_out` | `live` | 0.143 | 0.296 | 0.154 | [0.104, 0.489] | 2/9 |
| `heterogeneous` | `slot_local` | `dual2` | `swap` | `leave_one_out` | `silent` | 0.183 | -0.051 | -0.233 | [-0.174, 0.073] | 0/9 |

## Communication and heterogeneity checks

`all_natural−permuted` is the causal message check: positive values mean that shuffling messages across a worker's complete goal block changes behavior. `heterogeneous_minus_homogeneous` isolates the effect of population-level label conventions while holding semantic worlds, goals, partners, and random draws fixed.

| contrast | factors | mean difference | 95% CI | n |
|---|---|---:|---|---:|
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=mono4, support=full, channel=live | -0.012 | [-0.105, 0.081] | 9 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=mono4, support=full, channel=live | -0.011 | [-0.086, 0.063] | 9 |
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=mono4, support=full, channel=silent | -0.057 | [-0.184, 0.070] | 9 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=mono4, support=full, channel=silent | -0.057 | [-0.184, 0.070] | 9 |
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=mono4, support=leave_one_out, channel=live | -0.037 | [-0.096, 0.022] | 9 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=mono4, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 9 |
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=mono4, support=leave_one_out, channel=silent | -0.023 | [-0.077, 0.030] | 9 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=mono4, support=leave_one_out, channel=silent | -0.023 | [-0.077, 0.030] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=identity, support=full, channel=live | -0.035 | [-0.125, 0.054] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=identity, support=full, channel=silent | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=identity, support=leave_one_out, channel=live | -0.000 | [-0.016, 0.016] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=identity, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=swap, support=full, channel=live | -0.034 | [-0.140, 0.072] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=swap, support=full, channel=silent | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=swap, support=leave_one_out, channel=live | 0.037 | [-0.028, 0.102] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=mono4, mapping=swap, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 9 |
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=dual2, support=full, channel=live | 0.012 | [-0.063, 0.086] | 9 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=dual2, support=full, channel=live | -0.011 | [-0.120, 0.098] | 9 |
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=dual2, support=full, channel=silent | 0.184 | [-0.019, 0.387] | 9 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=dual2, support=full, channel=silent | 0.184 | [-0.019, 0.387] | 9 |
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=dual2, support=leave_one_out, channel=live | 0.023 | [-0.030, 0.076] | 9 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=dual2, support=leave_one_out, channel=live | 0.012 | [-0.015, 0.038] | 9 |
| `swap_minus_identity` | population_mode=homogeneous, representation=joint_history, form=dual2, support=leave_one_out, channel=silent | 0.035 | [-0.021, 0.091] | 9 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=joint_history, form=dual2, support=leave_one_out, channel=silent | 0.035 | [-0.021, 0.091] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=identity, support=full, channel=live | -0.023 | [-0.127, 0.081] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=identity, support=full, channel=silent | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=identity, support=leave_one_out, channel=live | 0.012 | [-0.015, 0.038] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=identity, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=swap, support=full, channel=live | -0.046 | [-0.136, 0.044] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=swap, support=full, channel=silent | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=swap, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | representation=joint_history, form=dual2, mapping=swap, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 9 |
| `swap_minus_identity` | population_mode=homogeneous, representation=slot_local, form=dual2, support=full, channel=live | -0.058 | [-0.165, 0.049] | 9 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=slot_local, form=dual2, support=full, channel=live | -0.046 | [-0.153, 0.061] | 9 |
| `swap_minus_identity` | population_mode=homogeneous, representation=slot_local, form=dual2, support=full, channel=silent | 0.011 | [-0.112, 0.135] | 9 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=slot_local, form=dual2, support=full, channel=silent | 0.011 | [-0.112, 0.135] | 9 |
| `swap_minus_identity` | population_mode=homogeneous, representation=slot_local, form=dual2, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 9 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=slot_local, form=dual2, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 9 |
| `swap_minus_identity` | population_mode=homogeneous, representation=slot_local, form=dual2, support=leave_one_out, channel=silent | -0.000 | [-0.069, 0.069] | 9 |
| `swap_minus_identity` | population_mode=heterogeneous, representation=slot_local, form=dual2, support=leave_one_out, channel=silent | -0.000 | [-0.069, 0.069] | 9 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=identity, support=full, channel=live | -0.012 | [-0.074, 0.051] | 9 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=identity, support=full, channel=silent | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=identity, support=leave_one_out, channel=live | -0.000 | [-0.160, 0.160] | 9 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=identity, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=swap, support=full, channel=live | 0.001 | [-0.079, 0.080] | 9 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=swap, support=full, channel=silent | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=swap, support=leave_one_out, channel=live | -0.000 | [-0.160, 0.160] | 9 |
| `heterogeneous_minus_homogeneous` | representation=slot_local, form=dual2, mapping=swap, support=leave_one_out, channel=silent | 0.000 | [0.000, 0.000] | 9 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=identity, support=full, channel=live | 0.070 | [-0.037, 0.176] | 9 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=identity, support=full, channel=live | 0.012 | [-0.073, 0.096] | 9 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=identity, support=full, channel=silent | -0.103 | [-0.236, 0.029] | 9 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=identity, support=full, channel=silent | 0.104 | [-0.034, 0.243] | 9 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=identity, support=leave_one_out, channel=live | -0.000 | [-0.016, 0.016] | 9 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=identity, support=leave_one_out, channel=live | 0.046 | [-0.060, 0.153] | 9 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=identity, support=leave_one_out, channel=silent | -0.023 | [-0.101, 0.054] | 9 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=identity, support=leave_one_out, channel=silent | 0.023 | [-0.043, 0.090] | 9 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=swap, support=full, channel=live | 0.093 | [-0.054, 0.240] | 9 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=swap, support=full, channel=live | -0.058 | [-0.139, 0.023] | 9 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=swap, support=full, channel=silent | 0.138 | [0.012, 0.264] | 9 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=swap, support=full, channel=silent | -0.068 | [-0.157, 0.020] | 9 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=swap, support=leave_one_out, channel=live | 0.060 | [-0.017, 0.137] | 9 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=swap, support=leave_one_out, channel=live | 0.023 | [-0.030, 0.077] | 9 |
| `dual2_minus_mono4` | population_mode=homogeneous, mapping=swap, support=leave_one_out, channel=silent | 0.035 | [-0.022, 0.091] | 9 |
| `slot_local_minus_joint_history` | population_mode=homogeneous, mapping=swap, support=leave_one_out, channel=silent | -0.012 | [-0.059, 0.036] | 9 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=identity, support=full, channel=live | 0.082 | [-0.006, 0.170] | 9 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=identity, support=full, channel=live | 0.023 | [-0.081, 0.127] | 9 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=identity, support=full, channel=silent | -0.103 | [-0.236, 0.029] | 9 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=identity, support=full, channel=silent | 0.104 | [-0.034, 0.243] | 9 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=identity, support=leave_one_out, channel=live | 0.012 | [-0.015, 0.038] | 9 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=identity, support=leave_one_out, channel=live | 0.035 | [-0.144, 0.214] | 9 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=identity, support=leave_one_out, channel=silent | -0.023 | [-0.101, 0.054] | 9 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=identity, support=leave_one_out, channel=silent | 0.023 | [-0.043, 0.090] | 9 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=swap, support=full, channel=live | 0.082 | [-0.014, 0.178] | 9 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=swap, support=full, channel=live | -0.012 | [-0.113, 0.090] | 9 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=swap, support=full, channel=silent | 0.138 | [0.012, 0.264] | 9 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=swap, support=full, channel=silent | -0.068 | [-0.157, 0.020] | 9 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=swap, support=leave_one_out, channel=live | 0.023 | [-0.030, 0.076] | 9 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=swap, support=leave_one_out, channel=live | 0.023 | [-0.146, 0.192] | 9 |
| `dual2_minus_mono4` | population_mode=heterogeneous, mapping=swap, support=leave_one_out, channel=silent | 0.035 | [-0.022, 0.091] | 9 |
| `slot_local_minus_joint_history` | population_mode=heterogeneous, mapping=swap, support=leave_one_out, channel=silent | -0.012 | [-0.059, 0.036] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=mono4, mapping=identity, channel=live | -0.254 | [-0.355, -0.154] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=mono4, mapping=identity, channel=silent | -0.312 | [-0.477, -0.147] | 9 |
| `live_minus_silent_all` | population_mode=homogeneous, representation=joint_history, form=mono4, mapping=identity, support=full | 0.243 | [0.202, 0.283] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=mono4, mapping=swap, channel=live | -0.279 | [-0.434, -0.124] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=mono4, mapping=swap, channel=silent | -0.278 | [-0.417, -0.139] | 9 |
| `live_minus_silent_all` | population_mode=homogeneous, representation=joint_history, form=mono4, mapping=swap, support=full | 0.243 | [0.204, 0.282] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=dual2, mapping=identity, channel=live | -0.324 | [-0.408, -0.239] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=dual2, mapping=identity, channel=silent | -0.232 | [-0.401, -0.063] | 9 |
| `live_minus_silent_all` | population_mode=homogeneous, representation=joint_history, form=dual2, mapping=identity, support=full | 0.232 | [0.179, 0.286] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=dual2, mapping=swap, channel=live | -0.312 | [-0.382, -0.243] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=joint_history, form=dual2, mapping=swap, channel=silent | -0.381 | [-0.540, -0.222] | 9 |
| `live_minus_silent_all` | population_mode=homogeneous, representation=joint_history, form=dual2, mapping=swap, support=full | 0.232 | [0.179, 0.286] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=slot_local, form=dual2, mapping=identity, channel=live | -0.289 | [-0.401, -0.177] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=slot_local, form=dual2, mapping=identity, channel=silent | -0.313 | [-0.439, -0.186] | 9 |
| `live_minus_silent_all` | population_mode=homogeneous, representation=slot_local, form=dual2, mapping=identity, support=full | 0.232 | [0.179, 0.285] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=slot_local, form=dual2, mapping=swap, channel=live | -0.231 | [-0.356, -0.106] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, representation=slot_local, form=dual2, mapping=swap, channel=silent | -0.324 | [-0.454, -0.195] | 9 |
| `live_minus_silent_all` | population_mode=homogeneous, representation=slot_local, form=dual2, mapping=swap, support=full | 0.233 | [0.180, 0.286] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=mono4, mapping=identity, channel=live | -0.219 | [-0.293, -0.145] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=mono4, mapping=identity, channel=silent | -0.312 | [-0.477, -0.147] | 9 |
| `live_minus_silent_all` | population_mode=heterogeneous, representation=joint_history, form=mono4, mapping=identity, support=full | 0.221 | [0.194, 0.248] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=mono4, mapping=swap, channel=live | -0.208 | [-0.248, -0.167] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=mono4, mapping=swap, channel=silent | -0.278 | [-0.417, -0.139] | 9 |
| `live_minus_silent_all` | population_mode=heterogeneous, representation=joint_history, form=mono4, mapping=swap, support=full | 0.220 | [0.193, 0.247] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=dual2, mapping=identity, channel=live | -0.289 | [-0.393, -0.185] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=dual2, mapping=identity, channel=silent | -0.232 | [-0.401, -0.063] | 9 |
| `live_minus_silent_all` | population_mode=heterogeneous, representation=joint_history, form=dual2, mapping=identity, support=full | 0.255 | [0.185, 0.325] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=dual2, mapping=swap, channel=live | -0.267 | [-0.357, -0.177] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=joint_history, form=dual2, mapping=swap, channel=silent | -0.381 | [-0.540, -0.222] | 9 |
| `live_minus_silent_all` | population_mode=heterogeneous, representation=joint_history, form=dual2, mapping=swap, support=full | 0.255 | [0.185, 0.326] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=slot_local, form=dual2, mapping=identity, channel=live | -0.278 | [-0.474, -0.081] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=slot_local, form=dual2, mapping=identity, channel=silent | -0.313 | [-0.439, -0.186] | 9 |
| `live_minus_silent_all` | population_mode=heterogeneous, representation=slot_local, form=dual2, mapping=identity, support=full | 0.232 | [0.135, 0.328] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=slot_local, form=dual2, mapping=swap, channel=live | -0.232 | [-0.435, -0.029] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, representation=slot_local, form=dual2, mapping=swap, channel=silent | -0.324 | [-0.454, -0.195] | 9 |
| `live_minus_silent_all` | population_mode=heterogeneous, representation=slot_local, form=dual2, mapping=swap, support=full | 0.232 | [0.136, 0.328] | 9 |

This finite tabular study measures protocol formation and repair under controlled perceptual conventions. It does not claim that the agents possess human language.
