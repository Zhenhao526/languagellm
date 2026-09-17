# Object-surface remapping

A parent learns on the identity mapping from visible surface label to semantic object type. A fresh worker receives the frozen sender codebook and learns under either the same mapping or a stable label swap. `dual2` and `mono4` have four complete message states; only `dual2` exposes two staged slots.

## Held-out live endpoint

| representation | form | mapping | support | channel | initial | final | gain | 95% CI | functional |
|---|---|---|---|---|---:|---:|---:|---|---:|
| `joint_history` | `mono4` | `identity` | `full` | `live` | 0.104 | 0.458 | 0.354 | [0.345, 0.572] | 2/9 |
| `joint_history` | `mono4` | `identity` | `full` | `silent` | 0.199 | 0.296 | 0.097 | [0.168, 0.423] | 0/9 |
| `joint_history` | `mono4` | `identity` | `leave_one_out` | `live` | 0.104 | 0.234 | 0.130 | [0.196, 0.271] | 0/9 |
| `joint_history` | `mono4` | `identity` | `leave_one_out` | `silent` | 0.199 | -0.005 | -0.203 | [-0.132, 0.122] | 0/9 |
| `joint_history` | `mono4` | `swap` | `full` | `live` | 0.243 | 0.482 | 0.238 | [0.385, 0.578] | 1/9 |
| `joint_history` | `mono4` | `swap` | `full` | `silent` | 0.176 | 0.238 | 0.062 | [0.153, 0.322] | 0/9 |
| `joint_history` | `mono4` | `swap` | `leave_one_out` | `live` | 0.243 | 0.245 | 0.002 | [0.235, 0.256] | 0/9 |
| `joint_history` | `mono4` | `swap` | `leave_one_out` | `silent` | 0.176 | -0.028 | -0.203 | [-0.148, 0.092] | 0/9 |
| `joint_history` | `dual2` | `identity` | `full` | `live` | 0.136 | 0.517 | 0.381 | [0.403, 0.630] | 4/9 |
| `joint_history` | `dual2` | `identity` | `full` | `silent` | 0.165 | 0.261 | 0.096 | [0.113, 0.409] | 0/9 |
| `joint_history` | `dual2` | `identity` | `leave_one_out` | `live` | 0.136 | 0.250 | 0.114 | [0.250, 0.250] | 0/9 |
| `joint_history` | `dual2` | `identity` | `leave_one_out` | `silent` | 0.165 | 0.019 | -0.146 | [-0.124, 0.162] | 0/9 |
| `joint_history` | `dual2` | `swap` | `full` | `live` | 0.228 | 0.551 | 0.323 | [0.477, 0.625] | 2/9 |
| `joint_history` | `dual2` | `swap` | `full` | `silent` | 0.177 | 0.227 | 0.050 | [0.114, 0.340] | 0/9 |
| `joint_history` | `dual2` | `swap` | `leave_one_out` | `live` | 0.228 | 0.250 | 0.022 | [0.250, 0.250] | 0/9 |
| `joint_history` | `dual2` | `swap` | `leave_one_out` | `silent` | 0.177 | -0.016 | -0.192 | [-0.130, 0.099] | 0/9 |
| `slot_local` | `dual2` | `identity` | `full` | `live` | 0.218 | 0.470 | 0.252 | [0.323, 0.617] | 2/9 |
| `slot_local` | `dual2` | `identity` | `full` | `silent` | 0.245 | 0.296 | 0.051 | [0.163, 0.430] | 0/9 |
| `slot_local` | `dual2` | `identity` | `leave_one_out` | `live` | 0.218 | 0.250 | 0.032 | [0.090, 0.410] | 1/9 |
| `slot_local` | `dual2` | `identity` | `leave_one_out` | `silent` | 0.245 | -0.027 | -0.272 | [-0.133, 0.079] | 0/9 |
| `slot_local` | `dual2` | `swap` | `full` | `live` | 0.161 | 0.424 | 0.263 | [0.264, 0.584] | 1/9 |
| `slot_local` | `dual2` | `swap` | `full` | `silent` | 0.130 | 0.239 | 0.109 | [0.145, 0.333] | 0/9 |
| `slot_local` | `dual2` | `swap` | `leave_one_out` | `live` | 0.161 | 0.250 | 0.089 | [0.090, 0.410] | 1/9 |
| `slot_local` | `dual2` | `swap` | `leave_one_out` | `silent` | 0.130 | -0.039 | -0.169 | [-0.151, 0.072] | 0/9 |

## All-goal communication check

The `natural−permuted` value uses the all-goal evaluation stream. A live positive value means the message changes behavior after the permutation control swaps messages across distinct goals.

| representation | form | mapping | support | channel | natural−permuted |
|---|---|---|---|---|---:|
| `joint_history` | `mono4` | `identity` | `full` | `live` | 0.415 |
| `joint_history` | `mono4` | `identity` | `full` | `silent` | 0.000 |
| `joint_history` | `mono4` | `identity` | `leave_one_out` | `live` | 0.399 |
| `joint_history` | `mono4` | `identity` | `leave_one_out` | `silent` | 0.000 |
| `joint_history` | `mono4` | `swap` | `full` | `live` | 0.416 |
| `joint_history` | `mono4` | `swap` | `full` | `silent` | 0.000 |
| `joint_history` | `mono4` | `swap` | `leave_one_out` | `live` | 0.411 |
| `joint_history` | `mono4` | `swap` | `leave_one_out` | `silent` | 0.000 |
| `joint_history` | `dual2` | `identity` | `full` | `live` | 0.439 |
| `joint_history` | `dual2` | `identity` | `full` | `silent` | 0.000 |
| `joint_history` | `dual2` | `identity` | `leave_one_out` | `live` | 0.417 |
| `joint_history` | `dual2` | `identity` | `leave_one_out` | `silent` | 0.000 |
| `joint_history` | `dual2` | `swap` | `full` | `live` | 0.439 |
| `joint_history` | `dual2` | `swap` | `full` | `silent` | 0.000 |
| `joint_history` | `dual2` | `swap` | `leave_one_out` | `live` | 0.417 |
| `joint_history` | `dual2` | `swap` | `leave_one_out` | `silent` | 0.000 |
| `slot_local` | `dual2` | `identity` | `full` | `live` | 0.393 |
| `slot_local` | `dual2` | `identity` | `full` | `silent` | 0.000 |
| `slot_local` | `dual2` | `identity` | `leave_one_out` | `live` | 0.394 |
| `slot_local` | `dual2` | `identity` | `leave_one_out` | `silent` | 0.000 |
| `slot_local` | `dual2` | `swap` | `full` | `live` | 0.392 |
| `slot_local` | `dual2` | `swap` | `full` | `silent` | 0.000 |
| `slot_local` | `dual2` | `swap` | `leave_one_out` | `live` | 0.394 |
| `slot_local` | `dual2` | `swap` | `leave_one_out` | `silent` | 0.000 |

## Paired contrasts

| contrast | factors | mean difference | 95% CI | n |
|---|---|---:|---|---:|
| `swap_minus_identity` | representation=joint_history, form=mono4, support=full, channel=live | 0.023 | [-0.030, 0.076] | 9 |
| `swap_minus_identity` | representation=joint_history, form=mono4, support=full, channel=silent | -0.058 | [-0.172, 0.056] | 9 |
| `swap_minus_identity` | representation=joint_history, form=mono4, support=leave_one_out, channel=live | 0.012 | [-0.015, 0.038] | 9 |
| `swap_minus_identity` | representation=joint_history, form=mono4, support=leave_one_out, channel=silent | -0.023 | [-0.076, 0.030] | 9 |
| `swap_minus_identity` | representation=joint_history, form=dual2, support=full, channel=live | 0.034 | [-0.063, 0.132] | 9 |
| `swap_minus_identity` | representation=joint_history, form=dual2, support=full, channel=silent | -0.034 | [-0.147, 0.079] | 9 |
| `swap_minus_identity` | representation=joint_history, form=dual2, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 9 |
| `swap_minus_identity` | representation=joint_history, form=dual2, support=leave_one_out, channel=silent | -0.035 | [-0.092, 0.023] | 9 |
| `swap_minus_identity` | representation=slot_local, form=dual2, support=full, channel=live | -0.047 | [-0.117, 0.024] | 9 |
| `swap_minus_identity` | representation=slot_local, form=dual2, support=full, channel=silent | -0.058 | [-0.178, 0.063] | 9 |
| `swap_minus_identity` | representation=slot_local, form=dual2, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 9 |
| `swap_minus_identity` | representation=slot_local, form=dual2, support=leave_one_out, channel=silent | -0.012 | [-0.087, 0.063] | 9 |
| `dual2_minus_mono4` | mapping=identity, support=full, channel=live | 0.058 | [-0.023, 0.140] | 9 |
| `dual2_minus_mono4` | mapping=identity, support=full, channel=silent | -0.035 | [-0.133, 0.063] | 9 |
| `dual2_minus_mono4` | mapping=identity, support=leave_one_out, channel=live | 0.016 | [-0.021, 0.054] | 9 |
| `dual2_minus_mono4` | mapping=identity, support=leave_one_out, channel=silent | 0.024 | [-0.043, 0.091] | 9 |
| `dual2_minus_mono4` | mapping=swap, support=full, channel=live | 0.070 | [-0.050, 0.190] | 9 |
| `dual2_minus_mono4` | mapping=swap, support=full, channel=silent | -0.011 | [-0.096, 0.074] | 9 |
| `dual2_minus_mono4` | mapping=swap, support=leave_one_out, channel=live | 0.005 | [-0.006, 0.015] | 9 |
| `dual2_minus_mono4` | mapping=swap, support=leave_one_out, channel=silent | 0.012 | [-0.051, 0.075] | 9 |
| `slot_local_minus_joint_history` | mapping=identity, support=full, channel=live | -0.047 | [-0.161, 0.067] | 9 |
| `slot_local_minus_joint_history` | mapping=identity, support=full, channel=silent | 0.035 | [-0.078, 0.149] | 9 |
| `slot_local_minus_joint_history` | mapping=identity, support=leave_one_out, channel=live | -0.000 | [-0.160, 0.160] | 9 |
| `slot_local_minus_joint_history` | mapping=identity, support=leave_one_out, channel=silent | -0.046 | [-0.105, 0.012] | 9 |
| `slot_local_minus_joint_history` | mapping=swap, support=full, channel=live | -0.128 | [-0.259, 0.003] | 9 |
| `slot_local_minus_joint_history` | mapping=swap, support=full, channel=silent | 0.012 | [-0.073, 0.096] | 9 |
| `slot_local_minus_joint_history` | mapping=swap, support=leave_one_out, channel=live | -0.000 | [-0.160, 0.160] | 9 |
| `slot_local_minus_joint_history` | mapping=swap, support=leave_one_out, channel=silent | -0.024 | [-0.077, 0.030] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=mono4, mapping=identity, channel=live | -0.225 | [-0.361, -0.088] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=mono4, mapping=identity, channel=silent | -0.301 | [-0.442, -0.159] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=mono4, mapping=swap, channel=live | -0.236 | [-0.339, -0.133] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=mono4, mapping=swap, channel=silent | -0.265 | [-0.372, -0.158] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=dual2, mapping=identity, channel=live | -0.267 | [-0.380, -0.153] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=dual2, mapping=identity, channel=silent | -0.242 | [-0.438, -0.046] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=dual2, mapping=swap, channel=live | -0.301 | [-0.375, -0.227] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=dual2, mapping=swap, channel=silent | -0.243 | [-0.356, -0.129] | 9 |
| `leave_one_out_minus_full` | representation=slot_local, form=dual2, mapping=identity, channel=live | -0.220 | [-0.313, -0.127] | 9 |
| `leave_one_out_minus_full` | representation=slot_local, form=dual2, mapping=identity, channel=silent | -0.324 | [-0.480, -0.167] | 9 |
| `leave_one_out_minus_full` | representation=slot_local, form=dual2, mapping=swap, channel=live | -0.174 | [-0.253, -0.094] | 9 |
| `leave_one_out_minus_full` | representation=slot_local, form=dual2, mapping=swap, channel=silent | -0.278 | [-0.347, -0.209] | 9 |
| `live_minus_silent_all` | representation=joint_history, form=mono4, mapping=identity, support=full | 0.229 | [0.193, 0.265] | 9 |
| `live_minus_silent_all` | representation=joint_history, form=mono4, mapping=identity, support=leave_one_out | 0.216 | [0.188, 0.245] | 9 |
| `live_minus_silent_all` | representation=joint_history, form=mono4, mapping=swap, support=full | 0.230 | [0.195, 0.265] | 9 |
| `live_minus_silent_all` | representation=joint_history, form=mono4, mapping=swap, support=leave_one_out | 0.219 | [0.193, 0.246] | 9 |
| `live_minus_silent_all` | representation=joint_history, form=dual2, mapping=identity, support=full | 0.231 | [0.178, 0.284] | 9 |
| `live_minus_silent_all` | representation=joint_history, form=dual2, mapping=identity, support=leave_one_out | 0.220 | [0.193, 0.247] | 9 |
| `live_minus_silent_all` | representation=joint_history, form=dual2, mapping=swap, support=full | 0.230 | [0.177, 0.284] | 9 |
| `live_minus_silent_all` | representation=joint_history, form=dual2, mapping=swap, support=leave_one_out | 0.221 | [0.194, 0.247] | 9 |
| `live_minus_silent_all` | representation=slot_local, form=dual2, mapping=identity, support=full | 0.207 | [0.126, 0.287] | 9 |
| `live_minus_silent_all` | representation=slot_local, form=dual2, mapping=identity, support=leave_one_out | 0.209 | [0.129, 0.289] | 9 |
| `live_minus_silent_all` | representation=slot_local, form=dual2, mapping=swap, support=full | 0.207 | [0.127, 0.288] | 9 |
| `live_minus_silent_all` | representation=slot_local, form=dual2, mapping=swap, support=leave_one_out | 0.209 | [0.129, 0.289] | 9 |

`live−silent` and `natural−permuted` are communication checks. The swap intervention holds latent goals and semantic scenes fixed while changing only visible surface labels. This finite tabular study measures protocol transfer and local repair; it does not claim that the agents have human language.
