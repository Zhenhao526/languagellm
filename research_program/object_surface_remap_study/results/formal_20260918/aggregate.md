# Object-surface remapping

A parent learns on the identity mapping from visible surface label to semantic object type. A fresh worker receives the frozen sender codebook and learns under either the same mapping or a stable label swap. `dual2` and `mono4` have four complete message states; only `dual2` exposes two staged slots.

## Held-out live endpoint

| representation | form | mapping | support | channel | held-out natural | 95% CI | functional |
|---|---|---|---|---|---:|---|---:|
| `joint_history` | `mono4` | `identity` | `full` | `live` | 0.458 | [0.345, 0.571] | 2/9 |
| `joint_history` | `mono4` | `identity` | `full` | `silent` | 0.295 | [0.168, 0.422] | 0/9 |
| `joint_history` | `mono4` | `identity` | `leave_one_out` | `live` | 0.234 | [0.196, 0.271] | 0/9 |
| `joint_history` | `mono4` | `identity` | `leave_one_out` | `silent` | -0.004 | [-0.132, 0.123] | 0/9 |
| `joint_history` | `mono4` | `swap` | `full` | `live` | 0.481 | [0.385, 0.578] | 1/9 |
| `joint_history` | `mono4` | `swap` | `full` | `silent` | 0.237 | [0.153, 0.320] | 0/9 |
| `joint_history` | `mono4` | `swap` | `leave_one_out` | `live` | 0.245 | [0.234, 0.256] | 0/9 |
| `joint_history` | `mono4` | `swap` | `leave_one_out` | `silent` | -0.028 | [-0.148, 0.093] | 0/9 |
| `joint_history` | `dual2` | `identity` | `full` | `live` | 0.516 | [0.402, 0.630] | 4/9 |
| `joint_history` | `dual2` | `identity` | `full` | `silent` | 0.261 | [0.114, 0.407] | 0/9 |
| `joint_history` | `dual2` | `identity` | `leave_one_out` | `live` | 0.250 | [0.250, 0.250] | 0/9 |
| `joint_history` | `dual2` | `identity` | `leave_one_out` | `silent` | 0.018 | [-0.125, 0.161] | 0/9 |
| `joint_history` | `dual2` | `swap` | `full` | `live` | 0.551 | [0.477, 0.625] | 2/9 |
| `joint_history` | `dual2` | `swap` | `full` | `silent` | 0.226 | [0.115, 0.337] | 0/9 |
| `joint_history` | `dual2` | `swap` | `leave_one_out` | `live` | 0.250 | [0.250, 0.250] | 0/9 |
| `joint_history` | `dual2` | `swap` | `leave_one_out` | `silent` | -0.016 | [-0.130, 0.099] | 0/9 |
| `slot_local` | `dual2` | `identity` | `full` | `live` | 0.470 | [0.323, 0.617] | 2/9 |
| `slot_local` | `dual2` | `identity` | `full` | `silent` | 0.295 | [0.161, 0.429] | 0/9 |
| `slot_local` | `dual2` | `identity` | `leave_one_out` | `live` | 0.250 | [0.090, 0.410] | 1/9 |
| `slot_local` | `dual2` | `identity` | `leave_one_out` | `silent` | -0.027 | [-0.133, 0.079] | 0/9 |
| `slot_local` | `dual2` | `swap` | `full` | `live` | 0.423 | [0.263, 0.583] | 1/9 |
| `slot_local` | `dual2` | `swap` | `full` | `silent` | 0.237 | [0.145, 0.330] | 0/9 |
| `slot_local` | `dual2` | `swap` | `leave_one_out` | `live` | 0.250 | [0.090, 0.410] | 1/9 |
| `slot_local` | `dual2` | `swap` | `leave_one_out` | `silent` | -0.039 | [-0.151, 0.073] | 0/9 |

## Paired contrasts

| contrast | factors | mean difference | 95% CI | n |
|---|---|---:|---|---:|
| `swap_minus_identity` | representation=joint_history, form=mono4, support=full, channel=live | 0.024 | [-0.030, 0.077] | 9 |
| `swap_minus_identity` | representation=joint_history, form=mono4, support=full, channel=silent | -0.058 | [-0.172, 0.056] | 9 |
| `swap_minus_identity` | representation=joint_history, form=mono4, support=leave_one_out, channel=live | 0.011 | [-0.015, 0.038] | 9 |
| `swap_minus_identity` | representation=joint_history, form=mono4, support=leave_one_out, channel=silent | -0.023 | [-0.076, 0.030] | 9 |
| `swap_minus_identity` | representation=joint_history, form=dual2, support=full, channel=live | 0.035 | [-0.063, 0.132] | 9 |
| `swap_minus_identity` | representation=joint_history, form=dual2, support=full, channel=silent | -0.035 | [-0.148, 0.078] | 9 |
| `swap_minus_identity` | representation=joint_history, form=dual2, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 9 |
| `swap_minus_identity` | representation=joint_history, form=dual2, support=leave_one_out, channel=silent | -0.034 | [-0.090, 0.022] | 9 |
| `swap_minus_identity` | representation=slot_local, form=dual2, support=full, channel=live | -0.047 | [-0.117, 0.024] | 9 |
| `swap_minus_identity` | representation=slot_local, form=dual2, support=full, channel=silent | -0.057 | [-0.178, 0.064] | 9 |
| `swap_minus_identity` | representation=slot_local, form=dual2, support=leave_one_out, channel=live | 0.000 | [0.000, 0.000] | 9 |
| `swap_minus_identity` | representation=slot_local, form=dual2, support=leave_one_out, channel=silent | -0.012 | [-0.086, 0.063] | 9 |
| `dual2_minus_mono4` | mapping=identity, support=full, channel=live | 0.058 | [-0.023, 0.140] | 9 |
| `dual2_minus_mono4` | mapping=identity, support=full, channel=silent | -0.034 | [-0.132, 0.064] | 9 |
| `dual2_minus_mono4` | mapping=identity, support=leave_one_out, channel=live | 0.016 | [-0.021, 0.054] | 9 |
| `dual2_minus_mono4` | mapping=identity, support=leave_one_out, channel=silent | 0.023 | [-0.044, 0.089] | 9 |
| `dual2_minus_mono4` | mapping=swap, support=full, channel=live | 0.069 | [-0.050, 0.189] | 9 |
| `dual2_minus_mono4` | mapping=swap, support=full, channel=silent | -0.011 | [-0.095, 0.073] | 9 |
| `dual2_minus_mono4` | mapping=swap, support=leave_one_out, channel=live | 0.005 | [-0.006, 0.016] | 9 |
| `dual2_minus_mono4` | mapping=swap, support=leave_one_out, channel=silent | 0.012 | [-0.050, 0.074] | 9 |
| `slot_local_minus_joint_history` | mapping=identity, support=full, channel=live | -0.046 | [-0.161, 0.068] | 9 |
| `slot_local_minus_joint_history` | mapping=identity, support=full, channel=silent | 0.034 | [-0.080, 0.148] | 9 |
| `slot_local_minus_joint_history` | mapping=identity, support=leave_one_out, channel=live | -0.000 | [-0.160, 0.160] | 9 |
| `slot_local_minus_joint_history` | mapping=identity, support=leave_one_out, channel=silent | -0.046 | [-0.103, 0.012] | 9 |
| `slot_local_minus_joint_history` | mapping=swap, support=full, channel=live | -0.128 | [-0.259, 0.003] | 9 |
| `slot_local_minus_joint_history` | mapping=swap, support=full, channel=silent | 0.011 | [-0.073, 0.096] | 9 |
| `slot_local_minus_joint_history` | mapping=swap, support=leave_one_out, channel=live | -0.000 | [-0.160, 0.160] | 9 |
| `slot_local_minus_joint_history` | mapping=swap, support=leave_one_out, channel=silent | -0.023 | [-0.076, 0.030] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=mono4, mapping=identity, channel=live | -0.224 | [-0.361, -0.087] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=mono4, mapping=identity, channel=silent | -0.299 | [-0.441, -0.157] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=mono4, mapping=swap, channel=live | -0.236 | [-0.339, -0.133] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=mono4, mapping=swap, channel=silent | -0.265 | [-0.372, -0.158] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=dual2, mapping=identity, channel=live | -0.266 | [-0.380, -0.152] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=dual2, mapping=identity, channel=silent | -0.242 | [-0.439, -0.046] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=dual2, mapping=swap, channel=live | -0.301 | [-0.375, -0.227] | 9 |
| `leave_one_out_minus_full` | representation=joint_history, form=dual2, mapping=swap, channel=silent | -0.242 | [-0.355, -0.128] | 9 |
| `leave_one_out_minus_full` | representation=slot_local, form=dual2, mapping=identity, channel=live | -0.220 | [-0.314, -0.126] | 9 |
| `leave_one_out_minus_full` | representation=slot_local, form=dual2, mapping=identity, channel=silent | -0.322 | [-0.480, -0.165] | 9 |
| `leave_one_out_minus_full` | representation=slot_local, form=dual2, mapping=swap, channel=live | -0.173 | [-0.253, -0.094] | 9 |
| `leave_one_out_minus_full` | representation=slot_local, form=dual2, mapping=swap, channel=silent | -0.276 | [-0.346, -0.207] | 9 |

`live−silent` and `natural−permuted` are communication checks. The swap intervention holds latent goals and semantic scenes fixed while changing only visible surface labels. This finite tabular study measures protocol transfer and local repair; it does not claim that the agents have human language.
