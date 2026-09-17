# Ecological factorization and novel-combination transfer

- parent runs: 36
- child runs: 216
- primary zero-shot metric: leave-one-out live heldout return

## Parent performance

| form | task | mean all-goal return |
|---|---|---:|
| `mono9` | `factorized` | 0.143 [0.114,0.173] |
| `mono9` | `holistic` | 0.131 [0.116,0.147] |
| `tri3` | `factorized` | 0.195 [0.168,0.222] |
| `tri3` | `holistic` | 0.167 [0.150,0.184] |

## Child held-out return

| representation | form | task | support | channel | heldout | pass count |
|---|---|---|---|---|---:|---:|
| `joint_history` | `mono9` | `factorized` | `full` | `live` | 0.206 [0.132,0.281] | n/a |
| `joint_history` | `mono9` | `factorized` | `full` | `silent` | 0.123 [0.043,0.203] | n/a |
| `joint_history` | `mono9` | `factorized` | `leave_one_out` | `live` | -0.044 [-0.121,0.033] | 0/9 |
| `joint_history` | `mono9` | `factorized` | `leave_one_out` | `silent` | -0.081 [-0.137,-0.026] | n/a |
| `joint_history` | `mono9` | `holistic` | `full` | `live` | 0.148 [0.077,0.219] | n/a |
| `joint_history` | `mono9` | `holistic` | `full` | `silent` | 0.108 [0.053,0.163] | n/a |
| `joint_history` | `mono9` | `holistic` | `leave_one_out` | `live` | -0.074 [-0.141,-0.006] | 0/9 |
| `joint_history` | `mono9` | `holistic` | `leave_one_out` | `silent` | -0.101 [-0.150,-0.053] | n/a |
| `joint_history` | `tri3` | `factorized` | `full` | `live` | 0.303 [0.238,0.368] | n/a |
| `joint_history` | `tri3` | `factorized` | `full` | `silent` | 0.088 [-0.015,0.191] | n/a |
| `joint_history` | `tri3` | `factorized` | `leave_one_out` | `live` | -0.023 [-0.126,0.080] | 0/9 |
| `joint_history` | `tri3` | `factorized` | `leave_one_out` | `silent` | -0.082 [-0.156,-0.007] | n/a |
| `joint_history` | `tri3` | `holistic` | `full` | `live` | 0.214 [0.102,0.327] | n/a |
| `joint_history` | `tri3` | `holistic` | `full` | `silent` | 0.112 [0.066,0.158] | n/a |
| `joint_history` | `tri3` | `holistic` | `leave_one_out` | `live` | -0.065 [-0.140,0.010] | 0/9 |
| `joint_history` | `tri3` | `holistic` | `leave_one_out` | `silent` | -0.094 [-0.150,-0.037] | n/a |
| `slot_local` | `tri3` | `factorized` | `full` | `live` | 0.293 [0.222,0.363] | n/a |
| `slot_local` | `tri3` | `factorized` | `full` | `silent` | 0.120 [0.031,0.209] | n/a |
| `slot_local` | `tri3` | `factorized` | `leave_one_out` | `live` | -0.036 [-0.147,0.075] | 0/9 |
| `slot_local` | `tri3` | `factorized` | `leave_one_out` | `silent` | -0.067 [-0.138,0.004] | n/a |
| `slot_local` | `tri3` | `holistic` | `full` | `live` | 0.253 [0.122,0.385] | n/a |
| `slot_local` | `tri3` | `holistic` | `full` | `silent` | 0.077 [0.020,0.135] | n/a |
| `slot_local` | `tri3` | `holistic` | `leave_one_out` | `live` | -0.063 [-0.163,0.037] | 0/9 |
| `slot_local` | `tri3` | `holistic` | `leave_one_out` | `silent` | -0.086 [-0.169,-0.002] | n/a |

## Pre-registered contrasts

| contrast | mean | 95% CI |
|---|---:|---:|
| `factorized_minus_holistic|slot_local|tri3|leave_one_out|live|heldout` | 0.027 | [-0.109,0.163] |
| `slot_local_minus_joint|tri3|factorized|leave_one_out|live|heldout` | -0.013 | [-0.057,0.031] |
| `tri3_minus_mono9|joint_history|factorized|leave_one_out|live|heldout` | 0.021 | [-0.045,0.086] |
| `leave_minus_full|slot_local|tri3|factorized|live|heldout` | -0.329 | [-0.468,-0.190] |

## Recombination gaps

| representation | task | support | raw factor gap | task-target gap |
|---|---|---|---:|---:|
| `joint_history` | `factorized` | `full` | -0.065 | -0.065 |
| `joint_history` | `factorized` | `leave_one_out` | -0.020 | -0.020 |
| `joint_history` | `holistic` | `full` | -0.048 | -0.074 |
| `joint_history` | `holistic` | `leave_one_out` | -0.019 | 0.004 |
| `slot_local` | `factorized` | `full` | -0.007 | -0.007 |
| `slot_local` | `factorized` | `leave_one_out` | -0.008 | -0.008 |
| `slot_local` | `holistic` | `full` | -0.072 | -0.065 |
| `slot_local` | `holistic` | `leave_one_out` | -0.034 | 0.051 |
