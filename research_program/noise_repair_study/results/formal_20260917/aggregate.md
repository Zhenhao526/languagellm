# Noisy transmission and protocol repair

The nine parent-composable seeds are fixed before this batch. `natural` is evaluated at the training noise rate; `incumbent` averages frozen workers 1–3.

| adaptation | receiver | noise | n | new natural | clean | silent | live−silent | recombined−natural | incumbent natural | composable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `worker_only` | `joint_history` | 0.00 | 9 | 0.667 | 0.667 | 0.149 | 0.517 | 0.000 | 0.667 | 9/9 |
| `worker_only` | `joint_history` | 0.10 | 9 | 0.583 | 0.667 | 0.149 | 0.434 | 0.000 | 0.583 | 0/9 |
| `worker_only` | `joint_history` | 0.25 | 9 | 0.458 | 0.667 | 0.149 | 0.309 | 0.000 | 0.459 | 0/9 |
| `worker_only` | `joint_history` | 0.40 | 9 | 0.334 | 0.667 | 0.149 | 0.184 | 0.000 | 0.332 | 0/9 |
| `worker_only` | `slot_local` | 0.00 | 9 | 0.667 | 0.667 | 0.149 | 0.517 | 0.000 | 0.667 | 9/9 |
| `worker_only` | `slot_local` | 0.10 | 9 | 0.583 | 0.667 | 0.149 | 0.434 | 0.000 | 0.583 | 0/9 |
| `worker_only` | `slot_local` | 0.25 | 9 | 0.458 | 0.667 | 0.149 | 0.309 | 0.000 | 0.459 | 0/9 |
| `worker_only` | `slot_local` | 0.40 | 9 | 0.334 | 0.667 | 0.149 | 0.184 | 0.000 | 0.332 | 0/9 |
| `coadapt` | `joint_history` | 0.00 | 9 | 0.667 | 0.667 | 0.149 | 0.517 | 0.000 | 0.667 | 9/9 |
| `coadapt` | `joint_history` | 0.10 | 9 | 0.583 | 0.667 | 0.149 | 0.434 | 0.000 | 0.583 | 0/9 |
| `coadapt` | `joint_history` | 0.25 | 9 | 0.458 | 0.667 | 0.149 | 0.309 | 0.000 | 0.459 | 0/9 |
| `coadapt` | `joint_history` | 0.40 | 9 | 0.334 | 0.667 | 0.149 | 0.184 | 0.000 | 0.332 | 0/9 |
| `coadapt` | `slot_local` | 0.00 | 9 | 0.667 | 0.667 | 0.149 | 0.517 | 0.000 | 0.667 | 9/9 |
| `coadapt` | `slot_local` | 0.10 | 9 | 0.583 | 0.667 | 0.149 | 0.434 | 0.000 | 0.583 | 0/9 |
| `coadapt` | `slot_local` | 0.25 | 9 | 0.458 | 0.667 | 0.149 | 0.309 | 0.000 | 0.459 | 0/9 |
| `coadapt` | `slot_local` | 0.40 | 9 | 0.334 | 0.667 | 0.149 | 0.184 | 0.000 | 0.332 | 0/9 |

| receiver | noise | coadapt−worker-only natural | coadapt−worker-only incumbent | coadapt−worker-only recombined gap |
|---|---:|---:|---:|---:|
| `joint_history` | 0.00 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] |
| `joint_history` | 0.10 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] |
| `joint_history` | 0.25 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] |
| `joint_history` | 0.40 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] |
| `slot_local` | 0.00 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] |
| `slot_local` | 0.10 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] |
| `slot_local` | 0.25 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] |
| `slot_local` | 0.40 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] |

A coadaptation gain with a simultaneous incumbent loss is evidence of renegotiation rather than faithful transmission. This is a conditional stability study on parent-composable protocols, not an emergence-rate estimate.
