# Two-generation compositional transmission

The sample is the nine pre-registered composable parent seeds. Each row is evaluated with natural, closed/silent, partner-permuted and slot-recombined messages.

| lineage | generation | g1 channel | g2/eval channel | n | natural | recombined−natural | composable |
|---|---:|---|---|---:|---:|---:|---:|
| `worker_then_sender` | 1 | `live` | `live` | 9 | 0.667 | 0.000 | 9/9 |
| `worker_then_sender` | 1 | `live` | `silent` | 0 | n/a | n/a | 0/0 |
| `worker_then_sender` | 1 | `silent` | `live` | 0 | n/a | n/a | 0/0 |
| `worker_then_sender` | 1 | `silent` | `silent` | 9 | 0.127 | 0.000 | 0/9 |
| `worker_then_sender` | 2 | `live` | `live` | 9 | 0.667 | 0.000 | 9/9 |
| `worker_then_sender` | 2 | `live` | `silent` | 9 | 0.283 | -0.029 | 0/9 |
| `worker_then_sender` | 2 | `silent` | `live` | 9 | 0.177 | -0.018 | 0/9 |
| `worker_then_sender` | 2 | `silent` | `silent` | 9 | 0.166 | 0.006 | 0/9 |
| `sender_then_worker` | 1 | `live` | `live` | 9 | 0.667 | 0.000 | 9/9 |
| `sender_then_worker` | 1 | `live` | `silent` | 0 | n/a | n/a | 0/0 |
| `sender_then_worker` | 1 | `silent` | `live` | 0 | n/a | n/a | 0/0 |
| `sender_then_worker` | 1 | `silent` | `silent` | 9 | 0.180 | -0.068 | 0/9 |
| `sender_then_worker` | 2 | `live` | `live` | 9 | 0.667 | 0.000 | 9/9 |
| `sender_then_worker` | 2 | `live` | `silent` | 9 | 0.152 | 0.000 | 0/9 |
| `sender_then_worker` | 2 | `silent` | `live` | 9 | 0.314 | 0.002 | 0/9 |
| `sender_then_worker` | 2 | `silent` | `silent` | 9 | 0.167 | -0.012 | 0/9 |

| lineage | generation | g1 channel | live−silent | live−permuted |
|---|---:|---|---:|---:|
| `worker_then_sender` | 1 | `live` | 0.511 [0.478,0.543] | 0.414 [0.404,0.424] |
| `worker_then_sender` | 1 | `silent` | n/a | n/a |
| `worker_then_sender` | 2 | `live` | 0.511 [0.478,0.544] | 0.417 [0.412,0.421] |
| `worker_then_sender` | 2 | `silent` | -0.071 [-0.135,-0.007] | 0.029 [-0.009,0.066] |
| `sender_then_worker` | 1 | `live` | 0.482 [0.453,0.510] | 0.414 [0.404,0.424] |
| `sender_then_worker` | 1 | `silent` | n/a | n/a |
| `sender_then_worker` | 2 | `live` | 0.506 [0.481,0.531] | 0.417 [0.412,0.421] |
| `sender_then_worker` | 2 | `silent` | 0.153 [0.063,0.244] | 0.063 [-0.014,0.141] |

The generation-2 rows condition on the corresponding generation-1 endpoint. They therefore test cumulative transmission rather than fresh emergence.
