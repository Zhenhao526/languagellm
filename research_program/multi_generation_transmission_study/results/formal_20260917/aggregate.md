# Two-generation compositional transmission

The sample is the nine pre-registered composable parent seeds. Each row is evaluated with natural, closed/silent, partner-permuted and slot-recombined messages.

| lineage | generation | channel | n | natural | recombined−natural | composable |
|---|---:|---|---:|---:|---:|---:|
| `worker_then_sender` | 1 | `live` | 9 | 0.667 | 0.000 | 9/9 |
| `worker_then_sender` | 1 | `silent` | 9 | 0.127 | 0.000 | 0/9 |
| `worker_then_sender` | 2 | `live` | 18 | 0.422 | -0.009 | 9/18 |
| `worker_then_sender` | 2 | `silent` | 18 | 0.224 | -0.012 | 0/18 |
| `sender_then_worker` | 1 | `live` | 9 | 0.667 | 0.000 | 9/9 |
| `sender_then_worker` | 1 | `silent` | 9 | 0.180 | -0.068 | 0/9 |
| `sender_then_worker` | 2 | `live` | 18 | 0.490 | 0.001 | 9/18 |
| `sender_then_worker` | 2 | `silent` | 18 | 0.159 | -0.006 | 0/18 |

| lineage | generation | live−silent | live−permuted |
|---|---:|---:|---:|
| `worker_then_sender` | 1 | 0.511 [0.478,0.543] | 0.414 [0.404,0.424] |
| `worker_then_sender` | 2 | 0.220 [0.066,0.374] | 0.223 [0.121,0.325] |
| `sender_then_worker` | 1 | 0.482 [0.453,0.510] | 0.414 [0.404,0.424] |
| `sender_then_worker` | 2 | 0.330 [0.228,0.432] | 0.240 [0.141,0.339] |

The generation-2 rows condition on the corresponding generation-1 endpoint. They therefore test cumulative transmission rather than fresh emergence.
