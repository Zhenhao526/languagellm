# Shared-parity redundancy pressure

Both action stages require the same hidden parity. `triple2` and `atomic8` each have eight raw message states; `dual2` has four. The atomic corruption probability is matched to the probability that at least one of three binary coordinates flips.

| form | adaptation | noise | natural | clean | silent | live−silent | class Hamming | within-class | functional | error-correcting candidate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dual2` | `scratch` | 0.00 | 0.667 | 0.667 | 0.182 | 0.485 | 2.00 | 0.00 | 9/9 | 0/9 |
| `dual2` | `scratch` | 0.10 | 0.582 | 0.667 | 0.182 | 0.401 | 2.00 | 0.00 | 0/9 | 0/9 |
| `dual2` | `scratch` | 0.25 | 0.460 | 0.667 | 0.182 | 0.278 | 2.00 | 0.00 | 0/9 | 0/9 |
| `dual2` | `worker_only` | 0.00 | 0.667 | 0.667 | 0.159 | 0.508 | 1.89 | 0.00 | 9/9 | 0/9 |
| `dual2` | `worker_only` | 0.10 | 0.584 | 0.667 | 0.159 | 0.425 | 1.89 | 0.00 | 0/9 | 0/9 |
| `dual2` | `worker_only` | 0.25 | 0.459 | 0.667 | 0.159 | 0.301 | 1.89 | 0.00 | 0/9 | 0/9 |
| `dual2` | `coadapt` | 0.00 | 0.667 | 0.667 | 0.159 | 0.508 | 1.89 | 0.00 | 9/9 | 0/9 |
| `dual2` | `coadapt` | 0.10 | 0.584 | 0.667 | 0.159 | 0.425 | 1.89 | 0.00 | 0/9 | 0/9 |
| `dual2` | `coadapt` | 0.25 | 0.460 | 0.667 | 0.159 | 0.301 | 1.89 | 0.00 | 0/9 | 0/9 |
| `triple2` | `scratch` | 0.00 | 0.667 | 0.667 | 0.166 | 0.500 | 2.78 | 0.00 | 9/9 | 9/9 |
| `triple2` | `scratch` | 0.10 | 0.626 | 0.667 | 0.166 | 0.460 | 2.78 | 0.00 | 7/9 | 7/9 |
| `triple2` | `scratch` | 0.25 | 0.506 | 0.667 | 0.166 | 0.340 | 3.00 | 0.00 | 0/9 | 0/9 |
| `triple2` | `worker_only` | 0.00 | 0.667 | 0.667 | 0.179 | 0.488 | 2.89 | 0.00 | 9/9 | 9/9 |
| `triple2` | `worker_only` | 0.10 | 0.637 | 0.667 | 0.179 | 0.458 | 2.89 | 0.00 | 8/9 | 8/9 |
| `triple2` | `worker_only` | 0.25 | 0.531 | 0.667 | 0.179 | 0.352 | 2.89 | 0.00 | 0/9 | 0/9 |
| `triple2` | `coadapt` | 0.00 | 0.667 | 0.667 | 0.179 | 0.488 | 2.89 | 0.00 | 9/9 | 9/9 |
| `triple2` | `coadapt` | 0.10 | 0.637 | 0.667 | 0.179 | 0.458 | 2.89 | 0.00 | 8/9 | 8/9 |
| `triple2` | `coadapt` | 0.25 | 0.531 | 0.667 | 0.179 | 0.352 | 2.89 | 0.00 | 0/9 | 0/9 |
| `atomic8` | `scratch` | 0.00 | 0.667 | 0.667 | 0.147 | 0.520 | 1.00 | 0.00 | 9/9 | 0/9 |
| `atomic8` | `scratch` | 0.10 | 0.538 | 0.667 | 0.147 | 0.391 | 1.00 | 0.22 | 0/9 | 0/9 |
| `atomic8` | `scratch` | 0.25 | 0.319 | 0.452 | 0.147 | 0.172 | 0.89 | 0.89 | 0/9 | 0/9 |
| `atomic8` | `worker_only` | 0.00 | 0.667 | 0.667 | 0.139 | 0.527 | 1.00 | 0.00 | 9/9 | 0/9 |
| `atomic8` | `worker_only` | 0.10 | 0.539 | 0.667 | 0.139 | 0.400 | 1.00 | 0.00 | 0/9 | 0/9 |
| `atomic8` | `worker_only` | 0.25 | 0.393 | 0.667 | 0.139 | 0.254 | 1.00 | 0.00 | 0/9 | 0/9 |
| `atomic8` | `coadapt` | 0.00 | 0.667 | 0.667 | 0.139 | 0.527 | 1.00 | 0.00 | 9/9 | 0/9 |
| `atomic8` | `coadapt` | 0.10 | 0.539 | 0.667 | 0.139 | 0.400 | 1.00 | 0.00 | 0/9 | 0/9 |
| `atomic8` | `coadapt` | 0.25 | 0.393 | 0.667 | 0.139 | 0.254 | 1.00 | 0.00 | 0/9 | 0/9 |

| form | noise | adaptation | triple2−atomic8 natural |
|---|---:|---|---:|
| `triple2` vs `atomic8` | 0.00 | `scratch` | 0.000 [0.000,0.000] |
| `triple2` vs `atomic8` | 0.00 | `worker_only` | 0.000 [0.000,0.000] |
| `triple2` vs `atomic8` | 0.00 | `coadapt` | 0.000 [0.000,0.000] |
| `triple2` vs `atomic8` | 0.10 | `scratch` | 0.088 [0.070,0.106] |
| `triple2` vs `atomic8` | 0.10 | `worker_only` | 0.097 [0.081,0.114] |
| `triple2` vs `atomic8` | 0.10 | `coadapt` | 0.098 [0.081,0.114] |
| `triple2` vs `atomic8` | 0.25 | `scratch` | 0.188 [0.144,0.231] |
| `triple2` vs `atomic8` | 0.25 | `worker_only` | 0.137 [0.116,0.159] |
| `triple2` vs `atomic8` | 0.25 | `coadapt` | 0.138 [0.117,0.159] |

| form | noise | coadapt−worker_only new | coadapt−worker_only incumbent |
|---|---:|---:|---:|
| `dual2` | 0.00 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] |
| `dual2` | 0.10 | -0.000 [-0.001,0.000] | 0.000 [0.000,0.000] |
| `dual2` | 0.25 | 0.000 [-0.000,0.000] | 0.000 [0.000,0.000] |
| `triple2` | 0.00 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] |
| `triple2` | 0.10 | -0.000 [-0.000,0.000] | 0.000 [0.000,0.000] |
| `triple2` | 0.25 | -0.000 [-0.000,0.000] | 0.000 [0.000,0.000] |
| `atomic8` | 0.00 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] |
| `atomic8` | 0.10 | -0.000 [-0.002,0.001] | 0.000 [0.000,0.000] |
| `atomic8` | 0.25 | -0.001 [-0.002,0.001] | 0.000 [0.000,0.000] |

The analysis separates a code-space effect (`triple2` versus `atomic8`), a maintenance effect (`worker_only`) and a renegotiation effect (`coadapt`). It is a mechanism test, not a claim that any learned code is natural language.
