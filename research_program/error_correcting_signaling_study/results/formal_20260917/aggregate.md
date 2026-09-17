# Emergent redundancy and noisy protocol repair

`triple2` and `atomic8` each have eight raw message states; `dual2` has four. The atomic corruption probability is matched to the probability that at least one of three binary coordinates flips.

| form | adaptation | noise | natural | clean | silent | live−silent | min Hamming | functional | error-correcting candidate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dual2` | `scratch` | 0.00 | 0.457 | 0.457 | 0.180 | 0.277 | 0.00 | 0/9 | 0/9 |
| `dual2` | `scratch` | 0.10 | 0.414 | 0.456 | 0.180 | 0.234 | 0.00 | 0/9 | 0/9 |
| `dual2` | `scratch` | 0.25 | 0.357 | 0.466 | 0.180 | 0.177 | 0.00 | 0/9 | 0/9 |
| `dual2` | `worker_only` | 0.00 | 0.458 | 0.458 | 0.205 | 0.253 | 0.00 | 0/9 | 0/9 |
| `dual2` | `worker_only` | 0.10 | 0.415 | 0.458 | 0.205 | 0.211 | 0.00 | 0/9 | 0/9 |
| `dual2` | `worker_only` | 0.25 | 0.352 | 0.457 | 0.205 | 0.147 | 0.00 | 0/9 | 0/9 |
| `dual2` | `coadapt` | 0.00 | 0.457 | 0.457 | 0.205 | 0.253 | 0.00 | 0/9 | 0/9 |
| `dual2` | `coadapt` | 0.10 | 0.416 | 0.458 | 0.205 | 0.212 | 0.00 | 0/9 | 0/9 |
| `dual2` | `coadapt` | 0.25 | 0.352 | 0.458 | 0.205 | 0.147 | 0.00 | 0/9 | 0/9 |
| `triple2` | `scratch` | 0.00 | 0.504 | 0.504 | 0.161 | 0.343 | 0.11 | 1/9 | 0/9 |
| `triple2` | `scratch` | 0.10 | 0.462 | 0.492 | 0.161 | 0.301 | 0.11 | 0/9 | 0/9 |
| `triple2` | `scratch` | 0.25 | 0.395 | 0.522 | 0.161 | 0.235 | 0.11 | 0/9 | 0/9 |
| `triple2` | `worker_only` | 0.00 | 0.481 | 0.481 | 0.177 | 0.304 | 0.11 | 1/9 | 0/9 |
| `triple2` | `worker_only` | 0.10 | 0.454 | 0.481 | 0.177 | 0.277 | 0.11 | 0/9 | 0/9 |
| `triple2` | `worker_only` | 0.25 | 0.391 | 0.480 | 0.177 | 0.214 | 0.11 | 0/9 | 0/9 |
| `triple2` | `coadapt` | 0.00 | 0.481 | 0.481 | 0.177 | 0.304 | 0.11 | 1/9 | 0/9 |
| `triple2` | `coadapt` | 0.10 | 0.455 | 0.481 | 0.177 | 0.278 | 0.11 | 0/9 | 0/9 |
| `triple2` | `coadapt` | 0.25 | 0.392 | 0.481 | 0.177 | 0.215 | 0.11 | 0/9 | 0/9 |
| `atomic8` | `scratch` | 0.00 | 0.503 | 0.503 | 0.160 | 0.343 | 0.00 | 0/9 | 0/9 |
| `atomic8` | `scratch` | 0.10 | 0.442 | 0.530 | 0.160 | 0.282 | 0.00 | 0/9 | 0/9 |
| `atomic8` | `scratch` | 0.25 | 0.305 | 0.413 | 0.160 | 0.145 | 0.44 | 0/9 | 0/9 |
| `atomic8` | `worker_only` | 0.00 | 0.551 | 0.551 | 0.161 | 0.389 | 0.11 | 1/9 | 0/9 |
| `atomic8` | `worker_only` | 0.10 | 0.456 | 0.551 | 0.161 | 0.295 | 0.11 | 0/9 | 0/9 |
| `atomic8` | `worker_only` | 0.25 | 0.350 | 0.545 | 0.161 | 0.188 | 0.11 | 0/9 | 0/9 |
| `atomic8` | `coadapt` | 0.00 | 0.505 | 0.505 | 0.161 | 0.344 | 0.00 | 0/9 | 0/9 |
| `atomic8` | `coadapt` | 0.10 | 0.426 | 0.505 | 0.161 | 0.264 | 0.00 | 0/9 | 0/9 |
| `atomic8` | `coadapt` | 0.25 | 0.337 | 0.505 | 0.161 | 0.176 | 0.11 | 0/9 | 0/9 |

| form | noise | adaptation | triple2−atomic8 natural |
|---|---:|---|---:|
| `triple2` vs `atomic8` | 0.00 | `scratch` | 0.000 [-0.071,0.072] |
| `triple2` vs `atomic8` | 0.00 | `worker_only` | -0.070 [-0.152,0.012] |
| `triple2` vs `atomic8` | 0.00 | `coadapt` | -0.024 [-0.095,0.048] |
| `triple2` vs `atomic8` | 0.10 | `scratch` | 0.019 [-0.016,0.055] |
| `triple2` vs `atomic8` | 0.10 | `worker_only` | -0.002 [-0.058,0.054] |
| `triple2` vs `atomic8` | 0.10 | `coadapt` | 0.029 [-0.021,0.080] |
| `triple2` vs `atomic8` | 0.25 | `scratch` | 0.090 [0.056,0.125] |
| `triple2` vs `atomic8` | 0.25 | `worker_only` | 0.042 [0.012,0.072] |
| `triple2` vs `atomic8` | 0.25 | `coadapt` | 0.055 [0.024,0.086] |

| form | noise | coadapt−worker_only new | coadapt−worker_only incumbent |
|---|---:|---:|---:|
| `dual2` | 0.00 | -0.000 [-0.001,0.000] | 0.000 [0.000,0.000] |
| `dual2` | 0.10 | 0.001 [-0.000,0.002] | 0.000 [0.000,0.000] |
| `dual2` | 0.25 | 0.000 [-0.001,0.002] | 0.000 [0.000,0.000] |
| `triple2` | 0.00 | 0.000 [-0.001,0.001] | 0.000 [0.000,0.000] |
| `triple2` | 0.10 | 0.001 [0.000,0.001] | 0.000 [0.000,0.000] |
| `triple2` | 0.25 | 0.001 [-0.001,0.002] | 0.000 [0.000,0.000] |
| `atomic8` | 0.00 | -0.046 [-0.085,-0.007] | -0.044 [-0.081,-0.007] |
| `atomic8` | 0.10 | -0.031 [-0.057,-0.004] | -0.031 [-0.057,-0.005] |
| `atomic8` | 0.25 | -0.013 [-0.028,0.002] | -0.015 [-0.028,-0.002] |

The analysis separates a code-space effect (`triple2` versus `atomic8`), a maintenance effect (`worker_only`) and a renegotiation effect (`coadapt`). It is a mechanism test, not a claim that any learned code is natural language.
