# Action-dependent equilibrium analysis

- runs: 256
- composable-equilibrium rule: natural ≥ 0.60 and |recombined−natural| ≤ 0.02

| task | protocol | composable seeds | rate | Wilson 95% | natural | recombined−natural |
|---|---|---:|---:|---|---:|---:|
| `factorized` | `simultaneous` | 3/32 | 0.094 | [0.032, 0.242] | 0.494 | -0.150 |
| `factorized` | `staged` | 9/32 | 0.281 | [0.156, 0.454] | 0.517 | -0.135 |
| `entangled` | `simultaneous` | 0/32 | 0.000 | [0.000, 0.107] | 0.471 | -0.137 |
| `entangled` | `staged` | 10/32 | 0.312 | [0.180, 0.486] | 0.523 | -0.111 |


| task | staged−sim natural | staged−sim recombination gap | McNemar staged-only / simultaneous-only | exact p |
|---|---:|---:|---:|---:|
| `factorized` | 0.022 [-0.016,0.060] | 0.015 | 7 / 1 | 0.0703 |
| `entangled` | 0.052 [0.016,0.088] | 0.027 | 10 / 0 | 0.0020 |


The equilibrium rule was fixed before the additional 16 seeds. These counts describe convergence into a high-performing, recombination-preserving basin; they are not a claim that every staged run develops a language.
