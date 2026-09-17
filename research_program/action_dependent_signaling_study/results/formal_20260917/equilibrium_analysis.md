# Action-dependent equilibrium analysis

- runs: 256
- composable-equilibrium rule: natural ≥ 0.60 and |recombined−natural| ≤ 0.02

| task | protocol | composable seeds | rate | Wilson 95% | natural | recombined−natural |
|---|---|---:|---:|---|---:|---:|
| factorized | simultaneous | 3/32 | 0.094 | [0.032, 0.242] | 0.494 | -0.150 |
| factorized | staged | 9/32 | 0.281 | [0.156, 0.454] | 0.517 | -0.135 |
| entangled | simultaneous | 0/32 | 0.000 | [0.000, 0.107] | 0.471 | -0.137 |
| entangled | staged | 10/32 | 0.312 | [0.180, 0.486] | 0.523 | -0.111 |

The seed-level equilibrium flag is a descriptive balance-selection readout, not a test of zero-shot compositional generalization.

- factorized: staged−simultaneous natural = +0.022 [-0.016, +0.060]; McNemar staged-only/simultaneous-only = 7/1, exact two-sided p=0.0703.
- entangled: staged−simultaneous natural = +0.052 [+0.016, +0.088]; McNemar staged-only/simultaneous-only = 10/0, exact two-sided p=0.0020.
