# Ternary attribute and message-capacity extension

The live/silent contrast is paired by seed. `mono9` has the same 9-symbol capacity as two ternary slots but no slot recombination readout; `tri3` exposes two 3-valued slots.

| form | task | protocol | live natural | silent natural | live−silent | recombined−natural | composable |
|---|---|---|---:|---:|---:|---:|---:|
| `mono9` | `factorized` | `simultaneous` | 0.148 | 0.110 | 0.038 [0.031,0.046] | n/a | 0/9 |
| `mono9` | `factorized` | `staged` | 0.143 | 0.110 | 0.033 [0.026,0.039] | n/a | 0/9 |
| `mono9` | `entangled` | `simultaneous` | 0.153 | 0.111 | 0.042 [0.027,0.058] | n/a | 0/9 |
| `mono9` | `entangled` | `staged` | 0.146 | 0.111 | 0.035 [0.021,0.049] | n/a | 0/9 |
| `tri3` | `factorized` | `simultaneous` | 0.182 | 0.111 | 0.071 [0.055,0.087] | -0.026 | 0/9 |
| `tri3` | `factorized` | `staged` | 0.193 | 0.111 | 0.081 [0.071,0.091] | -0.014 | 0/9 |
| `tri3` | `entangled` | `simultaneous` | 0.180 | 0.111 | 0.069 [0.054,0.085] | -0.034 | 0/9 |
| `tri3` | `entangled` | `staged` | 0.177 | 0.111 | 0.066 [0.047,0.085] | -0.016 | 0/9 |

| form | task | staged−simultaneous natural |
|---|---|---:|
| `mono9` | `factorized` | -0.005 [-0.012,0.001] |
| `mono9` | `entangled` | -0.008 [-0.011,-0.005] |
| `tri3` | `factorized` | 0.010 [-0.003,0.023] |
| `tri3` | `entangled` | -0.003 [-0.027,0.021] |

The ternary extension tests scaling to three-valued attributes and distinguishes a capacity-matched atomic token from an exposed slot structure.
