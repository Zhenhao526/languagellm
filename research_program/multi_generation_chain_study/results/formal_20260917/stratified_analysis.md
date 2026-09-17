# Three-generation chain stratification

| representation | generation | parent stratum | n | live held-out | live−silent | recombined gap | composable |
|---|---:|---|---:|---:|---:|---:|---:|
| `joint_history` | 1 | composable | 9 | 0.250 | +0.044 | +0.000 | 0/9 |
| `joint_history` | 1 | non-composable | 23 | 0.250 | +0.095 | +0.000 | 0/23 |
| `joint_history` | 2 | composable | 9 | 0.204 | +0.032 | +0.000 | 0/9 |
| `joint_history` | 2 | non-composable | 23 | 0.159 | +0.005 | +0.000 | 0/23 |
| `joint_history` | 3 | composable | 9 | 0.218 | +0.025 | +0.000 | 0/9 |
| `joint_history` | 3 | non-composable | 23 | 0.164 | +0.013 | +0.000 | 0/23 |
| `slot_local` | 1 | composable | 9 | 0.667 | +0.447 | +0.000 | 9/9 |
| `slot_local` | 1 | non-composable | 23 | 0.187 | +0.038 | +0.000 | 0/23 |
| `slot_local` | 2 | composable | 9 | 0.296 | +0.104 | +0.000 | 2/9 |
| `slot_local` | 2 | non-composable | 23 | 0.141 | -0.002 | +0.000 | 4/23 |
| `slot_local` | 3 | composable | 9 | 0.296 | +0.101 | +0.000 | 2/9 |
| `slot_local` | 3 | non-composable | 23 | 0.159 | +0.035 | +0.000 | 4/23 |


| representation | parent stratum | transition | False→False | False→True | True→False | True→True |
|---|---|---|---:|---:|---:|---:|
| `joint_history` | composable | g1→g2 | 9 | 0 | 0 | 0 |
| `joint_history` | composable | g2→g3 | 9 | 0 | 0 | 0 |
| `joint_history` | non-composable | g1→g2 | 23 | 0 | 0 | 0 |
| `joint_history` | non-composable | g2→g3 | 23 | 0 | 0 | 0 |
| `slot_local` | composable | g1→g2 | 0 | 0 | 7 | 2 |
| `slot_local` | composable | g2→g3 | 7 | 0 | 0 | 2 |
| `slot_local` | non-composable | g1→g2 | 19 | 4 | 0 | 0 |
| `slot_local` | non-composable | g2→g3 | 18 | 1 | 1 | 3 |
