# Tabular signaling compact aggregation

- split: `training_support`
- runs: 144
- confidence interval: paired-by-seed Student-t 95%

| condition | natural | closed | permuted | Δ natural−closed | Δ natural−permuted | MI |
|---|---:|---:|---:|---:|---:|---:|
| `recurrent_abundant_FI_live_persistent` | 0.569 | 0.220 | 0.563 | 0.349 | 0.006 | 0.792 |
| `recurrent_abundant_FI_silent_persistent` | 0.571 | 0.571 | 0.571 | NA | NA | 0.000 |
| `recurrent_abundant_PI_live_persistent` | 0.571 | 0.223 | 0.309 | 0.348 | 0.262 | 0.751 |
| `recurrent_abundant_PI_silent_persistent` | 0.313 | 0.313 | 0.313 | NA | NA | 0.000 |
| `recurrent_scarce_FI_live_persistent` | 0.256 | 0.145 | 0.253 | 0.111 | 0.003 | 0.600 |
| `recurrent_scarce_FI_silent_persistent` | 0.265 | 0.265 | 0.265 | NA | NA | 0.000 |
| `recurrent_scarce_PI_live_persistent` | 0.208 | 0.111 | 0.208 | 0.097 | 0.000 | 0.621 |
| `recurrent_scarce_PI_live_switching` | 0.259 | 0.152 | 0.207 | 0.107 | 0.051 | 0.700 |
| `recurrent_scarce_PI_silent_persistent` | 0.208 | 0.208 | 0.208 | NA | NA | 0.000 |
| `recurrent_scarce_PI_silent_switching` | 0.209 | 0.209 | 0.209 | NA | NA | 0.000 |
| `stateless_abundant_FI_live_persistent` | 0.571 | 0.521 | 0.571 | 0.050 | 0.000 | 0.833 |
| `stateless_abundant_FI_silent_persistent` | 0.571 | 0.571 | 0.571 | NA | NA | 0.000 |
| `stateless_abundant_PI_live_persistent` | 0.313 | 0.297 | 0.313 | 0.016 | 0.000 | 0.792 |
| `stateless_abundant_PI_silent_persistent` | 0.313 | 0.313 | 0.313 | NA | NA | 0.000 |
| `stateless_scarce_FI_live_persistent` | 0.268 | 0.255 | 0.268 | 0.013 | -0.001 | 0.809 |
| `stateless_scarce_FI_silent_persistent` | 0.265 | 0.265 | 0.265 | NA | NA | 0.000 |
| `stateless_scarce_PI_live_persistent` | 0.210 | 0.224 | 0.210 | -0.014 | 0.000 | 0.897 |
| `stateless_scarce_PI_silent_persistent` | 0.208 | 0.208 | 0.208 | NA | NA | 0.000 |

The natural-minus-closed and natural-minus-permuted values are paired within seed. A nonzero token MI is descriptive and is not counted as causal evidence without the corresponding intervention.
