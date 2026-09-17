# Tabular signaling compact aggregation

- split: `heldout`
- runs: 144
- confidence interval: paired-by-seed Student-t 95%

| condition | natural | closed | permuted | Δ natural−closed | Δ natural−permuted | MI |
|---|---:|---:|---:|---:|---:|---:|
| `recurrent_abundant_FI_live_persistent` | 0.573 | 0.220 | 0.567 | 0.353 | 0.006 | 0.794 |
| `recurrent_abundant_FI_silent_persistent` | 0.575 | 0.575 | 0.575 | NA | NA | 0.000 |
| `recurrent_abundant_PI_live_persistent` | 0.574 | 0.223 | 0.312 | 0.351 | 0.263 | 0.750 |
| `recurrent_abundant_PI_silent_persistent` | 0.315 | 0.315 | 0.315 | NA | NA | 0.000 |
| `recurrent_scarce_FI_live_persistent` | 0.258 | 0.146 | 0.254 | 0.112 | 0.003 | 0.603 |
| `recurrent_scarce_FI_silent_persistent` | 0.267 | 0.267 | 0.267 | NA | NA | 0.000 |
| `recurrent_scarce_PI_live_persistent` | 0.210 | 0.113 | 0.210 | 0.097 | 0.000 | 0.621 |
| `recurrent_scarce_PI_live_switching` | 0.202 | 0.153 | 0.209 | 0.049 | -0.007 | 0.875 |
| `recurrent_scarce_PI_silent_persistent` | 0.210 | 0.210 | 0.210 | NA | NA | 0.000 |
| `recurrent_scarce_PI_silent_switching` | 0.209 | 0.209 | 0.209 | NA | NA | 0.000 |
| `stateless_abundant_FI_live_persistent` | 0.575 | 0.524 | 0.575 | 0.052 | 0.000 | 0.830 |
| `stateless_abundant_FI_silent_persistent` | 0.575 | 0.575 | 0.575 | NA | NA | 0.000 |
| `stateless_abundant_PI_live_persistent` | 0.314 | 0.299 | 0.314 | 0.015 | 0.000 | 0.790 |
| `stateless_abundant_PI_silent_persistent` | 0.315 | 0.315 | 0.315 | NA | NA | 0.000 |
| `stateless_scarce_FI_live_persistent` | 0.269 | 0.256 | 0.270 | 0.013 | -0.000 | 0.808 |
| `stateless_scarce_FI_silent_persistent` | 0.267 | 0.267 | 0.267 | NA | NA | 0.000 |
| `stateless_scarce_PI_live_persistent` | 0.211 | 0.226 | 0.211 | -0.014 | 0.000 | 0.895 |
| `stateless_scarce_PI_silent_persistent` | 0.210 | 0.210 | 0.210 | NA | NA | 0.000 |

The natural-minus-closed and natural-minus-permuted values are paired within seed. A nonzero token MI is descriptive and is not counted as causal evidence without the corresponding intervention.
