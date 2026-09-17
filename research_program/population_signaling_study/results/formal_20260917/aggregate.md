# Population signaling compact aggregation

- runs: 128
- split: heldout

| condition | active workers | natural | closed | permuted | Δ live−silent |
|---|---:|---:|---:|---:|---:|
| `fixed_hidden_live_abundant` | 8 | 0.800 | 0.211 | 0.298 | 0.500 |
| `fixed_hidden_live_scarce` | 8 | 0.384 | 0.205 | 0.133 | 0.084 |
| `fixed_hidden_silent_abundant` | 8 | 0.300 | 0.300 | 0.300 | NA |
| `fixed_hidden_silent_scarce` | 8 | 0.300 | 0.300 | 0.300 | NA |
| `fixed_visible_live_abundant` | 8 | 0.800 | 0.211 | 0.298 | 0.500 |
| `fixed_visible_live_scarce` | 8 | 0.384 | 0.205 | 0.133 | 0.084 |
| `fixed_visible_silent_abundant` | 8 | 0.300 | 0.300 | 0.300 | NA |
| `fixed_visible_silent_scarce` | 8 | 0.300 | 0.300 | 0.300 | NA |
| `rotating_hidden_live_abundant` | 32 | 0.800 | 0.201 | 0.300 | 0.501 |
| `rotating_hidden_live_scarce` | 32 | 0.307 | 0.182 | 0.212 | 0.007 |
| `rotating_hidden_silent_abundant` | 32 | 0.299 | 0.299 | 0.299 | NA |
| `rotating_hidden_silent_scarce` | 32 | 0.300 | 0.300 | 0.300 | NA |
| `rotating_visible_live_abundant` | 32 | 0.755 | 0.201 | 0.296 | 0.456 |
| `rotating_visible_live_scarce` | 32 | 0.300 | 0.182 | 0.296 | 0.000 |
| `rotating_visible_silent_abundant` | 32 | 0.299 | 0.299 | 0.299 | NA |
| `rotating_visible_silent_scarce` | 32 | 0.300 | 0.300 | 0.300 | NA |

Natural-minus-closed and natural-minus-permuted are within-run intervention effects; natural-minus-silent compares a live run with a from-scratch silent run. The partner alignment fields are descriptive readouts from final checkpoints.

## Codebook readout

| condition | sender-token agreement | semantic success mean | semantic success minimum |
|---|---:|---:|---:|
| `fixed_hidden_live_abundant` | 1.000 | 0.531 | 0.000 |
| `fixed_hidden_live_scarce` | 1.000 | 0.547 | 0.000 |
| `fixed_hidden_silent_abundant` | 1.000 | 0.320 | 0.000 |
| `fixed_hidden_silent_scarce` | 1.000 | 0.312 | 0.000 |
| `fixed_visible_live_abundant` | 0.781 | 0.477 | 0.000 |
| `fixed_visible_live_scarce` | 0.766 | 0.477 | 0.000 |
| `fixed_visible_silent_abundant` | 1.000 | 0.320 | 0.000 |
| `fixed_visible_silent_scarce` | 1.000 | 0.312 | 0.000 |
| `rotating_hidden_live_abundant` | 1.000 | 1.000 | 1.000 |
| `rotating_hidden_live_scarce` | 1.000 | 1.000 | 1.000 |
| `rotating_hidden_silent_abundant` | 1.000 | 0.320 | 0.000 |
| `rotating_hidden_silent_scarce` | 1.000 | 0.312 | 0.000 |
| `rotating_visible_live_abundant` | 0.516 | 1.000 | 1.000 |
| `rotating_visible_live_scarce` | 0.484 | 0.969 | 0.812 |
| `rotating_visible_silent_abundant` | 1.000 | 0.320 | 0.000 |
| `rotating_visible_silent_scarce` | 1.000 | 0.312 | 0.000 |