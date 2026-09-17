# Incumbent-worker surface transfer

A parent learns a two-slot protocol on identity object labels. The sender is then frozen. An `incumbent` child copies parent worker 0; a `fresh` child receives a new worker. Each child faces identity or a stable swap of visible labels, with live and silent channels.

## Held-out transfer and repair

| initialization | mapping | channel | initial | final | repair gain | 95% CI final | functional |
|---|---|---|---:|---:|---:|---|---:|
| `fresh` | `identity` | `live` | 0.146 | 0.597 | 0.451 | [0.484, 0.710] | 7/9 |
| `fresh` | `identity` | `silent` | 0.165 | 0.319 | 0.154 | [0.193, 0.445] | 0/9 |
| `fresh` | `swap` | `live` | 0.216 | 0.609 | 0.393 | [0.518, 0.699] | 7/9 |
| `fresh` | `swap` | `silent` | 0.177 | 0.307 | 0.130 | [0.217, 0.397] | 0/9 |
| `incumbent` | `identity` | `live` | 0.586 | 0.574 | -0.012 | [0.433, 0.715] | 7/9 |
| `incumbent` | `identity` | `silent` | 0.139 | 0.249 | 0.111 | [0.085, 0.413] | 0/9 |
| `incumbent` | `swap` | `live` | -0.086 | 0.667 | 0.752 | [0.667, 0.667] | 9/9 |
| `incumbent` | `swap` | `silent` | 0.208 | 0.331 | 0.123 | [0.244, 0.418] | 0/9 |

## Communication check

The all-goal stream gives each hidden partner a complete block of the four goals. `natural−permuted` therefore measures whether the worker uses the sender message rather than a fixed action policy.

| initialization | mapping | channel | all natural | all natural−permuted | live−silent all (identity) |
|---|---|---|---:|---:|---:|
| `fresh` | `identity` | `live` | 0.621 | 0.579 | 0.372 |
| `fresh` | `identity` | `silent` | 0.249 | 0.000 | 0.372 |
| `fresh` | `swap` | `live` | 0.621 | 0.579 | 0.372 |
| `fresh` | `swap` | `silent` | 0.248 | 0.000 | 0.372 |
| `incumbent` | `identity` | `live` | 0.620 | 0.579 | 0.371 |
| `incumbent` | `identity` | `silent` | 0.250 | 0.000 | 0.371 |
| `incumbent` | `swap` | `live` | 0.620 | 0.579 | 0.371 |
| `incumbent` | `swap` | `silent` | 0.249 | 0.000 | 0.371 |

## Paired contrasts

| contrast | factors | mean difference | 95% CI | n |
|---|---|---:|---|---:|
| `initial_swap_minus_identity` | initialization=fresh, channel=live | 0.070 | [-0.069, 0.209] | 9 |
| `swap_minus_identity` | initialization=fresh, channel=live | 0.012 | [-0.051, 0.074] | 9 |
| `gain_swap_minus_identity` | initialization=fresh, channel=live | -0.058 | [-0.179, 0.063] | 9 |
| `live_minus_silent_all` | initialization=fresh, mapping=identity | 0.372 | [0.302, 0.442] | 9 |
| `communication_swap_minus_identity` | initialization=fresh, channel=live | 0.000 | [-0.000, 0.001] | 9 |
| `initial_swap_minus_identity` | initialization=fresh, channel=silent | 0.012 | [-0.129, 0.153] | 9 |
| `swap_minus_identity` | initialization=fresh, channel=silent | -0.012 | [-0.106, 0.081] | 9 |
| `gain_swap_minus_identity` | initialization=fresh, channel=silent | -0.024 | [-0.188, 0.139] | 9 |
| `live_minus_silent_all` | initialization=fresh, mapping=identity | 0.372 | [0.302, 0.442] | 9 |
| `communication_swap_minus_identity` | initialization=fresh, channel=live | 0.000 | [-0.000, 0.001] | 9 |
| `initial_swap_minus_identity` | initialization=incumbent, channel=live | -0.671 | [-0.922, -0.421] | 9 |
| `swap_minus_identity` | initialization=incumbent, channel=live | 0.093 | [-0.049, 0.234] | 9 |
| `gain_swap_minus_identity` | initialization=incumbent, channel=live | 0.764 | [0.651, 0.877] | 9 |
| `live_minus_silent_all` | initialization=incumbent, mapping=identity | 0.371 | [0.300, 0.441] | 9 |
| `communication_swap_minus_identity` | initialization=incumbent, channel=live | 0.000 | [0.000, 0.000] | 9 |
| `initial_swap_minus_identity` | initialization=incumbent, channel=silent | 0.069 | [-0.101, 0.239] | 9 |
| `swap_minus_identity` | initialization=incumbent, channel=silent | 0.082 | [-0.087, 0.251] | 9 |
| `gain_swap_minus_identity` | initialization=incumbent, channel=silent | 0.013 | [-0.081, 0.106] | 9 |
| `live_minus_silent_all` | initialization=incumbent, mapping=identity | 0.371 | [0.300, 0.441] | 9 |
| `communication_swap_minus_identity` | initialization=incumbent, channel=live | 0.000 | [0.000, 0.000] | 9 |
| `incumbent_minus_fresh_initial` | mapping=identity, channel=live | 0.440 | [0.331, 0.549] | 9 |
| `incumbent_minus_fresh_final` | mapping=identity, channel=live | -0.023 | [-0.077, 0.030] | 9 |
| `incumbent_minus_fresh_gain` | mapping=identity, channel=live | -0.463 | [-0.554, -0.372] | 9 |
| `incumbent_minus_fresh_initial` | mapping=identity, channel=silent | -0.026 | [-0.138, 0.086] | 9 |
| `incumbent_minus_fresh_final` | mapping=identity, channel=silent | -0.070 | [-0.176, 0.036] | 9 |
| `incumbent_minus_fresh_gain` | mapping=identity, channel=silent | -0.044 | [-0.180, 0.093] | 9 |
| `incumbent_minus_fresh_initial` | mapping=swap, channel=live | -0.301 | [-0.441, -0.162] | 9 |
| `incumbent_minus_fresh_final` | mapping=swap, channel=live | 0.058 | [-0.033, 0.149] | 9 |
| `incumbent_minus_fresh_gain` | mapping=swap, channel=live | 0.359 | [0.285, 0.434] | 9 |
| `incumbent_minus_fresh_initial` | mapping=swap, channel=silent | 0.031 | [-0.073, 0.135] | 9 |
| `incumbent_minus_fresh_final` | mapping=swap, channel=silent | 0.024 | [-0.107, 0.156] | 9 |
| `incumbent_minus_fresh_gain` | mapping=swap, channel=silent | -0.007 | [-0.123, 0.109] | 9 |

The incumbent/swap initial contrast is the zero-shot cost of changing visible object labels after a protocol has been learned. The final contrast and repair gain measure reward-based local adaptation; fresh children separate this from ordinary child learning.
