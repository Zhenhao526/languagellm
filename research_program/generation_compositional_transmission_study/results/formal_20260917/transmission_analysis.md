# Stratified compositional transmission analysis

- parent composable: 9/32
- child rule: natural ≥ 0.60 and |recombined−natural| ≤ 0.02

| role | parent group | n | child live composable | live natural | live recombined−natural | sender sequence exact |
|---|---|---:|---:|---:|---:|---:|
| `worker` | `all` | 32 | 9/32 | 0.524 | -0.143 | 32/32 |
| `worker` | `parent_composable` | 9 | 9/9 | 0.667 | 0.000 | 9/9 |
| `worker` | `parent_noncomposable` | 23 | 0/23 | 0.468 | -0.199 | 23/23 |
| `sender` | `all` | 32 | 9/32 | 0.517 | -0.135 | 32/32 |
| `sender` | `parent_composable` | 9 | 9/9 | 0.667 | 0.000 | 9/9 |
| `sender` | `parent_noncomposable` | 23 | 0/23 | 0.458 | -0.188 | 23/23 |


| role | parent group | live−silent | live−permuted |
|---|---|---:|---:|
| `worker` | `all` | 0.357 [0.311,0.402] | 0.298 [0.251,0.345] |
| `worker` | `parent_composable` | 0.499 [0.441,0.556] | 0.419 [0.350,0.488] |
| `worker` | `parent_noncomposable` | 0.301 [0.262,0.340] | 0.251 [0.204,0.297] |
| `sender` | `all` | 0.285 [0.229,0.342] | 0.372 [0.299,0.446] |
| `sender` | `parent_composable` | 0.393 [0.300,0.486] | 0.613 [0.539,0.688] |
| `sender` | `parent_noncomposable` | 0.243 [0.181,0.306] | 0.278 [0.217,0.339] |


The parent stratum is fixed by the parent endpoint before child training. The analysis therefore separates recovery of a compositional parent protocol from recovery of a degenerate whole-code protocol.
