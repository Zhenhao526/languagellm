# Receiver initialization × surface mapping: supplemental analysis

This paired analysis reads the frozen 288-row transfer result table. It uses a two-sided Student t interval with 8 degrees of freedom across nine seed-level contrasts.

## Receiver-only replacement: held-out double-new endpoint

| Architecture | Mapping | Initialization | Initial | Final | Repair gain | Functional final | Recombined final |
|---|---|---|---:|---:|---:|---:|---:|
| factorized | identity | fresh | -0.176 | 1.000 | 1.176 | 9/9 | 1.000 |
| factorized | identity | incumbent_receiver | 1.000 | 1.000 | 0.000 | 9/9 | 1.000 |
| factorized | reverse | fresh | -0.122 | 1.000 | 1.122 | 9/9 | 1.000 |
| factorized | reverse | incumbent_receiver | -0.250 | 1.000 | 1.250 | 9/9 | 1.000 |
| holistic | identity | fresh | 0.196 | -0.250 | -0.446 | 0/9 | -0.250 |
| holistic | identity | incumbent_receiver | -0.250 | -0.250 | 0.000 | 0/9 | -0.250 |
| holistic | reverse | fresh | -0.143 | -0.250 | -0.107 | 0/9 | -0.250 |
| holistic | reverse | incumbent_receiver | -0.250 | -0.250 | 0.000 | 0/9 | -0.250 |

## Paired contrasts

`surface-by-initialization-interaction` is (reverse−identity among incumbent receivers) minus (reverse−identity among fresh receivers), paired by seed.

| Contrast | Mean | Paired 95% t interval |
|---|---:|---:|
| `factorized/incumbent_receiver:reverse-minus-identity/initial:new_double` | -1.250 | [-1.250, -1.250] |
| `factorized/surface-by-initialization-interaction/initial:new_double` | -1.304 | [-1.663, -0.945] |
| `factorized/incumbent_receiver:reverse-minus-identity/final:new_double` | 0.000 | [0.000, 0.000] |
| `factorized/surface-by-initialization-interaction/final:new_double` | 0.000 | [0.000, 0.000] |
| `holistic/incumbent_receiver:reverse-minus-identity/initial:new_double` | 0.000 | [0.000, 0.000] |
| `holistic/surface-by-initialization-interaction/initial:new_double` | 0.340 | [-0.043, 0.722] |
| `holistic/incumbent_receiver:reverse-minus-identity/final:new_double` | 0.000 | [0.000, 0.000] |
| `holistic/surface-by-initialization-interaction/final:new_double` | 0.000 | [0.000, 0.000] |

## Interpretation boundary

The mapping intervention permutes tabular attribute labels; it does not use images or a learned visual encoder. The factorized result concerns whether a copied receiver can re-ground a compositional message protocol after a stable label change. It is not evidence that agents invented natural language.
