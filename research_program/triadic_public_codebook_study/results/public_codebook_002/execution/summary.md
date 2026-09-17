# Public/private codebook pilot summary

Plan hash: `999d5b65e3b2f2a4bbf326939b469f31d2179f19cff91095a7b966923a176560`

## Endpoint and AUC contrasts

| contrast | endpoint Q mean [95% CI] | Q AUC mean [95% CI] | endpoint reward mean | endpoint execution mean |
|---|---:|---:|---:|---:|
| identity_minus_silent | 5.33 pp [2.64, 8.02] | 4.73 pp [1.61, 7.85] | 24.11 pp | 38.21 pp |
| public_minus_silent | 6.41 pp [4.28, 8.55] | 5.01 pp [2.58, 7.44] | 24.57 pp | 39.32 pp |
| private_minus_silent | 4.71 pp [2.27, 7.16] | 4.56 pp [2.06, 7.06] | 22.69 pp | 37.03 pp |
| public_minus_identity | 1.08 pp [-0.16, 2.33] | 0.28 pp [-0.59, 1.15] | 0.46 pp | 1.11 pp |
| private_minus_identity | -0.62 pp [-3.23, 2.00] | -0.17 pp [-2.60, 2.25] | -1.42 pp | -1.18 pp |
| public_minus_private | 1.70 pp [-0.50, 3.90] | 0.45 pp [-1.40, 2.30] | 1.88 pp | 2.29 pp |

## Endpoint Q by condition

| condition | Q mean [95% CI] | reward mean | execution mean | packet entropy bits |
|---|---:|---:|---:|---:|
| silent | 5.62 pp [4.47, 6.77] | 0.272 | 0.494 | 0.365 |
| identity_live | 10.95 pp [8.56, 13.34] | 0.513 | 0.876 | 1.703 |
| public_live | 12.04 pp [10.46, 13.61] | 0.518 | 0.887 | 1.623 |
| private_live | 10.33 pp [8.16, 12.51] | 0.499 | 0.864 | 1.764 |

The pilot uses eight paired initialization blocks. Intervals are descriptive across seeds; they do not establish a language-formation claim. Saved compact files retain the full partition's messages, wire messages, actions, and exact conditional probability vectors; `audit.py` replayed all 288 files with zero numerical error.
