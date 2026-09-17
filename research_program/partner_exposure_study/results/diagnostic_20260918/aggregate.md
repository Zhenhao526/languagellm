# Multi-partner sender alignment under perceptual heterogeneity

A parent population first learns a hidden-partner protocol. A fresh sender is then inserted while the four workers are retained. Every child batch exposes that sender to all four rotating partners; `hidden` and `visible` control whether the sender can condition on partner identity, while `sender_only` and `coadapt` control whether workers are frozen or updated together. `leave_one_out` masks one joint goal during adaptation.

## Parent endpoint

| population | all natural | all natural−permuted | sender consistency |
|---|---:|---:|---:|
| `heterogeneous` | 0.361 | 0.143 | 1.000 |
| `homogeneous` | 0.341 | 0.089 | 1.000 |
| `heterogeneous` | 0.294 | 0.051 | 1.000 |
| `homogeneous` | 0.328 | 0.157 | 1.000 |

Parent contrasts are paired by seed.

| contrast | factors | mean | 95% CI | n |
|---|---|---:|---|---:|
| `heterogeneous_minus_homogeneous_parent` | value=all_natural | -0.006 | [-0.069, 0.056] | 2 |
| `heterogeneous_minus_homogeneous_parent` | value=all_natural_minus_permuted | -0.026 | [-0.210, 0.158] | 2 |
| `heterogeneous_minus_homogeneous_parent` | value=sender_consistency | 0.000 | [0.000, 0.000] | 2 |

## Fresh sender endpoint

| population | visibility | adaptation | support | initial | final | gain | silent | natural−permuted | consistency | functional |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `homogeneous` | `hidden` | `sender_only` | `leave_one_out` | 0.303 | 0.303 | 0.000 | 0.233 | 0.000 | 1.000 | 0/2 |
| `homogeneous` | `visible` | `sender_only` | `leave_one_out` | 0.226 | 0.226 | 0.000 | 0.233 | 0.000 | 0.458 | 0/2 |
| `heterogeneous` | `hidden` | `sender_only` | `leave_one_out` | 0.382 | 0.382 | 0.000 | 0.103 | 0.000 | 1.000 | 0/2 |
| `heterogeneous` | `hidden` | `coadapt` | `leave_one_out` | 0.382 | -0.140 | -0.522 | 0.103 | 0.000 | 1.000 | 0/2 |
| `heterogeneous` | `visible` | `sender_only` | `leave_one_out` | 0.200 | 0.200 | 0.000 | 0.103 | 0.000 | 0.438 | 0/2 |

## Paired contrasts

| contrast | factors | mean | 95% CI | n |
|---|---|---:|---|---:|
| `heterogeneous_minus_homogeneous` | value=heldout_natural, visibility=hidden, adaptation=sender_only, support=leave_one_out | 0.078 | [-0.221, 0.378] | 2 |
| `heterogeneous_minus_homogeneous` | value=sender_consistency, visibility=hidden, adaptation=sender_only, support=leave_one_out | 0.000 | [0.000, 0.000] | 2 |
| `heterogeneous_minus_homogeneous` | value=all_natural_minus_permuted, visibility=hidden, adaptation=sender_only, support=leave_one_out | -0.059 | [-0.078, -0.041] | 2 |
| `heterogeneous_minus_homogeneous` | value=heldout_natural, visibility=visible, adaptation=sender_only, support=leave_one_out | -0.026 | [-0.087, 0.036] | 2 |
| `heterogeneous_minus_homogeneous` | value=sender_consistency, visibility=visible, adaptation=sender_only, support=leave_one_out | -0.021 | [-0.069, 0.027] | 2 |
| `heterogeneous_minus_homogeneous` | value=all_natural_minus_permuted, visibility=visible, adaptation=sender_only, support=leave_one_out | -0.098 | [-0.297, 0.101] | 2 |
| `visible_minus_hidden` | value=heldout_natural, population_mode=homogeneous, adaptation=sender_only, support=leave_one_out | -0.078 | [-0.141, -0.015] | 2 |
| `visible_minus_hidden` | value=sender_consistency, population_mode=homogeneous, adaptation=sender_only, support=leave_one_out | -0.542 | [-0.926, -0.157] | 2 |
| `visible_minus_hidden` | value=heldout_natural, population_mode=heterogeneous, adaptation=sender_only, support=leave_one_out | -0.182 | [-0.480, 0.117] | 2 |
| `visible_minus_hidden` | value=sender_consistency, population_mode=heterogeneous, adaptation=sender_only, support=leave_one_out | -0.562 | [-0.899, -0.226] | 2 |
| `coadapt_minus_sender_only` | value=heldout_natural, population_mode=heterogeneous, visibility=hidden, support=leave_one_out | -0.522 | [-0.818, -0.226] | 2 |
| `coadapt_minus_sender_only` | value=sender_consistency, population_mode=heterogeneous, visibility=hidden, support=leave_one_out | 0.000 | [0.000, 0.000] | 2 |

The central test is whether exposure to multiple perceptual conventions changes a fresh sender's ability to recover a reusable message code. A hidden sender must use one codebook across all partners; a visible sender can maintain partner-specific mappings. This is a protocol-alignment mechanism test, not evidence of human syntax or open-ended language.
