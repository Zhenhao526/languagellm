# Multi-partner sender alignment under perceptual heterogeneity

A parent population first learns a hidden-partner protocol. A fresh sender is then inserted while the four workers are retained. Every child batch exposes that sender to all four rotating partners; `hidden` and `visible` control whether the sender can condition on partner identity, while `sender_only` and `coadapt` control whether workers are frozen or updated together. `leave_one_out` masks one joint goal during adaptation.

## Parent endpoint

| population | all natural | all natural−permuted | sender consistency |
|---|---:|---:|---:|
| `heterogeneous` | 0.667 | 0.625 | 1.000 |
| `homogeneous` | 0.667 | 0.625 | 1.000 |
| `heterogeneous` | 0.458 | 0.415 | 1.000 |
| `homogeneous` | 0.458 | 0.417 | 1.000 |
| `heterogeneous` | 0.457 | 0.415 | 1.000 |
| `homogeneous` | 0.458 | 0.417 | 1.000 |
| `heterogeneous` | 0.667 | 0.625 | 1.000 |
| `homogeneous` | 0.458 | 0.417 | 1.000 |
| `heterogeneous` | 0.667 | 0.625 | 1.000 |
| `homogeneous` | 0.458 | 0.417 | 1.000 |
| `heterogeneous` | 0.459 | 0.417 | 1.000 |
| `homogeneous` | 0.458 | 0.417 | 1.000 |
| `heterogeneous` | 0.458 | 0.417 | 1.000 |
| `homogeneous` | 0.667 | 0.625 | 1.000 |
| `heterogeneous` | 0.458 | 0.417 | 1.000 |
| `homogeneous` | 0.667 | 0.625 | 1.000 |
| `heterogeneous` | 0.459 | 0.417 | 1.000 |
| `homogeneous` | 0.459 | 0.417 | 1.000 |

Parent contrasts are paired by seed.

| contrast | factors | mean | 95% CI | n |
|---|---|---:|---|---:|
| `heterogeneous_minus_homogeneous_parent` | value=all_natural | -0.000 | [-0.113, 0.113] | 9 |
| `heterogeneous_minus_homogeneous_parent` | value=all_natural_minus_permuted | -0.000 | [-0.114, 0.113] | 9 |
| `heterogeneous_minus_homogeneous_parent` | value=sender_consistency | 0.000 | [0.000, 0.000] | 9 |

## Fresh sender endpoint

| population | visibility | adaptation | support | initial | final | gain | silent | natural−permuted | consistency | functional |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `homogeneous` | `hidden` | `sender_only` | `full` | 0.160 | 0.513 | 0.353 | 0.190 | 0.000 | 1.000 | 5/9 |
| `homogeneous` | `hidden` | `sender_only` | `leave_one_out` | 0.160 | 0.160 | 0.000 | 0.190 | 0.000 | 1.000 | 1/9 |
| `homogeneous` | `hidden` | `coadapt` | `full` | 0.160 | 0.510 | 0.350 | 0.190 | 0.000 | 1.000 | 4/9 |
| `homogeneous` | `hidden` | `coadapt` | `leave_one_out` | 0.160 | 0.111 | -0.049 | 0.190 | 0.000 | 1.000 | 0/9 |
| `homogeneous` | `visible` | `sender_only` | `full` | 0.212 | 0.513 | 0.301 | 0.190 | 0.000 | 1.000 | 5/9 |
| `homogeneous` | `visible` | `sender_only` | `leave_one_out` | 0.212 | 0.212 | 0.000 | 0.190 | 0.000 | 0.824 | 0/9 |
| `homogeneous` | `visible` | `coadapt` | `full` | 0.212 | 0.516 | 0.304 | 0.190 | 0.000 | 1.000 | 6/9 |
| `homogeneous` | `visible` | `coadapt` | `leave_one_out` | 0.212 | 0.192 | -0.020 | 0.190 | 0.000 | 0.824 | 0/9 |
| `heterogeneous` | `hidden` | `sender_only` | `full` | 0.166 | 0.559 | 0.394 | 0.121 | 0.000 | 1.000 | 5/9 |
| `heterogeneous` | `hidden` | `sender_only` | `leave_one_out` | 0.166 | 0.166 | 0.000 | 0.121 | 0.000 | 1.000 | 1/9 |
| `heterogeneous` | `hidden` | `coadapt` | `full` | 0.166 | 0.557 | 0.391 | 0.121 | 0.000 | 1.000 | 5/9 |
| `heterogeneous` | `hidden` | `coadapt` | `leave_one_out` | 0.166 | 0.157 | -0.009 | 0.121 | 0.000 | 1.000 | 1/9 |
| `heterogeneous` | `visible` | `sender_only` | `full` | 0.227 | 0.559 | 0.333 | 0.121 | 0.000 | 1.000 | 5/9 |
| `heterogeneous` | `visible` | `sender_only` | `leave_one_out` | 0.227 | 0.227 | 0.000 | 0.121 | 0.000 | 0.824 | 0/9 |
| `heterogeneous` | `visible` | `coadapt` | `full` | 0.227 | 0.565 | 0.339 | 0.121 | 0.000 | 1.000 | 5/9 |
| `heterogeneous` | `visible` | `coadapt` | `leave_one_out` | 0.227 | 0.181 | -0.046 | 0.121 | 0.000 | 0.824 | 0/9 |

## Paired contrasts

| contrast | factors | mean | 95% CI | n |
|---|---|---:|---|---:|
| `heterogeneous_minus_homogeneous` | value=heldout_natural, visibility=hidden, adaptation=sender_only, support=full | 0.046 | [-0.162, 0.254] | 9 |
| `heterogeneous_minus_homogeneous` | value=sender_consistency, visibility=hidden, adaptation=sender_only, support=full | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | value=all_natural_minus_permuted, visibility=hidden, adaptation=sender_only, support=full | -0.000 | [-0.114, 0.113] | 9 |
| `heterogeneous_minus_homogeneous` | value=heldout_natural, visibility=hidden, adaptation=sender_only, support=leave_one_out | 0.006 | [-0.234, 0.245] | 9 |
| `heterogeneous_minus_homogeneous` | value=sender_consistency, visibility=hidden, adaptation=sender_only, support=leave_one_out | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | value=all_natural_minus_permuted, visibility=hidden, adaptation=sender_only, support=leave_one_out | -0.021 | [-0.190, 0.149] | 9 |
| `heterogeneous_minus_homogeneous` | value=heldout_natural, visibility=hidden, adaptation=coadapt, support=full | 0.046 | [-0.158, 0.250] | 9 |
| `heterogeneous_minus_homogeneous` | value=sender_consistency, visibility=hidden, adaptation=coadapt, support=full | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | value=all_natural_minus_permuted, visibility=hidden, adaptation=coadapt, support=full | 0.000 | [-0.113, 0.113] | 9 |
| `heterogeneous_minus_homogeneous` | value=heldout_natural, visibility=hidden, adaptation=coadapt, support=leave_one_out | 0.046 | [-0.204, 0.297] | 9 |
| `heterogeneous_minus_homogeneous` | value=sender_consistency, visibility=hidden, adaptation=coadapt, support=leave_one_out | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | value=all_natural_minus_permuted, visibility=hidden, adaptation=coadapt, support=leave_one_out | 0.023 | [-0.125, 0.172] | 9 |
| `heterogeneous_minus_homogeneous` | value=heldout_natural, visibility=visible, adaptation=sender_only, support=full | 0.046 | [-0.162, 0.254] | 9 |
| `heterogeneous_minus_homogeneous` | value=sender_consistency, visibility=visible, adaptation=sender_only, support=full | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | value=all_natural_minus_permuted, visibility=visible, adaptation=sender_only, support=full | -0.000 | [-0.114, 0.113] | 9 |
| `heterogeneous_minus_homogeneous` | value=heldout_natural, visibility=visible, adaptation=sender_only, support=leave_one_out | 0.014 | [-0.109, 0.138] | 9 |
| `heterogeneous_minus_homogeneous` | value=sender_consistency, visibility=visible, adaptation=sender_only, support=leave_one_out | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | value=all_natural_minus_permuted, visibility=visible, adaptation=sender_only, support=leave_one_out | -0.019 | [-0.139, 0.101] | 9 |
| `heterogeneous_minus_homogeneous` | value=heldout_natural, visibility=visible, adaptation=coadapt, support=full | 0.049 | [-0.154, 0.252] | 9 |
| `heterogeneous_minus_homogeneous` | value=sender_consistency, visibility=visible, adaptation=coadapt, support=full | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | value=all_natural_minus_permuted, visibility=visible, adaptation=coadapt, support=full | -0.000 | [-0.113, 0.113] | 9 |
| `heterogeneous_minus_homogeneous` | value=heldout_natural, visibility=visible, adaptation=coadapt, support=leave_one_out | -0.012 | [-0.141, 0.118] | 9 |
| `heterogeneous_minus_homogeneous` | value=sender_consistency, visibility=visible, adaptation=coadapt, support=leave_one_out | 0.000 | [0.000, 0.000] | 9 |
| `heterogeneous_minus_homogeneous` | value=all_natural_minus_permuted, visibility=visible, adaptation=coadapt, support=leave_one_out | -0.006 | [-0.123, 0.112] | 9 |
| `visible_minus_hidden` | value=heldout_natural, population_mode=homogeneous, adaptation=sender_only, support=full | 0.000 | [0.000, 0.000] | 9 |
| `visible_minus_hidden` | value=sender_consistency, population_mode=homogeneous, adaptation=sender_only, support=full | 0.000 | [0.000, 0.000] | 9 |
| `visible_minus_hidden` | value=heldout_natural, population_mode=homogeneous, adaptation=sender_only, support=leave_one_out | 0.052 | [-0.089, 0.194] | 9 |
| `visible_minus_hidden` | value=sender_consistency, population_mode=homogeneous, adaptation=sender_only, support=leave_one_out | -0.176 | [-0.228, -0.123] | 9 |
| `visible_minus_hidden` | value=heldout_natural, population_mode=homogeneous, adaptation=coadapt, support=full | 0.006 | [-0.003, 0.015] | 9 |
| `visible_minus_hidden` | value=sender_consistency, population_mode=homogeneous, adaptation=coadapt, support=full | 0.000 | [0.000, 0.000] | 9 |
| `visible_minus_hidden` | value=heldout_natural, population_mode=homogeneous, adaptation=coadapt, support=leave_one_out | 0.081 | [-0.073, 0.235] | 9 |
| `visible_minus_hidden` | value=sender_consistency, population_mode=homogeneous, adaptation=coadapt, support=leave_one_out | -0.176 | [-0.228, -0.123] | 9 |
| `visible_minus_hidden` | value=heldout_natural, population_mode=heterogeneous, adaptation=sender_only, support=full | 0.000 | [0.000, 0.000] | 9 |
| `visible_minus_hidden` | value=sender_consistency, population_mode=heterogeneous, adaptation=sender_only, support=full | 0.000 | [0.000, 0.000] | 9 |
| `visible_minus_hidden` | value=heldout_natural, population_mode=heterogeneous, adaptation=sender_only, support=leave_one_out | 0.061 | [-0.191, 0.313] | 9 |
| `visible_minus_hidden` | value=sender_consistency, population_mode=heterogeneous, adaptation=sender_only, support=leave_one_out | -0.176 | [-0.228, -0.123] | 9 |
| `visible_minus_hidden` | value=heldout_natural, population_mode=heterogeneous, adaptation=coadapt, support=full | 0.009 | [-0.005, 0.023] | 9 |
| `visible_minus_hidden` | value=sender_consistency, population_mode=heterogeneous, adaptation=coadapt, support=full | 0.000 | [0.000, 0.000] | 9 |
| `visible_minus_hidden` | value=heldout_natural, population_mode=heterogeneous, adaptation=coadapt, support=leave_one_out | 0.023 | [-0.168, 0.214] | 9 |
| `visible_minus_hidden` | value=sender_consistency, population_mode=heterogeneous, adaptation=coadapt, support=leave_one_out | -0.176 | [-0.228, -0.123] | 9 |
| `coadapt_minus_sender_only` | value=heldout_natural, population_mode=homogeneous, visibility=hidden, support=full | -0.003 | [-0.009, 0.004] | 9 |
| `coadapt_minus_sender_only` | value=sender_consistency, population_mode=homogeneous, visibility=hidden, support=full | 0.000 | [0.000, 0.000] | 9 |
| `coadapt_minus_sender_only` | value=heldout_natural, population_mode=homogeneous, visibility=hidden, support=leave_one_out | -0.049 | [-0.221, 0.123] | 9 |
| `coadapt_minus_sender_only` | value=sender_consistency, population_mode=homogeneous, visibility=hidden, support=leave_one_out | 0.000 | [0.000, 0.000] | 9 |
| `coadapt_minus_sender_only` | value=heldout_natural, population_mode=homogeneous, visibility=visible, support=full | 0.003 | [-0.004, 0.009] | 9 |
| `coadapt_minus_sender_only` | value=sender_consistency, population_mode=homogeneous, visibility=visible, support=full | 0.000 | [0.000, 0.000] | 9 |
| `coadapt_minus_sender_only` | value=heldout_natural, population_mode=homogeneous, visibility=visible, support=leave_one_out | -0.020 | [-0.095, 0.054] | 9 |
| `coadapt_minus_sender_only` | value=sender_consistency, population_mode=homogeneous, visibility=visible, support=leave_one_out | 0.000 | [0.000, 0.000] | 9 |
| `coadapt_minus_sender_only` | value=heldout_natural, population_mode=heterogeneous, visibility=hidden, support=full | -0.003 | [-0.015, 0.009] | 9 |
| `coadapt_minus_sender_only` | value=sender_consistency, population_mode=heterogeneous, visibility=hidden, support=full | 0.000 | [0.000, 0.000] | 9 |
| `coadapt_minus_sender_only` | value=heldout_natural, population_mode=heterogeneous, visibility=hidden, support=leave_one_out | -0.009 | [-0.145, 0.128] | 9 |
| `coadapt_minus_sender_only` | value=sender_consistency, population_mode=heterogeneous, visibility=hidden, support=leave_one_out | 0.000 | [0.000, 0.000] | 9 |
| `coadapt_minus_sender_only` | value=heldout_natural, population_mode=heterogeneous, visibility=visible, support=full | 0.006 | [-0.011, 0.023] | 9 |
| `coadapt_minus_sender_only` | value=sender_consistency, population_mode=heterogeneous, visibility=visible, support=full | 0.000 | [0.000, 0.000] | 9 |
| `coadapt_minus_sender_only` | value=heldout_natural, population_mode=heterogeneous, visibility=visible, support=leave_one_out | -0.046 | [-0.167, 0.074] | 9 |
| `coadapt_minus_sender_only` | value=sender_consistency, population_mode=heterogeneous, visibility=visible, support=leave_one_out | 0.000 | [0.000, 0.000] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, visibility=hidden, adaptation=sender_only | -0.353 | [-0.635, -0.072] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, visibility=hidden, adaptation=coadapt | -0.399 | [-0.627, -0.171] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, visibility=visible, adaptation=sender_only | -0.301 | [-0.495, -0.107] | 9 |
| `leave_one_out_minus_full` | population_mode=homogeneous, visibility=visible, adaptation=coadapt | -0.324 | [-0.513, -0.135] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, visibility=hidden, adaptation=sender_only | -0.394 | [-0.696, -0.091] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, visibility=hidden, adaptation=coadapt | -0.399 | [-0.664, -0.135] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, visibility=visible, adaptation=sender_only | -0.333 | [-0.489, -0.176] | 9 |
| `leave_one_out_minus_full` | population_mode=heterogeneous, visibility=visible, adaptation=coadapt | -0.385 | [-0.552, -0.217] | 9 |

The central test is whether exposure to multiple perceptual conventions changes a fresh sender's ability to recover a reusable message code. A hidden sender must use one codebook across all partners; a visible sender can maintain partner-specific mappings. This is a protocol-alignment mechanism test, not evidence of human syntax or open-ended language.
