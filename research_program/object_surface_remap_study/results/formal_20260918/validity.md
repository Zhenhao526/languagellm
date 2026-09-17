# Validity note

The 9-seed binary object-surface remapping run was fully replay-audited, but it is retained as a **diagnostic archive** rather than the final inferential result.

The evaluation stream originally assigned `partner_id = arange(count) % WORKERS` while the all-goal stream cycled goals in the same order. Consequently, every partner saw only one goal value in the all-goal evaluation. The `permuted` control rotates messages within a partner, so it became a no-op (`natural_minus_permuted = 0` for every live child). The fixed held-out endpoint and identity/swap training streams were still generated and paired, but the communication checks and all/seen summaries do not identify message causality under this schedule.

The raw outputs, checkpoints, and temporary replay trees were deleted only after the archive was hash-verified and all three replay audits passed. A corrective source patch groups four consecutive all-goal examples under the same partner, adds a regression test that the permutation control actually mixes goals, and will be used for the replacement formal run. The archived numbers must not be presented as evidence for a live-versus-permuted communication effect.
