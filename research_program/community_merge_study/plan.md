# Frozen plan: community merge

The parent stage trains one role-symmetric policy per community. The same tabular policy supplies a sender head and a receiver head, and role assignment is balanced during parent training. The conflict population exposes identity in community 0 and a fixed pair-swap surface permutation in community 1; the aligned population uses identity in both communities.

The child stage initializes a fresh policy and exposes it to all four incumbents. `alternating` assigns the fresh agent to sender and receiver roles in balanced blocks; `sender_only` is an ablation. `hidden` removes partner identity from the fresh policy, while `visible` gives separate partner-conditioned heads initialized from the same hidden parameters. `fresh_only` freezes incumbents; `coadapt` updates the incumbent that fills the opposite role.

Training support is `full`, `heldout_combo` (one pair absent while both factor values remain seen), or `heldout_value` (one value of the first factor absent from target training). Evaluation keeps held-out combination and held-out atomic value as separate endpoints. The pre-registered functional threshold is natural held-out return at least `0.60`; message causality is assessed with `natural−permuted` and `natural−silent`, and compositionality with slot recombination.

The finite message alphabet has four symbols and two slots, yielding 16 surface messages for nine object meanings. All source files, random streams, checkpoints and training rows are hash-auditable; no language model, teacher labels or external service is used.
