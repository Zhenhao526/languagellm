# Validity

The formal claims use the v4 reevaluated endpoints and the combined replay audit. Training was completed before the evaluator correction; the correction only changes the role=2 recombination mask and recomputes endpoint fields from the stored final checkpoints. All 54 parent and 135 child runs were replayed against the frozen v4 source with maximum absolute replay error 0.0. The strict zero-shot double-new evidence comes from `expanded_single_alternating` versus `expanded_single_pair`, where `(3,3)` is absent from both training supports.
