# Structured factor-sharing referential study

This study tests whether zero-shot combination recovery comes from the task ecology or from an explicit factor-sharing inductive bias. It reuses the role-symmetric four-object referential game with two three-valued attributes and a two-slot, four-symbol channel.

The `holistic` policy has one sender row per complete meaning and one receiver row per complete message. The `factorized` policy shares sender parameters by attribute value and makes the receiver score each candidate by adding slot-local evidence for the two attributes. Token identities remain unlabelled; the factorized arm is an explicit architectural intervention, not evidence that an unconstrained agent discovered compositionality.

Both architectures are trained as parents on all nine meanings. Children replace the fresh agent, alternate sender and receiver roles across four partners, and train with full, held-out-combination, or held-out-value support. Aligned and conflict populations, plus hidden and visible partner identity, separate factor sharing from community surface conflict and partner-specific codes.

The primary endpoint is held-out natural return with the pre-registered functional threshold 0.60, paired against holistic policies with the same seed and world streams. `natural−permuted`, `natural−silent`, cross-partner consistency, and factor-slot uniqueness are secondary readouts. A positive factorized result would show that structural sharing is sufficient for transfer; it would not show that an unconstrained population invented language.

Raw execution trees are temporary. The formal archive retains the frozen source snapshot, compact result, replay audit, and cleanup receipt.
