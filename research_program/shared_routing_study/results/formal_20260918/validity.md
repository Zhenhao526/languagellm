# Validity record

- The four architecture arms share the same world, goal, partner, role, message-uniform and action-uniform streams within each seed and condition.
- Parent and child updates are replayed from checkpoint 0; parameter hashes and checkpoint hashes are checked at every recorded checkpoint.
- Three independent execution shards each replay with maximum absolute error 0.0. The combined audit sums their disjoint keys and matches the full 9-seed design.
- The strict held-out-value arm remains a negative control for unseen atomic meanings.
- Raw logs and checkpoints are not retained after archive verification; the compact endpoint and source snapshot are retained.
