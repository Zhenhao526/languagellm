# Formal incumbent-worker transfer archive (2026-09-18)

This archive contains the frozen plan, endpoint results, compact training-return curves, independent audit receipts, and hashes for the 9-seed incumbent-worker surface-transfer experiment. The formal matrix has 9 parent runs and 72 child runs (`fresh/incumbent × identity/swap × live/silent`), with 3,000 updates and checkpoints at 0/500/1000/2000/3000.

The parent learns a `dual2` protocol on identity object labels. The sender is frozen. An incumbent child copies parent worker 0; a fresh child receives an independent worker. A stable swap changes only visible surface labels. The primary endpoint is full-support live held-out natural return at initialization and after child training; repair gain and all-goal `natural−permuted` are secondary endpoints.

Raw `training.jsonl` files and checkpoints were replay-audited and are intentionally not retained in the repository. `aggregate.json` and `learning_curves.json` remain in this local compact checkout. The complete `results.json` payload is retained in the remote repository and removed locally after hash verification.
