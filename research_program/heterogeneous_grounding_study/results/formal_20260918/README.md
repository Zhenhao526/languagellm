# Heterogeneous-grounding formal archive

This archive contains the nine-seed formal matrix for the population-level perceptual-heterogeneity study. It includes 36 parent runs and 432 child runs across 48 child conditions, with 3,000 updates per run. The full final-result payload is retained here for remote reproducibility; raw training logs and checkpoints were removed after replay audit. `pilot_curves.json` is an exploratory three-seed, 500-update learning-curve summary and is kept separate from the formal endpoint.

The execution was audited independently in three shards. The audit replays every training stream, gradient, parameter hash, checkpoint, and live/silent plus identity/swap pairing. `aggregate.json` and `aggregate.md` are computed from the merged final results. The `execution_plan.json` records the source freeze used by the three executions; `plan.json` records the final analysis source after a statistics-only de-duplication of contrast rows.
