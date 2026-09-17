# Triadic conflict-penalty study

This package tests whether a continuous cost on an engaged unmatched third
actor changes the routed-message effect on partner selection. `c0` through
`c100` use reciprocal physical execution and multiply the native reward by
`1 - penalty` when a third proposal is present. `strict` is a hard-constraint
anchor that requires the third actor to be neutral.

The design crosses six constraint levels with PL/FI information, static/
rematched partner schedules and live/silent channels on eight new seeds. The
primary dose-response statistic is the per-seed slope of the live−silent
target-pair legality gain over the five continuous penalty levels; strict is
reported as a structural anchor and is not included in that slope.

The local model is a NumPy float64 tanh MLP with two sender windows and a
factorized neutral/engage plus 16-way proposal action head. Messages are
eight-way symbols chosen by the experiment, not a natural language system.

Workflow:

```text
PYTHONPATH=. .venv/bin/python -m research_program.triadic_conflict_penalty_study.runner prepare --out <dir>
PYTHONPATH=. .venv/bin/python -m research_program.triadic_conflict_penalty_study.runner execute --out <dir>
PYTHONPATH=. .venv/bin/python -m research_program.triadic_conflict_penalty_study.audit --source <dir> --output <audit-dir>
```

The preparation freezes all source hashes, data partitions, streams and the
budget. The audit replays checkpoints and final evaluations independently.
