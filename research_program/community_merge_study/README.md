# Community codebook conflict and role-symmetric referential game

This study asks whether agents that inherit incompatible community conventions can form a shared, reusable code while alternating between sender and receiver roles. Two parent communities use the same latent referential task but may expose different bijections over a four-symbol, two-slot alphabet (`@`, `#`, `￥`, `%`). A fresh agent is trained with all four incumbent partners in every alternating batch. Partner identity is hidden or visible, incumbents are frozen or coadapt, and training support either includes all meanings or masks one joint combination or one atomic value.

Each episode is a referential game: the sender sees the target object's two attributes, the receiver sees a four-object scene, and the receiver selects the target. The receiver's action is the only task channel. Evaluation separates ordinary referential return, `live−silent`/`natural−permuted` message effects, role-conditioned performance, cross-partner sender consistency, community codebook convergence, and slot-wise recombination. The design is finite and tabular; it does not claim that human syntax or open vocabulary has emerged.

## Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.community_merge_study.runner prepare --out /tmp/community_merge_prepared
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.community_merge_study.runner execute --prepared /tmp/community_merge_prepared --out /tmp/community_merge_execution --updates 3000
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.community_merge_study.audit --prepared /tmp/community_merge_prepared --execution /tmp/community_merge_execution/execution --out /tmp/community_merge_audit.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.community_merge_study.aggregate --results /tmp/community_merge_execution/execution/results.json --out /tmp/community_merge_analysis.json --markdown /tmp/community_merge_analysis.md
```

The formal grid has nine seeds, two independently trained communities per population mode, and 48 child conditions. Raw logs and checkpoints are temporary and should be removed only after the replay audit and compact archive have been verified.
