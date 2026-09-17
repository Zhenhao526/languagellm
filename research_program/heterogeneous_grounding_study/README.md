# Population-level perceptual heterogeneity study

This round tests whether a shared symbolic protocol is easier or harder to form when workers use different conventions for mapping visible labels to latent object types. A sender observes a two-object goal and emits a short discrete message. The sender does not observe partner identity. Workers observe only the surface labels of the scene and use the message to collect the requested object at two stages.

The **homogeneous** population maps labels identically for all workers. The **heterogeneous** population gives workers 1 and 3 a stable two-label swap, while workers 0 and 2 retain the identity mapping. A child replaces worker 0 and adapts its worker policy while the sender codebook and other workers are frozen. The child mapping is either identity or swap, so the study separates population-level convention diversity from the target worker's local remapping.

The grid crosses population mode, message form (`mono4` versus staged `dual2`), worker representation (`joint_history` versus `slot_local`), target mapping, support (`full` versus leave-one-out), and channel (`live` versus silent). Every paired arm shares semantic scenes, goals, partner IDs, and random draws. The all-goal permutation control groups four goals per partner; shuffling messages within that block changes only the message-to-goal alignment.

This is a controlled protocol-formation experiment, not a claim that the tabular agents possess human language. The main readout is the child's live held-out return after leave-one-out training. `live−silent` and `natural−permuted` are causal communication checks; `heterogeneous−homogeneous` is the population-level intervention.

## Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.heterogeneous_grounding_study.runner prepare --out /tmp/hgs_prepared
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.heterogeneous_grounding_study.runner execute --prepared /tmp/hgs_prepared --out /tmp/hgs_execution --updates 3000
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.heterogeneous_grounding_study.audit --prepared /tmp/hgs_prepared --execution /tmp/hgs_execution --out /tmp/hgs_audit.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.heterogeneous_grounding_study.aggregate --results /tmp/hgs_execution/execution/results.json --out /tmp/hgs_analysis.json --markdown /tmp/hgs_analysis.md
```

The formal design uses nine fixed seeds and 48 child conditions. A smaller seed or condition subset is suitable for a local smoke test; the audit still checks replay and pairwise stream invariants for the executed subset.
