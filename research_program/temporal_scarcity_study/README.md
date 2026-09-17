# Temporal scarcity communication study

This package is the next controlled experiment in the language-emergence program. It uses two randomly initialized, non-linguistic policy networks in a five-round resource game. Private demands, independent resource patches, persistent inventory, and a two-round token channel create a pressure for coordination without giving the agents a language or a teacher.

Run the invariant tests from the repository root:

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.temporal_scarcity_study.tests.test_game
```

Freeze a plan and run a small subset first:

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.temporal_scarcity_study.runner prepare \
  --out research_program/temporal_scarcity_study/results/temporal_001
PYTHONPATH=. .exp_venv/bin/python -m research_program.temporal_scarcity_study.runner execute \
  --prepared research_program/temporal_scarcity_study/results/temporal_001 \
  --out research_program/temporal_scarcity_study/results/temporal_001_smoke \
  --updates 20 --seeds 68101 \
  --conditions stateless_scarce_PI_silent,stateless_scarce_PI_live
```

The frozen design and causal interpretation rules are in [plan.md](plan.md). Results include a training-support split and a held-out transition split, central-oracle normalization, and natural/closed/permuted channel controls.
