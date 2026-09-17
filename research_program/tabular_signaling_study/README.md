# Tabular signaling control

This package is a low-variance positive control for the division-of-labour
experiments. It uses a scout's private request, one discrete token, a worker,
persistent resource inventory and causal channel controls.

The eight-seed persistent confirmation and the matched FI controls are
summarized in `tabular_joint_summary_2026-09.json` and
`tabular_joint_summary_2026-09.md` when the compact archive is assembled. The
abundant PI persistent arm shows a positive live-versus-silent effect and a
positive natural−permuted effect; the scarce arm does not show either effect
once the from-scratch silent control is used. The closed intervention remains
useful as a diagnostic, but it is not the primary channel-necessity estimate.

The four-seed switching confirmation is summarized in
[tabular_003_switching_结果与下一步.md](tabular_003_switching_结果与下一步.md).
Training-support permutation loss is clear, while held-out switching transfer
attenuates and crosses zero in the scarce arm.

Run tests from the repository root:

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.tabular_signaling_study.tests.test_game
```

Freeze a plan and run a smoke subset:

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.tabular_signaling_study.runner prepare \
  --out research_program/tabular_signaling_study/results/tabular_001
PYTHONPATH=. .exp_venv/bin/python -m research_program.tabular_signaling_study.runner execute \
  --prepared research_program/tabular_signaling_study/results/tabular_001 \
  --out research_program/tabular_signaling_study/results/tabular_001_smoke \
  --updates 100 --seeds 68101 \
  --conditions recurrent_abundant_PI_live_persistent
```
