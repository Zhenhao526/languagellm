# Division-of-labour signaling study

This package tests whether a shared token convention is useful when one agent
privately knows a request and a partner must execute it. It is the next
controlled stage after the temporal scarcity and partner-demand pilots.

Run the invariant tests from the repository root:

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.role_signaling_study.tests.test_game
```

Prepare and run a smoke subset:

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.role_signaling_study.runner prepare \
  --out research_program/role_signaling_study/results/role_001
PYTHONPATH=. .exp_venv/bin/python -m research_program.role_signaling_study.runner execute \
  --prepared research_program/role_signaling_study/results/role_001 \
  --out research_program/role_signaling_study/results/role_001_smoke \
  --updates 20 --seeds 68101 \
  --conditions recurrent_scarce_PI_live_persistent
```
