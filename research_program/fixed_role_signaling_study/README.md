# Fixed-role signaling study

This package tests whether a shared token convention is useful when one agent
privately knows a request and a partner must execute it. It is the next
controlled stage after the temporal scarcity and partner-demand pilots.

The four-seed fixed-role confirmation is summarized in
[fixed_002_confirmatory_结果与下一步.md](fixed_002_confirmatory_结果与下一步.md);
raw executions stay local and `audit.py` performs the independent checks.

The FI ability control is summarized in
[fixed_004_fi_control_结果与下一步.md](fixed_004_fi_control_结果与下一步.md).

The explicit non-learning capacity gate is summarized in
[fixed_ability_gate_结果与下一步.md](fixed_ability_gate_结果与下一步.md). It
matches the environment oracle exactly on persistent and switching held-out
episodes, so the next learning control should expose the target to the worker
before returning to private-information signaling.

The same-architecture supervised FI action control is summarized in
[fi_supervised_control_结果与下一步.md](fi_supervised_control_结果与下一步.md).
It reaches the finite-horizon oracle on four held-out seeds, isolating
self-play credit assignment from network expressivity.

Run the invariant tests from the repository root:

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.fixed_role_signaling_study.tests.test_game
```

Prepare and run a smoke subset:

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.fixed_role_signaling_study.runner prepare \
  --out research_program/fixed_role_signaling_study/results/fixed_001
PYTHONPATH=. .exp_venv/bin/python -m research_program.fixed_role_signaling_study.runner execute \
  --prepared research_program/fixed_role_signaling_study/results/fixed_001 \
  --out research_program/fixed_role_signaling_study/results/fixed_001_smoke \
  --updates 20 --seeds 68101 \
  --conditions recurrent_scarce_PI_live_persistent
```
