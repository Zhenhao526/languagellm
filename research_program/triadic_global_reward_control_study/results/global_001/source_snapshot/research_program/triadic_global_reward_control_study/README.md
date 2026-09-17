# Global reward-scale null control

This paired control asks whether the c75 effect from the preceding conflict
penalty study can be explained by multiplying all rewards by a constant. The
three rules are:

- `c0`: reciprocal execution with native rewards;
- `c75`: reciprocal execution, with only an engaged unmatched third proposal
  receiving a 0.25 multiplier;
- `global25`: reciprocal execution, with every native reward receiving a 0.25
  multiplier.

All rules are crossed with PL/FI information, static/rematched training and
live/silent channels on eight new seeds. Worlds, batches, message uniforms and
initial parameters are paired within seed. The learner maximizes mean log exact
expected reward plus the unchanged action entropy term. Under this objective a
state-independent reward multiplier shifts `log_J` by a constant and leaves the
receiver gradient and paired sender advantage unchanged. The training control
tests this prediction numerically rather than treating it as an assumption.

```text
PYTHONPATH=. .venv/bin/python -m research_program.triadic_global_reward_control_study.runner prepare --out <dir>
PYTHONPATH=. .venv/bin/python -m research_program.triadic_global_reward_control_study.runner execute --out <dir>
PYTHONPATH=. .venv/bin/python -m research_program.triadic_global_reward_control_study.audit --source <dir> --output <audit-dir> --workers 2
PYTHONPATH=. .venv/bin/python -m research_program.triadic_global_reward_control_study.summarize --source <dir> --output <summary-dir>
```

This is a protocol and optimization control. It does not provide evidence for
lexical meaning, grammar, cultural transmission or human language origin.
