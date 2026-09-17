# Triadic rule–communication study

This package runs the preregistered 128-run CPU experiment described in
[`plan.md`](plan.md). It is a task-coordination experiment over a discrete
resource world. The two settlement rules are deliberately explicit: strict
settlement requires a neutral third actor, while reciprocal settlement ignores
the unmatched third proposal. The package does not claim that its symbols are
language.

The workflow is:

```text
python -m research_program.triadic_rule_communication_study.runner prepare --out <dir>
python -m research_program.triadic_rule_communication_study.runner execute --out <dir>
```

`prepare` freezes source hashes and the full factorial design. `execute` trains
independent tanh MLPs with NumPy only and writes checkpoints, paired logs and
JSON results. The separate audit and summary scripts are intentionally added
after execution and are not part of the frozen training source.
