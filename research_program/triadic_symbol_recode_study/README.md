# Symbol and packet-position recoding controls

This package tests whether the response observed in the four-choice content probe survives simple recodings of the donor packet. It reuses the completed formation policies and the frozen content cases; there is no training or new Qwen call.

The live intervention replaces only the selected sender's first-window outward packet and greedily recomputes W2 and all action heads. The three fixed transformations are:

- `global_symbol_permutation`: apply the bijection `(3, 7, 1, 6, 0, 4, 2, 5)` to every token;
- `position_rotation`: map `[x0,x1,x2,x3]` to `[x1,x2,x3,x0]`;
- `position_reverse`: map `[x0,x1,x2,x3]` to `[x3,x2,x1,x0]`.

The primary report quantity for each mode is the same-group natural live `M` minus recoded live `M`, averaged over strict and reciprocal rules within each of 16 paired seeds. A positive value means that the frozen policy is sensitive to that recoding. It does not show that a symbol is a word or that a compositional grammar exists.

## Reproduce

```bash
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_symbol_recode_study.dataset prepare \
  --out research_program/triadic_symbol_recode_study/results/recode_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_symbol_recode_study.audit \
  --run research_program/triadic_symbol_recode_study/results/recode_001 \
  --plan-sha 0cc41ef6e34ba526a376189de534c08b896e33ee0c941ac46cc8b1947e9ab9b --freeze
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_symbol_recode_study.runner execute \
  --out research_program/triadic_symbol_recode_study/results/recode_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_symbol_recode_study.audit \
  --run research_program/triadic_symbol_recode_study/results/recode_001 \
  --plan-sha 0cc41ef6e34ba526a376189de534c08b896e33ee0c941ac46cc8b1947e9ab9b \
  --out research_program/triadic_symbol_recode_study/results/recode_001/audit_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_symbol_recode_study.summarize \
  --run research_program/triadic_symbol_recode_study/results/recode_001 \
  --audit research_program/triadic_symbol_recode_study/results/recode_001/audit_001/verification.json
research_program/.plotting_venv/bin/python -m research_program.triadic_symbol_recode_study.plot_results \
  --run research_program/triadic_symbol_recode_study/results/recode_001
```

The audited result is in `results/recode_001/execution/results.json`; the Chinese report is in `结果与下一步.md`.
