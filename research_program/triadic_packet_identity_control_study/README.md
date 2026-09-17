# Triadic packet identity controls

This package runs a frozen, no-training follow-up to the four-choice content probe. It uses the completed formation policies in `triadic_rule_formation_study/results/formation_001` and the static cases in `triadic_content_response_study/results/content_001`.

The control replaces only the selected sender's first-window outward packet and greedily recomputes W2 and all action heads in live conditions. `endpoint_cycle` uses the next endpoint in the same group and background. `cross_group_cycle` uses the same endpoint from the next group in the same sender-listener × attribute layer. Silent conditions reuse natural action probabilities and perform no forward pass.

The primary quantity is the natural same-group live `M` minus endpoint-cycle live `M`, averaged over strict and reciprocal rules within each of 16 paired seeds. The cross-group contrast is retained as a fixed control. A positive value indicates endpoint-specific packet response; it does not establish a lexicon, compositionality, or human-language origin.

## Reproduce

```bash
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_packet_identity_control_study.dataset prepare \
  --out research_program/triadic_packet_identity_control_study/results/control_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_packet_identity_control_study.audit \
  --run research_program/triadic_packet_identity_control_study/results/control_001 \
  --plan-sha 73583dc793de3c06ad635251b028a59eb82c77a228146e2bb053a93363f4606d --freeze
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_packet_identity_control_study.runner execute \
  --out research_program/triadic_packet_identity_control_study/results/control_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_packet_identity_control_study.audit \
  --run research_program/triadic_packet_identity_control_study/results/control_001 \
  --plan-sha 73583dc793de3c06ad635251b028a59eb82c77a228146e2bb053a93363f4606d \
  --out research_program/triadic_packet_identity_control_study/results/control_001/audit_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_packet_identity_control_study.summarize \
  --run research_program/triadic_packet_identity_control_study/results/control_001 \
  --audit research_program/triadic_packet_identity_control_study/results/control_001/audit_001/verification.json
research_program/.plotting_venv/bin/python -m research_program.triadic_packet_identity_control_study.plot_results \
  --run research_program/triadic_packet_identity_control_study/results/control_001
```

The audited result is in `results/control_001/execution/results.json`; the Chinese report is in `结果与下一步.md`.
