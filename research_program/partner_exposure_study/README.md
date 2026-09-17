# Multi-partner sender alignment

This study fixes the identifiability problem in the previous population-heterogeneity experiment. A parent sender and four workers first form a hidden-partner protocol. A fresh sender then interacts with all four rotating workers in every training batch while worker surface conventions remain homogeneous or heterogeneous. The sender either cannot see partner identity (`hidden`) or may maintain a partner-specific code (`visible`). Workers are frozen (`sender_only`) or coadapt with the new sender (`coadapt`); `leave_one_out` masks one joint goal during adaptation.

The main question is whether exposure to multiple perceptual conventions selects a shared, reusable sender code. `natural−permuted`, `live−silent`, visible−hidden, and sender-code consistency separate message causality, partner-specific codes, and shared alignment. The design is a finite tabular protocol experiment and makes no claim about human syntax or open-ended language.

## Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.partner_exposure_study.runner prepare --out /tmp/partner_exposure_prepared
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.partner_exposure_study.runner execute --prepared /tmp/partner_exposure_prepared --out /tmp/partner_exposure_execution --updates 3000
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.partner_exposure_study.audit --prepared /tmp/partner_exposure_prepared --execution /tmp/partner_exposure_execution/execution --out /tmp/partner_exposure_audit.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.partner_exposure_study.aggregate --results /tmp/partner_exposure_execution/execution/results.json --out /tmp/partner_exposure_analysis.json --markdown /tmp/partner_exposure_analysis.md
```

The formal grid uses nine seeds, two parent modes, and sixteen fresh-sender conditions per population mode. Full logs and checkpoints are kept only in the temporary execution tree until the compact archive is verified.
