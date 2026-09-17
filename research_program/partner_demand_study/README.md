# Private-demand cooperation study

This package tests whether a shared token convention becomes causally useful
when each agent privately requests a resource that only its partner's action
can satisfy. See [plan.md](plan.md) for the frozen factors and interpretation
rules.

The `partner_005` two-seed pilot is summarized in
[partner_005_结果与下一步.md](partner_005_结果与下一步.md). Its raw outputs are
local only; the independent checks are implemented in `audit.py`.

The four-cell `partner_006` credit-assignment longprobe is summarized in
[partner_006_longprobe_结果与下一步.md](partner_006_longprobe_结果与下一步.md).

Run the low-cost invariants from the repository root:

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.partner_demand_study.tests.test_game
```

Prepare and execute a smoke subset:

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.partner_demand_study.runner prepare \
  --out research_program/partner_demand_study/results/partner_001
PYTHONPATH=. .exp_venv/bin/python -m research_program.partner_demand_study.runner execute \
  --prepared research_program/partner_demand_study/results/partner_001 \
  --out research_program/partner_demand_study/results/partner_001_smoke \
  --updates 20 --seeds 68101 \
  --conditions stateless_scarce_PI_live_persistent
```
