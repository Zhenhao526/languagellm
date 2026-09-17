# Compositional signaling study

这是一个低方差 tabular 对照，用来检验消息长度和任务因素结构是否影响共同符号的形成。sender 看到两个私有二值因子；worker 只能看到每个子任务的本地地点类型，依次完成两个子任务。`mono4` 发送一个四值 token，`dual2` 发送两个二值 token；没有教师、语言模型或预置词典。

完整设计见 [plan.md](plan.md)。形式、伙伴生态和通道的定义在冻结 `design.py` 中。运行接口测试：

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.compositional_signaling_study.tests.test_game
```

冻结并运行 smoke：

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.compositional_signaling_study.runner prepare --out /tmp/compositional_prepared
PYTHONPATH=. .exp_venv/bin/python -m research_program.compositional_signaling_study.runner execute --prepared /tmp/compositional_prepared --out /tmp/compositional_smoke --updates 20 --seeds 75101 --conditions dual2_rotating_hidden_live_factorized,dual2_rotating_hidden_silent_factorized
```

`natural−silent` 是通道必要性的主对照；`natural−permuted` 检验 sender/worker 配对；dual2 的 `recombined−natural` 只作为槽位可组合性的结构读数。即使这些指标为正，也只说明本任务中的共同离散协议，不等于自然语言或人类语言起源。
