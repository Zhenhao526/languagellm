# Generation transmission study

该包检验一个已形成的离散协议能否被新 agent 从零恢复。先训练 rotating-hidden 的 parent population，再替换 sender 或一个 worker；新主体只通过环境中的 token、行动后果和奖励适应。`live`、`silent`、`scrambled` 三个 child channel 共享随机流，便于把通信收益和一般奖励学习分开。

运行接口测试：

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.generation_transmission_study.tests.test_game
```

冻结并执行 smoke：

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.generation_transmission_study.runner prepare --out /tmp/generation_prepared
PYTHONPATH=. .exp_venv/bin/python -m research_program.generation_transmission_study.runner execute --prepared /tmp/generation_prepared --out /tmp/generation_smoke --updates 20 --seeds 76101 --conditions worker_live,worker_silent
```

`live−silent` 是新主体的主要通信必要性对照；`live−scrambled` 检验稳定 token 是否比随机符号扰动更有用；学习曲线和 parent fidelity 测量传递速度与协议保持。结果仍是离散协议的文化传递诊断，不等于自然语言起源证据。
