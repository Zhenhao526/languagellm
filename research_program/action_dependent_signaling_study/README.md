# Action-dependent signaling study

本实验检验行动依赖是否是组合性共同符号形成的必要条件。发送者知道两个私有二元目标，worker 依次完成两个资源子任务，消息形式固定为两个二值 token。

- `simultaneous`：两个 token 在第一个行动前都到达，允许任意整体码本。
- `staged`：slot 0 只在第一个子任务前到达，slot 1 只在第二个子任务前到达；这使两个槽位承担不同的行动后果。

两种协议使用相同世界、目标、伙伴和抽样流；每个协议都有 live/silent、factorized/entangled 条件。主指标仍是配对的 `natural−silent`、`natural−permuted` 和 `recombined−natural`，并保留独立回放审计。

运行测试：

```bash
PYTHONPATH=. .exp_venv/bin/python -m research_program.action_dependent_signaling_study.tests.test_game
```
