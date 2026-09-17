# 联合组合留出语义转移探针

这是一项冻结策略的后验干预实验。A 只见部分三人需求联合组合，B/C 见过全部联合组合；在留出支持内沿 `kind`、`length` 改变一个发送者的需求，比较供体 aligned 消息和同背景 cyclic placebo 消息是否把接收者推向供体独有合法方案。

权威结果在 `results/transfer_002/`，独立审计在 `audit_transfer_002/`，汇总和图表分别在 `results/summary_transfer_002/` 与 `results/figures_transfer_002/`。完整解释见 [组合留出语义转移_结果与下一步.md](组合留出语义转移_结果与下一步.md)。

在仓库根目录运行：

```bash
PYTHONPATH=/Users/xia/Documents/ChatGPT/语言 \
/Users/xia/Documents/ChatGPT/语言/qwen_collect_pilot/.venv/bin/python \
-m research_program.triadic_compositional_holdout_semantic_transfer_probe.probe \
execute --out research_program/triadic_compositional_holdout_semantic_transfer_probe/results/transfer_002 \
--workers 4
```

`transfer_002` 已冻结，不应覆盖；`transfer_001` 保留为导入修复前的开发批次。审计、汇总和绘图命令见报告及各自脚本；所有实验均为 NumPy 本地策略前向，没有新的训练更新或 Qwen 调用。
