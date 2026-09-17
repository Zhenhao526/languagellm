# 因素组合留出形成实验

## 新增后验机制探针：semantic_transfer_heldout_001

在 `factorial_001_corrected` 的冻结策略上，`semantic_transfer_heldout_001` 把源端首窗消息移植到目标端，并与同层循环错配 placebo 比较。16 个种子、2 个训练臂、2 个规则、2 个通道共 128 个策略块，覆盖 328,704 条有向案例和 11,833,344 个世界；没有训练更新。独立重放通过，`max_abs_error=0`。

主要 `aligned−placebo source_pull` 交互为 strict **−0.1336 pp**（t(15) CI **[−0.3513,+0.0841]**），reciprocal **+0.0068 pp**（**[−0.1828,+0.1963]**）。结果没有支持未见对象×长度组合出现额外的源端需求—消息对齐；消息改变行动并不等于恢复了源端内容。完整设计、结果和审计见[`semantic_transfer_heldout_结果与下一步.md`](semantic_transfer_heldout_结果与下一步.md)、[`results/summary_semantic_transfer_heldout_001/汇总表.md`](results/summary_semantic_transfer_heldout_001/汇总表.md)和[`audit_semantic_transfer_heldout_001/verification.json`](audit_semantic_transfer_heldout_001/verification.json)。

## 已完成批次：factorial_001_corrected

128 次训练（16 个配对初始化 × factorial/saturated × strict/reciprocal × live/silent）已完成。原始 `factor_response` 的二维广播错误已由 [`correction.py`](correction.py) 以 JSON/NPZ-only 方式校正，未重新训练；校正后主量为未见对象×长度资源上的通信交互 **−1.1221 个百分点**，近似 t15 95% 区间 **[−1.9474, −0.2968]**。这表示因素留出臂没有获得高于饱和臂的通信增益，不能称组合语言形成。完整结果和下一步见[`结果与下一步.md`](结果与下一步.md)。

校正运行树为[`factorial_001_corrected`](results/factorial_001_corrected/)，含1,152个评价文件、六个形成检查点和128条训练日志。`audit_corrected_006` 已完成所有逐文件科学重放后只在行数 scope 断言处退出；[`audit_corrected_scope_001/verification.json`](../../audit_corrected_scope_001/verification.json) 独立确认该断言的配对行数（192,000）与四条件日志总行数（768,000）差异，未再次神经前向。结果、校正、汇总和图表收据均保留。

本包检验通信是否支持未见的对象×属性组合。`factorial_holdout` 训练臂排除资源5（wood-long）和6（fiber-short），`saturated` 训练臂包含全部支持需求；两臂在未见资源、未见布局和双留出分区中用完整世界评估。每个配对初始化同时运行 strict/reciprocal × live/silent 四种条件，16个初始化共128次训练。

执行顺序固定为：

```text
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_factorial_formation_study.runner prepare --out research_program/triadic_factorial_formation_study/results/factorial_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_factorial_formation_study.runner execute --out research_program/triadic_factorial_formation_study/results/factorial_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_factorial_formation_study.audit --run research_program/triadic_factorial_formation_study/results/factorial_001 --plan-sha <冻结计划哈希> --out audit_execution_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_factorial_formation_study.summarize --run research_program/triadic_factorial_formation_study/results/factorial_001 --audit audit_execution_001/verification.json --output summary_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_factorial_formation_study.plot_results --summary research_program/triadic_factorial_formation_study/results/factorial_001/summary_001/summary.json --output figures_001
```

上面的命令保留原始生产流程以便追溯；论文分析应使用已校正的 `results/factorial_001_corrected/` 运行树及其 `summary_001`、`figures_001`。原始 `factorial_001` 的 `factor_response` 不可直接用于科学结论。

正式结果、独立重放和图表完成前，不把消息熵、ARI 或单个成功率称作语言形成证据。
