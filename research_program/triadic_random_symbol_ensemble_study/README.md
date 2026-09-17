# 六个随机符号双射的集成对照

本包检验上一轮固定全局符号置换是否只是某个特殊变换。它复用已经完成并冻结的 `formation_001` 64 条策略和四选一内容案例，不训练、不调用 Qwen，也不改变对象、属性、搭档、目的地、位置或消息长度。

六个模式 `perm_00`–`perm_05` 是在读取策略输出前固定的 8 符号随机双射。每个模式只把指定发送者的首窗 W1 对外包逐 token 映射，live 条件随后重算 W2 与三人的行动；silent 条件是自然轨迹别名。每个模式覆盖 16 个配对初始化、strict/reciprocal×live/silent 四条件和 0/100/500/1500/3000/6000 六个检查点。

主量是 6000 步时自然同组 live `M` 减置换 live `M`，先在每个种子内平均两条执行规则，再在 16 个种子上统计；`M` 是四候选接收动作的目标概率边际，先对背景和端点均衡。正值表示策略对该双射敏感，不表示符号具有词义，也不证明组合性或语言起源。

## 复现

```bash
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_random_symbol_ensemble_study.dataset prepare \
  --out research_program/triadic_random_symbol_ensemble_study/results/ensemble_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_random_symbol_ensemble_study.audit \
  --run research_program/triadic_random_symbol_ensemble_study/results/ensemble_001 \
  --plan-sha 6105a720b0f6ab84af131c4a893c04d42dea284331eda9dd31715d3729ecbfd7 --freeze
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_random_symbol_ensemble_study.runner execute \
  --out research_program/triadic_random_symbol_ensemble_study/results/ensemble_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_random_symbol_ensemble_study.audit \
  --run research_program/triadic_random_symbol_ensemble_study/results/ensemble_001 \
  --plan-sha 6105a720b0f6ab84af131c4a893c04d42dea284331eda9dd31715d3729ecbfd7 \
  --out research_program/triadic_random_symbol_ensemble_study/results/ensemble_001/audit_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_random_symbol_ensemble_study.summarize \
  --run research_program/triadic_random_symbol_ensemble_study/results/ensemble_001 \
  --audit research_program/triadic_random_symbol_ensemble_study/results/ensemble_001/audit_001/verification.json
research_program/.plotting_venv/bin/python -m research_program.triadic_random_symbol_ensemble_study.plot_results \
  --run research_program/triadic_random_symbol_ensemble_study/results/ensemble_001
```

权威结果在 `results/ensemble_001/execution/results.json`，独立逐条重放在 `results/ensemble_001/audit_001/verification.json`，汇总在 `results/ensemble_001/summary_001/summary.json`，中文报告在 `结果与下一步.md`。
