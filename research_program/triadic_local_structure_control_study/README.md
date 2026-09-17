# 局部包结构对照

本包把随机符号双射的整包效应进一步拆分为三种预注册控制：固定首槽单符号循环、保留符号数值秩的包内规范化、只保留首现等式模式的局部重标记。三种变换均在读取策略输出前固定，复用已冻结的 `formation_001` 64 条策略和四选一内容案例，不训练、不调用 Qwen。

live 条件只替换指定发送者首窗 W1 对外包，并重算 W2 与三人的动作；silent 条件是自然轨迹别名。所有条件保留对象、属性、搭档、目的地、位置、包长和背景。每个模式覆盖 16 个配对初始化、四个 strict/reciprocal×live/silent 条件和六个检查点，共 1152 条记录（576 live、576 silent）。

## 复现

```bash
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_local_structure_control_study.dataset prepare \
  --out research_program/triadic_local_structure_control_study/results/local_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_local_structure_control_study.audit \
  --run research_program/triadic_local_structure_control_study/results/local_001 \
  --plan-sha 3ee301d349267248a82b1423d75958fffe0e7ac44288005b8e45149ba43205e6 --freeze
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_local_structure_control_study.runner execute \
  --out research_program/triadic_local_structure_control_study/results/local_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_local_structure_control_study.audit \
  --run research_program/triadic_local_structure_control_study/results/local_001 \
  --plan-sha 3ee301d349267248a82b1423d75958fffe0e7ac44288005b8e45149ba43205e6 \
  --out research_program/triadic_local_structure_control_study/results/local_001/audit_001
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_local_structure_control_study.summarize \
  --run research_program/triadic_local_structure_control_study/results/local_001 \
  --audit research_program/triadic_local_structure_control_study/results/local_001/audit_001/verification.json
research_program/.plotting_venv/bin/python -m research_program.triadic_local_structure_control_study.plot_results \
  --run research_program/triadic_local_structure_control_study/results/local_001
```

权威结果在 `results/local_001/execution/results.json`，独立审计在 `results/local_001/audit_001/verification.json`，汇总在 `results/local_001/summary_001/summary.json`，中文报告在 `结果与下一步.md`。
