# v0.45 陌生主体社会学习与协议恢复

本目录把 v0.43 已形成的 resident 文化作为社会环境，重置一个 newcomer 的通信模块，测试它能否仅凭共同任务回报恢复 resident 协议。视觉前端和 resident 参数冻结；训练在本机 PyTorch 上完成，未调用 LLM 或外部 API。

## 设计

- 4 个 seed × 3 个视觉 partition × 6 个资源列排列 × 6 类 resident 文化，共 432 条正式链；
- resident 文化：静态/随机角色顺序 × 固定 A、轮换 A/B、随机 A/B/C 形成拓扑；
- newcomer 为 identity 0，只重置通信模块；三个 resident 冻结；
- 适应阶段统一使用随机 A/B/C 伙伴日程，训练 300 次更新，检查点 0、100、300；
- 每个 sender 输出 2 个 token（词表 7），receiver 预测 3 个资源的站点，保存 literal/equivariant J、角色 spread、token agreement 和 position NMI。

完整设计见 [newcomer_adaptation_design.json](newcomer_adaptation_design.json)，研究结果见 [陌生主体社会学习与协议恢复研究报告.md](results/newcomer_adaptation_001/陌生主体社会学习与协议恢复研究报告.md)。

## 已封存输出

- [training_complete.json](results/newcomer_adaptation_001/training_complete.json)：432 条链、300 次更新；
- [newcomer_adaptation_analysis.json](results/newcomer_adaptation_001/newcomer_adaptation_analysis.json)：独立 NumPy 重算；
- [newcomer_adaptation_audit.json](results/newcomer_adaptation_001/newcomer_adaptation_audit.json)：独立动作/采样/分数审计；
- [01_newcomer_adaptation.png](results/newcomer_adaptation_001/figures/01_newcomer_adaptation.png)：六面板图；
- [结果审查.md](结果审查.md)：科学审查和限制。

## 复现实验

以下命令会向空目录写入正式输出；正式训练约需十余分钟，输出包含大量 NPZ 文件。

```bash
../.venv/bin/python newcomer_adaptation_train.py --out results/newcomer_adaptation_001
../.venv/bin/python newcomer_adaptation_analysis.py --out results/newcomer_adaptation_001
../.venv/bin/python newcomer_adaptation_audit.py --out results/newcomer_adaptation_001
../.venv/bin/python plot_newcomer_adaptation.py --out results/newcomer_adaptation_001
../.venv/bin/python visual_qa.py --out results/newcomer_adaptation_001
../.venv/bin/python build_newcomer_adaptation_report.py --out results/newcomer_adaptation_001
../.venv/bin/python build_review.py --out .
../.venv/bin/python package_results.py --out results/newcomer_adaptation_001
```

`results/newcomer_adaptation_smoke` 是早期 5-update 冒烟运行，不属于正式批次；正式封存只使用 `newcomer_adaptation_001`。
