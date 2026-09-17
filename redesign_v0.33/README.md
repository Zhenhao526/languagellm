# redesign_v0.33：互补观察下的双发送者通信

v0.33 把上一轮“共同收益”操纵改成一个更直接的信息结构检验：同一接收者的食物和水行动由两个 sender 分别贡献，food sender 只能看到 food-only 视图，water sender 只能看到 water-only 视图。冗余双发送者条件让两个 sender 都看到完整视图；single_full 是单 sender 完整视图基线。

正式结果位于 [results/team_001/互补观察下的双发送者共同符号形成研究报告.md](<results/team_001/互补观察下的双发送者共同符号形成研究报告.md>)。互补条件目标12 J 为39.931%，冗余双发送者为12.355%，四个 seed 的差值全部为正；但固定接收者做跨团队 token 替换后，all-cross J 只有5.076%，因此目前得到的是配对特定的 grounded 组合协议，不是群体共享词典。

## 运行链

```bash
.venv/bin/python -m py_compile redesign_v0.33/*.py
.venv/bin/python redesign_v0.33/check_support.py
.venv/bin/python redesign_v0.33/check_views.py
.venv/bin/python redesign_v0.33/check_dual.py
.venv/bin/python redesign_v0.33/run_support.py --dev --updates 40 --out redesign_v0.33/results/smoke_001
.venv/bin/python redesign_v0.33/analyze_results.py --out redesign_v0.33/results/smoke_001
.venv/bin/python redesign_v0.33/audit_results.py --out redesign_v0.33/results/smoke_001
.venv/bin/python redesign_v0.33/preflight.py
.venv/bin/python redesign_v0.33/run_support.py --out redesign_v0.33/results/team_001
.venv/bin/python redesign_v0.33/analyze_results.py --out redesign_v0.33/results/team_001
.venv/bin/python redesign_v0.33/audit_results.py --out redesign_v0.33/results/team_001
.venv/bin/python redesign_v0.33/cross_team.py --out redesign_v0.33/results/team_001
.venv/bin/python redesign_v0.33/plot_results.py --out redesign_v0.33/results/team_001
.venv/bin/python redesign_v0.33/message_examples.py --out redesign_v0.33/results/team_001
.venv/bin/python redesign_v0.33/build_report.py --out redesign_v0.33/results/team_001
.venv/bin/python redesign_v0.33/package_results.py --out redesign_v0.33/results/team_001
```

正式运行已经完成；上面的链条用于复核，不应在已有输出目录直接重复封存。`analysis.py` 只读取保存的 sender/receiver 表，`audit_results.py` 重放边界协议与抽样训练轨迹，`cross_team.py` 专门检查 sender token 是否能跨团队复用。

## 文件说明

- `support_design.json`、`固定执行方案.md`：冻结的条件、采样、收益和预定指标。
- `views.py`、`dual.py`、`run_support.py`：互补视图、双 sender 损失和训练/评估流程。
- `analysis.py`、`audit_results.py`、`cross_team.py`：独立统计、边界重放和跨团队替换检查。
- `results/team_001/analysis.json`、`cross_team.json`、`audit_execution.json`：正式数值证据。
- `results/team_001/figures/`：设计图、结果图及 `visual_qa.json`。
- `结果审查.json`、`结果审查.md`：带限制条件的科学审查。
- `results/team_001/completion_manifest.json`：最终文件清单与哈希。

下一轮应先在保持互补视图与收益不变的前提下轮换 sender–receiver 配对，测量原配对、已见角色新伙伴和完全跨团队三种迁移，再决定是否增加第三种资源或延迟转发。
