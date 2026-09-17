# redesign_v0.34：伙伴拓扑轮换与同命名空间 fixed-A 对照

v0.34 在 v0.33 的互补观察任务上加入伙伴拓扑轮换：dual_complementary 训练时交替使用 A/B 两套 sender–receiver 拓扑，终点评估从未训练过的 C。随后在完全相同的 34034 fixture 命名空间下补跑 fixed-A control，只把训练拓扑固定为 A。正式配对报告见 [results/rotation_001/伙伴拓扑轮换与跨伙伴迁移研究报告.md](<results/rotation_001/伙伴拓扑轮换与跨伙伴迁移研究报告.md>)。

主要结果是：fixed-A native A 目标 J 为 40.625%，rotating-A 为 29.109%；同命名空间 rotating−fixed 为 -11.516 个百分点。另一方面，A 拓扑 all-cross J 从 fixed-A 的 6.037% 升到 rotating 的 20.779%，配对差值为 +14.742 个百分点；同类型完整消息一致率从 12.708% 升到 43.125%。伙伴轮换提高了跨团队形式可读性，却牺牲了 native grounded 性能，说明形式收敛、语义后果和社会迁移是不同指标。

## 运行链

```bash
.venv/bin/python -m py_compile redesign_v0.34/*.py
.venv/bin/python redesign_v0.34/check_support.py
.venv/bin/python redesign_v0.34/check_views.py
.venv/bin/python redesign_v0.34/check_dual.py
.venv/bin/python redesign_v0.34/run_support.py --dev --updates 40 --out redesign_v0.34/results/smoke_001
.venv/bin/python redesign_v0.34/analyze_results.py --out redesign_v0.34/results/smoke_001
.venv/bin/python redesign_v0.34/audit_results.py --out redesign_v0.34/results/smoke_001
.venv/bin/python redesign_v0.34/preflight.py
.venv/bin/python redesign_v0.34/run_support.py --out redesign_v0.34/results/rotation_001
.venv/bin/python redesign_v0.34/analyze_results.py --out redesign_v0.34/results/rotation_001
.venv/bin/python redesign_v0.34/audit_results.py --out redesign_v0.34/results/rotation_001
.venv/bin/python redesign_v0.34/rotation_transfer.py --out redesign_v0.34/results/rotation_001
.venv/bin/python redesign_v0.34/cross_topology.py --out redesign_v0.34/results/rotation_001
.venv/bin/python redesign_v0.34/plot_rotation.py --out redesign_v0.34/results/rotation_001
.venv/bin/python redesign_v0.34/message_rotation.py --out redesign_v0.34/results/rotation_001
.venv/bin/python redesign_v0.34/run_fixed_control.py --out redesign_v0.34/results/fixed_001
.venv/bin/python redesign_v0.34/fixed_control_analysis.py --out redesign_v0.34/results/fixed_001
.venv/bin/python redesign_v0.34/fixed_control_cross.py --out redesign_v0.34/results/fixed_001
.venv/bin/python redesign_v0.34/audit_fixed_control.py --out redesign_v0.34/results/fixed_001
.venv/bin/python redesign_v0.34/build_report.py --out redesign_v0.34/results/rotation_001
.venv/bin/python redesign_v0.34/package_results.py --out redesign_v0.34/results/rotation_001
```

rotating 和 fixed-A 训练使用相同 source/input hashes、world、照片、初始化和训练预算。`fixed_control_analysis.py` 只做 NumPy 统计和 saved-table 重算；`audit_fixed_control.py` 进一步检查 endpoint、agreement、categorical trace、reward 算术和 completion hashes。两个脚本都不导入生产训练模块，不调用模型。

## 文件说明

- `support_design.json`、`固定执行方案_v0.34.md`：冻结拓扑、遮罩、收益、采样和解释规则。
- `support.py`、`run_support.py`：A/B 轮换训练与 A/B/C 终点协议保存。
- `run_fixed_control.py`：同命名空间 fixed-A 训练包装器。
- `analysis.py`、`audit_results.py`：rotating 三条件 native 统计与边界重放。
- `rotation_transfer.py`、`cross_topology.py`：rotating 拓扑迁移和 sender token 兼容性。
- `fixed_control_analysis.py`、`fixed_control_cross.py`、`audit_fixed_control.py`：fixed-A 统计、配对差分和原始 trace 审计。
- `results/rotation_001/`、`results/fixed_001/`：正式批次原始表、分析、审计和回执。
- `figures/`：设计图、结果图和 visual QA。
- `结果审查.json`、`结果审查.md`：含同命名空间配对结论和限制的科学审查。
- `results/rotation_001/completion_manifest.json`：最终文件清单与哈希，包含 fixed-A control。

下一轮应在保留 fixed-A/rotating paired 框架的同时，加入未见 sender 或 receiver 身份留出，区分拓扑泛化与身份泛化，再考虑第三种资源或延迟中继。
