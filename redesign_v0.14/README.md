# v0.14 私人视觉准备的迁移作用

本轮补充同来源的私人接口重置对照。复用v0.13的12个control参考，只新增12个reset社会运行；来源仍为原4个种子，属于旧图片机制开发。

正式批次results/reset_001已完成全部12次reset社会学习，并通过独立执行及结果核验。保留私人接口的sealed自然成功22.04%，重置后4.44%，四来源保留优势均正、平均+17.60个百分点；old也96.62%对86.53%，只能先解释综合迁移收益。下一候选为old训练支持上的单标量尺度校准，不在本批追加。

- [完整研究报告](results/reset_001/私人非语言经验的迁移研究报告.md)、[学习曲线与分析](results/reset_001/保留与重置接口_分析草稿.md)、[消息实例](results/reset_001/固定消息形成样例.md)
- [正式执行核验](results/reset_001/audit_execution.json)：427874项通过，0失败；28,800更新和96检查点核对。
- [独立NumPy重算](results/reset_001/independent_recount.json)：23424标量；[分析比较](results/reset_001/independent_recount_comparison.json)：31195项吻合。
- [图表视觉QA](results/reset_001/figure_qa.json)、[纯显示版本记录](results/reset_001/display_revision.json)：仅补刻度，181947数值不变。
- [贡献边界预审](贡献边界预审.md)：解释框架未读取正式结果，不冒充正式运行前预注册；私人准备消融本身不足以构成论文。

- [固定执行方案](固定执行方案.md)、[科学审查](前置科学审查.md)、[执行审查要求](前置审查.md)
- [运行器](run_reset.py)、[原参考文件来源](source_receipt.json)
- [无训练的接口撤销检查](manipulation_002/result.json)：24人、原私人logits逐位重放；固定旧头接reset的结果仅是读出兼容性，不是重训可达能力。
- [已保留的前向浮点差异](manipulation_001/failure_receipt.json)：恢复原requires_grad标记后逐位一致，始终no_grad、没有优化器。
- [保留臂短流程逐位复现](results/smoke_003/root_retained_replay.json)

正式固定命令记录（该批已经完成；重现必须改用不存在的输出目录）：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.14/run_reset.py --out redesign_v0.14/results/reset_001
```

memory在本任务作单步视觉编码。重置不会移除DINO和资源投影知识，也不是移除时间记忆。主要比较为retained−reset的sealed终点自然双成功，辅助观察全程AUC、共同支持消息区分和实际使用。每源内三分区与两方向平均，独立单位n=4。

正式分析与核查入口为analyze_reset.py、independent_recount.py和audit_reset.py，默认读取已完成批次。短流程retained_replay只作工程复现；正式12次新运行仅reset，旧control引用不能计作新训练。smoke001/002绑定前一版只读操纵脚本，smoke003/004绑定最终执行源码并共同通过preflight_qa.json。审计器跨smoke累计计数的失败结果和修正前源码已保留；成功审计源码SHA37c226a60fc3a77b040c3606e200d1fe01f989a23929fd58e138f50bb9d2a815。
