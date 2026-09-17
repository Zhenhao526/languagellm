# v0.17 发送基线与场景对应关系

正式前360,606项独立预检通过，现按固定预算启动12个新社会运行，复用v0.15的12个matched参考。全部完成后才汇总正式效应。唯一主要比较为matched−shuffled的2400步sealed自然双目标N；四个继承来源内先平均三分区、两个方向，n=4。当前仍是旧图片开发证据。

新条件保持策略/价值两支路原固定幅度，仅在每人256个发送世界内置换进入策略优势的神经基线。价值MSE仍按原世界拟合，接收公式不变；世界、照片、消息/动作随机流及训练预算匹配。置换允许固定点，记录实际数值变化；只保持各自当前batch多重集，不声称两臂全程基线分布相同。

- [固定执行方案](固定执行方案.md)、[前置科学审查](前置科学审查.md)、[独立前置审查](前置审查.md)
- [运行器](run_baseline.py)、[基线干预模块](baseline_interface.py)、[独立执行审计器](audit_baseline.py)
- [成功预检](preflight_qa.json)、[来源绑定](source_receipt.json)
- [完整分析程序](analyze_baseline.py)、[NumPy原始复算](independent_recount.py)
- [裁剪日志检查](clip_activity.py)、[置换与操纵强度独立检查](independent_baseline_diagnostics.py)

训练命令（已启动；查看结果无需再次运行；复跑须指定不存在的输出目录）：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.17/run_baseline.py --out redesign_v0.17/results/baseline_001
```

每新运行2400×512通信，共14,745,600条新通信、29,491,200次资源行动，无新增私人训练、倍率校准、视觉主干微调或完整参考重跑。官方冻结DINOv2 ViT-L/14沿用既有缓存，仍在当前Mac本地运行。

先前v0.16裁剪未激活并不保证新臂也如此。本轮预定检查全程裁剪，若触发即原样报告，不事后修改阈值或删除运行。对基线方差的直接测量仅在科学审查中提出，尚未执行；总干预成绩不等于其方差机制或中介比例证据。
