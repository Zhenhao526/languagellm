# v0.15 单一全局幅度校准

当前已固定一个新条件reset_scaled。它只在v0.14重置编码的observe输出乘每人一个常量，检验全局幅度差是否足以解释先前私人准备的迁移收益。训练支持校准已完成，短流程及2101步开发流程完成；正式社会批次等待独立预检。

- [固定方案](固定执行方案.md)、[科学审查](前置科学审查.md)、[执行审查](前置审查.md)
- [缩放与校准模块](scaled_interface.py)、[运行器](run_scaled.py)
- [正式24人校准](calibration_001/calibration_complete.json)：每人old18×22×22训练照片场景，倍率约4.98–5.68；test/未训练地图不参与。
- [源文件绑定](source_receipt.json)：原v13保留和v14重置参考已见过，不计新独立样本。
- [倍率1的开发重放](results/smoke_001/root_unit_replay.json)：原参数/Adam、日志和行为逐位一致；新增buffer导致state哈希与旧版不同是预期。

正式命令记录（输出目录必须不存在；必须先通过preflight_qa.json）：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.15/run_scaled.py --out redesign_v0.15/results/scaled_001
```

本轮新运行仅12次reset_scaled社会训练，复用12个retained和12个reset参考。每组2400×512通信。主比较为retained−reset_scaled的sealed自然双目标终点剩余差；修复差reset_scaled−reset为辅助。四来源内三分区、两方向平均，n=4。范数校正不等于能力匹配、完整几何匹配或移除知识；沿原7×2离散消息和冻结DINO路线。
