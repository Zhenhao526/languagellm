# 固定分工 FI 能力对照：结果与下一步

## 目的

在固定 scout/worker 环境中，先给 worker 直接看到 scout 的需求（FI），再判断任务和优化器是否能学到基本的执行能力。4 个 seed、`recurrent_scarce_FI_{silent,live,persistent}`、2,000 更新，共 8 个 run；live 仍保存 natural、closed、permuted。

## 结果

4 个 FI live held-out cell 的 natural−closed 均值为 **−2.475 个百分点**，natural−permuted 为 **−0.352 个百分点**。四个 seed 的 natural−closed 为 **+2.153、+0.106、−10.331、−1.827 pp**；natural−permuted 为 **−0.248、+0.289、−2.486、+1.038 pp**。直接给需求没有产生稳定的正能力增益，消息互信息均值约 **0.314 bit**。

这轮没有通过能力门槛：当前环境、奖励尺度或 score-function 优化仍不足以让 worker 稳定执行已知需求。因而 PI 消息的零效应不能解释为“语言没有形成”，也不能继续靠增加 seed 把结果包装成语言结论。

## 审计

独立 `audit.py` 按 `fixed_004` 源快照检查 8 个 run、16,000 条训练日志和 48 个 final evaluation block；结果副本、计数、silent alias、有限性、日志哈希和 oracle 一致性通过，非有限诊断为 0，oracle 最大误差为 0。

## 下一步

先从语言实验中抽离一个最小的显式计划执行测试：固定一个 scout request、固定一个 worker target，去掉消息采样和 recurrent 状态，确认 worker 能在 held-out patch type 上稳定选择正确地点；再逐步恢复需求变化、库存和 blackout。只有显式计划在同一结算规则下过预设能力门槛，才重新引入离散 token，并把 natural−permuted 作为主要因果读数。
