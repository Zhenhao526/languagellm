# 因素留出 aligned/placebo 消息转移探针

## 研究问题

前面的 `factorial_001_corrected` 主实验比较了 `factorial_holdout` 与 `saturated` 两个训练臂，但团队成功率的变化不能直接说明消息是否携带了源端需求内容。本轮做一个冻结策略上的后验机制探针：把源端首个消息窗口移植到目标端的同一公共布局和背景中，观察目标行动和结算是否向源端需求移动；再把消息换成同层、同发送者、同方向的循环错配消息作为 placebo。`aligned−placebo` 是主要的内容对齐读数，绝对的 aligned 反应只表示任务敏感性。

案例来自 `factorial_holdout` 的 `heldout_both` 分区。每条无向需求边展开为两个方向，覆盖 kind、length、destination 三个变化轴；每个策略块使用 36 个背景。策略、checkpoint、自然行动、消息和需求案例全部读取冻结文件，没有训练更新。

本轮共 16 个配对初始化 × 2 个训练臂 × 2 个结算规则 × 2 个通信通道 = 128 个策略块。每个策略块有 2,568 条有向案例，合计 328,704 行、11,833,344 个世界；live 和 silent 各 164,352 行，模型前向 71,000,064 次，优化器更新为 0。aligned 替换目标端跨观察者可见的发送者首窗消息并保留目标自视输入；placebo 在相同 changed-person × 轴 × 方向层内循环错配消息。严格结算和 reciprocal 结算都重新计算 W2、行动和物理结果。

## 结果

下表是先在三个轴内等权、再在 16 个种子上计算的均值。`source_pull` 定义为源端搭档命中率增量减去目标端搭档命中率增量；数值为百分点。区间是种子层描述性 t(15) 区间。

| 训练臂 | 规则 | 通道 | aligned | placebo | aligned−placebo |
|---|---|---|---:|---:|---:|
| factorial_holdout | strict | live | −0.1828 | −0.1390 | **−0.0438** [−0.1377, +0.0501] |
| factorial_holdout | reciprocal | live | +2.4691 | +2.4741 | **−0.0050** [−0.0893, +0.0793] |
| saturated | strict | live | −0.0508 | −0.1406 | **+0.0898** [−0.0743, +0.2540] |
| saturated | reciprocal | live | +2.4461 | +2.4579 | **−0.0117** [−0.1630, +0.1395] |

四个 silent 条件的 aligned 和 placebo 逐行相等，`source_pull` 与动作改变率均为 0；这确认闭通道对照没有被消息替换逻辑污染。live 条件下消息移植确实改变了一部分动作和物理执行。例如 factorial/reciprocal 的 aligned−placebo 动作改变率为 −4.12 个百分点、物理执行率差为 +2.76 个百分点，但 source_pull 接近 0，说明行动受消息影响不等于源端需求内容被恢复。

预先关心的训练臂交互是：

```text
(factorial live−silent) − (saturated live−silent)
```

在 `aligned−placebo source_pull` 上，strict 为 **−0.1336 个百分点**，t(15) 区间 **[−0.3513, +0.0841]**；reciprocal 为 **+0.0068 个百分点**，区间 **[−0.1828, +0.1963]**。因此没有证据表明，排除 wood-long 与 fiber-short 后，未见对象×长度组合产生额外的源端消息—需求对齐。这个结果与主实验的通信交互阴性方向一致，但本探针的估计量是内容对齐而不是团队 Q。

## 审计与边界

独立审计脚本没有导入生成器，重新构造了案例、同层循环 placebo、消息路由、W2/action rollout、严格与 reciprocal 结算以及源/目标搭档指标。128 个策略块、328,704 行、11,833,344 个世界全部通过，`max_abs_error=0`；128 个 checkpoint、128 个自然数据文件和 2,568 个 placebo 映射均核对通过。审计收据见 [`audit_semantic_transfer_heldout_001/verification.json`](audit_semantic_transfer_heldout_001/verification.json)，汇总见 [`summary_semantic_transfer_heldout_001/汇总表.md`](results/summary_semantic_transfer_heldout_001/汇总表.md)。

这项探针只能回答冻结任务中的消息是否使行动朝源端需求移动。它没有建立符号指称、可组合语法、公共词典或人类语言起源；消息是四槽八符号的研究者预设接口，案例仍来自已有训练分布的需求边。aligned−placebo 的小量和宽区间也不能证明不存在更弱的内容结构。主要风险包括固定动作接口、整包消息替换、目标需求分布不均和同一策略在多个案例中的重复使用。

## 下一步

1. 在新的独立群体中预注册同一 aligned/placebo 读数，并把消息替换限制到单发送者、单槽位和跨需求组合三种层级，区分整包任务敏感性与可定位内容。
2. 将接收者独有方案、源端独有方案和 neutral 行动分开记录，避免物理执行率变化遮蔽内容选择。
3. 加入 full-information、closed-channel 和随机重编码对照；若 aligned−placebo 仍稳定为正，再进入新主体或代际传递实验。

完整机器可读结果、冻结哈希和执行收据保留在本目录；本探针是 `factorial_001_corrected` 的后验机制分析，不改写其主结果。

SHA-256 清单见 [`results/semantic_transfer_heldout_001/provenance.json`](results/semantic_transfer_heldout_001/provenance.json)，校验结果为 `provenance_hash_errors=0`。
