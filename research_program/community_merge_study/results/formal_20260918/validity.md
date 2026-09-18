# 有效性收据

- 独立 replay audit：`passed`。
- 覆盖 `36` parent 与 `432` child；训练日志 `108,000` / `1,296,000`；checkpoint `144` / `1728`；最大绝对 replay 误差 `0.0`。
- 配对重放行数：visibility `648,000`，adaptation `1,296,000`，population `648,000`，support `864,000`，role `648,000`。
- aligned 与 conflict 使用同一 parent 初始化和 paired stream；conflict 仅改变 community 1 的 surface token permutation。

## 解释边界

这批次证明角色对称 referential game 中消息可以获得行为因果作用，并能识别社区表面冲突、伙伴身份可见性和 coadapt 的影响；严格 held-out combination/value 的零样本恢复仍未稳定出现。该结果支持机制边界，不等同于人类语言已被模拟。
