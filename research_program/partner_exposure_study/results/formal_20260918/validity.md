# 有效性收据

- 全量 replay audit：`passed`。
- 统计量：18 parent、144 child；54,000/432,000 条训练日志；72/576 个 checkpoint；visibility 216,000、adaptation 432,000、population 432,000、support 216,000 条配对行；`max_abs_replay_error=0.0`。
- parent 的均质/异质总体回报几乎相同（约 0.528），说明本轮表面异质性没有自动造成总体失败。
- fresh sender 在 full support 上达到约 0.51–0.57 held-out natural return，并有约 0.30–0.39 的训练增益；leave-one-out 约 0.11–0.23，未形成稳定的未见联合目标零样本恢复。
- hidden sender 的码本跨伙伴一致率固定为 1；visible sender 在 leave-one-out 中一致率约 0.824，显示伙伴可见性允许局部专用码，但没有带来稳定组合泛化。
- coadapt 相对 sender-only 在 full support 上几乎没有增益，在 leave-one-out 中也没有稳定改善。

## 解释边界

本轮识别了多伙伴暴露、伙伴身份可见性和共同码本一致性的关系，但独立 joint-goal sender 参数仍不具备因子参数共享，因此 leave-one-out 失败不能解释为一般语言能力缺失。下一步需要把新对象/新组合、角色对称和可学习的因子复用正交化，并在视觉主体上复现。
