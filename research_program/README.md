# 当前研究包

- [tabular_signaling_study](tabular_signaling_study/README.md)：144 个 run 的私有需求、记忆、容量与通道正对照。
- [temporal_scarcity_study](temporal_scarcity_study/temporal_003_formal_结果与下一步.md)：128 个 run 的 neural temporal-scarcity 矩阵和负边界。
- [compositional_signaling_study](compositional_signaling_study/README.md)：80 个筛查性 run 加 80 个 3000-update 完整 run，比较一个四值 token 与两个二值 token，并做槽位重组检验。
- [action_dependent_signaling_study](action_dependent_signaling_study/README.md)：32 seeds × 8 conditions 的 simultaneous/staged 行动依赖对照，预注册 seed-level 可组合平衡指标。
- [generation_compositional_transmission_study](generation_compositional_transmission_study/README.md)：32 个 staged dual2 parent、192 个 child run，分层检验整体协议和槽位重组能否传给新 worker 或 sender。
- [heldout_composition_study](heldout_composition_study/README.md)：128 个 child run，在冻结 parent sender 后留出一个目标组合，检验新 worker 的零样本组合恢复。
- [receiver_representation_study](receiver_representation_study/README.md)：256 个 child run，加入 full-support 对照并按 parent 可组合性分层，检验接收者表示对组合留出的影响。
- [population_signaling_study](population_signaling_study/README.md)：128 个 run 的固定/轮换伙伴与公共 token 对照。

三个 signaling 包的 `results/formal_20260917/` 只保留紧凑汇总、冻结 JSON、审计和哈希收据；temporal 包保留源码、冻结方案、紧凑审计和结果报告，不保留原始日志或 checkpoint。完整阶段历史仍可从 Git 历史中的旧提交恢复；当前工作树只展开继续实验所需的代码和记录。
