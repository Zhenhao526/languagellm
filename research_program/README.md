# 当前研究包

- [tabular_signaling_study](tabular_signaling_study/README.md)：144 个 run 的私有需求、记忆、容量与通道正对照。
- [temporal_scarcity_study](temporal_scarcity_study/temporal_003_formal_结果与下一步.md)：128 个 run 的 neural temporal-scarcity 矩阵和负边界。
- [compositional_signaling_study](compositional_signaling_study/README.md)：80 个筛查性 run 加 80 个 3000-update 完整 run，比较一个四值 token 与两个二值 token，并做槽位重组检验。
- [action_dependent_signaling_study](action_dependent_signaling_study/README.md)：32 seeds × 8 conditions 的 simultaneous/staged 行动依赖对照，预注册 seed-level 可组合平衡指标。
- [generation_compositional_transmission_study](generation_compositional_transmission_study/README.md)：32 个 staged dual2 parent、192 个 child run，分层检验整体协议和槽位重组能否传给新 worker 或 sender。
- [heldout_composition_study](heldout_composition_study/README.md)：128 个 child run，在冻结 parent sender 后留出一个目标组合，检验新 worker 的零样本组合恢复。
- [receiver_representation_study](receiver_representation_study/README.md)：256 个 child run，加入 full-support 对照并按 parent 可组合性分层，检验接收者表示对组合留出的影响。
- [multi_generation_transmission_study](multi_generation_transmission_study/README.md)：9 个可组合 parent、两种替换顺序、两代 live/silent 链，检验组合协议能否连续传递。
- [multi_generation_chain_study](multi_generation_chain_study/README.md)：32 个 seed、三代替换链和严格 leave-one-out 组合留出，检验协议在连续替换后的稳定性。
- [noise_repair_study](noise_repair_study/plan.md)：可组合 parent 上的 0–40% 信道噪声、worker-only/coadapt 和两种接收表示，检验旧协议的维持与协同修复。
- [ternary_composition_study](ternary_composition_study/plan.md)：三值属性、等容量 `mono9`/`tri3`、任务耦合和行动时序矩阵，检验组合结构随容量扩展的边界。
- [error_correcting_signaling_study](error_correcting_signaling_study/README.md)：容量匹配的冗余编码、信道噪声与 sender–receiver 修复；包含 243 个 child run 的正式紧凑归档，并保留一次因理论阈值不可达而拒绝的设计记录。
- [shared_redundancy_study](shared_redundancy_study/README.md)：让同一隐藏意义跨两个行动阶段重复有效，检验重复后果是否足以诱导 `triple2` 形成 parity 冗余码；包含 243 个 child run 的正式紧凑归档。
- [repetition_pressure_study](repetition_pressure_study/README.md)：在唯一意义、局部重复和全局重复三种任务语义之间做容量匹配的冗余压力曲线；正式矩阵包含 81 个 parent 和 729 个 child run，结果归档于 `results/formal_20260918/`。
- [population_signaling_study](population_signaling_study/README.md)：128 个 run 的固定/轮换伙伴与公共 token 对照。

各 signaling 包的 `results/formal_20260917/` 只保留紧凑汇总、冻结 JSON、审计和哈希收据；temporal 包保留源码、冻结方案、紧凑审计和结果报告，不保留原始日志或 checkpoint。完整阶段历史仍可从 Git 历史中的旧提交恢复；当前工作树只展开继续实验所需的代码和记录。
