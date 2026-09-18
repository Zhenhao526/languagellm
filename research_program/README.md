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
- [meaning_capacity_study](meaning_capacity_study/README.md)：把重复压力曲线扩展到 8 个意义类，比较 `triple2`、等容量 `quad2`/`atomic16`；正式矩阵已完成 81 个 parent 和 729 个 child run，结果显示低熵重复意义仍可形成冗余码，而 `unique8`/`repeat4` 没有满足候选标准。
- [ecological_factorization_study](ecological_factorization_study/README.md)：9 个三值目标组合、因子化/整体置换任务、等容量 `tri3`/`mono9` 和新 worker 的 leave-one-combination-out 传递；正式矩阵包含 36 个 parent 和 216 个 child，结果显示三值任务有弱通信收益，但没有零样本新组合恢复。
- [object_surface_remap_study](object_surface_remap_study/README.md)：二值 `dual2`/`mono4` 容量匹配、固定 object-label permutation、slot-local/joint-history 与 leave-one-combination-out 传递；首轮通信对照失效的 diagnostic archive 保留，修正版九种子正式结果与审计位于 `object_surface_remap_study/results/formal_20260918_corrected/`。
- [object_surface_transfer_study](object_surface_transfer_study/README.md)：在冻结 sender 后直接复制或重新初始化 worker，比较 identity 与 stable surface swap 下的零样本迁移成本和奖励修复；9 个 parent、72 个 child 的正式 endpoint、学习曲线和三片独立审计位于 `object_surface_transfer_study/results/formal_20260918/`。
- [heterogeneous_grounding_study](heterogeneous_grounding_study/README.md)：把稳定的 surface-label 异质性提升到整个 parent population，比较隐藏伙伴、消息形式、接收表示和新 worker 的 leave-one-out 修复；9 个 seed、36 个 parent、432 个 child 的正式摘要和三片审计位于 `heterogeneous_grounding_study/results/formal_20260918/`。
- [heterogeneous_grounding_study 诊断归档](heterogeneous_grounding_study/results/diagnostic_20260918/)：一枚种子的 3,000-update 候选干预通过 replay audit，但因目标 worker 覆盖使 child 流相同，暂不作为正式群体效应估计。
- [partner_exposure_study](partner_exposure_study/README.md)：修正 target-worker 不可识别性，让 fresh sender 在每个 batch 接触四个实际伙伴；正式矩阵为 18 个 parent、144 个 child，交叉 hidden/visible、sender-only/coadapt 和 full/leave-one-out，完整 replay audit 通过。
- [community_merge_study](community_merge_study/README.md)：在四符号双槽位的角色对称 referential game 中，控制 aligned/conflict 社区码本、伙伴身份可见性、fresh-only/coadapt、full/heldout-combination/heldout-value 和 alternating/sender-only；正式矩阵为 36 个 parent、432 个 child，独立 replay audit 通过，结果归档于 `results/formal_20260918/`。
- [structured_factorization_study](structured_factorization_study/README.md)：保持社区冲突 referential game 不变，因果操纵 holistic 完整表格与 factorized slot/attribute 参数共享；正式矩阵为 72 个 parent、216 个 child，独立 replay audit 通过，factorized 在 aligned hidden 条件下实现组合留出而 holistic 对照失败。
- [routed_factorization_study](routed_factorization_study/README.md)：在同一 referential game 中加入不固定 slot—attribute 对应关系的可学习 routing；9 seed、108 parent、324 child 的正式结果显示部分槽位对齐但组合泛化对优化盆地敏感，另存低梯度 routing 基线。
- [shared_routing_study](shared_routing_study/README.md)：在 holistic、fixed factorized、independent routed 之外加入 sender/receiver 共享 routing 的角色对称干预；9 seed、144 parent、432 child 的正式矩阵检验共享组合坐标是否能恢复 held-out combination。
- [routing_initialization_control_study](routing_initialization_control_study/README.md)：让 independent routing 从同一矩阵初始化但独立更新，用于排除 shared routing 的初始化混淆；9 seed、54 parent、81 child 的 aligned/hidden 正式控制与 replay audit 位于 `results/formal_20260918/`。
- [open_world_expansion_study](open_world_expansion_study/README.md)：把父代限制在 3×3 旧本体，让子代首次接触第 4 个值，并严格留出双新值组合；54 parent、135 child、9 seed 的正式端点、source snapshot 与 replay audit 位于 `results/formal_20260918/`。
- [open_world_cultural_transmission_study](open_world_cultural_transmission_study/README.md)：冻结 single-new 扩展后替换 sender、receiver、双方或不训练的新 worker；54 parent、27 expansion child、108 transfer run 的正式端点与六片 replay audit 位于 `results/formal_20260918/`。
- [open_world_grounding_transfer_study](open_world_grounding_transfer_study/README.md)：在同一开放世界文化传递矩阵中加入 fresh receiver 的 identity/reverse 对象表面置换；54 parent、27 expansion child、216 transfer run，三片 expansion 与三片 transfer replay audit 均通过。
- [repetition_pressure_study](repetition_pressure_study/README.md)：在唯一意义、局部重复和全局重复三种任务语义之间做容量匹配的冗余压力曲线；正式矩阵包含 81 个 parent 和 729 个 child run，结果归档于 `results/formal_20260918/`，详见 `meaning_capacity_formal_结果与下一步.md`。
- [population_signaling_study](population_signaling_study/README.md)：128 个 run 的固定/轮换伙伴与公共 token 对照。

各 signaling 包的 `results/formal_20260917/` 只保留紧凑汇总、冻结 JSON、审计和哈希收据；temporal 包保留源码、冻结方案、紧凑审计和结果报告，不保留原始日志或 checkpoint。完整阶段历史仍可从 Git 历史中的旧提交恢复；当前工作树只展开继续实验所需的代码和记录。
