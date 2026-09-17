# redesign_v0.42：双 token 端点干预

本轮对 v0.41 的 72 条双 token 形成终点做端点反事实干预，检验第二个 token 的功能依赖、资源块置换的结构性等变，以及空间重定位下的协议稳定性。没有新增训练，因此本轮回答的是“已形成协议具有什么可干预结构”，不改变 v0.41 的形成结果。

- 输入：72 条 v0.41 endpoint，4 seeds × 3 partitions × 2 resource assignments × 3 formation topologies
- 评估：A/B/C 三种伙伴拓扑、4 个 team、120 张测试地图，共 864 条 team-schedule 记录
- token 干预：token0/token1 的全部 7 种替换值、token 顺序交换、跨世界 token1 循环打乱
- 资源块干预：三个双 token 资源块的全部 6 种排列，同时报告 literal 与同步置换目标的 equivariant 分数
- 空间干预：identity 加 5 个六站点循环重定位；照片身份保持不变，位置在 sender 编码前移动

正式报告：[三资源双token端点干预研究报告.md](results/sequence_intervention_001/三资源双token端点干预研究报告.md)

正式结果：[sequence_intervention_analysis.json](results/sequence_intervention_001/sequence_intervention_analysis.json)、[sequence_intervention_audit.json](results/sequence_intervention_001/sequence_intervention_audit.json)、[completion_manifest.json](results/sequence_intervention_001/completion_manifest.json)

结果图：[01_sequence_intervention.png](results/sequence_intervention_001/figures/01_sequence_intervention.png)

主要结果：token0 是当前联合任务的主要信息载体，删除或交换 token0/token1 会使功能大幅下降；轮换和随机伙伴形成条件下，资源块同步置换的 equivariant J 远高于 literal J，说明存在可重排的资源轴结构。位置重定位反而提高 target J，但该结果受坐标编码和原 target 关系划分交互影响，不能称为新地点泛化。

本轮不声称形成自然语言、语法或开放组合性。下一轮需要在形成阶段遍历全部资源/角色排列并随机化顺序，再测试这些干预结构能否跨新主体代际传递。
