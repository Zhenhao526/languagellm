# v0.48 有界居民共同适应与陌生主体恢复

本轮研究陌生主体在既有离散通信协议中恢复时，居民端的有限可塑性是否改变恢复过程和最终结构。陌生主体的发送端与接收端都重置；居民端分为完全冻结、只更新发送端、只更新接收端和双侧稀疏更新。居民更新每 20 个适应步最多执行一次，最多 30 次，视觉前端、记忆和未选中的模块保持冻结。

## 设计

- 4 个 seed × 3 个视觉 partition × 6 个资源身份分配 = 72 个配对视觉组；
- 6 类居民形成文化：静态/随机角色结构 × 固定 A、轮换 A/B、随机 A/B/C 日程；
- 4 种居民适应模式：`resident_frozen`、`resident_sender_sparse`、`resident_receiver_sparse`、`resident_both_sparse`；
- 完整交叉共 1,728 条链，每条 600 个更新，检查点 0、100、300、600；
- 每个检查点保存 3 种日程、4 个团队拓扑和 6 种角色置换；
- 本地 PyTorch 控制模型，无 LLM、API 或外部模型调用。

正式设计见 [coadaptation_design.json](coadaptation_design.json)，研究结果见 [有界居民共同适应与陌生主体恢复研究报告.md](results/coadaptation_001/有界居民共同适应与陌生主体恢复研究报告.md)。

## 复现

```bash
../.venv/bin/python coadaptation_train.py --out results/coadaptation_001
../.venv/bin/python coadaptation_analysis.py --out results/coadaptation_001
../.venv/bin/python coadaptation_statistics.py --out results/coadaptation_001
../.venv/bin/python coadaptation_audit.py --out results/coadaptation_001
../.venv/bin/python plot_coadaptation.py --out results/coadaptation_001
../.venv/bin/python visual_qa.py --out results/coadaptation_001
../.venv/bin/python build_coadaptation_report.py --out .
../.venv/bin/python build_review.py --out .
../.venv/bin/python package_results.py --out .
```

正式结果目录包含 13,824 个训练轨迹、82,944 个协议快照和完整的源/输入哈希。`results/coadaptation_smoke`、`results/coadaptation_probe_*` 是开发期小规模运行，不属于正式批次。

## 解释边界

本实验是六站点、三资源、双 token 的有限 grounded protocol。它测量居民可塑性对陌生主体恢复的影响，不能单独证明开放语义、自然语言句法、延迟生存压力或代际传递。后续应加入延迟资源后果、冲突修复和新一代主体。
