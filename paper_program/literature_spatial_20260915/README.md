# v0.13 空间能力近邻文献核查

2026-09-15 定向联网审查；仅整理文献，不读取 v0.13 正式成绩、不训练、不操作独立照片框。

主文：[近邻方法与贡献边界](近邻方法与贡献边界.md)。结论是已有研究充分覆盖“认知/表征先验影响涌现通信”这一宽泛方向；本轮“先私人行动等变准备、冻结接口、再形成新离散协议”的具体比较尚未被本次已核查方法直接覆盖，但新意未得到全面证明。

七篇 PDF 已按年份、第一作者、主题重命名，六篇完成方法全文核查；Imai 2024 日文抽取乱码，仅摘要级。来源和方法页码在主文逐篇列出。

- [候选及阅读状态](triage.json)
- [五篇 arXiv 候选元数据](targeted_arxiv_candidates.json)
- [radar 人工评审排序](reports/radar_ranked_review.md)（阅读优先级，不是质量榜）
- [原始 PDF 下载记录](primary/manifest.json)与[补充下载记录](primary/supplementary_manifest.json)
- [packet 缓存来源及成功记录](packet_provenance.json)
- [本地文件检查](qa.json)

原版 radar 的 20 篇候选收集遇 HTTP 429，未成功生成该池，未继续循环请求。同次研究改用官方页面定向核查五篇 arXiv 候选，再运行原版 packet、record-review 和 finalize；另两篇期刊/会议候选没有伪造 arXiv ID。详细限制与失败日志见主文。

可复查命令与脚本：[配置](config.toml)、[元数据子集脚本](prepare_radar_subset.py)、[packet 入口](run_packets.py)、[人工评审记录](reviews/)。这些脚本不会运行模型。正式 ACL PDF 用于两篇 packet 的缓存，版本区别已在来源记录中披露。
