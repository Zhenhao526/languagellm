# 多伙伴暴露实验同步与清理

本轮新增 `partner_exposure_study`，把上一轮 target-worker child 的不可识别性改为可识别的 fresh-sender 多伙伴暴露设计：每个 child batch 实际混合四个轮换伙伴，交叉 hidden/visible 伙伴身份、sender-only/coadapt 适应方式和 full/leave-one-out 支持。

正式批次已完成并通过全量 replay audit：9 个 seed、18 个 parent、144 个 child；54,000/432,000 条训练日志、72/576 个 checkpoint，visibility/adaptation/population/support 配对流完整，最大回放误差为 0。正式归档保留完整 `results.json`、聚合统计、冻结方案和审计收据；smoke 归档单独保存，不进入正式统计。

结果显示，多伙伴暴露确实可以测量 shared codebook 与 partner-specific code 的差异：hidden sender 的码本跨伙伴一致，visible sender 在 leave-one-out 中出现较低的一致率；但独立 joint-goal 参数仍不能产生未见联合目标的稳定零样本恢复。该负边界已写入主报告，下一步转向独立社区码本冲突、角色对称 referential game 和新对象/新组合正交化。

归档位置：

- `research_program/partner_exposure_study/results/formal_20260918/`
- `research_program/partner_exposure_study/results/diagnostic_20260918/`

源代码提交：`11582a826ce829d7fc30c8c0d7856340a0f0f971`
远端仓库：`git@github.com:Zhenhao526/language.git`
