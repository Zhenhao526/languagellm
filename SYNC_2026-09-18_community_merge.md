# 社区码本冲突正式批次同步与清理

日期：2026-09-18。远端仓库：`git@github.com:Zhenhao526/languagellm.git`。本批次使用源代码提交 `ec3b9e19bb262b3b84eb7e64b3d22aa43e44c9ab`，正式归档位于 `research_program/community_merge_study/results/formal_20260918/`。

正式矩阵包含 9 个 seed、36 个 parent 和 432 个 child，每个 run 3,000 updates。child 交叉 aligned/conflict 社区、hidden/visible 伙伴身份、fresh-only/coadapt、full/heldout-combination/heldout-value 和 alternating/sender-only。独立 replay audit 已通过：训练日志为 108,000 + 1,296,000 行，checkpoint 为 144 + 1728 个，visibility/adaptation/population/support/role 配对行分别为 648,000/1,296,000/648,000/864,000/648,000，最大绝对回放误差为 0.0。

主要结果是：parent 全体 natural return 均值约 0.584、message gap 约 0.566；aligned/hidden/fresh-only/alternating 的 full-support return 为 0.400，conflict 对照为 0.177；conflict−aligned 的 heldout-combination 差为 -0.278（95% CI [-0.536, -0.021]），message-gap 差为 -0.232；conflict/hidden/coadapt 相对 fresh-only 的 full-support heldout-combination 提升约 0.060。伙伴可见性降低跨伙伴一致率，但没有稳定恢复严格留出组合或原子值。

本地只保留源代码、冻结设计、紧凑正式结果、审计和本收据。已删除 25 个临时路径，共释放 1,702,302,369 bytes；包括 formal 原始 execution tree 和此前 pilot/long/smoke 诊断。删除前已核对归档 manifest、结果哈希和 audit 状态，删除后所有列出的临时路径均不存在。
