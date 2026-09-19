# 2026-09-19 同步与本地清理记录

正式批次 `open_world_grounding_incumbent_20260919` 已同步到 `git@github.com:Zhenhao526/languagellm.git` 的 `main`。实验代码和报告在提交 `9cf2acbe`，冻结结果归档在提交 `41dd3c91bd8bddca494471efaf651bbd40d79dd1`；推送后通过 `git ls-remote` 核实远端 `main` 指向该归档提交，之后才清理临时原始数据。实验冻结基于前序仓库提交 `0a83039f69874026070f970fdfc8bc3ef7857b5a`，精确代码快照与源文件哈希保存在 `research_program/open_world_grounding_incumbent_study/results/formal_20260919/`。

归档包含 36 个 parent、18 个 expansion child 和 288 个 transfer run 的紧凑结果，六片 replay audit（最大误差 0）以及 144 次 incumbent receiver 参数复制核验。factorized incumbent receiver 在 reverse 表面下，未见 double-new 初始回报为 −0.250，single-new 反馈后为 1.000（9/9）；factorized 的表面×初始化交互为 −1.304，配对 95% t 区间 [−1.663, −0.945]。该环境只使用表格对象标签置换，不含图像或视觉模型。

远端核实后，本地删除 34 个临时路径，共 2,478 个文件、1,161,803,240 字节。完整文件清单、路径缺失核验和归档 manifest 哈希见 [本地清理收据](LOCAL_CLEANUP_RECEIPT_2026-09-19_open_world_grounding_incumbent.json)。本地保留研究代码、冻结方案、紧凑结果和审计记录。
