# 开放世界 grounding transfer 同步与清理

本轮新增 `open_world_grounding_transfer_study`，把已形成的 single-new 文化传递与 fresh receiver 的对象表面双射放入同一正式矩阵。父代 54 个、expansion child 27 个、transfer 216 个；identity/reverse mapping 共用语义场景、目标、伙伴和随机流。三片 expansion 与三片 transfer replay audit 均通过，最大绝对重放误差为 0，incumbent 参数哈希未改变。

正式归档：`research_program/open_world_grounding_transfer_study/results/formal_20260918/`。归档 manifest 载荷为 35 个文件、7949602 bytes；清理前已逐文件验证哈希。

清理删除了 2179 个文件、974114379 bytes 的临时准备目录、训练日志、checkpoint、合并中间结果和审计临时文件；完整历史仍在 Git 提交中，当前工作树只保留源码、冻结方案、紧凑结果、报告和回执。
