# Structured factor-sharing 正式批次同步与清理

正式批次使用源代码提交 `6480b69ca505fab007ff9fc0c1e88785bda0c32e`，归档位于 `research_program/structured_factorization_study/results/formal_20260918/`。任务和消息容量与社区冲突 referential game 完全相同，只改变策略的参数共享结构：`holistic` 使用完整意义/消息表，`factorized` 按属性值共享发送参数并按 slot 加和接收证据。

矩阵包含 9 个 seed、72 个 parent、216 个 child，每个 run 3,000 updates。独立 replay audit 通过，最大回放误差为 0；重放训练日志 216,000/648,000 行，checkpoint 288/864 个；architecture/visibility/population/support 配对流分别为 324,000/324,000/324,000/432,000 行。

主要结果：factorized/aligned/hidden 的 held-out-combination return 为 0.974，9/9 达到 0.60；holistic 对照为 -0.220，0/9 达到；配对差为 1.194（95% CI [1.127, 1.260]）。factorized/aligned/hidden 的 held-out-value return 为 0.122，0/9 达到，区分了已见原子值的组合复用与未见原子值的词义学习。factorized/conflict/hidden 的 held-out-combination 为 0.434；伙伴 identity 可见时为 0.974，说明社区冲突和专用码本可以通过身份条件被分离。

这不是无先验主体自发发明组合语法的证明，而是一个可识别的架构干预：当前环境压力单独不足以产生组合复用，显式 factor sharing 可以使组合协议传递。

本地已删除 19 个临时路径，共释放 1,080,825,210 bytes；删除前已验证 archive manifest 和 audit，删除后所有临时路径均不存在。
