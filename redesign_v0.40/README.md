# redesign_v0.40：三资源三 token 协议的代际传递

本轮把 v0.39 的三资源、三 token 形成终点接入代际替换，检验共同符号在新主体加入后的保真度与漂移。

- 正式矩阵：3 个形成条件 × 3 个传递条件 × 4 seeds × 3 partitions = 108 条链
- 每条链：代际 0–4，依次替换主体 0、1、2、3；每代训练 300 updates
- 世界：6 个地点中的 3 个有序且互不相同的位置，共 120 张地图；每个 partition 使用 60 张训练地图和 60 张 target 地图
- 通信：三个资源 sender 各发送一个 7 值 token，receiver 解码 343 个三 token 组合
- 训练：私有视觉前端冻结；新主体只训练随机重置的通信头；共享 grounded 奖励

正式报告：[三资源三token代际传递研究报告.md](results/triad_transfer_001/三资源三token代际传递研究报告.md)

正式结果：[triad_transfer_analysis.json](results/triad_transfer_001/triad_transfer_analysis.json)、[triad_transfer_audit.json](results/triad_transfer_001/triad_transfer_audit.json)、[completion_manifest.json](results/triad_transfer_001/completion_manifest.json)

源代码：[triad_transfer_train.py](triad_transfer_train.py)、[triad_transfer_analysis.py](triad_transfer_analysis.py)、[triad_transfer_audit.py](triad_transfer_audit.py)、[plot_triad_transfer.py](plot_triad_transfer.py)

本轮仍是有限 grounded 协议的机制探针，不等同于开放式自然语言。代际 0 继承自 v0.39；下一轮应先加入资源/角色置换控制，再测试每个 sender 的两个 token 与分布式社会学习。
