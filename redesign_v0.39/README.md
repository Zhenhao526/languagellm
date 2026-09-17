# redesign_v0.39：三资源三 token 共同符号形成

本轮将双资源互补任务扩展为三个资源槽位：apple、banana、orange 各由一个 sender 私有观察并发送一个 7 值 token，receiver 只读取有序三 token 组合并恢复三个地点。

- 正式矩阵：4 seeds × 3 panels × 3 topology schedules = 36 条链
- 世界：6 个地点中的 3 个有序且互不相同的位置，共 120 张地图
- 训练：60 张 partition-specific 地图；测试：全部 120 张地图
- 消息空间：7³ = 343 个三 token 码

第 1200 步 target60 联合 J：fixed-A 为 A/B/C 评估 30.451%/0.451%/0.833%；rotating-AB 为 24.340%/24.097%/14.618%；random-ABC 为 26.667%/26.944%/26.806%。A 评估下三 token joint 一致率分别为 0%、15.521% 和 65.000%。单资源正确率明显高于联合 J，显示了三槽位组合解码瓶颈。

正式报告：[三资源三token共同符号形成研究报告.md](results/triad_001/三资源三token共同符号形成研究报告.md)

可复核文件：[triad_design.json](triad_design.json)、[triad_train.py](triad_train.py)、[triad_analysis.py](triad_analysis.py)、[triad_audit.py](triad_audit.py)、[triad_analysis.json](results/triad_001/triad_analysis.json)、[triad_audit.json](results/triad_001/triad_audit.json)、[completion_manifest.json](results/triad_001/completion_manifest.json)。

本轮仍是机制探针：视觉前端从前序两资源训练继承并冻结，通信头从随机状态开始；共同奖励和梯度更新是中心化的，尚未测试同一 sender 内的序列语法。
