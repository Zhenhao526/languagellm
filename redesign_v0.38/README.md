# redesign_v0.38：形成协议的代际传递

本轮把 v0.37 的三种无教师形成终点接入同一套有限社会学习瓶颈，形成条件与传递条件做 3×3 全因子比较。

- 形成条件：`origin_fixed_A`、`origin_rotating_AB`、`origin_random_ABC`
- 传递条件：`fixed_A`、`rotating_AB`、`random_ABC`
- 正式矩阵：4 seeds × 3 panels × 3 × 3 = 108 条链
- 每条链：依次替换 4 个身份，每次 300 次通信模块更新，评估 A/B/C 三种拓扑
- 任务：两个互补资源，各 sender 发一个 7 值 token，receiver 输出资源位置

主要结果：fixed-A 形成的局部协议在固定 A 传递下由 46.094% 提升到 60.359%，在轮换/随机传递下分别降至 13.600%/9.057%；random-ABC 形成的协议在三种传递条件下第 4 代 A 评估仍为 62.934%–66.291%，并在 B/C 上保持相近读出。这说明协议形成时的伙伴覆盖与代际传递时的伙伴覆盖存在交互。

正式报告：[形成协议的代际传递研究报告.md](results/transfer_001/形成协议的代际传递研究报告.md)

可复核文件：

- [transfer_design.json](transfer_design.json)
- [transfer_train.py](transfer_train.py)
- [transfer_analysis.py](transfer_analysis.py)
- [transfer_audit.py](transfer_audit.py)
- [transfer_analysis.json](results/transfer_001/transfer_analysis.json)
- [transfer_audit.json](results/transfer_001/transfer_audit.json)
- [completion_manifest.json](results/transfer_001/completion_manifest.json)

本轮仍是机制探针：私有视觉编码器冻结，通信头才是随机初始化和更新部分；共同反馈由实验者定义，群体只有四个主体和两个资源，不涉及自然语言语法。
