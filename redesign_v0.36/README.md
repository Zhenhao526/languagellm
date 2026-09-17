# redesign_v0.36：有限社会学习瓶颈下的代际传递

本轮将 v0.35 的一次性 newcomer 适应推进到序列化文化传递。四个主体从同一个 v0.34 fixed-A 终点开始；每一代按 `0,1,2,3,0,1,2,3` 重置一个身份的通信模块，只通过当前三名居民的 grounded 任务反馈训练 300 次，然后把新主体安装回群体。所有条件共享 generation 0，条件差异从第一次替换开始。

正式矩阵为 4 个 seed × 3 个坐标面板 × 3 个训练日程，共 36 条链、288 次替换。比较 fixed-A、rotating-AB 和确定性 random-ABC；每代评估 A/B/C 三套伙伴拓扑，并记录相邻代 token 输出变化。

第 8 代结果：fixed-A 的 A/B/C J 为 58.536%/2.431%/2.431%；rotating-AB 为 23.264%/20.920%/5.122%；random-ABC 为 36.111%/34.201%/34.722%。random-ABC 让三种拓扑接近，但 A 协议相邻代 token 变化率仍为 9.031%，高于 fixed-A 的 2.370%。

报告：[有限社会学习瓶颈下的代际传递研究报告.md](/Users/xia/Documents/ChatGPT/语言/redesign_v0.36/results/iterated_001/有限社会学习瓶颈下的代际传递研究报告.md)

分析：[iterated_analysis.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.36/results/iterated_001/iterated_analysis.json)；独立重算：[iterated_raw_validation.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.36/results/iterated_001/iterated_raw_validation.json)；审计：[iterated_audit.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.36/results/iterated_001/iterated_audit.json)。

图形：[01_iterated_outcomes.png](/Users/xia/Documents/ChatGPT/语言/redesign_v0.36/results/iterated_001/figures/01_iterated_outcomes.png)；审查：[结果审查.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.36/结果审查.json)。

这仍是文化传递机制探针：generation 0 已有协议，没有大规模开源 VLM/LLM、端到端视觉学习、人口选择、词汇扩张或组合语法。下一轮从随机通信模块开始，让多个主体共同建立协议，再把形成的群体接入这条代际链。
