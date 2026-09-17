# redesign_v0.37：无教师条件下的共同符号形成

本轮把所有通信模块重新随机初始化，四个主体在 grounded 双资源任务中同时更新；私有视觉编码器保持冻结。三种伙伴拓扑日程分别为 fixed-A、rotating-AB 和确定性 random-ABC，评估 A/B/C 三套终点拓扑。

正式矩阵为 4 个 seed × 3 个坐标面板 × 3 个日程，共 36 个运行、1200 更新。结果显示：fixed-A 在 A 上达到 46.094%，但 B/C 仍约 1%；rotating-AB 在 A/B/C 达到 30.382%/30.208%/16.696%；random-ABC 在 A/B/C 达到 48.438%/48.032%/48.669%。fixed-A 的 A 终点 grounded J 约 46%，同类型 joint token agreement 只有 1.435%；random-ABC 同时达到约 48% grounded J 和 89.699% token agreement。

报告：[无教师条件下的共同符号形成研究报告.md](/Users/xia/Documents/ChatGPT/语言/redesign_v0.37/results/origin_001/无教师条件下的共同符号形成研究报告.md)

分析：[origin_analysis.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.37/results/origin_001/origin_analysis.json)；独立重算：[origin_raw_validation.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.37/results/origin_001/origin_raw_validation.json)；审计：[origin_audit.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.37/results/origin_001/origin_audit.json)。

图形：[01_origin_outcomes.png](/Users/xia/Documents/ChatGPT/语言/redesign_v0.37/results/origin_001/figures/01_origin_outcomes.png)；审查：[结果审查.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.37/结果审查.json)。

本轮说明冻结的资源感知、离散 token、共同 grounded 后果和同步学习已经足以在受控任务中形成非随机协议，但 fixed-A 的局部配对码与 random-ABC 的群体共享码不同。下一轮应把形成的三类群体接入代际替换，并加入无中心化梯度、可观察 episode 反馈和更大的资源/序列空间。
