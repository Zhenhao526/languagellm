# redesign_v0.35：身份留出与在线社会适应

这一轮把 v0.34 的伙伴拓扑结果推进到“新主体能否进入既有符号系统”。同一 v0.34-34034 世界和私有视觉接口下，一个 newcomer 保留私有视觉编码器、重置通信模块。第一部分做零样本身份留出；第二部分冻结居民，只允许 newcomer 的通信模块通过 grounded 反馈在线更新 600 次。

正式矩阵为 4 个 seed × 3 个坐标面板 × 2 个训练条件 × 2 个留出身份，共 48 个适应运行；fixed-A 每一步训练 A，rotating-AB 交替训练 A/B，A/B/C 均在终点评估。两资源、双 sender、单 receiver、每个 sender 一个 7 值 token 的任务沿用 v0.34。

主要结果：零样本 newcomer 的目标联合 J 约为随机水平，说明共享私有视觉类型不足以直接读懂 resident 协议；在线适应后 fixed-A 在 A 达到 45.920%，但 B/C 仍约 2%；rotating-AB 在 A/B/C 分别达到 31.105%、31.829% 和 14.815%。相对 fixed-A 的配对终点差异为 A −14.815 个百分点、B +29.442 个百分点、C +13.238 个百分点，显示拓扑轮换牺牲专门化并扩大迁移范围。

报告：[新主体身份留出与在线适应研究报告.md](/Users/xia/Documents/ChatGPT/语言/redesign_v0.35/results/adapt_001/新主体身份留出与在线适应研究报告.md)

零样本结果：[identity_analysis.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.35/results/identity_001/identity_analysis.json)；在线适应结果：[adaptation_analysis.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.35/results/adapt_001/adaptation_analysis.json)。

独立重算：[identity_raw_validation.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.35/results/identity_001/identity_raw_validation.json)、[adaptation_raw_validation.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.35/results/adapt_001/adaptation_raw_validation.json)；审计：[identity_audit.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.35/results/identity_001/identity_audit.json)、[adaptation_audit.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.35/results/adapt_001/adaptation_audit.json)。

图形：[01_adaptation_outcomes.png](/Users/xia/Documents/ChatGPT/语言/redesign_v0.35/results/adapt_001/figures/01_adaptation_outcomes.png)；结果审查：[结果审查.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.35/结果审查.json)。

当前批次仍是小型机制探针：没有大规模开源 VLM/LLM、人口替换、代际传递、端到端视觉学习或自然语言语法。下一轮应加入有限社会学习瓶颈的迭代替换，比较协议保真度、漂移、身份泛化和组合结构的跨代变化。
