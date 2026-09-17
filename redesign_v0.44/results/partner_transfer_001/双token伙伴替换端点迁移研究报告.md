# v0.44 双 token 伙伴替换端点迁移：共同符号兼容性研究报告

## 摘要

v0.43 显示，形成阶段的角色随机化可以降低协议对固定资源槽位的依赖。本轮进一步问：一个独立形成的文化中的 sender，能否被另一个 receiver 直接读懂？在相同 seed、partition 和资源列排列内，保留相同冻结视觉前端，只替换通信头，比较不替换、替换一个 sender 槽位和替换全部 sender。结果很清楚：同一文化内的 native endpoint 平均等变 target-60 J 为 41.16%，跨文化只替换一个 sender 时下降到约 9%–13%，替换全部 sender 时降到 0.62%–1.21%；跨文化 token pair agreement 只有 3.44%–5.21%。

这说明 v0.43 的角色对称性并不自动形成跨独立训练运行的公共符号。角色随机化改善的是单一文化内部对角色排列的稳定性；不同随机初始化的文化仍然各自建立了不可直接互读的码本。对语言起源研究而言，这一步把“共同符号能形成”与“共同符号能被新主体/新文化继承”区分开来，并为下一轮真正的社会学习提供了明确基线。

## 研究问题和设计

本轮使用 v0.43 的 432 个形成终点。每个 seed×partition×资源列排列构成一个视觉组，共 72 个组；每组有六个独立文化终点：静态/随机角色顺序 × 固定 A、轮换 A/B、随机 A/B/C 形成拓扑。receiver 来自 recipient culture，sender 可来自 donor culture；donor 与 recipient 在视觉组内共享冻结视觉前端，因此主要测量通信协议兼容性。没有进行任何替换后再训练。

每个组合都在 A/B/C 三种评估拓扑、六种角色排列和四个 team 上测试。替换模式中的 `slot0/1/2` 指评估角色排列下 team 中相应的 sender 位置；`all` 同时替换三个 sender。每个 sender 输出两个 token，词表大小为 7；receiver 输出三个站点动作。报告同时给出 literal 与 target-axis-synchronized equivariant J，避免把资源轴重排误判为兼容性。

正式输出包含 72 个视觉组、6 个文化终点、6 种角色排列和 5 种替换模式。设计：[partner_transfer_design.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.44/partner_transfer_design.json)；输入绑定：[invocation.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.44/results/partner_transfer_001/invocation.json)。

## 结果一：独立文化之间的完整替换几乎不能直接读懂

下表把所有评估拓扑、角色排列、team、seed、partition 和资源列排列合并，报告不同文化距离下的等变 target-60 J。

| donor 与 recipient 关系 | 不替换 | 替换 sender 槽 0 | 替换 sender 槽 1 | 替换 sender 槽 2 | 替换全部 sender |
|---|---:|---:|---:|---:|---:|
| 同一文化 | 41.16% | 41.16% | 41.16% | 41.16% | 41.16% |
| 同角色模式、不同拓扑 | 41.16% | 12.84% | 12.16% | 11.18% | 1.21% |
| 不同角色模式、同拓扑 | 41.16% | 10.89% | 10.30% | 9.48% | 0.74% |
| 角色模式和拓扑都不同 | 41.16% | 10.38% | 9.70% | 8.87% | 0.62% |

同一文化的 native 对照为 41.16%；不替换时 donor 标签不参与消息构造，所有 donor 副本完全一致。不同文化只替换一个 sender 时仍保留约 8.87%–12.84% 的联合 J，全部 sender 替换后只剩 0.62%–1.21%，接近有限任务的低基线。literal 与 equivariant 的差异并不能解释这个下降：问题是 donor 的 token—receiver 码本不兼容，而不是资源轴是否同步。

## 结果二：替换规模揭示组合解码的脆弱性

`all` 替换的 schedule-A、role-012 兼容性矩阵如下。这是一个固定诊断切片，完整平均结果见上一表和图。对角线是同一文化，非对角线是独立文化。

| recipient \ donor | 静态/A | 静态/A-B | 静态/A-B-C | 随机/A | 随机/A-B | 随机/A-B-C |
|---|---:|---:|---:|---:|---:|---:|
| 静态/A | 32.8% | 0.3% | 0.2% | 0.2% | 0.1% | 0.1% |
| 静态/A-B | 0.2% | 24.3% | 0.3% | 0.1% | 0.1% | 0.1% |
| 静态/A-B-C | 0.2% | 0.3% | 26.7% | 0.1% | 0.2% | 0.1% |
| 随机/A | 0.9% | 0.5% | 0.6% | 99.7% | 1.1% | 0.9% |
| 随机/A-B | 0.5% | 1.1% | 0.5% | 1.2% | 41.8% | 2.3% |
| 随机/A-B-C | 0.6% | 0.7% | 0.6% | 1.0% | 2.0% | 34.8% |

矩阵的非对角线接近零，而对角线保留各文化自己的 endpoint 性能；随机角色/A 的对角线在这个切片达到约 99.7%，因为其整体 J 仍受不同评估拓扑和角色排列平均影响。这个对角线/非对角线结构是独立文化码本的直接证据：视觉表征相同并不足以让符号自动公共化。

单 sender 替换的三个位点存在小幅不对称：跨文化平均 J 在 slot0 高于 slot2。它与 sender 位置、receiver actor 输入顺序和当前训练分布共同相关，不能解释为语言中的固定语法层级；后续需要对 sender 槽位做平衡化和随机化。

## 结果三：消息形式的跨文化相似度接近机会水平

在同一视觉组内，比较相同 agent identity、资源和测试世界上的消息。pair agreement 要求两个 token 同时相同。

| 文化关系 | token pair agreement |
|---|---:|
| 同一文化 | 100.00% |
| 同角色模式、不同拓扑 | 5.21% |
| 不同角色模式、同拓扑 | 4.04% |
| 角色模式和拓扑都不同 | 3.44% |

同一文化的 pair agreement 为 100% 是同一 endpoint 的自比较；跨文化关系只有 3.44%–5.21%。词表大小为 7、消息长度为 2，独立随机码的 pair 机会水平约为 1/49=2.04%；观测值略高于机会，但远低于可直接互读所需的公共码本。这与 all-sender replacement 的低 J 一致，表明功能崩溃不是 receiver 偶然对少量 token 过拟合造成的。

## 结果四：角色随机化的作用边界

v0.43 的角色随机化把单一文化内部六种角色排列的 spread 压低，并提高 identity endpoint J；本轮的跨文化替换结果显示，随机角色模式的跨文化 all-sender J 仍只有约 0.53%–1.67%，与静态模式同量级。也就是说，角色随机化解决了“同一文化内部的角色轴稳定性”，没有解决“不同独立文化共享同一符号约定”。

这一区分对机制解释很关键。共同回报和角色置换足以让一组相互训练的 agent 形成可用协议，但独立群体之间没有共享历史或社会学习时，符号的公共性不会凭空出现。要研究语言诞生的社会条件，下一步必须引入真实的跨主体学习路径，而不是只比较独立 endpoint。

## 对核心问题的回答

在当前 grounded 任务中，非语言的视觉能力、离散采样和共同后果反馈足以支持一个文化内部的共同符号系统；它们不足以让独立文化直接互读。协议公共性需要至少一种额外机制：共享初始主体、直接观察/模仿、跨群体交互、代际 bottleneck，或替换后通过共同结果重新学习。角色随机化主要改变协议的对称结构，不能替代文化传递。

这也给“用 agent 模拟语言诞生”的实验设定一个可检验分层：先测同一群体内符号形成，再把一个陌生 sender 或陌生 receiver 引入并允许有限社会学习，最后测新主体是否保留旧码本、是否产生修复与折衷。直接 endpoint 替换是零社会学习基线。

## 限制

- 本轮没有替换后的适应训练，不能给出新主体重新学习协议所需的更新量或代际保真度；
- donor 与 recipient 共享 seed、partition 和资源列排列，隔离了视觉前端差异，但没有测试新 agent 的新感知系统；
- 任务仍是六站点、三资源、双 token 的有限协议，receiver 可能在有限码表上查表；没有开放词汇、指称扩展或生产性组合；
- reward、视觉前端和通信架构沿用 v0.43，仍由实验者集中实现；没有库存、延迟后果、冲突和真实分工；
- schedule-A/role-012 矩阵只是诊断切片，主要结论以跨拓扑、角色和 team 的全量平均为准；
- 资源图片 strata 的统计并不完全匹配，仍需更平衡的视觉 bank。

## 下一轮实验

下一步应在本轮端点基线上加入短程社会学习，而不是重新假设公共符号已经存在：固定一个 resident receiver 和两个 resident sender，只替换一个陌生 sender；给 newcomer 100/300/600 个更新，比较固定、轮换、随机伙伴日程下的恢复曲线。随后再做 receiver 替换和全群体替换，记录 token 是否向 resident 约定收敛、是否出现折衷码本，以及功能恢复和形式变化的时间差。

实验应继续保留 v0.43 的角色 spread、资源排列和 token-position 干预，并把新主体的初始兼容性、重学样本效率、跨代保留率作为预注册主指标。只有当陌生主体可以通过有限社会交互恢复公共符号，才有必要再引入延迟资源库存、互补分工和代际瓶颈。

## 可复核性

独立 NumPy 分析：[partner_transfer_analysis.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.44/results/partner_transfer_001/partner_transfer_analysis.json)，完成 2,271,608 项检查和 2,985,984 个标量比较，native v0.43 provenance mismatch=0，最大保存指标差异 2.78e-08。
独立动作/分数审计：[partner_transfer_audit.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.44/results/partner_transfer_001/partner_transfer_audit.json)，覆盖 72 个视觉组、933,120 个 score cell，重算最大误差 0.0e+00；审计未导入生产模型。
图形：[01_partner_transfer.png](/Users/xia/Documents/ChatGPT/语言/redesign_v0.44/results/partner_transfer_001/figures/01_partner_transfer.png)；视觉 QA：[visual_qa.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.44/results/partner_transfer_001/visual_qa.json)；审查：[结果审查.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.44/结果审查.json)。

本轮是 endpoint 兼容性实验，不是从零训练或代际传递实验。它的作用是为下一轮社会学习提供一个可量化的公共性基线。
