# 三资源三 token 共同符号形成：v0.39 实验报告

## 研究问题

v0.37/v0.38 在双资源任务中显示，伙伴拓扑可以决定协议是局部配对码还是跨拓扑共享码。本轮增加第三种资源和第三个互补 sender，让 receiver 只从一个有序三 token 组合恢复三个资源的位置。目标是检查共同符号形成是否能扩展到三个槽位，以及形式一致、分资源正确率和联合 grounded 成功是否仍然分离。

本轮所有通信模块从随机状态开始；私有视觉前端冻结。三种训练日程仍为 fixed-A、rotating-AB 和 random-ABC，因此可以把三资源结果与之前的双资源结果直接对照。

## 实验条件

世界由 6 个地点中的 3 个有序且互不相同的位置组成，共 120 种地图。三个 sender 分别只观察 apple、banana、orange 三种资源视图，各发送一个 7 值 token；receiver 读取 3 token tuple（343 个可能码）并输出三个地点。收益为 `(1/6)*sum(correct_resources)+(1/2)*all_three_correct`，随机通信头没有 token 监督。训练地图为每个 partition 的 60 个三元组，测试包含全部 120 个三元组。

正式矩阵为 4 个 seed × 3 个坐标面板 × 3 个伙伴拓扑，共 36 条链；每条链训练 1200 更新，在 0、100、600、1200 更新评估 A/B/C。

执行绑定：[invocation.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.39/results/triad_001/invocation.json)；训练完成：[training_complete.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.39/results/triad_001/training_complete.json)。

## 结果一：三资源组合协议可以从随机头形成

下表给出第 1200 步 target60 联合 J；括号内是跨 12 条 seed×panel 运行的 SD。

| 训练日程 | A 评估 | B 评估 | C 评估 |
|---|---:|---:|---:|
| fixed_A | 30.451%（5.141%） | 0.451%（0.996%） | 0.833%（1.066%） |
| rotating_AB | 24.340%（10.061%） | 24.097%（9.679%） | 14.618%（7.737%） |
| random_ABC | 26.667%（11.836%） | 26.944%（11.435%） | 26.806%（11.844%） |

random-ABC 在三种评估拓扑上形成了接近的联合 grounded 读出；fixed-A 和 rotating-AB 的联合 J 需要结合单资源结果判断，因为单个资源可能已经学会，而三者尚未同时正确。

## 结果二：三 token 的组合一致率与任务成功仍然是两个维度

下表给出 A 评估下同类型 sender 的 token 一致率。`joint` 要求三个资源槽位同时一致。

| 训练日程 | 资源 0 | 资源 1 | 资源 2 | 三 token joint |
|---|---:|---:|---:|---:|
| fixed_A | 16.667%（7.946%） | 16.667%（7.946%） | 16.667%（7.946%） | 0.000%（0.000%） |
| rotating_AB | 54.861%（12.542%） | 54.861%（12.542%） | 55.556%（12.975%） | 15.521%（15.498%） |
| random_ABC | 87.500%（10.952%） | 87.500%（10.952%） | 87.500%（10.952%） | 65.000%（28.683%） |

若三 token 只是在表面上趋同而没有同时提高联合 J，它们不能被称为共享语言。相反，如果联合 J 提高但 token 一致率不高，说明 sender 可以与特定 receiver 形成关系特定的组合码。两类指标必须同时报告。

## 结果三：分资源正确率显示组合瓶颈的位置

下表是 A 评估下第 1200 步的三个资源正确率。

| 训练日程 | 资源 0 | 资源 1 | 资源 2 |
|---|---:|---:|---:|---:|
| fixed_A | 61.076% | 60.972% | 95.000% |
| rotating_AB | 55.243% | 58.438% | 88.819% |
| random_ABC | 57.292% | 58.924% | 91.632% |

三资源任务比双资源任务增加了一个必要的联合约束：即使每个 sender 的槽位输出都出现局部规律，receiver 仍需要学习 343 码中的组合到三地点动作的映射。若单资源正确率提高而联合 J 仍接近随机，瓶颈位于组合解码而非词槽形成。

## 机制解释

1. **组合槽位是比双资源互补更强的检验。** 三个 sender 的局部输出只有在 receiver 的三 token 解码中共同产生后果，才构成有效协议。
2. **伙伴覆盖仍是形成条件。** fixed-A 只覆盖一种角色排列；rotating-AB 和 random-ABC 迫使同一槽位在更多主体关系中保持可读。
3. **一致率不能替代 grounded 结果。** 三 token 的形式收敛可能是无功能的常量；联合 J 也可能来自关系特定编码而非群体公共码。
4. **任务复杂度开始触及容量瓶颈。** 从 49 个双 token 码扩展到 343 个三 token 码后，需要独立报告每个资源槽位、联合动作和留出地图上的性能。

## 与语言起源问题的关系

本轮把“共同符号形成”从两个互补信息源推进到三个信息源的组合任务。若协议在三资源条件下仍能形成并跨拓扑读出，说明共同 grounded 反馈与伙伴覆盖足以支持有限的多槽位符号组合；若性能显著下降，则提供一个可量化的组合复杂度边界。两种结果都比单纯增加训练步数更能说明非语言能力与协议结构之间的关系。

## 限制

- 三种资源视图使用已有 apple/banana/orange 图像特征；视觉编码器继承自前序两资源训练并被冻结。
- 每个 sender 只发一个 token，尚未测试同一 sender 内的序列语法或词汇增长。
- 共同奖励和联合梯度仍由实验者中心化计算；没有完全分布式的社会学习。
- 四个主体、六个地点和固定角色日程是受控机制探针，不是人类社会历史重建。

## 可复核性

独立 NumPy 重算通过 91747 项检查和 20736 个标量比较，最大指标误差为 0.0；覆盖 1728 张协议表。trace 审计通过 18883 项检查，覆盖 69120 行 trace、288 个 trace 文件。

图形：[01_triad_outcomes.png](/Users/xia/Documents/ChatGPT/语言/redesign_v0.39/results/triad_001/figures/01_triad_outcomes.png)；图形源数据：[figure_source.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.39/results/triad_001/figures/figure_source.json)；视觉 QA：[visual_qa.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.39/results/triad_001/figures/visual_qa.json)。
分析：[triad_analysis.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.39/results/triad_001/triad_analysis.json)；独立重算：[triad_raw_validation.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.39/results/triad_001/triad_raw_validation.json)；审计：[triad_audit.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.39/results/triad_001/triad_audit.json)。
设计：[triad_design.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.39/triad_design.json)；执行：[triad_train.py](/Users/xia/Documents/ChatGPT/语言/redesign_v0.39/triad_train.py)。

## 下一步实验

将三资源协议接入 v0.38 的代际替换，测量三槽位组合在主体替换后的保真度；随后增加每个 sender 的第二个 token和完全分布式 episode 反馈，区分槽位组合、序列结构和社会传递三种复杂度来源。
