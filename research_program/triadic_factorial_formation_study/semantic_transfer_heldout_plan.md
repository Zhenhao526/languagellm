# 因素留出策略的 aligned/placebo 消息转移探针

## 状态

这是对已完成并校正的 `factorial_001_corrected` 冻结策略的后验机制探针，不重新训练，也不修改原因素留出主量。它把原实验的未见对象×长度资源边，转化为“源端消息是否比同层错配消息更能推动目标端行动”的接收读数。

## 冻结输入

- 16 个配对初始化：62101–62116。
- 两个训练覆盖臂：`factorial_holdout`（训练中排除 wood-long 与 fiber-short）和 `saturated`（包含全部支持需求）。
- 两种结算规则：`strict`、`reciprocal`；两种通道：`PL_live`、`PL_silent`。
- 只读取每个策略 `heldout_both` 的 6 个留出布局×6 个站点所有者完整评价文件和 6000 步 checkpoint。
- 候选边来自冻结 `need_response_cases`：改变一个主体的一个因素、两端合法搭档对不同的 1,284 条无向边；每条边展开为两个方向，共 2,568 个有向案例。每案 36 个背景。

## 干预

aligned 把源端改变主体的首窗消息放入目标世界的同一发送者位置，保留目标自视角和其他路由块，重新计算第二窗与三主体动作。placebo 在 `changed_person × axis × direction` 层内循环错配源消息，保持发送者、轴、方向和消息边际而打破源需求对齐。`PL_silent` 两种干预都是自然别名。

对每个目标世界，用冻结 settlement kernel 得到自然和干预的实际搭档对。`source_pull` 定义为源端搭档对命中率增量减去目标端搭档对命中率增量。主机制量是 aligned−placebo；绝对 aligned 只作为描述性任务敏感性。

## 统计

每个种子、训练覆盖臂、结算规则和掩码内先对九个主体×轴结构等权，再对三个轴等权。预先报告：aligned、placebo、aligned−placebo 的 `source_pull`，动作／搭档改变率，以及 `(factorial live−silent)−(saturated live−silent)` 的机制交互。种子层区间为四个配对初始化组内或全16种子的描述性 Student-t 区间；世界和背景不是独立社会样本。

这项探针不能单独证明词义、组合语法或语言起源。只有未见组合上正确源消息相对错配消息稳定为正，并且在新布局、新主体或新组合上复用，才构成进一步进入形成与代际实验的理由。
