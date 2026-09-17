# 多搭档生态的信息权限与通信通道对照

## 研究问题

前一轮 PL 训练中，live 消息显著提高物理执行，却没有稳定提高执行后的搭档选择。本轮加入 full-information（FI）训练，对照“主体是否需要通过通信获得其他人的需求事实”。如果 FI silent 已能达到较高的执行后选择，而 FI live 没有额外收益，则可把 PL live 的作用定位到信息缺口和实时通道；如果 FI live 仍改变行为，则说明通道本身或整包编码具有独立作用。

## 冻结条件

- 16 个独立种子 66701–66716；每个种子比较 `static/rematched × FI_live/FI_silent`，共64次训练、每次6,000步。
- 任务、两个不同搭档对的 full-success 方案、neutral/engage 意图头、16类 proposal、两个四符号窗口、strict结算和六选一重配均与 `altpair_001` 相同。
- FI 观察包含三名主体的需求和公共布局；主体之间仍没有直接跨人通道。FI silent 是信息充分且闭通道的能力上界参照，FI live 检验已有完整事实时实时消息是否仍有边际作用。
- PL live/PL silent 结果直接绑定已审计的 `altpair_001`，只在汇总阶段与本轮 FI 结果比较，不改写原运行树。

## 预注册读数

本轮内部主量是 FI 条件的重配×通信交互：

`[(rematched FI_live − rematched FI_silent) − (static FI_live − static FI_silent)]`

按每个种子的六个检查点做中心化时间 AUC，再在16个种子上计算近似 t15 区间。次量为 FI live/silent 的团队 Q、`Q|physical`、物理执行率、目标搭档率、proposal 合法率、engage/neutral 和第三人 neutral。跨信息层的探索性比较为 `FI_silent − PL_silent` 以及 `FI_live − PL_live`，不得把它们当作同一训练群体内的因果差异。

## 数据完整性与边界

FI 与 PL 共享任务、初始化规则、世界和消息随机流；但它们是不同观察权限下的独立训练策略。每个 FI 发送者前向使用 `x_FI`，并核查 FI 标志和三人需求块确实可见；不使用监督动作或研究者标签。独立审计逐 checkpoint 重放四臂 FI 网格和配对流。

FI 能力上界只能说明任务在给定接口下可解，不能证明形成了词义或语言。任何 live/silent 差异仍可能来自固定整包代码、策略扰动或奖励信用分配；只有跨需求、跨搭档和重编码下稳定的接收行为才值得进入新主体或代际传递。
