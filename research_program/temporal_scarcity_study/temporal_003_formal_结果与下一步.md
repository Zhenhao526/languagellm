# Temporal scarcity：私有需求、持久库存与记忆的正式确认

这批实验检验一个直接的形成压力：两个 agent 只能看到自己的需求和本地资源，资源库存跨回合持续，通信在前两轮之后进入 blackout。正式矩阵为 `stateless/recurrent × scarce/abundant × PI/FI × silent/live`，8 个 seed、128 个 run、2,000 updates/run；每个 run 保存五个 checkpoint，训练支持和留出分割都比较 natural、closed、token-permuted，另有从头训练的 silent 对照。

独立审计在原始运行树清理前通过：256,000 条训练日志、512 个 checkpoint，最大回放误差为 0。紧凑收据见 [`temporal_003_audit.json`](temporal_003_audit.json)。原始 `training.jsonl` 和 `checkpoint_*.npz` 已在远端同步后从本地删除；当前保留源码、冻结方案、审计收据和本报告。

## Held-out live 结果

| memory | scarcity | information | natural−closed | natural−permuted | token MI |
|---|---|---|---:|---:|---:|
| recurrent | scarce | PI | −0.552 pp [−2.664,+1.559] | −0.005 pp [−0.036,+0.027] | 0.290 bit |
| recurrent | abundant | PI | +1.224 pp [−0.704,+3.153] | +0.025 pp [−0.033,+0.083] | 0.377 bit |
| recurrent | scarce | FI | −0.053 pp [−0.817,+0.711] | +0.098 pp [−0.062,+0.257] | 0.175 bit |
| recurrent | abundant | FI | +1.032 pp [+0.205,+1.860] | −0.062 pp [−0.395,+0.270] | 0.310 bit |
| stateless | scarce | PI | +0.120 pp [−0.164,+0.404] | 0 | 0.162 bit |
| stateless | abundant | PI | 0 | 0 | 0.284 bit |
| stateless | scarce | FI | +0.390 pp [−0.176,+0.956] | −0.060 pp [−0.157,+0.038] | 0.185 bit |
| stateless | abundant | FI | +0.925 pp [+0.522,+1.327] | +0.156 pp [−0.040,+0.352] | 0.217 bit |

训练支持分割更弱：除 stateless/abundant/FI 的小幅差异外，live 的 natural−closed 和 natural−permuted 均接近 0。因而本批没有得到稳定的操作性通信签名；持久库存和 recurrent memory 没有自动产生可置换破坏的公共约定。token 与需求的互信息约为 0.162–0.377 bit，只是描述统计，不能替代干预。

## 解释边界

这不是“语言没有可能”，而是当前 neural self-play 在这个任务和优化预算下没有复现 tabular 正对照的因果信号。更合理的下一步是先做动作能力 warm-start／监督控制，再把 PI 消息差异作为形成主量；随后将对象、属性、搭档、目的地的组合留出接回持久库存环境。该批不涉及 Qwen 或外部模型调用。
