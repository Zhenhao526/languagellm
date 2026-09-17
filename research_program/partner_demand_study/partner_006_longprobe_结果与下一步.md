# Private-demand 改进信用分配 longprobe

`partner_006` 使用当前工作树的改进消息信用分配：消息的 score target 使用伙伴的未来收益，并在可观察状态组内做 leave-one-out baseline；错误动作代价为 −0.25，策略仍不接收结果正确性反馈。该批只选 1 个 seed（68101）、4 个条件、2,000 更新，目的是检查新梯度是否能把消息相关性转成伙伴收益。

held-out 终点中，recurrent＋scarce＋PI 的 live 与 silent 分别为 **0.1044** 和 **0.1239**，live−silent 为 **−1.95 个百分点**；live 的 natural、closed、permuted 三种评价相同。recurrent＋abundant＋PI 的 live 与 closed 也相同（约 **0.372**），stateless＋abundant＋PI 的 live 与 closed 也相同（约 **0.373**）。消息互信息仅在第一个 live 条件约 **0.077 bit**，其余接近零。

独立审计通过 4 个 run、8,000 条训练日志和 24 个最终评价块，非有限诊断为 0，oracle 一致性最大误差为 0。该批只有一个 seed 和四个选择条件，不能作正式零结果；它没有显示改进信用分配已经产生可操作的自然通信增益。

下一步固定 `partner_006` 源快照，先做 4 个 seed 的 PI×scarce×{persistent,switching} 与 FI 对照，再决定是否扩展到完整 32 条件。主量仍是 natural−closed 与 natural−permuted 的 paired contrast；只要两者不分离，就不把消息互信息或 token 熵当作语言证据。
