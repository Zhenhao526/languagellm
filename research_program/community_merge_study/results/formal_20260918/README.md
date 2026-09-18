# 社区码本冲突与角色对称 referential game：正式归档

本归档对应源代码提交 `ec3b9e19bb262b3b84eb7e64b3d22aa43e44c9ab`，远端仓库为 `git@github.com:Zhenhao526/languagellm.git`。任务让 sender 只观察目标对象的两个三值属性，receiver 观察四个候选对象，并通过四符号、双槽位消息选择目标。parent 在两个角色间轮换；child 通过四个伙伴进行 `alternating` 或 `sender_only` 适应。aligned 与 conflict 的唯一区别是 community 1 的外部 token 双交换。

- 设计：9 个 seed、36 个 parent、432 个 child；每个 run 3000 updates。
- 审计：`passed`；重放误差 `0.0`；训练日志 `108,000` + `1,296,000` 行；checkpoint `144` + `1728` 个。
- 配对流：visibility `648,000`、adaptation `1,296,000`、population `648,000`、support `864,000`、role `648,000` 行。
- parent 全体 natural return 均值约 `0.584`，message gap 约 `0.566`。
- aligned/hidden/fresh-only/alternating 的 full-support return 为 `0.400`；conflict 对照为 `0.177`。conflict−aligned 的 heldout-combination 差为 `-0.278`，95% CI [-0.536, -0.021]；message-gap 差为 `-0.232`。
- conflict/hidden/coadapt 相对 fresh-only 的 full-support heldout-combination 提升约 `0.060`；visible/hidden 的 fresh-sender 跨伙伴一致率分别为 `0.481` 与 `1.000`。

`results.json` 是完整 formal 汇总结果；原始逐步日志和 checkpoint 没有放进归档，审计和清理收据记录了其完整性与删除。
