# 全局奖励尺度零假设对照

## 问题

上一轮的 `c75` 条件把第三方冲突世界的回报乘以 0.25，观察到参与结构和通信增益改变。这里检验这个效应是否只是“所有奖励变小”造成的。若训练目标是逐状态的 `log` 精确期望回报，统一乘以常数只会给 `log_J` 加上常数；接收端对策略 logits 的导数和两条轨迹的发送端优势都应保持不变。

## 条件

`c0` 使用 reciprocal 结算和原生 0/.5/1 回报；`c75` 只对匹配搭档之外仍 engaged 的第三人分支乘 0.25；`global25` 对所有 native reward 统一乘 0.25，但物理结算与 `c0` 完全相同。三者都交叉 PL/FI、static/rematched 和 live/silent。8 个新初始化 67001–67008，每格 3000 更新，检查点为 0/100/500/1500/3000。世界、重配、消息随机流和初始参数在每个种子内配对。

## 主要读数

主对照是 `c75 − global25` 的 live−silent 目标搭档合法率，以及物理执行率；按种子先在 static/rematched 两种日程内平均，再以检查点真实间距做中心化梯形 AUC和终点差。零假设读数是 `global25 − c0`，其置信区间应覆盖零，且模型参数／行为轨迹只出现机器精度级差异。Q、reward、提案合法率、第三人冲突率和绝对终点为预定次量。

另外在新 logits 上逐项检查 `global25` 的期望回报缩放、`log_J` 常数平移、解析梯度和有限差分。该检查不是训练结果的替代，而是确认实现与理论零假设一致。

## 解释边界

若 `global25 − c0` 接近零而 `c75 − global25` 保持上一轮方向，支持“冲突位置而非全局回报尺度”这一窄机制解释。若全局对照也偏离，则必须把数值、熵项、裁剪或优化器尺度作为未排除混淆。无论结果如何，本实验只测任务级协调和优化机制，不能推出词义、组合语法、公共词典、文化传承或人类语言起源。

## 复现

```sh
PYTHONPATH=. .venv/bin/python research_program/triadic_global_reward_control_study/tests/test_design.py
PYTHONPATH=. .venv/bin/python -m research_program.triadic_global_reward_control_study.runner prepare --out research_program/triadic_global_reward_control_study/results/global_001
PYTHONPATH=. .venv/bin/python -m research_program.triadic_global_reward_control_study.runner execute --out research_program/triadic_global_reward_control_study/results/global_001
PYTHONPATH=. .venv/bin/python -m research_program.triadic_global_reward_control_study.audit --source research_program/triadic_global_reward_control_study/results/global_001 --output research_program/triadic_global_reward_control_study/results/audit_global_001 --workers 2
PYTHONPATH=. .venv/bin/python -m research_program.triadic_global_reward_control_study.summarize --source research_program/triadic_global_reward_control_study/results/global_001 --output research_program/triadic_global_reward_control_study/results/summary_global_001
```
