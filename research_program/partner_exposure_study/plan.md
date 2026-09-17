# 实验计划：多伙伴暴露下的发送者协议对齐

## 核心问题

上一轮群体异质性实验把 child 固定在 worker 0，导致 worker 1、3 的表面约定没有进入 child 流。本轮把替换角色改为发送者：新发送者在每个 batch 中实际面对四个轮换伙伴。这样异质性、伙伴身份可见性和共同码本是否形成都可识别。

## 环境与训练

- 两阶段、两对象任务，发送者看到目标二元组，worker 只能看到经伙伴约定映射后的表面位置。
- 消息固定为两个二值 token，在第 1、3 个时间步到达；奖励来自两个阶段的取物结果。
- parent 同时训练隐藏伙伴身份的发送者和四个 worker；均质群体的四个 worker 使用 identity，异质群体中 worker 1、3 使用 swap。
- child 复制 parent 的四个 worker，重新初始化发送者；每个 child batch 混合四个伙伴。`hidden` 要求一个发送码本服务全部伙伴，`visible` 允许按伙伴维护码本。
- `sender_only` 冻结 worker，`coadapt` 同时更新发送者和全部 worker；`leave_one_out` 在发送者适应期间遮蔽一个联合目标，检验留出组合。

## 预注册读数

1. 主要读数：fresh sender 的 leave-one-out、hidden/visible、live held-out natural return。
2. 对齐机制：visible−hidden 的回报差和发送码本跨伙伴一致率。
3. 群体干预：heterogeneous−homogeneous，环境流逐 seed 成对共享。
4. 通信因果性：natural−permuted、live−silent 和 all-goal 的消息作用。
5. 组合性：双 token 的 raw slot recombination 与 pairwise Hamming 结构。

`0.60` 只作功能计数阈值，连续回报和配对区间是主要统计。结果只能支持有限离散协议的形成或对齐，不等同于人类语言或开放语法。

## 复现命令

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.partner_exposure_study.runner prepare --out /tmp/partner_exposure_prepared
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.partner_exposure_study.runner execute --prepared /tmp/partner_exposure_prepared --out /tmp/partner_exposure_execution --updates 3000
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.partner_exposure_study.audit --prepared /tmp/partner_exposure_prepared --execution /tmp/partner_exposure_execution/execution --out /tmp/partner_exposure_audit.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ./.exp_venv/bin/python -m research_program.partner_exposure_study.aggregate --results /tmp/partner_exposure_execution/execution/results.json --out /tmp/partner_exposure_analysis.json --markdown /tmp/partner_exposure_analysis.md
```
