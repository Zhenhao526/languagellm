# 因子化中性行动的搭档集合方向探针

本包冻结并读取 `triadic_factorized_neutral_altpartner_study/results/altpair_001` 的64个最终策略。它修正前一轮单轴近邻的识别边界：单轴改变不会改变合法搭档集合，本轮只保留一个主体的两轴复合需求端点，并要求供体与接收世界的合法搭档集合不同。

- [冻结方案](plan.md)
- [执行结果与下一步](结果与下一步.md)
- [独立审计](results/audit_partner_direction_001/verification.json)
- [JSON-only汇总](results/aggregation_partner_direction_001/results.json)

案例不按策略行为筛选。live 只替换指定发送者的首窗口外发包并重算第二窗口与行动，silent 是闭通道自然参照；每个发送者前256行做逐项 sham。

即使搭档转移为正，也只能支持任务内的消息—行为响应，不能称词义、组合语言、代际传递或人类语言起源。
