# 多搭档生态的信息权限与通信通道对照

本包训练同一多搭档任务的 FI（全需求可见）策略，比较 static/rematched × live/silent 四格，并在汇总阶段与已审计的 PL `altpair_001` 结果对照。FI silent 是闭通道的信息充分参照，FI live 检验完整事实下实时消息是否仍改变协调。

- [冻结方案](plan.md)
- [执行结果与下一步](结果与下一步.md)
- [独立审计](results/audit_fi_001/verification.json)
- [JSON-only汇总](results/aggregation_fi_001/results.json)
- [单槽位 payload／符号探针结果](single_slot_结果与下一步.md)
- [单槽位独立审计](results/audit_single_slot_fi_001/verification.json)
- [单槽位汇总](results/single_slot_fi_001/summary_001/summary.json)
- [单槽位 provenance](results/single_slot_fi_001/provenance.json)

本轮不使用LLM、教师标签或外部API；即使FI策略达到高Q，也只能说明任务可解，不能称词义、组合语言或语言起源。
