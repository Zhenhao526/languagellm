# Public/private symbol-codebook training study

This package turns the earlier post-hoc token-recoding control into a training-time comparison. It keeps the audited PL reciprocal task and compares an identity wire, a stable public token bijection, sender-private token bijections, and a from-scratch silent control. All arms share the seed-specific initialization and exogenous world/message streams.

The wire map is applied immediately before messages enter the routed receiver input. It is stable for the whole run and frozen before training. The study measures task execution and later cross-sender content-transfer probes separately; neither is treated as a language score.


本轮已完成[训练期公共／私有码本 pilot](结果与下一步.md)：32 个 run、192 个 checkpoint 和 288 个完整分区紧凑评估，独立回放 max_abs_error=0。public_live−private_live 的目标 Q 终点为 +1.70 pp，t(7) 区间跨零；跨发送者四选一内容 probe 也没有公共码本优势。下一步把发送者身份线索作为独立因素操纵，再检验共享约定是否能迁移到新发送者。
