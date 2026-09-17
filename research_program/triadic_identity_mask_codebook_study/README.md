# 身份歧义 × 公私码本实验

该包是三人有限符号协调研究的下一轮判别实验。它只改变发送者槽位是否稳定可识别，并交叉比较共享公共八符号双射与三套发送者私有双射。

- `design.py`：冻结 seed、条件、码本和分区。
- `wire.py`：消息重编码、随机槽位路由和梯度包装。
- `runner.py`：准备、验证、训练、评价和聚合。
- `audit.py`：独立重算指标、消息映射、路由排列和哈希链。
- `plan.md`：研究问题、控制、主量和主张边界。
- `artifacts/`：本地生成的参数、日志和 NPZ，不提交到 Git。

运行入口：

```bash
PYTHONPATH=. .exp_venv/bin/python -m \
  research_program.triadic_identity_mask_codebook_study.runner \
  prepare --out research_program/triadic_identity_mask_codebook_study/artifacts/identity_mask_001
PYTHONPATH=. .exp_venv/bin/python -m \
  research_program.triadic_identity_mask_codebook_study.runner \
  verify --out research_program/triadic_identity_mask_codebook_study/artifacts/identity_mask_001
PYTHONPATH=. .exp_venv/bin/python -m \
  research_program.triadic_identity_mask_codebook_study.runner \
  execute --out research_program/triadic_identity_mask_codebook_study/artifacts/identity_mask_001
PYTHONPATH=. .exp_venv/bin/python -m \
  research_program.triadic_identity_mask_codebook_study.runner \
  summarize --out research_program/triadic_identity_mask_codebook_study/artifacts/identity_mask_001
PYTHONPATH=. .exp_venv/bin/python -m \
  research_program.triadic_identity_mask_codebook_study.audit \
  --out research_program/triadic_identity_mask_codebook_study/artifacts/identity_mask_001
```

主量是联合留出上的
`(masked_public_live − masked_silent) − (masked_private_live − masked_silent)`。
即便该量为正，也只表示身份歧义下的约定对齐迹象；本包不把它称为词义、组合语法或语言起源。
