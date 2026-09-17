# v0.7 视觉地点局部性与共同消息

状态：52组社会实验及12组个人能力诊断全部完成。24个诊断主体正常观察时均为9600/9600，清空视觉表示后均为1600/9600。独立执行审计874,023项检查通过，0失败；全部104个通信方向完成预定10,400次整体重编码参照。

- [完整实验报告](results/binding_001/analysis/视觉地点局部性实验报告.md)
- [协议结构与干预结果](results/binding_001/protocol_analysis.md)
- [完整执行审计](results/binding_001/audit_execution.json)
- [独立汇总及逐种子记录](results/binding_001/analysis/summary.json)

四种子先各自平均三划分和两方向，再等权汇总：

| 视觉输入 | 训练图单目标 | 留出图单目标 | 留出自然双目标 | 人工拼接双目标 | 拼接重编码参照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| I 保留地点槽 | 85.30% | 47.69% | 2.82% | 21.79% | 0.67% |
| P 置换地点槽 | 80.97% | 43.93% | 1.91% | 14.39% | 0.52% |
| Q 正交混合地点 | 80.77% | 44.59% | 1.65% | 15.54% | 0.62% |

预定I−Q留出单目标差为+3.09个百分点，四种子中一个反向；P−Q为−0.66个百分点。P同样保留局部槽却未稳定优于Q，结果不足以支持一般的局部表示优势。全30图通信I/P/Q分别为81.90%、81.15%、84.60%，关闭通道为16.67%；Q也能学会功能通信。三种表示均出现有限片段复用，但自然消息可靠表达未见资源组合的比例仍低，人工拼接不能替代自主产出。这些是四种子、旧照片上的探索结果。

研究问题：相同原始视觉信息进入共享非线性编码器时，保留地点局部性是否更有利于主体自主表达新的资源组合？这是提供表示方式的比较，不将可逆混合解释为移除世界知识。

- [训练前固定的执行方案](固定执行方案.md)
- [独立设计审查](design_review.md)
- [执行审查](execution_review.md)
- [独立测试](test_binding_independent.py)

主体仍使用官方冻结DINOv2 ViT-L/14和旧44训练/16开发留出照片。六地点、两资源、三个匹配留出划分、两符号49码、直接训练2400×512均固定。三种条件的可训练参数量、参数初值及接收器完全相同，只改私有固定矩阵：identity保留地点槽、permute置换地点槽、orthogonal稠密正交混合地点。矩阵同时变换各地点的64维视觉与存在位，然后经过共享Linear65→65+Tanh及相同96维GRU；不存在原始位置或存在位旁路。

原始变换在非线性编码前可逆，后续激活、信息保留和学习难度不必等价。P对照检查固定地点编号的影响；主比较预先定为identity−orthogonal，不择优挑identity/permute。四个新种子26101–26104；三划分为种子内重复。

个人诊断覆盖全部正式私人矩阵，在独立副本上学习看图采集，保存每人≥90%的整数门槛及连续成绩。诊断后的phi/GRU/行动头不回灌社会模型；它验证有限预算下的可学习性，不表示社会训练开始前的新接口已经掌握完整地图任务。个人每人每更新512次选择，社会每主体对每更新512段，两种预算单位分开记载。

## 复现

在项目根目录使用现有环境执行。先跑个人诊断，再执行主社会训练；独立重跑使用新目录。已有完整且来源一致的社会运行会跳过，部分结果不覆盖。

```bash
redesign_v0.3/deployment/.venv/bin/python -m unittest discover -s redesign_v0.7 -p 'test_*.py' -v
redesign_v0.3/deployment/.venv/bin/python redesign_v0.7/run_controls.py --out redesign_v0.7/results/binding_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.7/run_experiment.py --out redesign_v0.7/results/binding_001 --seeds 26101 26102 26103 26104 --updates 2400 --batch 512 --eval-n 9600
redesign_v0.3/deployment/.venv/bin/python redesign_v0.7/analyze_protocols.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.7/audit_execution.py
.venv/bin/python redesign_v0.7/analyze_binding.py
```

原始目录保存私人变换矩阵、独立可训练/接收初始化指纹、训练批次、源代码与方案哈希、检查点、优化器、五模式终点轨迹和12组独立个人诊断。`smoke_001`仅为开发，三组各60更新不进入正式52组统计。v0.6及更早结果保留。
