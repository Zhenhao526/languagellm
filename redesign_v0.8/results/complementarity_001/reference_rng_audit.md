# 完整码重编码参考RNG审计

本记录在主训练和原协议分析完成后，对已保存的参考种子、完整码双射及记录数作只读复核。没有加载模型、进行推理、生成新随机数或重算参考评分；原冻结RNG规则、协议计算和缓存均保持原样。

| 项目 | 数量 |
| --- | ---: |
| 正式训练运行 | 60 |
| 协议方向 | 120 |
| 已保存的条件化参考评分 | 12000 |
| 每方向有效双射及评分 | 100 |
| 不同参考RNG种子 | 90 |
| 主划分方向 / 主划分内不同参考种子 | 72 / 72 |
| 共享参考种子组 | 30 |

原规则采用 `910000000 + training_seed*100 + condition_index*10 + scout`。条件数为15时，部分全图或阻断条件的索引偏移超过相邻训练种子的间隔，因此会与下一训练种子的部分主条件复用参考随机序列。下表列出全部30组，每组恰两个方向；逐条核验100个已存49码双射序列完全一致。

**主划分72方向彼此没有参考种子碰撞**；其中部分仍与另一个训练种子的全图或阻断方向共享序列。每个协议方向内部保留100个不同且有效的完整码双射，原条件化评分仍然有效。相同重编码作用在不同的已学协议上，并不要求其评分相同。

这项复用影响的是参考重命名序列之间的依赖关系，不表示主训练主体使用了同一个训练随机种子，也不改变自然消息成绩。报告应称“12,000次条件化参考评分”，不能称“12,000个独立实验样本”。四个训练种子27101–27104仍是重复单位；划分、方向、菜单和参考重编码都不能增加训练主体样本量。

| 参考RNG种子 | 第一个协议方向 | 第二个协议方向 | 已存序列核对 |
| ---: | --- | --- | --- |
| 912710200 | s27101_full_mixed 0→1 | s27102_split1_additive 0→1 | 100/100一致 |
| 912710201 | s27101_full_mixed 1→0 | s27102_split1_additive 1→0 | 100/100一致 |
| 912710210 | s27101_full_joint 0→1 | s27102_split1_mixed 0→1 | 100/100一致 |
| 912710211 | s27101_full_joint 1→0 | s27102_split1_mixed 1→0 | 100/100一致 |
| 912710220 | s27101_blocked_additive 0→1 | s27102_split1_joint 0→1 | 100/100一致 |
| 912710221 | s27101_blocked_additive 1→0 | s27102_split1_joint 1→0 | 100/100一致 |
| 912710230 | s27101_blocked_mixed 0→1 | s27102_split2_additive 0→1 | 100/100一致 |
| 912710231 | s27101_blocked_mixed 1→0 | s27102_split2_additive 1→0 | 100/100一致 |
| 912710240 | s27101_blocked_joint 0→1 | s27102_split2_mixed 0→1 | 100/100一致 |
| 912710241 | s27101_blocked_joint 1→0 | s27102_split2_mixed 1→0 | 100/100一致 |
| 912710300 | s27102_full_mixed 0→1 | s27103_split1_additive 0→1 | 100/100一致 |
| 912710301 | s27102_full_mixed 1→0 | s27103_split1_additive 1→0 | 100/100一致 |
| 912710310 | s27102_full_joint 0→1 | s27103_split1_mixed 0→1 | 100/100一致 |
| 912710311 | s27102_full_joint 1→0 | s27103_split1_mixed 1→0 | 100/100一致 |
| 912710320 | s27102_blocked_additive 0→1 | s27103_split1_joint 0→1 | 100/100一致 |
| 912710321 | s27102_blocked_additive 1→0 | s27103_split1_joint 1→0 | 100/100一致 |
| 912710330 | s27102_blocked_mixed 0→1 | s27103_split2_additive 0→1 | 100/100一致 |
| 912710331 | s27102_blocked_mixed 1→0 | s27103_split2_additive 1→0 | 100/100一致 |
| 912710340 | s27102_blocked_joint 0→1 | s27103_split2_mixed 0→1 | 100/100一致 |
| 912710341 | s27102_blocked_joint 1→0 | s27103_split2_mixed 1→0 | 100/100一致 |
| 912710400 | s27103_full_mixed 0→1 | s27104_split1_additive 0→1 | 100/100一致 |
| 912710401 | s27103_full_mixed 1→0 | s27104_split1_additive 1→0 | 100/100一致 |
| 912710410 | s27103_full_joint 0→1 | s27104_split1_mixed 0→1 | 100/100一致 |
| 912710411 | s27103_full_joint 1→0 | s27104_split1_mixed 1→0 | 100/100一致 |
| 912710420 | s27103_blocked_additive 0→1 | s27104_split1_joint 0→1 | 100/100一致 |
| 912710421 | s27103_blocked_additive 1→0 | s27104_split1_joint 1→0 | 100/100一致 |
| 912710430 | s27103_blocked_mixed 0→1 | s27104_split2_additive 0→1 | 100/100一致 |
| 912710431 | s27103_blocked_mixed 1→0 | s27104_split2_additive 1→0 | 100/100一致 |
| 912710440 | s27103_blocked_joint 0→1 | s27104_split2_mixed 0→1 | 100/100一致 |
| 912710441 | s27103_blocked_joint 1→0 | s27104_split2_mixed 1→0 | 100/100一致 |

来源：[protocol_analysis.json](/Users/xia/Documents/ChatGPT/语言/redesign_v0.8/results/complementarity_001/protocol_analysis.json)；SHA-256：`30df51a774ba304aa13f9b99e9b310e0c98766b4de557ba81ef0c97e3bd3be86`。

复现脚本：[audit_reference_rng.py](/Users/xia/Documents/ChatGPT/语言/redesign_v0.8/audit_reference_rng.py)；SHA-256：`1b3a36adc5638fe002c3daea20109b2316e7f6b65af493e2eed43f90de4dfaeb`。

JSON保留全部120个方向的参考种子、序列哈希和共享组成员，便于逐项复核。
