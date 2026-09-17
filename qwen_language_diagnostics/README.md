# 规则判断、历史干扰与地图呈现的短诊断

**最新完成：三人续跑 CONT。** 180 次调用、两臂各 3 个物理步；保留历史臂交付 0/2，清空起始历史臂交付 1/2，并实际完成一次长木材共同搬运与共同交付。原臂 90 次调用和 3 步精确复现，两臂均在 t=11 达到观察上限，环境仍运行。见[结果与下一步](/Users/xia/Documents/ChatGPT/语言/qwen_language_diagnostics/results/20260915_CONT_01/结果与下一步.md)及[独立核验](/Users/xia/Documents/ChatGPT/语言/qwen_language_diagnostics/results/20260915_CONT_01/独立核验.md)。能力门槛未通过；下一步先处理规划与行动可靠性，本轮未检验符号约定。

以下为此前诊断记录，保留原说明。

**最新完成：地图呈现对照MAP。** 原说明与逐条列出道路均3/4，匹配共同搬运仍判错；四组答案含义均未改变。16次调用全部完成，J1原版精确复现上轮M1。见[结果与下一步](/Users/xia/Documents/ChatGPT/语言/qwen_language_diagnostics/results/20260915_MAP_01/结果与下一步.md)。本批用独立模块`map_presentation.py`运行，路径见`LATEST_MAP`；旧模块和结果保留。

**动作语义诊断M已完成：** 16次调用、8次正式判断中6次正确；错误集中于M1的一步共同搬运陈述。见[结果与下一步](/Users/xia/Documents/ChatGPT/语言/qwen_language_diagnostics/results/20260915_M_01/结果与下一步.md)。M用独立模块`action_semantics.py`运行，原R/S模块及其结果保留。

**较早结果：** R/S诊断已完成，16次调用；规则题4/4，未触发规则卡。删除历史后B改为携纤维到营地，C改拿错误短木材，未统一改善。见[结果与下一步](/Users/xia/Documents/ChatGPT/语言/qwen_language_diagnostics/results/20260915_R_S_01/结果与下一步.md)。

以下说明保留最初R/S诊断的设计和运行方式。原v3实验、七个核心文件、群体历史和失败数据保持原样。这里不启动新的符号群体，也不把短题通过视为语言形成。

默认4次规则选择加4次历史对照选择，每次两阶段推理，共16次调用。完整规则下任何正式答案错误时，追加一次预先写好的短规则卡同4题，最多24次调用。规则卡替换完整规则，不追加强调句；无论卡片结果如何都完成原定S对照，不重试到成功。长短、选项顺序及规则卡条件使用同一预设种子，每题独立输入。

R是在固定合成局部状态中判断干长/干短纤维能否单人拿取，两个选项各交换一次编号。全部题目的正确含义都是“可以”，4/4仍无法排除固定作答偏向。S选自已知失败案例：首场B第2步和C第9步，分别保留原历史或仅将过去历史清空。当前观察、四窗广播、菜单和种子保留；动作前分析各自重新生成。研究者在模型提交后固定另外两人的原动作结算一步，不是三人重新协商。

`prepare`先重建旧输入、核对实际调用和源码，再保存全部新题目、短卡、两臂输入、研究者诊断动作、源数据哈希和源码快照；`execute`仅运行完全匹配且尚未执行的目录。模型推理是单进程本地MLX，沿用Qwen3.5-9B-8bit、分析256 token、温度0、关闭原生thinking、每次新cache。测试使用FakeBackend或纯环境，无模型调用。

```sh
qwen_collect_pilot/.venv/bin/python -m pytest qwen_language_diagnostics/tests -q
qwen_collect_pilot/.venv/bin/python -m qwen_language_diagnostics.experiment prepare --out <新的结果目录>
/usr/bin/caffeinate -i qwen_collect_pilot/.venv/bin/python -m qwen_language_diagnostics.experiment execute --out <该结果目录>
```

最新独立批次路径见`LATEST`。批次内的`prepared_cases.json`含研究者记录和参考输出，不能整份送入模型；只有明确列出的prompt及empty_history_prompt为模型输入。实际输入和输出在`inference.jsonl`，每次结算在`decisions.jsonl`，完整结果在`results.json`。

比较原历史重放时，分别报告分析输入及渲染SHA、分析输出、正式输入及渲染SHA、正式答案和实际动作是否重现。分数不变也可能伴随成功移动或共同搬运。历史删除同时改变过去信息和上下文长度；规则卡同时改变内容、长度与突出程度，不据单次差异认定唯一机制。
