# v3启动审计

审查时间：2026-09-15 15:24:17—15:24:57，Asia/Shanghai。本次为一次有界启动检查：核对启动元数据及完整信息能力控制的**首12次调用**，覆盖第1场、第1步的前两个广播窗口；这不是审查时刻的全量调用统计。未加载模型，未修改7个冻结核心、配置、提示或运行结果，未等待后续成绩。

**启动配置、源码和初始上下文核对一致。尚不能据此判断能力门槛是否通过；本样本甚至未包含首次物理动作结算。**

| 核对项 | 结果 |
|---|---|
| 锁定配置 | 根目录、batch及capability_full_information目录的三份experiment_locked.json均符合冻结摘要`c16dd82f6d4dc8bbbc859ea741609803a45fd0a29f66f9463f05db24b5978d12` |
| 源码 | environment、agents、backend、protocol、run、probe、execute_locked_pipeline共7个文件：当前文件、运行code_snapshot及manifest记录均符合冻结SHA-256；三份core_source_frozen.json内容相同 |
| 实际启动设计 | calibration/full_information，group17，2场，seed/variant为92015/0与92038/22，history_mode=independent_episodes；与batch启动命令和锁定配置一致 |
| 初始历史 | 第1场before checkpoint为A/B/C各空列表；首12次实际提示的“你自己的历史”均为空，当前观察agent均对应本次调用所有者 |
| 同步消息 | 窗1的6次生成均看不到任何当前广播；窗2只读到本轮窗1正式发布的三条消息，没有提前读到窗2伙伴输出 |
| 简化规则 | 实际system文本明确“无需加工”“没有加工操作、斧具或其他工具”及“所有道路……永久双向开放”；未发现旧版加工前置或可能封路的规则被误输入。长木双人规则与纤维单人规则均明确保留 |
| 初始世界身份 | seed92015、variant0、max_steps12；事件为kind=none、enabled=false、occurred=false；两项需求为干长木和干长纤维各1件送营地 |
| 模型 | manifest、锁定配置与既有模型清单的revision均为`16daa4818c54ce5f5436f929d52542eb65bbed9d`；实际本地路径一致且存在，量化配置为8-bit affine、group_size64 |
| 生成标记 | 首12次日志全部valid=true、fresh_cache=true、native_thinking=false；6次分析温度0，6次自然语言正式输出温度0.7。冻结backend确实传入prompt_cache=None，禁止trust_remote_code |

模型目录为`/Users/xia/Documents/ChatGPT/语言/qwen_collect_pilot/models/Qwen3.5-9B-8bit`。本次检查的是路径、版本记录、配置和首批实际调用；未重新计算约10.4GB权重的完整哈希，也不将revision声明当成重新下载验证。

需要单独保留一个能力观察：call1的A和call3的B在私有分析中已经把长纤维说成需要两人搬运，随后正式广播沿用；它们当时的历史和收到的广播均为空，而实际规则明确所有纤维可单人拿取。因此“长纤维双人”的说法来自这些模型输出，不能写成框架注入了旧规则或旧场景记忆。它提示规则使用仍需检查，但这12次调用不能说明后续是否会纠正，更不能代替完成两场后的门槛判定。

实际首轮没有旧任务段经历输入；提示中的“身份跨任务段持续”“过去物品句柄不能代替当前观察”是通用规则文字，不等于存在非空旧历史。分析原文会写入研究者inference.jsonl，并用于同次正式生成；本轮没有增加持续私有笔记。后续场景fresh histories及最终结果需在完成后另审。

依据：[启动命令](pipeline_status.json)、[锁定配置](experiment_locked.json)、[冻结摘要](core_source_frozen.json)、[运行清单](capability_full_information/manifest.json)、[实际首批推理日志](capability_full_information/inference.jsonl)、[第1场初始历史](capability_full_information/checkpoint_calibration_full_information_17_1_before.json)。
