# 正式数字输出审计

**在这两个失败判断题上，没有发现ChoiceMask把模型原本会输出的正确数字变成错误数字。无约束生成也直接给出同一个错误数字。仅把最后的“本地动作编号”要求改为明确的真假判断要求，两题均恢复正确。**

本次是对已知失败案例的探索诊断，不是新的独立准确率评测。共12次独立、fresh-cache、温度0的正式输出推理；没有重新生成私有分析，没有把预期答案加入提示，也没有用程序替换模型输出。模型推理合计约6.75秒，加载约1.60秒；此次没有暂停。

## 对照设置

源记录：[rule_check_20260914_194041/inference.jsonl](/Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/rule_check_20260914_194041/inference.jsonl)。取`ground_is_not_carried`和`processing_preserves_attributes`的正式输出`messages`，固定原规则、题目、编号映射和已经生成的分析，仅按下表改变最后指令或输出前缀。

| 变体 | 最后user指令 | assistant生成前缀 | 输出限制 |
|---|---|---|---|
| original_masked | 原文：“只输出你选择的一个本地动作编号，不要解释。” | 标准模板 | 原ChoiceMask，仅0/1后接EOS |
| original_raw | 与原文相同 | 标准模板 | 不限制词表，最多96 token |
| explicit_masked | “请回答上面的真假判断题，不选择物理动作。只输出一个数字：0表示否，1表示是。不要解释。” | 标准模板 | 原ChoiceMask |
| explicit_raw | 与explicit_masked相同 | 标准模板 | 不限制词表，最多96 token |
| prefix_masked | 与原文相同 | 标准模板后追加“答案：” | 原ChoiceMask |
| prefix_raw | 与原文相同 | 标准模板后追加“答案：” | 不限制词表，最多96 token |

原始两种变体的渲染prompt SHA与源记录完全相同。明确指令重述了原题已经给出的0/1映射，没有增加题目答案；“答案：”前缀也不包含答案信息。

## 原始输出

| 案例 | 正确判断 | 原始masked | 原始raw | 明确指令masked | 明确指令raw | 前缀masked | 前缀raw |
|---|---:|---:|---:|---:|---:|---:|---:|
| ground_is_not_carried | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| processing_preserves_attributes | 1 | 0 | 0 | 1 | 1 | 1 | 1 |

所有raw输出都是单个数字后直接EOS，没有先输出空格、解释或“编号”等前缀。因此，这两个案例不支持“硬屏蔽把正常文本前缀删除，导致数字选择改变”的解释。原始错误在屏蔽前的首token分布里已经存在。

## mask实现和logits检查

分词器验证`0`为token 15、`1`为token 16；两个字符均为可原样往返的单token。实际处理器轨迹显示，第一步仅允许`[15,16]`，生成一位数字后仅允许EOS token 248046。流式生成还有一个被丢弃的EOS后预取调用；它也只允许EOS，没有改变已返回的数字。

| 案例／条件 | 首token原始logit：0 | 首token原始logit：1 | 屏蔽前最高token |
|---|---:|---:|---|
| ground／原始 | 18.500 | 19.500 | 1 |
| ground／明确指令 | 23.875 | 19.125 | 0 |
| ground／答案前缀 | 17.625 | 16.625 | 0 |
| processing／原始 | 19.875 | 19.250 | 0 |
| processing／明确指令 | 18.125 | 24.125 | 1 |
| processing／答案前缀 | 15.9375 | 17.500 | 1 |

每个条件下masked和raw的首token原始logits相同。数值只是该次推理的未归一化输出，不是置信度或准确率；不宜跨题比较其绝对大小。

此次实测只覆盖单字符0/1选择，不作为多位动作编号所有前缀边界都正确的完整证明。`ChoiceMask`对多位编号的逻辑应继续由已有单元测试及真正动作任务验证。

## 最小有据改动

1. **为规则判断检查设置专用的最终输出指令**，明确这是对题目的真假判断并重述0=否、1=是，不复用“本地动作编号”。两例的直接对照支持这项修改。
2. 保留主任务的动作编号mask。主任务确实是在菜单中选物理动作；本次判断题的措辞错位不能证明主任务同样因为最后指令而失败。
3. “答案：”前缀是本次有效的替代方法，但并非必要，也尚未在完整题组或动作任务上验证。优先修复任务类型与指令的一致性，比给所有正式输出统一增加前缀更有针对性。
4. 修改后应重新运行原8题并保留两个版本结果。由于这两题已用于探索，之后还需独立的新题或状态变体，才能评价修改后的泛化。

本次不能把前一轮6/8结果直接改记成8/8；旧运行仍是旧接口下的真实记录。也不能据此断言原始自然语言控制的观察误读已经解决：该控制在第一次私有分析中就把地面物品认作携带物，而这里诊断的是明确题目与分析之后的数字输出。

审计脚本：[audit_action_output.py](/Users/xia/Documents/ChatGPT/语言/qwen_language_v2/audit_action_output.py)。完整原始提示、输出token、首token分布、mask轨迹：[inference.jsonl](/Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/action_output_audit_20260914_194433/inference.jsonl)；汇总：[results.json](/Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/action_output_audit_20260914_194433/results.json)。
