# 语言形成先导运行：描述性分析

运行目录：`/Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/calibration_clarified_20260914`

分析时间：2026-09-14T11:35:27.237534+00:00

本报告描述任务完成和字符串使用，不判定语言已经形成。独立实验单位为群体；没有显著性检验。

## 运行状态与数据完整性

运行配置（manifest.json）：

```json
{
  "schema_version": 1,
  "phase": "calibration",
  "conditions": [
    "natural",
    "full_information"
  ],
  "groups": [
    17
  ],
  "episodes_per_group": 1,
  "scenarios": [
    {
      "episode": 1,
      "seed": 6101,
      "variant": 0
    }
  ],
  "model": "/Users/xia/Documents/ChatGPT/语言/qwen_collect_pilot/models/Qwen3.5-9B-8bit",
  "model_commit": "16daa4818c54ce5f5436f929d52542eb65bbed9d",
  "alphabet": "@#%&*+=~",
  "max_message_symbols": 32,
  "per_agent_step_symbols": 64,
  "windows": 4,
  "private_analysis_max_tokens": 256,
  "private_analysis_temperature": 0,
  "message_temperature": 0.7,
  "action_temperature": 0,
  "natural_message_max_tokens": 96,
  "history": "all own raw episodic observations/actions/feedback and delivered broadcasts; no analysis",
  "controls": "independent histories; natural language is not bandwidth matched",
  "created": "2026-09-14T19:25:58.380834",
  "preregistered_order": "group then episode then alternating condition order",
  "scope": "engineering and descriptive pilot; no statistical language-emergence claim"
}
```

运行状态（status.json）：

```json
{
  "status": "stopped_for_rule_clarity",
  "error_type": "KeyboardInterrupt",
  "error": "",
  "backend": {
    "calls": 161,
    "load_seconds": 1.8562603750033304,
    "inference_seconds": 537.020300089207,
    "generation_tokens_by_mode": {
      "private_analysis": 8774,
      "natural_message": 2739,
      "action": 30
    },
    "max_prompt_tokens": 5624,
    "peak_mlx_memory_gb": 10.478795338
  },
  "stop_reason": "Manual stop after identifying that the agent-facing rules should explicitly require kind/length/condition/destination to match and state fiber processing and joint carrying capacity. This is an incomplete capability control, not a completed failure score.",
  "stopped_at": "2026-09-14T19:35:27.232895",
  "completed_physical_steps": 5
}
```

推理统计（backend_stats.json）：

```json
{
  "calls": 161,
  "load_seconds": 1.8562603750033304,
  "inference_seconds": 537.020300089207,
  "generation_tokens_by_mode": {
    "private_analysis": 8774,
    "natural_message": 2739,
    "action": 30
  },
  "max_prompt_tokens": 5624,
  "peak_mlx_memory_gb": 10.478795338
}
```

已纳入0条任务段记录、0个条件内群体。

- episodes.jsonl不存在；尚无可分析的任务段记录。

缺失值用“—”显示。未完成运行不能据此判断两个条件的最终差异。

## 条件比较

各阶段分别比较；下列成绩先在群体内求均值，再对有记录的群体等权汇总。最小—最大是群体取值范围，不是置信区间。

| 条件 | 阶段 | 有记录群体 | 任务段 | 群体平均score的均值（范围） | 群体成功率的均值（范围） |
|---|---|---:|---:|---|---|

score沿用环境记录的尺度；本分析器不将其重新定义为语言能力分数。

## 各群体结果

## 字符串使用与相邻窗口变化

以下按条件合并消息，只描述所记录文本。更长的任务段会贡献更多消息，不能把消息条数当作独立样本量。

## 解释边界

- 独立实验单位是群体，任务段和消息不作为独立重复；本报告不做显著性检验。
- 条件成绩按阶段分别汇总各群体均值，群体等权；消息汇总仅描述已记录字符串。
- 成功、重复、变长或相邻窗口改变不能证明共同意义、澄清、组合性或语法。
- 相邻窗口仅比较同一任务段、主体、物理动作步内编号连续的窗口；不比较跨段相邻文本。
- 尚未出现、丢失或损坏的任务段不算失败，缺失消息不算沉默；运行中报告可能不完整。
- 同时报告所有任务段步数与成功段步数；不据不同完成度的平均步数直接判断效率。

逐阶段、逐群体的完整消息统计和读取的运行元数据见analysis.json。
