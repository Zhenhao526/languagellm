# 语言形成先导运行：描述性分析

运行目录：`/Users/xia/Documents/ChatGPT/语言/qwen_language_v3/results/20260915_152343/capability_full_information`

分析时间：2026-09-15T08:04:56.803076+00:00

本报告描述任务完成和字符串使用，不判定语言已经形成。独立实验单位为群体；没有显著性检验。

## 运行状态与数据完整性

运行配置（manifest.json）：

```json
{
  "schema_version": 1,
  "phase": "calibration",
  "conditions": [
    "full_information"
  ],
  "groups": [
    17
  ],
  "episodes_per_group": 2,
  "scenarios": [
    {
      "episode": 1,
      "seed": 92015,
      "variant": 0
    },
    {
      "episode": 2,
      "seed": 92038,
      "variant": 22
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
  "history_mode": "independent_episodes",
  "environment_family": "river_valley_v0.3",
  "ecology_changes": [
    "all wood initially processed",
    "no axe or tool operations",
    "roads always open"
  ],
  "source_sha256": {
    "__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "agents.py": "f7d924ccbe210d01186dee7e989b7f2497055145aeea62e8682924e87fffc97f",
    "analyze.py": "fa62f5a550adc2f4c77503d83566cd9a4e600df0deacb7672c369a4c96b93b8d",
    "backend.py": "0aa38e9c5a53c9f3275a8208912bfc356181839e7d338083570dd32c332c7e23",
    "environment.py": "8c149d6effadad3cbe465c7d32f1962b1170b663b38ca251320bd12bfe551d81",
    "execute_locked_pipeline.py": "84bd58b5414220c77be80cbaf43272b3cbbfcf527631e2cc9c01652a513b4e0f",
    "inspect_run.py": "a586202183aa343f51c44267320369ad036021255722b8a5acf97a4eb35f437a",
    "probe.py": "74e97dda0d3f3bd46b6c3e89d14dc170822880906035a9a93d648c6262255cd3",
    "protocol.py": "2bc85ff7723459f09ea4aa1ba2020c19a2c042f8b5e6366431afa2ab1d80fab7",
    "replay_snapshot.py": "2f85683151f960631f684b7870bb70c785eef6a3553340f0f986adad962acf87",
    "report_suite.py": "10a65aa99ec3ff9b9ecd77f1ef242240bb28fad5e4a5c755ebb9de32ff548f9a",
    "run.py": "eff1e4fa3f8803b43d95c32381646e9c0f39b87a212ed1de110ede45ae4dcd7b"
  },
  "controls": "independent histories; natural language is not bandwidth matched",
  "created": "2026-09-15T15:23:43.356151",
  "preregistered_order": "group then episode then alternating condition order",
  "scope": "engineering and descriptive pilot; no statistical language-emergence claim"
}
```

运行状态（status.json）：

```json
{
  "status": "completed",
  "backend": {
    "calls": 510,
    "load_seconds": 3.242175874998793,
    "inference_seconds": 2468.720485502854,
    "generation_tokens_by_mode": {
      "private_analysis": 27294,
      "natural_message": 6309,
      "action": 102
    },
    "max_prompt_tokens": 13173,
    "peak_mlx_memory_gb": 10.96469364
  }
}
```

推理统计（backend_stats.json）：

```json
{
  "calls": 510,
  "load_seconds": 3.242175874998793,
  "inference_seconds": 2468.720485502854,
  "generation_tokens_by_mode": {
    "private_analysis": 27294,
    "natural_message": 6309,
    "action": 102
  },
  "max_prompt_tokens": 13173,
  "peak_mlx_memory_gb": 10.96469364
}
```

已纳入2条任务段记录、1个条件内群体。

缺失值用“—”显示。未完成运行不能据此判断两个条件的最终差异。

## 条件比较

各阶段分别比较；下列成绩先在群体内求均值，再对有记录的群体等权汇总。最小—最大是群体取值范围，不是置信区间。

| 条件 | 阶段 | 有记录群体 | 任务段 | 群体平均score的均值（范围） | 群体成功率的均值（范围） |
|---|---|---:|---:|---|---|
| full_information | calibration | 1 | 2 | 0.500（0.500—0.500；有效n=1） | 50.0%（50.0%—50.0%；有效n=1） |

score沿用环境记录的尺度；本分析器不将其重新定义为语言能力分数。

## 各群体结果

### full_information / 17

记录中的seed：92015, 92038

| 阶段 | 任务段 | 平均score | 成功/有效记录 | 所有段平均步数 | 仅成功段平均步数 |
|---|---:|---:|---:|---:|---:|
| calibration | 2 | 0.500 | 1/2 | 8.500 | 5.000 |

逐任务段记录：

| 阶段 | 任务段 | seed | score | 步数 | success |
|---|---|---|---|---|---|
| calibration | 1 | 92015 | 0.0 | 12 | False |
| calibration | 2 | 92038 | 1.0 | 5 | True |

消息：204条，非空204条（100.0%）；平均长度39.578，非空消息平均长度39.578。

自然语言能力控制不适用8字符字母表、32字符单条上限和64字符每步额度；字符长度不等于模型token数。

## 字符串使用与相邻窗口变化

以下按条件合并消息，只描述所记录文本。更长的任务段会贡献更多消息，不能把消息条数当作独立样本量。

### full_information

共204条消息，非空比例100.0%。所有消息平均长度39.578，非空平均长度39.578。

长度分布（字符数: 条数）：5: 1, 18: 1, 23: 7, 24: 8, 25: 3, 26: 7, 27: 4, 28: 2, 29: 6, 30: 7, 31: 7, 32: 13, 33: 9, 34: 4, 35: 7, 36: 3, 37: 3, 38: 4, 39: 8, 40: 10, 41: 8, 42: 5, 43: 10, 44: 2, 45: 7, 46: 9, 47: 3, 48: 6, 49: 7, 50: 7, 51: 6, 52: 3, 53: 2, 54: 1, 55: 1, 56: 2, 59: 1, 60: 1, 63: 1, 65: 1, 66: 1, 67: 1, 68: 1, 70: 1, 78: 1, 85: 1, 122: 1

自然语言能力控制不适用符号字母表与符号预算审计；生成token数另见推理日志。

非空完整字符串共有178种，首次出现以后的重复记录26条（占非空消息12.7%）。该统计把同一主体和不同主体的重复都计入，不证明共享意义。

出现最多的非空完整字符串（最多20种）：

| 字符串 | 次数 |
|---|---:|
| `静候 C 抵达河滩，二人共同交付干长纤维至营地。` | 5 |
| `静候 C 抵达河滩，随后二人共同交付干长纤维至营地。` | 5 |
| `我在营地静候。请 C 速带干长木材（物 10）抵达，我们立即共同交付。完成后请 C 转河滩助 B。` | 4 |
| `本步放下物 297，拿取物 284。下步移营地与 A 交付，完成后转河滩助 B。` | 3 |
| `本步拿取物 10（干长木材）。下步与 C 共同交付至营地，完成后请 C 转河滩助 B。` | 3 |
| `本步静默。待 C 抵达河滩后，二人共同交付干长纤维物 143 至营地。` | 2 |
| `静候 C 抵达河滩，二人共同交付干长纤维物 143 至营地。` | 2 |
| `本步放下物 294，移向林地取物 284。随后回营地与 A 交付，再赴河滩助 B。` | 2 |
| `我在营地静候。请 C 放下手中湿材，速取林地干长木材（物 10），与我共同交付。完成后请赴河滩助 B。` | 2 |
| `静候 C 抵达河滩，随后二人共同交付干长纤维物 143 至营地。` | 2 |
| `本步拿取物 284。下步与 A 共同交付木材，随后赴河滩助 B。` | 2 |
| `本步静默。待 C 抵达河滩后，二人共同交付干长纤维至营地。` | 2 |
| `静候 C 抵达河滩。二人同步共同交付干长纤维（物 143）至营地。` | 2 |
| `本步静候。待 C 抵达河滩后，二人共同交付干长纤维至营地。` | 2 |
| `本步静默。等待 C 行动或 B 完成纤维交付。` | 2 |
| `本步静默。等待 C 行动或 B 完成纤维交付，以便后续调整策略。` | 2 |
| `我去河滩，与 B 共同搬运干长纤维（物 32）至营地。` | 1 |
| `我在河滩。需求需干长纤维（2人共搬）和干长木材（A在营地）。请A来河滩助我搬干长纤维，或C来营地交接。其他伙伴请保持静默以免干扰。` | 1 |
| `目标：营地。需求：1干长木材，1干长纤维。我在地拿干长木材，去河滩协同B拿干长纤维。请同步。` | 1 |
| `我在营地。C 若能把干长木材运到营地，请与我共同交付。我本步等待。` | 1 |

同一任务段、主体、动作步内，编号连续的窗口对共153对。

| 相邻窗口文本变化 | 对数 |
|---|---:|
| 两条均为空 | 0 |
| 空→非空 | 0 |
| 非空→空 | 0 |
| 非空且完全相同 | 12 |
| 两条非空且不同 | 141 |

两条均非空时完全相同比例：7.8%；所有窗口对平均字符编辑距离：21.333；平均长度变化（后减前）：-2.242。
这些变化不能自动标注为回应、修复或新的语法结构；它们也可能来自预算耗尽、任务状态或重复策略。
缺少可用主体/步/窗口信息的消息0条；重复窗口键的额外消息0条，相关窗口不作配对。

## 解释边界

- 独立实验单位是群体，任务段和消息不作为独立重复；本报告不做显著性检验。
- 条件成绩按阶段分别汇总各群体均值，群体等权；消息汇总仅描述已记录字符串。
- 成功、重复、变长或相邻窗口改变不能证明共同意义、澄清、组合性或语法。
- 相邻窗口仅比较同一任务段、主体、物理动作步内编号连续的窗口；不比较跨段相邻文本。
- 尚未出现、丢失或损坏的任务段不算失败，缺失消息不算沉默；运行中报告可能不完整。
- 同时报告所有任务段步数与成功段步数；不据不同完成度的平均步数直接判断效率。

逐阶段、逐群体的完整消息统计和读取的运行元数据见analysis.json。
