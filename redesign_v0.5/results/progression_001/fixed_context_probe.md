# 固定当前情境的消息干预

在正常轨迹上保持世界、当前需求、库存、自己的两次采集历史和菜单不变。供体消息只在相同方向、步数、需求、库存、历史内重排，不模拟后续环境。单例层及重复消息保留，所以该干预不保证完全移除信息。

| 阶段 | 条件 | 种子 | 正常即时收益 | 替换后即时收益 | 差（百分点） |
| --- | --- | --- | ---: | ---: | ---: |
| B | delay_memory | 24001 | 79.98% | 25.33% | 54.65 |
| B | delay_replay | 24001 | 80.38% | 25.32% | 55.07 |
| B | delay_reset | 24001 | 24.23% | 24.23% | 0.00 |
| B | immediate_continue | 24001 | 80.38% | 25.32% | 55.07 |
| B | delay_memory | 24002 | 67.27% | 24.76% | 42.52 |
| B | delay_replay | 24002 | 67.10% | 24.38% | 42.72 |
| B | delay_reset | 24002 | 24.73% | 24.73% | 0.00 |
| B | immediate_continue | 24002 | 67.10% | 24.38% | 42.72 |
| B | delay_memory | 24003 | 87.60% | 24.37% | 63.23 |
| B | delay_replay | 24003 | 85.66% | 24.63% | 61.02 |
| B | delay_reset | 24003 | 25.40% | 25.40% | 0.00 |
| B | immediate_continue | 24003 | 85.66% | 24.63% | 61.02 |
| C | persistent_blocked | 24001 | 39.81% | 39.81% | 0.00 |
| C | persistent_channel_removed | 24001 | 40.09% | 40.09% | 0.00 |
| C | persistent_communication | 24001 | 85.03% | 31.08% | 53.95 |
| C | persistent_delayed | 24001 | 82.52% | 31.36% | 51.16 |
| C | persistent_known | 24001 | 82.82% | 32.78% | 50.03 |
| C | persistent_blocked | 24002 | 39.18% | 39.18% | 0.00 |
| C | persistent_channel_removed | 24002 | 39.16% | 39.16% | 0.00 |
| C | persistent_communication | 24002 | 77.27% | 31.74% | 45.53 |
| C | persistent_delayed | 24002 | 68.23% | 31.45% | 36.77 |
| C | persistent_known | 24002 | 84.19% | 30.82% | 53.37 |
| C | persistent_blocked | 24003 | 39.54% | 39.54% | 0.00 |
| C | persistent_channel_removed | 24003 | 39.55% | 39.55% | 0.00 |
| C | persistent_communication | 24003 | 89.08% | 32.87% | 56.21 |
| C | persistent_delayed | 24003 | 86.12% | 33.16% | 52.96 |
| C | persistent_known | 24003 | 91.61% | 31.99% | 59.63 |

已检查466,944条正常轨迹的模型动作与独立重算收益；运行数27/27。
这项检验支持当前消息的行为作用，不能单独证明词义、组合性或长期总收益的因果效应。
