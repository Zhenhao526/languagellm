# Smoke 有效性收据

- 2 seeds、4 parent、10 child、300 updates。
- replay audit 通过，最大绝对误差为 0；visibility、adaptation、population 配对均通过。
- 由于 smoke 没有同时包含 full 与 leave-one-out 对照，`paired_support_rows=0`；不用于效应估计。
