# 双因子消息长度与可组合性对照

本实验把“消息长度增加”与“可组合表达”分开。sender 观察两个私有二值因子，worker 依次完成两个资源子任务。`mono4` 用一个四值 token；`dual2` 用两个二值 token。两种形式的原子符号容量相同（4 个组合），但只有 dual2 有可检验的槽位结构。

矩阵为 `mono4/dual2 × rotating/fixed partner × hidden/visible identity × live/silent × factorized/entangled task` 的预注册子集，共 10 个条件、8 个 seed。rotating-hidden 是公共码主臂；visible 是伙伴专用控制；fixed 是单伙伴校准；silent 是从头训练通道对照。每个 run 共享世界、目标、伙伴抽样和动作/消息随机流。

factorized 任务的两个目标位直接对应两个私有因子；entangled 任务的第二个目标位为两个因子的 XOR。dual2 的 held-out `recombined` 读数把 slot 0 从 `(g0,0)` 的 sender 状态取出、slot 1 从 `(0,g1)` 的 sender 状态取出，再放到同一世界中评价。它检验槽位是否分别携带因子作用；这是结构诊断，不把成功直接称为语法。

主要因果量为 live natural−silent 与 natural−permuted；`recombined−natural`、跨 worker sequence agreement 和 semantic success 是结构描述。只有在公共码因果收益、置换损失和双因子留出重组同时稳定时，才把结果写成候选组合机制；不把互信息、单次动作成功或 token entropy 当作语言证据。
