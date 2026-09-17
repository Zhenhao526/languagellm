# Action-dependent signaling plan

核心比较是 simultaneous 与 staged。两者只有消息到达时间不同：simultaneous 在第一个行动前给出完整 dual2 序列；staged 在两个子任务开始前分别给出一个槽位。发送者没有教师或预置词典，worker 只能看到局部资源状态。

组合性判据预注册为：staged 相对于 simultaneous 的槽位重组保真度提升、natural−silent 的稳定正效应、natural−permuted 的配对损失，以及新目标组合上的 held-out 成功。seed-level 的“可组合平衡”定义为 held-out natural ≥ 0.60 且 `|recombined−natural|≤0.02`；该阈值在 16-seed 探索批次后固定，用于新增 seed 的确认。若 staged 仍只形成整体码本，则行动依赖不足以产生组合性；若 staged 重组成功而 simultaneous 失败，则支持“可分解的行动后果是组合性形成的必要条件”这一机制假设。
