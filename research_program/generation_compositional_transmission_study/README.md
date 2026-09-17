# Compositional parent-to-child transmission

该包把 staged `dual2` 协议传递给新 worker 或新 sender，比较 live、silent 和 partner-permuted 通道，并审计整体码本和槽位重组是否恢复。

父代 checkpoint 由 `action_dependent_signaling_study` 生成并在冻结配置中记录哈希。原始 parent/child execution 只在临时目录用于审计，仓库只保留压缩结果、冻结方案和收据。
