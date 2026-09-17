# v0.8 同照片形成轨迹：执行前核对

这是在部分及最终训练结果已可见后提出的事后探针。它沿用全部 60 组，不增训、不挑条件，不改变原协议或旧结果。当前文件是准备说明；是否已执行以新输出目录的 `execution/results.json` 为准。

范围固定为 4 种子 × 15 条件 × 8 检查点（0、100、300、600、1200、1800、2100、2400）× 2 方向，共 960 个方向检查点。`prepare` 只读源码、JSON 和权重文件字节计算 SHA；只有 `execute` 恢复已存小接口并前向推断。使用原 Python 3.12.14 / Torch 2.14.0 环境、CPU 单线程和 `weights_only=True`；不加载 DINO 骨干，只读取原缓存照片特征。每个检查点严格恢复双方参数；冻结投影与私有变换必须逐张量不变，才能复用该运行的照片投影。

接收者固定空库存、空历史、完整六地点菜单，枚举 49 码 × 2 私有目标，保存 49 × 2 × 6 物理 logits、最大值个数及前两名间隔。源码中菜单只在最终 `gather` 出现；只有所有最大值严格唯一，才由排列不变性证明免去 720 菜单重复计算。任何并列、非有限数或源锚不符都立即停止整个探针，保存异常和已完成记录，不换规则、不使用 v0.6 的菜单回退，也不从成功子集发布总体结果。

发送者每方向只评 30 地图 × 16 照片对，一次无目标贪心发送；16 对固定为原 validation 列表，所有检查点相同。原协议另外重复的两个 sender-goal 查询在本条件都被置零，本探针不把它们再当额外样本。逐照片保存 emitted、delivered、解码行动及两个正确性位。封锁组的 delivered 恒为零码，仍保存原发信。完整 49 码覆盖 U 与封锁通道实际只允许零码的覆盖分别记录，不能把前者称为封锁组可实现效果。

每地图记录同码双目标覆盖、两边际各自可达但无共同码、仅一个边际可达、两个均不可达，以及最优码的均匀双目标平均正确率。自然正确率 N 与 U 在相同地图及照片上对应；自然失败再分为通道内无正确码、存在正确码但未使用。U 的基本分母是地图，N 是地图 × 照片；条件失败比例另用自然失败数作分母。贪心 U=0 不表示随机策略成功概率为零，也不表示物理任务或信道容量不足。

强不变量：同种子、同方向在 update0 的全部 15 条件必须有逐项相同的接收 logits 和 emitted；blocked 的 delivery 不参与这个相等断言。每组 checkpoint2400 与 final.pt 逐张量相同；每个终点的全部 49 解码行、完整 720 菜单表的物理哈希，以及 30 地图 × 两个重复 sender-goal 的消息频数和正确数必须匹配已有完整 protocol。原 protocol 没有保存逐照片消息对应关系，因此终点能锚定完整解码与每地图频数，不能声称旧逐照片发信已经精确复现。

汇总先在种子内平均方向和划分，再列出 4 个独立训练种子值。split 条件每种子 6 个方向；full/blocked 每种子只有 2 个方向，不复制为三个划分。检查点、地图和照片不是额外独立训练单位。U 在随机初始化也可很高，必须连同 N 及失败分解读；本探针未干预消息，不单独确证消息因果作用或约定形成，也不能定位两个存盘点之间的形成时刻。与原随机 9600 例曲线不作同照片配对。

执行入口（由根代理在独立审查后调用，新目录不复用）：

```sh
redesign_v0.3/deployment/.venv/bin/python research_program/v08_formation_trajectory.py prepare --out research_program/v08_formation_trajectory_001
redesign_v0.3/deployment/.venv/bin/python research_program/v08_formation_trajectory.py verify --out research_program/v08_formation_trajectory_001
redesign_v0.3/deployment/.venv/bin/python research_program/v08_formation_trajectory.py execute --out research_program/v08_formation_trajectory_001
```

输出包括冻结 plan/source SHA 与源码副本、960 份逐方向记录、120 个终点核对及四种子汇总。`execution/` 排他创建，不支持选择性恢复或自动重试；准备和纯合成验证不运行策略。实测阶段只在全部检查都通过后写 `status=complete`。
