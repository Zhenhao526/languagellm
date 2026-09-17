"""Render the completed receiver probe without policy or training calls."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receiver_coverage_trajectory_v06_20260915_py312'
data = json.loads((OUT / 'execution/results.json').read_text())
assert data['status'] == 'completed'
assert data['receiver_checkpoint_count'] == 896
assert data['endpoint_anchors_exact'] == 112
assert data['tie_receiver_checkpoint_count'] == 0
rows = {(r['family'], r['update'], r['subset']): r for r in data['summaries']['by_family']}
updates = (0, 100, 300, 600, 1200, 1800, 2100, 2400)
families = ('course', 'mixed', 'direct', 'atomic_direct')


def pct(value):
    return f'{100 * value:.2f}%'


def value(family, update, subset, metric='same_code_both_correct'):
    return rows[family, update, subset]['means'][metric]


lines = [
    '# 接收解释范围的训练轨迹：完成记录', '',
    '本轮恢复已有v0.6的全部56个运行、448个检查点，评估896个接收者函数。共896次小接口批调用、87808个消息—需求输入；没有新增训练、自然发信、图像特征计算或DINO骨干加载。所有112个2400终点与原完整接收表逐项及SHA完全一致。全部检查点均无最大值并列，菜单排列等价成立，因此本轮没有“按菜单选码”与“固定一条码”的上界口径差异。', '',
    '读数是：固定接收者贪心动作、空库存与历史，是否有同一条消息能在两个私有需求下都指向正确地点。初始化的小通信接口未经双方协调，即使遍历49条码偶然找到正确动作，也不能称为已经建立的约定。', '',
    '## 留出地图的同码双目标覆盖', '',
    '| 更新次数 | 课程 | 混排 | 直接 | 单个49选1原子码 |',
    '| ---: | ---: | ---: | ---: | ---: |',
]
for update in updates:
    lines.append('| ' + str(update) + ' | ' + ' | '.join(pct(value(f, update, 'heldout')) for f in families) + ' |')
lines += [
    '', '四个独立种子先分别对两个方向和三个划分等权平均，再对种子平均。各条件族每个时点有144个地图—方向—划分—种子单元；同种子内的多个时点、方向及划分不能当独立训练重复。三个序列条件共享初始化，所以它们的初始相同数值不是三份独立证据。原子码的参数化及初始化不同，不能把起点差异归于学习结果。', '',
    '## 直接条件：训练地图与留出地图分开看', '',
    '| 更新次数 | 训练同码覆盖 | 留出同码覆盖 | 留出两边际可达但缺共同码 | 留出全码平均收益上界 |',
    '| ---: | ---: | ---: | ---: | ---: |',
]
for u in updates:
    metrics = [value('direct', u, 'train'), value('direct', u, 'heldout'),
               value('direct', u, 'heldout', 'both_marginals_but_no_joint_code'),
               value('direct', u, 'heldout', 'full_code_uniform_goal_ceiling')]
    lines.append('| ' + str(u) + ' | ' + ' | '.join(map(pct, metrics)) + ' |')
lines += ['', '直接条件各独立种子的留出覆盖：', '',
          '| 种子 | 初始化 | 300更新 | 2400更新 | 终点减初始化 |',
          '| ---: | ---: | ---: | ---: | ---: |']
for i, seed in enumerate(rows['direct', 0, 'heldout']['training_seeds']):
    values = [rows['direct', u, 'heldout']['seed_values']['same_code_both_correct'][i] for u in (0, 300, 2400)]
    lines.append(f'| {seed} | ' + ' | '.join(map(pct, values)) + f' | {100*(values[-1]-values[0]):+.2f}个百分点 |')
lines += [
    '', '这四个种子的终点留出覆盖均低于初始化。群体平均在早期下降，此后部分恢复而没有回到起点；所存轨迹并不单调。与此同时，直接条件训练地图的覆盖终点为63.72%，高于起点47.22%。这些是事后描述，没有把四个种子扩充为数百个独立样本，也未新增显著性或形成时刻检验。', '',
    '## 对解释的约束', '',
    '1. 终点缺口不是“初始化完全没有正确动作，随后逐步覆盖”的简单过程。49条未经协调的消息在随机映射中已有较广的动作组合；训练后的自然通信可能更有效，同时有些未见组合失去可用完整码。这里没有测量沿途自然消息，因此后一种共存解释仍需将同一检查点的自然发送与消息干预配对验证。',
    '2. 完整地图直接训练也从47.92%降至300更新的28.75%，最终升至66.25%。所以早期覆盖下降并不专属于组合留出；不能直接称为留出导致的机制。封锁通道时，遍历未使用码也可能显示较广覆盖，而其实际允许的零码平均收益上界始终是16.67%。这些控制再次表明，全码存在性不是有效语言指标。',
    '3. 固定贪心接收者的覆盖可反驳“只改选码就能修复”的解释，却不能定位训练原因。可能涉及消息使用分布、接收者更新、奖励目标和双方共同经历；本轮没有单独操纵任何一项。',
    '4. 8个检查点只显示保存时点，不确定两点之间何时产生或丢失某种解释，也不保证中间没有反复。不能把任一覆盖阈值首次越过命名为语言诞生。', '',
    '## 下一步', '',
    '在新的同任务生态实验中，成组保存自然消息及真实后果、消息干预和完整接收表，使N（自然双目标成功）与U（同码正确解存在）在相同照片、地图、伙伴与检查点对应。区别随机映射的偶然覆盖、双方实际使用的约定，以及未见组合的自主表达。训练收益单独保留；不要把严格双目标指标作为模型本来已经收到的训练奖励。', '',
    '新任务仍须保留对象、属性、搭档和目的地的区分，并在训练前验证这些区分何时有用。现有两主体资源定位仅为机制诊断。先冻结信息和行动关系的明确操纵，再做新群体确认；本轮不是新增确认训练。', '',
    '## 来源与执行记录', '',
    '- [执行结果](execution/results.json)及[逐检查点记录](execution/records.jsonl)。',
    '- [冻结计划](冻结计划.md)、[执行前口径说明](执行前口径说明.md)。原Python 3.9准备包未执行；使用Python 3.12重新准备是运行时兼容修复，686个源文件SHA与评估范围相同。',
    '- [独立核验](独立核验.md)。本报告由`summarize_receiver_trajectory.py`只读生成，未追加任何策略调用。',
    '',
]
(OUT / '结果与下一步.md').write_text('\n'.join(lines), encoding='utf-8')
print(OUT / '结果与下一步.md')
