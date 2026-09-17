"""Plot actual replay probabilities and entropies, no new model training."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': ['PingFang SC', 'Arial Unicode MS', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
curve = json.loads((ROOT / 'telemetry_replay/communicate_s101/learning_curve.json').read_text())
x = [r['update'] for r in curve]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
axes[0].plot(x, [r['normal']['mean_reward_per_step'] for r in curve], 'o-', label='每次取最大概率动作', color='#2563eb')
axes[0].plot(x, [r['stochastic']['mean_reward_per_step'] for r in curve], 'o-', label='按概率随机采样动作', color='#0d9488')
axes[0].axhline(5/7, color='#64748b', ls=':', label='无消息策略的期望上界')
axes[0].set(xlabel='训练更新次数', ylabel='相同测试场景的成功率', ylim=(.5, .78), title='固定选择没变，选择概率仍在变化')
axes[0].legend(loc='lower right', fontsize=9)
action_entropy = [np.mean([a['mixed_action_entropy_nats'] for a in r['policy_diagnostics']]) / np.log(2) for r in curve]
sender_entropy = [np.mean([a['sender_entropy_nats'] for a in r['policy_diagnostics']]) / np.log(5) for r in curve]
axes[1].plot(x, action_entropy, 'o-', label='资源选择的随机程度', color='#ea580c')
axes[1].plot(x, sender_entropy, 'o-', label='发送符号的随机程度', color='#9333ea')
axes[1].set(xlabel='训练更新次数', ylabel='熵 / 最大可能熵（0=确定，1=均匀）', ylim=(0, 1.06), title='行动快速确定，发送仍接近随机')
axes[1].legend(loc='center right', fontsize=9)
fig.suptitle('原实验种子 101 的前 100 次更新复跑：只增加记录，终点权重与原实验逐位相同')
fig.savefig(ROOT / 'flow_diagnosis.png', dpi=180)
fig.savefig(ROOT / 'flow_diagnosis.pdf')
plt.close(fig)
