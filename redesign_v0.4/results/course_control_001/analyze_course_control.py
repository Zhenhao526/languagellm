from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT.parent/'curriculum_001'
SEEDS=[101,202,303]
for font in ['/System/Library/Fonts/PingFang.ttc','/System/Library/Fonts/Supplemental/Arial Unicode.ttf']:
 if Path(font).exists():
  font_manager.fontManager.addfont(font)
  plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name();break
plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors={'课程':'#087E8B','直接完整任务':'#D07839'}
all_results={}
summary=[]
fig,axes=plt.subplots(2,3,figsize=(14,7.4),sharex=True,constrained_layout=True)
for column,seed in enumerate(SEEDS):
 pairs={}
 for label,folder in [('课程',REFERENCE),('直接完整任务',ROOT)]:
  r=json.loads((folder/f'baseline_s{seed}/result.json').read_text())
  curve=json.loads((folder/f'baseline_s{seed}/learning_curve.json').read_text())
  pairs[label]=r
  xs=[x['update'] for x in curve]
  success=[100*x['full']['normal']['balanced_gathering'] for x in curve]
  gaps=[100*(x['full']['normal']['balanced_gathering']-x['full']['shuffle']['balanced_gathering']) for x in curve]
  axes[0,column].plot(xs,success,color=colors[label],lw=2.2,marker='o',ms=3.5,label=label)
  axes[1,column].plot(xs,gaps,color=colors[label],lw=2.2,marker='o',ms=3.5,label=label)
 for row in range(2):
  axes[row,column].axvline(600,color='#aaaaaa',ls=':',lw=1)
  axes[row,column].axvline(900,color='#aaaaaa',ls=':',lw=1)
  axes[row,column].grid(axis='y',color='#eeeeee')
 axes[0,column].axhline(100*5/7,color='#888888',ls='--',lw=1,label='无消息期望上限 71.43%' if column==0 else None)
 axes[0,column].set_title(f'配对种子 {seed}')
 axes[0,column].set_ylim(45,102)
 axes[1,column].set_ylim(-2,42)
 axes[1,column].set_xlabel('训练更新次数')
 axes[1,column].set_xticks([0,200,400,600,900,1200])
 d,c=pairs['直接完整任务'],pairs['课程']
 dt=[json.loads(x) for x in (ROOT/f'baseline_s{seed}/training_metrics.jsonl').read_text().splitlines()]
 ct=[json.loads(x) for x in (REFERENCE/f'baseline_s{seed}/training_metrics.jsonl').read_text().splitlines()]
 matches=[a['world_input_sha256']==b['world_input_sha256'] for a,b in zip(dt,ct)]
 assert not any(matches[:600]) and all(matches[600:])
 summary.append({'seed':seed,'direct_full':{k:v['balanced_gathering'] for k,v in d['full'].items()},'curriculum':{k:v['balanced_gathering'] for k,v in c['full'].items()},'curriculum_minus_direct_normal':c['full']['normal']['balanced_gathering']-d['full']['normal']['balanced_gathering'],'direct_message_gap':d['full']['normal']['balanced_gathering']-d['full']['shuffle']['balanced_gathering'],'curriculum_message_gap':c['full']['normal']['balanced_gathering']-c['full']['shuffle']['balanced_gathering'],'last_600_training_world_batches_identical':True,'direct_same_observation_resource_sensitivity':[x['observed_symbol_sensitivity']['fraction_cases_resource_changes_for_some_used_symbol'] for x in d['intervention']['directions']]})
 all_results[seed]=pairs
axes[0,0].set_ylabel('完整任务成功率（%）')
axes[1,0].set_ylabel('正常 − 打乱消息（百分点）')
axes[0,0].legend(loc='lower right',fontsize=9)
fig.suptitle('课程对照：探索、起点和预算相同，只改变前 600 次场景',fontsize=17)
fig.supxlabel('600 次：原课程组切换完整任务；900 次：两组都停止行动熵奖励。曲线为固定检查点评估，终点另用 8,192 个案例。',fontsize=10)
fig.savefig(ROOT/'course_control_comparison.png',dpi=180)
fig.savefig(ROOT/'course_control_comparison.pdf')
plt.close(fig)
means={label:{metric:float(np.mean([all_results[s][label]['full'][metric]['balanced_gathering'] for s in SEEDS])) for metric in ['normal','shuffle','blank','stochastic']} for label in ['课程','直接完整任务']}
report={'runs':summary,'means_over_three_paired_seeds':means,'mean_curriculum_advantage':float(np.mean([x['curriculum_minus_direct_normal'] for x in summary])),'no_message_expectation_bound':5/7,'all_seeds_retained':True,'all_control_runs_complete':True}
(ROOT/'course_control_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
lines=['# 课程对照结果','', '三个配对种子一致显示：在相同个体准备、行动探索和总预算下，先学习一方受限的课程能建立有效通信；从第一步直接学习完整任务，仍停留在不使用消息的资源选择策略。这个结论限于当前任务、算法、参数和三个种子。','', '## 完整任务终点','', '|种子|课程：正常|直接完整：正常|课程优势|课程：打乱消息|直接完整：打乱消息|','|---|---:|---:|---:|---:|---:|']
for x in summary:
 lines.append(f"|{x['seed']}|{x['curriculum']['normal']:.2%}|{x['direct_full']['normal']:.2%}|{100*x['curriculum_minus_direct_normal']:.2f} 个百分点|{x['curriculum']['shuffle']:.2%}|{x['direct_full']['shuffle']:.2%}|")
lines += [f"|三种子均值|{means['课程']['normal']:.2%}|{means['直接完整任务']['normal']:.2%}|{100*report['mean_curriculum_advantage']:.2f} 个百分点|{means['课程']['shuffle']:.2%}|{means['直接完整任务']['shuffle']:.2%}|",'',f"随机采样策略的完整任务均值：课程 {means['课程']['stochastic']:.2%}，直接完整任务 {means['直接完整任务']['stochastic']:.2%}。表中的正常条件采用最大概率动作，随机采样结果单独报告。",'', '直接组的正常、打乱、恒符号 0 三种最终成功率在每个种子内完全相同；两个受限方方向都在约 50%。固定接收者照片和公共状态，在实际使用过的符号之间替换消息，三个种子的两个方向均未改变所取资源。这里不能从微小概率变化推断已形成有用通信。','', '课程组完整任务的消息打乱落差为 32.89–37.92 个百分点，双方向均有作用；种子 303 的一个方向较弱。直接组则在第 50 次检查点起，最大概率动作下的消息落差均为零，直到第 1200 次。这与再次收敛为固定资源偏好相符。','', '![课程对照学习曲线](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/results/course_control_001/course_control_comparison.png)','', '## 配对与独立核查','', '- 原训练函数未经修改，两边都是可通信且无发信辅助的 baseline。三个社会学习前起点逐位相同，准备检查点哈希相同。','- 每组均为 1200×1024 个联合案例；前 900 次行动熵系数 0.05，最后 300 次为 0。优化器、学习率、主体、照片划分和缩放相同。','- 前 600 次场景有意不同；实际记录显示后 600 次连训练世界／照片批次哈希也相同。这是核查发现，不是额外筛选条件。','- 两组终点和干预评估外部案例哈希一致；18 组配对轨迹逐行核对照片、场景、公共库存与时间，完全一致。','- 新三组的 18 份最终轨迹共 147,456 个案例，已从资源选择独立重算成功和奖励，并用保存的原始特征、缩放与最终检查点重新推理全部发信和动作，均一致。','- 正常／打乱／恒符号条件共用相同案例和原始发信；打乱保持每个发送方向、每个公共时间内的符号计数。全部测试照片属于留出划分，训练／测试不存在相同字节图片。','', '## 可以与不能据此推断的内容','', '这补齐了课程的配对对照：当前设置中，有课程的训练日程相对于直接完整任务产生了约 21 个百分点的完整任务优势，也保留了消息的实际作用。它支持课程对当前学习过程有帮助。它不说明课程是所有 agent 产生通信的必要条件，也不能分离“减小场景复杂度”和“提高通信压力”各自的贡献，因为该课程同时改变了这两点。','', '两边行动探索已经匹配，因此这里的差异不能归因于两组探索系数不同；但本实验没有检验取消探索后的课程效果，也没有隔离 DINO 预训练的贡献。正常高分代表有限照片库上的任务通信，尚不能称为共同词典或人类语言从零诞生。','', '旧课程结果在设计对照前已知，这是补充识别实验。三个种子全部保留，未按结果延长训练或改终点；三个配对种子不支持广泛总体推断。8192 个评估案例是照片库中的重复组合，不等于8192张独立新照片。','', '恒符号0可能已有习得含义，不能视为行为中性的空白；71.43%为无当前消息策略的期望上限，有限样本略高于它不构成流程错误。','', '## 文件','', '- `config.json`、`preflight_validation.json`、`source_hashes.json`：固定配置和来源。','- `course_comparison.json`：与旧课程组的逐种子配对核查。','- `独立课程对照核查.json`、`audit_direct_control.py`：独立评分与检查点重放核查。','- `course_control_summary.json`：终点汇总。','- `course_control_comparison.png/.pdf`：学习过程图。','- `baseline_s*/`：完整检查点、训练记录、终点评估及逐案例轨迹。']
(ROOT/'课程对照_结果.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(means,ensure_ascii=False));print('advantage',report['mean_curriculum_advantage'])
