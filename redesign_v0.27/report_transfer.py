"""Report and plot the prespecified frozen-protocol material probe."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.append(str(PROJECT/'redesign_v0.9/.analysis_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ARMS=('old_old','all_old','all_all');LABELS=('A: private18/social18','B: private30/social18','C: private30/social30')
CELLS=('old_old','new_old','old_new','new_new');NAMES=('原食物＋原水','新食物＋原水','原食物＋新水','新食物＋新水')

def main(out):
    data=json.loads((out/'summary.json').read_text());complete=json.loads((out/'completion.json').read_text())
    assert json.loads((out/'audit_qa.json').read_text())['passed']
    idx={(r['arm'],r['cell']):r for r in data['overall']};mean=data['mean_contrasts']
    material=[]
    for cell,label in zip(CELLS,NAMES):
        v=[idx[a,cell]['scores']['new12']['pooled'] for a in ARMS]
        material.append(f"| {label} | {v[0]['J']:.2%} | {v[1]['J']:.2%} | {v[2]['J']:.2%} | {v[1]['Q']:.2%} |")
    source=[]
    for r in data['contrasts']:
        source.append(f"| {r['seed']} | {r['B_original_J']:.2%} | {r['B_new_J']:.2%} | {100*r['primary_B_material_delta']:+.3f} | {100*r['aux_new_B_minus_A']:+.3f} |")
    sensitivity=[]
    for arm,label in zip(ARMS,('A','B','C')):
        r=idx[arm,'new_new'];s=r['scores'];c=r['changes']
        sensitivity.append(f"| {label} | {s['old']['pooled']['J']:.2%} | {s['common30']['pooled']['J']:.2%} | {c['old']['pooled']['message_TV']:.4f} | {s['new12']['pooled']['J']:.2%} | {c['new12']['pooled']['message_TV']:.4f} | {c['new12']['pooled']['action_TV']:.4f} |")
    control=[]
    for arm,label in zip(ARMS,('A','B','C')):
        s=idx[arm,'new_new']['scores']['new12']['pooled']
        control.append(f"| {label} | {s['J']:.2%} | {s['blank_J']:.2%} | {s['shuffle_J']:.2%} | {s['visible']:.2%} | {s['historical']:.2%} |")
    lines=[
        '# 私人行动与共同通信在新图片上的迁移', '',
        '2026-09-16。v27冻结协议材料探针。新图通信结果读取前固定主要量和全部A/B/C条件；本轮没有新增训练。', '',
        f"B组在目标12种布局上的自然双目标成功率由{mean['B_original_J']:.2%}变为{mean['B_new_J']:.2%}；预定主要材料差为{100*mean['primary_B_material_delta']:+.3f}个百分点，四来源均为小幅正值。上一轮私人all在相同新图上的行动为100%，本轮C组为{mean['C_new_J']:.2%}。因此，这批材料没有出现预设检验可能发现的通信退化；私人成功与B组低自然表达的差距仍存在，不能把低水平下的变化小称为充分稳健或等效。", '',
        '## 条件与主结果', '',
        'A：私人经验覆盖18布局、通信覆盖18；B：私人30、通信18；C：私人30、通信30。B/C继承相同私人状态，社会协议来自各自既有训练终点。new12对B是私人熟悉、通信未训练的组合；C已在通信中训练过这12种布局，C的高分不属于未训练布局泛化。', '',
        '每臂4个初始化来源×3个分区×2个方向，三臂共72方向。固定官方DINOv2-L底座及原私人/社会接口，复用v26缓存，不重新调用视觉底座；发送者只读h96，接收者只读两枚整数符号。9食物图与2水图均已在v26用于私人能力探针，是有限流程材料。四格中照片对、30布局、两末帧mask等权；分析先在来源内平均六个方向，再汇总四来源。', '',
        '| 测试材料 | A自然J | B自然J | C自然J | B概率Q |', '|---|---:|---:|---:|---:|',*material,'',
        'J要求接收者同时正确定位食物和水，使用逐token贪心消息及贪心行动。Q积分49个完整码及正确接收动作的概率；二者不能混称成功率。', '',
        '| 初始化来源 | B原图J | B全部新图J | 主要差（百分点） | 新图B−A（辅助，百分点） |', '|---|---:|---:|---:|---:|',*source,'',
        f"辅助的新图B−A均差为{100*mean['aux_new_B_minus_A']:+.3f}个百分点，旧图为{100*mean['aux_original_B_minus_A']:+.3f}个百分点；交互为{100*mean['aux_experience_material_interaction']:+.3f}个百分点。新图来源33101已反向，不能复述原v23“四来源私人经验收益均正”。这些冻结反应也不是在新材料上重新学习的经验效应。", '',
        '![通信成绩与消息敏感性](figures/communication_transfer.png)', '',
        '左图实线是四来源均值，淡点是各来源。右图总变差TV衡量固定布局和mask下旧/新照片引出的完整贪心码分布差异，范围0–1；它不是成功率或语法分数。', '',
        '## 消息变化和行为变化', '',
        '| 协议 | 新图old18 J | 新图全部30 J | old18消息TV | 新图new12 J | new12消息TV | new12行动TV |', '|---|---:|---:|---:|---:|---:|---:|',*sensitivity,'',
        'A/B在通信训练过的18布局上保持约97%成功，消息分布对这次图片替换的变化很小；同一协议在未训练12布局上，码和行动分布变化较大，但总成功率并未下降。这里的变化可能主要重新分配错误，不能把消息TV直接解释为任务失败概率。C在已训练全部布局上表现高且消息变化小。', '',
        '每固定布局/mask先计算照片对上的49码直方图，再按原接收器推送到36种双行动；TV取分布差绝对值和的一半。完整记录还保存跨图片一致概率和各格内部重复抽样一致概率。行动TV不大于消息TV是确定性映射的数学约束，不是新发现；码不同而行动相同只能说明原解码器的有限行为等价，不能推出人类式同义或组合结构。', '',
        '| 协议 | 自然J | 固定空码J | 完整消息打乱J | 可见资源正确 | 历史资源正确 |', '|---|---:|---:|---:|---:|---:|',*control,'',
        '以上控制均为全部新图/new12。打乱取格子全部贪心完整消息直方图的独立抽样解析期望；固定[0,0]可能本就是合法符号，不等于重新训练的无通信基线。接收器不看照片，所有格子共用同一49码行动表。', '',
        '## 对研究主线的影响', '',
        '本轮支持的有限结论是：在这些图片、已有主体和训练预算下，私人行动读出能够成功，仍不足以保证原协议表达通信训练未覆盖的组合；把目标布局纳入社会经验后得到的另一套协议，在这些新图片上也保持较高成绩。它未证明语言起源的必要条件、DINO知识的必要性或组合语言已经出现。', '',
        '上游视觉能力与通信能力分离、通信依赖非概念视觉信息、外观不变性压力改变消息内容，都已有直接先例。见[Bouchacourt与Baroni，2018](https://aclanthology.org/D18-1119/)、[Rodríguez Luna等，2020](https://aclanthology.org/2020.findings-emnlp.397/)、[Garcia等，2022](https://aclanthology.org/2022.naacl-main.335/)以及[本轮近邻定位](../../../paper_program/confirmation_readiness_20260916/视觉材料与通信读出的近邻定位.md)。本项目接收者无视觉，不能直接套用两端看图的比较捷径；当前结果也没有单独识别背景、颜色、图像身份或某一网络层的原因。', '',
        '应结束这批11图上的继续探针细分，转入预先保留的新训练/测试材料上的完整A/B/C形成过程检验。这个步骤需重做相关私人准备和社会学习，不能把再加一个冻结测量称为形成复现。若独立材料仍不足，应明确报告数据缺口并执行有限的数据准备；当前研究尚不足以认定达到ICLR论文质量。', '',
        '## 验证与材料限制', '',
        f"完成288张协议表、{complete['sender_worlds']:,}个发送世界前向，生产耗时{complete['seconds']:.2f}秒。72方向前端张量与对应私人来源逐位相同；复用h与旧社会cache有最高约1.19×10⁻⁶差别，故未声称缓存逐位一致。全部72原图对照的自然token完全相同，49码对数概率、归一化概率及接收表通过预定容差；最大对数概率差约7.63×10⁻⁶、归一化概率差约8.08×10⁻⁷，接收表无差别。", '',
        '独立核查覆盖全表统计、世界、来源差和消息变化；模型前向只在预定33101/p1/d0的A/B/C四格共14040个世界与三张49码接收表上重放，未重做全部72方向或DINO。本次生产及独立审计均通过，没有失败重跑；模型实验没有训练更新、网络请求、新图片读取或确认48像素访问。', '',
        '11张流程图只有2张水图，旧食物测试子类与新图不完全平衡；已知作者/原作键排除及像素近重复检查不能保证完全独立，也不能证明DINO预训练未见。四个初始化来源是同一批既有主体，不是新增独立训练来源。所有图片和终点保留，不能用这次成绩筛选未来确认材料。', '',
        '[固定方案](../../固定执行方案.md) · [完整汇总](summary.json) · [全部方向与逐布局变化](raw_summaries.json) · [独立复核](audit_qa.json) · [来源核验](frontend_inheritance.json) · [旧协议对照](legacy_control.json) · [运行记录](completion.json)', '',
        '从项目根目录执行，使用尚不存在的新目录：', '', '```bash',
        'redesign_v0.3/deployment/.venv/bin/python redesign_v0.27/run_transfer.py --out redesign_v0.27/results/replay_001',
        'redesign_v0.3/deployment/.venv/bin/python redesign_v0.27/audit_transfer.py --out redesign_v0.27/results/replay_001',
        'redesign_v0.3/deployment/.venv/bin/python redesign_v0.27/report_transfer.py --out redesign_v0.27/results/replay_001', '```',''
    ]
    (out/'私人行动与共同通信在新图片上的迁移研究报告.md').write_text('\n'.join(lines))
    fig,(a,b)=plt.subplots(1,2,figsize=(12,4.4),layout='constrained',gridspec_kw={'width_ratios':[1.3,1]})
    colors=('#718799','#D27B44','#459686')
    for arm,label,color in zip(ARMS,LABELS,colors):
        vals=[idx[arm,c]['scores']['new12']['pooled']['J']*100 for c in CELLS]
        a.plot(range(4),vals,'o-',label=label,color=color,lw=2)
        for j,c in enumerate(CELLS):
            values=[r['scores']['new12']['pooled']['J']*100 for r in data['sources'] if r['arm']==arm and r['cell']==c]
            a.scatter(np.arange(4)*.03-.045+j,values,color=color,alpha=.35,s=15)
    a.set(xticks=range(4),xticklabels=['Old/old','New/old','Old/new','New/new'],ylabel='Joint success on new12 layouts (%)',xlabel='Food/water image material',ylim=(0,105),title='Private success does not ensure natural expression')
    a.legend(loc='center left',fontsize=8)
    for offset,group,label in [(-.17,'old','Socially trained old18'),(.17,'new12','Target new12')]:
        values=[idx[arm,'new_new']['changes'][group]['pooled']['message_TV'] for arm in ARMS]
        b.bar(np.arange(3)+offset,values,width=.32,label=label,color=('#738FA7' if group=='old' else '#CEA07F'))
    b.set(xticks=range(3),xticklabels=['A','B','C'],ylabel='Message distribution total variation',title='Message sensitivity by layout support',ylim=(0,.26));b.legend(fontsize=8)
    for ax in (a,b):ax.spines[['top','right']].set_visible(False)
    (out/'figures').mkdir(exist_ok=True);fig.savefig(out/'figures/communication_transfer.png',dpi=180);fig.savefig(out/'figures/communication_transfer.pdf');plt.close(fig)
    print(str(out/'私人行动与共同通信在新图片上的迁移研究报告.md'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
