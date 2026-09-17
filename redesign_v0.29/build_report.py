"""Build the v29 report from independently recomputed finite statistics."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent

def read(p):return json.loads(Path(p).read_text())
def pct(x):return f'{100*x:.3f}%'
def pp(x):return f'{100*x:+.3f}' if abs(x)>1e-12 else '0.000'
def direction(values):
    return f"{sum(x>1e-12 for x in values)}正、{sum(x< -1e-12 for x in values)}负、{sum(abs(x)<=1e-12 for x in values)}零"
def link(p,label):return f'[{label}]({p.resolve()})'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve()
    x=read(out/'analysis.json');qa=read(out/'raw_validation.json');audit=read(out/'audit_execution.json');run=read(out/'training_complete.json')
    assert qa['passed'] and audit['passed'] and run['status']=='complete'
    elapsed=read(out/'terminal_receipt.json').get('reported_program_seconds',run['seconds'])
    arms=['target_paths3','target_paths2'];a,b=[x['aggregate'][c] for c in arms]
    s=lambda block,g,k='J',m='pooled':block['scores'][g][m][k]
    ds=[r['difference'] for r in x['primary']];ads=[r['auc_difference'] for r in x['primary']]
    supported=all(v>1e-12 for v in ds)
    finding=('四个来源均为预期方向，但仍只有两个固定支持图及四个开发来源，不能据此建立普适图规律。' if supported else
        '来源结果未一致支持事前预测“每个目标有3条关联路径优于2条”。本轮不支持该预测的稳健版本；这不证明支持结构完全无效或两组等效。')
    lines=['# 训练共现结构与共同符号迁移研究报告','',
      '2026-09-16 · v0.29 · 本地、固定预算的开发实验','',
      f"已完成24次新双主体通信训练。共同未训练P6的自然双资源成功率：路径3组{pct(s(a,'common_target6'))}，路径2组{pct(s(b,'common_target6'))}，事前主差{pp(x['primary_mean'])}个百分点，四初始化来源{direction(ds)}。{finding}",
      '', '## 研究问题与实验变化','',
      '本轮检验：在私人能力、训练布局数量、每位置实际曝光次数、消息容量和预算相同时，社会经验中的资源共现结构是否改变新配对的表达。主体沿用v28已获得有限私人资源定位能力的冻结接口，全部通信从新初值学习。',
      '两组均在18个食物—水位置对上训练，每个资源位置每步恰好出现42次。两组均匀分布的边际熵、联合熵及1bit互信息相同，只改变具体哪些位置曾一起出现。共同训练13个布局，各5个独有；共同未训练7个布局，预先选位置边际均匀的P6为主目标，额外1图单列。',
      '路径3/路径2指每个P目标在二部训练支持图中有3/2条长度3关联路径；图中四环数同时为3/6。该操纵改变整体支持结构，不能单独识别某个路径统计量的机制，也不是主体经历三步动作。',
      '',f'![两种经验支持图]({out}/figures/01_support_design.png)','',
      '## 预定主结果与学习过程','',
      '| 初始化来源 | 路径3组P6 J | 路径2组P6 J | 终点差（百分点） | P6 AUC差（百分点） |',
      '|---|---:|---:|---:|---:|']
    for r in x['primary']:lines.append(f"| {r['seed']} | {pct(r['path3'])} | {pct(r['path2'])} | {pp(r['difference'])} | {pp(r['auc_difference'])} |")
    lines += [f"| 来源均值 | {pct(s(a,'common_target6'))} | {pct(s(b,'common_target6'))} | {pp(x['primary_mean'])} | {pp(x['primary_auc_mean'])} |",'',
      f"主量使用2400步自然逐token贪心发送与贪心接收，两个资源都正确才计成功。AUC是固定0/100/600/1200/2100/2400检查点的归一梯形面积，来源{direction(ads)}，仅作辅助，不能替代终点主量。",'',
      '| 评价支持 | 路径3组自然J | 路径2组自然J | 路径3组概率Q | 路径2组概率Q |','|---|---:|---:|---:|---:|']
    scopes=[('train18','各自训练18'),('shared_train13','共同训练13'),('common_target6','共同主目标P6'),('common_unseen7','全部共同未训练7'),('extra_common_unseen1','额外共同未训练1'),('other_held6','各自其他留出6'),('held12','各自留出12'),('common30','全30')]
    for key,label in scopes:lines.append(f"| {label} | {pct(s(a,key))} | {pct(s(b,key))} | {pct(s(a,key,'Q'))} | {pct(s(b,key,'Q'))} |")
    lines += ['', '各自训练18、其他留出6和留出12所含具体布局不同，不能把它们的组差当同一组考题上的迁移效应。Q对49个完整消息和两个动作概率精确求和，不等同自然贪心J。','',
      '| 主目标P6的分项 | 路径3组 | 路径2组 |','|---|---:|---:|']
    for key,label in [('food','食物位置正确'),('water','水位置正确'),('blank_J','固定空码00双成功'),('shuffle_J','全30整码频率打乱'),('within_shuffle_J','P6内整码频率打乱')]:
        lines.append(f'| {label} | {pct(s(a,"common_target6",key))} | {pct(s(b,"common_target6",key))} |')
    for m,label in [('food_only','末帧仅食物：双成功'),('water_only','末帧仅水：双成功')]:
        lines.append(f'| {label} | {pct(s(a,"common_target6",m=m))} | {pct(s(b,"common_target6",m=m))} |')
    lines += ['', 'P6是完美匹配，在该评价子集中一个资源位置就确定另一个。固定无信息动作对的最优P6成功率可达1/6；不能以1/36作为普适无通信上界。即使自然J较高，也不能仅凭这个集合证明两个资源被独立组合编码。','',
      f'![学习过程与来源差]({out}/figures/02_support_outcomes.png)','',
      '## 片段重组与任意整码标签参照','',
      '对每个P6目标，固定相同测试图片与末帧mask，从训练布局取3个同食物位置供体和3个同水位置供体，遍历9种组合。FW取食物供体第一个符号及水供体第二个符号；WF反向，两种均预先固定。这里的“训练供体”指布局已训练，测试照片本身未作为通信训练输入。',
      '对199个预定49码随机双射，发送完整消息与接收表同时重编码，完整消息行为保持相同；随后才在新标签的两个坐标上拼接。这样能区分原协议的符号片段组织与任意整码标签效果。','',
      '| 条件 | 拼接方向 | 原始重组J | 199双射均值 | 原始减均值（百分点） | 上尾参照比例 |','|---|---|---:|---:|---:|---:|']
    for c,label in zip(arms,['路径3','路径2']):
        for orient in ('FW','WF'):
            v=x['aggregate'][c]['recombination_null']['pooled']['recombine_'+orient+'_J']
            lines.append(f"| {label} | {orient} | {pct(v['observed'])} | {pct(v['null_mean'])} | {pp(v['excess'])} | {v['upper_tail_fraction']:.3f} |")
    lines += ['', '上尾比例=(1+#随机标签分数≥观察分数)/200，是完整码随机标签下的描述性参照，不是以四个独立来源为单位的统计显著性。完整199项、每来源及两mask见analysis.json。', '',
      '两组、两种预定槽位方向的平均人工重组均高于全部199个完整码随机双射。它支持一个有限结论：分析者按资源真值选取供体后，可以利用当前协议中的部分片段对应。该参照只保持完整码行为，不保持原神经网络的参数化或学习难度；它没有分离任务学习与双符号、嵌入和分支解码架构的偏置，也不是另一种训练随机化。',
      '这一迹象并非每个来源、每种方向都稳定：来源34101的路径3组FW重组为0，低于标签参照；来源34102同格原始重组1.235%，上尾参照比例0.31。不能把聚合结果泛化为所有主体都具有相同词序或自主组合能力。人工重组与自然发送使用不同流程，不作直接优劣因果比较。', '',
      '| 来源 | 路径3 FW超出均值（百分点） | 路径3 WF超出均值（百分点） | 路径2 FW超出均值（百分点） | 路径2 WF超出均值（百分点） |','|---|---:|---:|---:|---:|']
    for seed in x['seeds']:
        vals=[]
        for c in arms:
            r=next(r for r in x['seed_rows'] if r['seed']==seed and r['condition']==c)
            for o in ('FW','WF'):vals.append(pp(r['recombination_null']['pooled']['recombine_'+o+'_J']['excess']))
        lines.append('| '+str(seed)+' | '+' | '.join(vals)+' |')
    lines += ['',link(out/'实际消息示例.md','24条实际消息与动作记录')+'按固定首来源、首照片规则展示，未按成败挑选。','',
      '## 执行与可复算性','',
      '现有官方DINOv2-L主干（304,368,640参数）冻结。私人阶段的小型投影、记忆和空间接口已训练，本轮全部冻结，只新训练通信接口；不是把完整agent所有能力都归于视觉预训练。私人动作头不进入通信，接收者没有图像或目标位置。每人参数独立，双方只用离散符号和任务后验奖励进行学习。',
      '数据和私人接口复用v28：6食物×2水训练照片，3食物×1水测试照片；第一帧两资源、第二帧一种可见。实验者给定6个位置槽，不能称不带结构先验。每个来源含3坐标重复及2方向，外层仍只有4个初始化来源；p、方向及反复使用的测试图不增加独立样本量。',
      f"完整批次24×2400={run['pair_updates']:,}双主体更新，{run['messages']:,}消息、{run['actions']:,}资源动作，CPU单线程训练与评价记账{run['seconds']:.3f}秒（含最终清单写入的终端耗时{elapsed:.3f}秒）。本轮0次新私人拟合、0次DINO前向、0次图像或权重下载。两新臂每方向252消息，旧v28为256，旧结果不作此轮单因素等预算对照。",
      '优化与v21/v28核心更新顺序一致：Adam .0007，每人发送/接收损失平均，两角色分别clip2；R=.25(cF+cW)+.5cF*cW，基线.5，前2100步熵.02、末300步0。整批不按成绩筛种子、不续训单臂。',
      '联调审计最初在收尾枚举依赖文件时遇到Torch虚拟模块路径，修复为仅记录实际存在的Python源文件后通过；失败记录和当时审计源码保留。未改变训练代码、指标或筛选条件。',
      '8项支持/fixture检查、3项人工协议指标检查、独立种子的40步联调与独立统计/终点重放在完整批次之前通过。正式批次随后独立从全部原始表重算统计，重放全部终点发送与49码接收表。训练抽第1/2101/末步核fixture和保存trace，未重放完整Adam轨迹。',
      f"原始统计核对{qa['coverage']['protocol_tables']}个协议表、{qa['coverage']['protocol_worlds']:,}个世界行，最大指标差{qa['maximum_metric_absolute_difference']:.3g}；执行复核passed={audit['passed']}。检查数保证结果可复算，不增加科学样本数或新颖性。",'',
      '## 证据边界与下一步','',
      finding,
      '此前研究已发现训练组合密度、覆盖方式和频率影响泛化；本轮的较窄贡献候选是保持单位置实际曝光、信息量和共同测试集的支持图对照。没有找到直接同构的先例，不等于已证明首次性。私人能力和图片来源均已开发暴露，只有一个测试水图；本轮不是独立确认，也尚不足以形成ICLR级中心论证。',
      '下一项任务设计应优先解除P6一一匹配的识别限制：让共同未训练目标中的同一食物位置对应多个水位置，再要求新表达同时区分两资源；同时设置整码记忆与可复用片段确有行为差别的任务条件。先明确能排除的捷径与成功标准，再增加复杂度。当前两图分支保持结果，不继续按图统计或训练预算寻找正差。', '',
      '## 文件与复现','',
      '- '+link(ROOT/'固定执行方案.md','模型运行前固定方案')+'；'+link(ROOT/'preflight_qa.json','完整批次前置记录'),
      '- '+link(out/'analysis.json','独立统计及199项标签参照')+'；'+link(out/'raw_validation.json','原始表核对')+'；'+link(out/'audit_execution.json','执行复核'),
      '- '+link(out/'training_complete.json','训练完成与哈希')+'；'+link(out/'terminal_receipt.json','终端完成记录'),
      '- '+link(PROJECT/'paper_program/mechanism_pressure_20260916/近邻与可证伪预测.md','三篇近邻原文核查')+'；'+link(PROJECT/'paper_program/mechanism_pressure_20260916/设计边界.md','独立设计审查'),
      '', '从项目根用既有环境运行run_support.py --out 一个不存在的目录，再依次运行analyze_results.py、audit_results.py、plot_results.py、message_examples.py及build_report.py，均指定同一--out。完整输入来源以invocation.json为准。']
    path=out/'训练共现结构与共同符号迁移研究报告.md';path.write_text('\n'.join(lines)+'\n');print(path)
if __name__=='__main__':main()
