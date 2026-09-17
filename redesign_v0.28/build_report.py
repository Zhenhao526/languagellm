"""Report the complete fixed-budget matrix from independently checked results."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/formation_001'
def read(p):return json.loads(Path(p).read_text())
def pct(x):return f'{100*x:.3f}%'
def pp(x):return '0.000' if round(100*x,3)==0 else f'{100*x:+.3f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])

def main():
    a=read(OUT/'analysis.json');done=read(OUT/'training_complete.json');audit=read(OUT/'audit_execution.json');stats=read(OUT/'raw_validation.json')
    assert done['status']=='complete' and done['formal'] and audit['passed'] and stats['passed']
    assert read(OUT/'terminal_receipt.json')['exit_code']==0
    agg=a['aggregate'];rows=a['seed_rows'];seeds=a['seeds']
    get=lambda s,k:next(r for r in rows if r['seed']==s and r['condition']==k)
    score=lambda k,g='new12',m='pooled',v='J':agg[k]['scores'][g][m][v]
    names={'private_old':'私人18','private_all':'私人30','old_old':'A：私人18→通信18','all_old':'B：私人30→通信18','all_all':'C：私人30→通信30'}
    pt=table(['私人经验','old18 J','目标12 J','食物末帧 J','水末帧 J','目标12 Q'],[
        [names[k],pct(score(k,'old')),pct(score(k)),pct(score(k,m='food_only')),pct(score(k,m='water_only')),pct(score(k,v='Q'))] for k in ('private_old','private_all')])
    st=table(['通信条件','old18 J','目标12 J','全部30 J','目标12 Q','目标12常码 J','目标12置换 J'],[
        [names[k],pct(score(k,'old')),pct(score(k)),pct(score(k,'common30')),pct(score(k,v='Q')),pct(score(k,v='blank_J')),pct(score(k,v='shuffle_J'))] for k in ('old_old','all_old','all_all')])
    seed_table=table(['初始化来源','私人18 J','私人30 J','A J','B J','B−A（百分点）','C J'],[
        [s,*[pct(get(s,k)['scores']['new12']['pooled']['J']) for k in ('private_old','private_all','old_old','all_old')],
        pp(get(s,'all_old')['scores']['new12']['pooled']['J']-get(s,'old_old')['scores']['new12']['pooled']['J']),pct(get(s,'all_all')['scores']['new12']['pooled']['J'])] for s in seeds])
    mt=table(['通信条件','食物末帧 J','水末帧 J','new12归一化AUC'],[
        [names[k],pct(score(k,m='food_only')),pct(score(k,m='water_only')),pct(agg[k]['auc']['new12']['pooled']['J'])] for k in ('old_old','all_old','all_all')])
    private=read(OUT/'private_applicability.json');cap_table=table(['来源','食物末帧 J','水末帧 J','两项≥80%'],[
        [r['seed'],pct(r['mask_J']['food_only']),pct(r['mask_J']['water_only']),'通过' if r['passed'] else '未通过'] for r in private['source_rows']])
    assert abs(a['primary_mean']-(score('all_old')-score('old_old')))<1e-12
    data=read(ROOT/'data/selection.json')['images']
    material=table(['资源','训练图ID','测试图ID'],[
        [k,', '.join(r['id'] for r in data if r['stratum']==k and r['split']=='train'),', '.join(r['id'] for r in data if r['stratum']==k and r['split']=='test')] for k in ('apple','banana','orange','water')])
    interpretation=(ROOT/'结果解释.md').read_text()
    body=f'''# 新材料上的私人经验与共同通信形成研究报告

2026年9月16日，v0.28。完成4个新初始化来源的48次私人拟合和36次共同通信训练，使用独立于v23原图库的12张流程图片。主要比较为通信未见12布局上的终点自然双目标成功率：A为{pct(score('old_old'))}，B为{pct(score('all_old'))}，**B−A为{pp(a['primary_mean'])}个百分点**。C的对应成绩为{pct(score('all_all'))}，但这些布局已参与C的通信训练。

本轮是新材料上的完整形成**开发复现**。9张水果图此前用于v26/v27，3张水图本轮首次进入模型；测试图片对本次学习更新留出，整个研究并未对这些材料全程保持未揭盲。原确认48项及另4个水候选没有被本轮使用。

## 实验问题与材料

当通信经验相同，更广的私人资源布局经验是否帮助主体表达未在共同交流中训练过的状态？A私人只练18种布局、通信也只练18种；B私人练30种、通信仍练18种；C两个阶段都练30种。B中的目标12布局是私人已见、通信未见，C用于检查任务可学习性。主差不等于纯知识效应：固定预算下扩大布局支持，同时改变每种布局的复习频次。

上一轮通过来源、图像及双AI验收的3张水与全部9张水果组成12图。首次新水推理前，按固定来源簇与图ID哈希，每类2张训练、1张测试，没有按模型成绩选择或补换。

{material}

训练为6张食物×2张水，测试为3张食物×1张水。每个照片对覆盖30布局与2末帧mask，共720训练世界、180测试世界。**所有初始化来源共用同一张测试水**；180世界、三个划分和两个方向不能视为额外独立图片样本。来源检查只排除了已知关联，不能保证不存在未知转载或预训练重叠。

## 主体、任务与信息权限

官方DINOv2 ViT-L/14（304,368,640参数）保持冻结，仅输入图片像素。本轮重新编码12图，8张旧图编码对照与缓存最大差为0；特征均值及整体RMS仅由8张训练图估计。视觉底座没有使用文本提示，本项目训练的小接口承担资源后果学习、时序状态保持、行动和通信，不称大型视觉模型在完整原始社会中自主生存。

任务保留6个地点、食物和水各一处：第一帧完整观察，第二帧只显示其中一种资源，位置不变。发送者不知道接收者需求，只发送两枚0–6整数符号；接收者仅从这条消息分别作两种需求行动，不观察图片或对方行动。资源类别与正确地点编号用于环境构造、奖励及评价，不作为额外答案输入；地点结构仍由实验者预设的6个视觉槽位体现。这些空间先验已经给定。49条完整码可记忆30布局，成功本身不能证明组合句法。

每个新主体先做200步×64次资源选择奖励准备，仅继承本轮project到fresh Camp。随后从配对初值训练私人old/all接口，各2400步×512次所选需求动作；视觉投影冻结，slot_phi、GRU和私人头更新。反馈为所选需求的成功与否，未用地点分类监督。准备后会无梯度评价测试图，评价不参与更新或筛选。

私人阶段结束后丢弃私人头、冻结全部视觉与时序接口，并重新初始化通信模块。B/C共享私人all终点；A/B的实际通信世界相同。各社会条件2400次配对更新，每方向256条消息及512次需求行动，奖励`.25(cF+cW)+.5cF*cW`。学习率、熵日程、梯度裁剪、奖励及模型均沿v23，主要实现改动是非方形图片池与训练图归一化。

四个初始化来源34101–34104，各含3个既有布局划分、2个方向。来源内等权平均后再汇总四来源，不将36个社会运行当36个独立样本。所有主体与失败均保留，未因能力门槛或中间成绩延长训练。

## 全部主要结果

J为自然逐token贪心消息和接收动作的双目标成功；Q对49完整消息及两种随机行动精确积分。常码00和整码独立置换只作无更新参照，置换保留完整码的频率。私人J/Q分别由原私人行动头计算。

{pt}

{st}

{seed_table}

{mt}

过程AUC由0、100、600、1200、2100、2400六个固定时点评价作归一化梯形积分；B−A的目标12 AUC差为{pp(a['primary_auc_mean'])}个百分点，仅作为辅助结果。

私人all能力检查为四来源分别在目标12的两mask均达到80%，每项跨3划分与2人平均。本批整体{'通过' if private['passed'] else '未通过'}，逐项如下；该门槛不筛选主体，也不等于全部非语言能力已经具备。

{cap_table}

![私人能力与共同通信终点]({(OUT/'figures/01_private_and_communication.png').resolve()})

![固定预算的学习曲线]({(OUT/'figures/02_learning_curves.png').resolve()})

{interpretation}

## 实际符号与验证范围

[固定消息实例]({(OUT/'实际消息示例.md').resolve()})展示首个来源、划分1、方向0的同一照片对、地图0–5及两mask，在三个条件下共36例，成功和失败全部保留。字符仅是整数的报告别名，没有赋予单个符号预定意义。

正式阶段共{done['private_updates']:,}次私人更新、{done['private_actions']:,}次私人动作；{done['pair_updates']:,}次社会配对更新、{done['messages']:,}条消息、{done['actions']:,}次需求行动。另有{done['initial_preparation_updates']:,}次基础准备更新。CPU单线程训练与评价计时{done['seconds']:.3f}秒，DINO编码及独立复核另计。

正式启动前完成7项世界测试、40步完整开发运行、核心函数保真与配对初值检查。当时独立数值和终点复算仍在并行进行，[前置记录]({(ROOT/'preflight_qa.json').resolve()})保留其真实范围，未提前记为全部通过。开发及正式最终复算均通过。

[独立统计]({(OUT/'raw_validation.json').resolve()})重算全部保存评价表及汇总；[执行复核]({(OUT/'audit_execution.json').resolve()})核对实际720/180世界、8训练图归一化、配对初值和冻结，并重放全部终点私人与社会前向、完整接收码表。训练流身份仅抽固定第1、2101和末次更新；训练cache只核首末块，未独立重训整条轨迹或重放优化器。原始数值一致不能替代独立材料范围或新颖性证据。

完整形成开发复现已完成。小图片池、单测试水、四初始化来源及任务内先验仍限制结论；正式独立确认、论文中心贡献和统一投稿稿尚未完成。

[固定方案]({(ROOT/'固定执行方案.md').resolve()}) · [独立分析]({(OUT/'analysis.json').resolve()}) · [复现索引]({(ROOT/'README.md').resolve()})
'''
    target=OUT/'新材料上的私人经验与共同通信形成研究报告.md';target.write_text(body);print(str(target))

if __name__=='__main__':main()
