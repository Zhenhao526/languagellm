"""Research report from independent metrics and separately audited process logs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/joint_001'
def read(name):return json.loads((OUT/name).read_text())
def pct(x):return f'{100*x:.3f}%'
def pp(x):return f'{100*x:+.3f}'
def main():
    a=read('analysis.json');st=read('structural_assay.json');audit=read('audit_execution.json');process=read('process_summary.json');receipt=read('training_complete.json')
    assert audit['passed'] and receipt['social_runs']==12;g=a['aggregate'];score=lambda k,group='old',mask='pooled':g[k]['scores'][group][mask]
    main_rows='\n'.join(f"| {title} | {pct(score('mean',group)[metric])} | {pct(score('joint',group)[metric])} | {pp(score('joint',group)[metric]-score('mean',group)[metric])} |" for title,group,metric in [
        ('主量：old18联合J','old','J'),('old18随机政策Q','old','Q'),('辅助：new12联合J','new12','J'),('new12随机政策Q','new12','Q'),('共同30联合J','common30','J')])
    source_rows='\n'.join(f"| {r['seed']} | {pct(r['mean'])} | {pct(r['joint'])} | {pp(r['difference'])} | {pp(r['auc_difference'])} | {pp(r['new12_difference'])} |" for r in a['primary'])
    resource_rows='\n'.join(f"| {group} | {pct(score('mean',group)['food'])} | {pct(score('joint',group)['food'])} | {pct(score('mean',group)['water'])} | {pct(score('joint',group)['water'])} |" for group in ('old','new12'))
    structure_rows=[];structure_sources=[]
    for arm,label in [('mean','原均值'),('joint','新联合'),('attention','原注意力')]:
        for assignment,desc in [('food_water','第0槽食物／第1槽水'),('water_food','第0槽水／第1槽食物')]:
            z=st['aggregate'][arm][assignment]['new12']['pooled'];r=z['recoding'];q=r['quantiles']
            structure_rows.append(f"| {label} | {desc} | {pct(z['recombined_J'])} | {pct(r['mean'])} | {pct(q['0.025'])}–{pct(q['0.975'])} |")
    for assignment,desc in [('food_water','食物／水'),('water_food','水／食物')]:
        rows=st['comparisons']['joint_minus_mean']['assignments'][assignment]['new12']['pooled']['source_differences']
        structure_sources.append(f"{desc}分配的四来源重组J差依次为"+'、'.join(pp(r['recombined_J_difference']) for r in rows)+'个百分点。')
    structure_rows='\n'.join(structure_rows);structure_sources=' '.join(structure_sources)
    controls='\n'.join(f"| {name} | {group} | {pct(score(arm,group)['J'])} | {pct(score(arm,group)['blank_J'])} | {pct(score(arm,group)['shuffle_J'])} |" for arm,name in [('mean','均值'),('joint','联合')] for group in ('old','new12'))
    pr=process['aggregate']['joint'];auc_diff=sum(r['auc_difference'] for r in a['primary'])/len(a['primary'])
    report=f'''# 资源角色分别读入与共同通信学习：v25研究报告

本轮新增12次双主体训练。允许分别读取两类资源表示后，熟悉布局上的联合成功率由 **{pct(score('mean')['J'])}** 升到 **{pct(score('joint')['J'])}**，主差 **{pp(a['primary_mean'])}个百分点**，四个继承来源均改善；曲线AUC也提高。但通信未见12组合的成功率仍只有 **{pct(score('joint','new12')['J'])}**，来源变化不一致。结果支持当前接口的熟悉协议学习得到改善，尚未解决新组合表达。

本轮从原均值组完全相同的初始发信函数开始，用可学习的资源差分读取替代原来无任务梯度的矩阵。总参数数目不变，有效自由度与优化路径改变；不能把所得差异称为纯非语言知识效应或语言结构已经增强。

## 1. 为什么进行本轮诊断

[v24报告](../../../redesign_v0.24/results/attention_001/私人预测动态读取与共同符号结构研究报告.md)中，mean/attention连已进行通信训练的old18布局也只有76.157%/62.963%联合成功，而new12仅1.042%/4.514%。这使新组合失败同时受到熟悉任务学习不足的限制。

本轮先检验一个具体的读入限制：原均值方式给两个编码向量相同的融合权重，允许这两套权重从相同起点分别学习，是否更容易学会熟悉组合？主量因此在v25训练前固定为old18的joint−mean。这是由v24结果触发的新诊断，未回改v24的new12主量，也没有把old成绩当作整体研究目标。

## 2. 同一初始协议，允许分别学习两类资源的读取

继续使用官方冻结DINOv2 ViT-L/14（304,368,640参数）特征，以及v23私人all终点产生、v24已经封存的预测缓存。本轮不重新推理视觉模型或私人头。私人主体在原有限测试集能准确定位食物和水，社会发送者读取自身两组6地点预测概率，各附2位角色标记；这是额外可读任务结构，不能称原始视觉直接产生语言。

相同共享编码得到z0、z1后，令m=(z0+z1)/2，d=(z0−z1)/2。本轮joint输出为：

`out(tanh(fusion(concat(q,m)) + V*d))`

q仍由原LSTMCell及上一token产生。V是零初始化96×96矩阵，替换mean中单key softmax导致任务梯度恒零的bilinear矩阵。编码、LSTM、embedding、fusion、out及接收者的其他参数全部复制实际v24 mean初始检查点。

V=0时完整49码概率、逐token贪心消息和接收行为与mean相同。若融合层对m的权重为Wm，则对z0、z1的有效线性权重为(Wm+V)/2和(Wm−V)/2。这说明初始两者绑定，此后可以分别读取；不是给第一枚、第二枚符号预先指定资源语义。[函数关系与边界](../../联合读入的函数关系.md)

两组总sender均73,191参数、receiver6,364参数；mean的旧矩阵没有任务梯度，而joint的V可用。因此相同初始函数、总参数数目不等于相同有效容量、梯度或训练轨迹。虽然m,d能在融合前恢复两个编码向量，后续仍有96维瓶颈；原mean高维非线性编码也可能保留角色信息，未证明它已经丢失某种信息。该控制不是新的注意力方法。

## 3. 固定环境和训练预算

| 项目 | 设置 |
|---|---|
| 参照来源 | v24的mean与attention共24次既有训练，直接绑定已审结果 |
| 新训练 | joint：四来源33101–33104×三分区，共12次双主体训练 |
| 世界 | 六地点、食物和水各占不同地点，共30布局 |
| 观察 | 第一帧完整、第二帧只见一种资源，布局不移动 |
| 私人／社会经验 | 私人已见全部30，社会训练仅old18；new12只是社会未见 |
| 测试 | 960世界＝30布局×16旧池测试照片对×2遮挡 |
| 信道与行动 | 两枚7选1符号，同一消息供接收者两需求行动；接收者无图像或私人预测 |
| 每组预算 | 2400次成对更新，每方向256消息，每消息两个行动 |
| 本轮总量 | 28,800对更新、14,745,600消息、29,491,200行动 |

收益R=.25(cF+cW)+.5cFcW，REINFORCE常数基线.5；每人(sender+receiver)/2，Adam学习率.0007，每角色clip2，前2100更新熵系数.02、之后0。真实地点只在抽样行动后计算收益。没有正确答案监督、额外反馈、来源筛选或预算延长。

世界、照片、遮挡及抽样uniforms与历史配对mean相同。初态完整函数匹配经过正式全部23,040个方向世界验证。开发仅99523/p1一个joint、40步，完整流程和独立复核通过后才启动正式批次。细节见[固定方案](../../固定执行方案.md)、[科学前审](../../前置科学审查.md)及[正式门禁](../../preflight_qa.json)。

## 4. 熟悉协议学得更好，新组合表达仍弱

J为逐token贪心发信、两需求贪心行动同时正确率。Q对全部49条消息与两个正确行动的概率求和，是随机政策成功率。本轮old18仍用测试照片评价，不是重复训练batch的准确率。

| 指标 | 历史mean | 新joint | joint−mean，百分点 |
|---|---:|---:|---:|
{main_rows}

![来源配对结果](figures/01_source_comparison.png)

| 来源 | mean old18 J | joint old18 J | 主终点差，百分点 | old18归一AUC差，百分点 | new12 J差，百分点 |
|---|---:|---:|---:|---:|---:|
{source_rows}

old18归一AUC平均提高{pp(auc_diff)}个百分点，四来源均为正，支持所采样学习曲线上较早出现较好的熟悉协议。AUC只按0/100/600/1200/2100/2400六个固定点作梯形积分并除以2400；不声称准确识别收敛时间或某次迭代的“语言诞生时刻”。

![固定预算学习曲线](figures/02_learning_curves.png)

| 评价支持 | mean食物 | joint食物 | mean水 | joint水 |
|---|---:|---:|---:|---:|
{resource_rows}

资源改善并非所有分项都一致。两种末帧遮挡的new12 J在本批完全相同，概率Q仍有细微差异；不能将遮挡分支当作独立来源。历史attention的old18/new12为62.963%/4.514%，joint熟悉任务更高而新组合更低。该辅助参照不改变本轮主量，不提供一般架构排名。

只有四个继承来源。分区、方向与照片不是独立重复，没有利用它们制造更大的有效样本量。旧图像池已用于多轮开发，结果尚不是独立材料确认。

## 5. 消息内容与片段复用

| 条件 | 支持 | 自然消息J | 固定空码J | 完整消息错配期望J |
|---|---|---:|---:|---:|
{controls}

空码是[0,0]；完整错配按整个有限测试支持中的自然贪心消息频率计算精确期望。这检验已学消息的作用，不是新增训练的无通信对照。

继续沿v24固定供体重组：仅从old18、相同照片对与遮挡中取供体，一条食物相同而水不同，另一条水相同而食物不同；各取指定槽的符号，拼成消息后查询固定接收者。每目标先均匀平均供体对，再目标等权。两个槽位—资源分配方向均保留，未按new12成绩择优。

200个固定整码双射先重命名完整供体，拆拼后再逆映射接收器，保留自然整串消息的行为而改变字符坐标。逐双射完成来源层级汇总后取分位数；旧基线直接复用已审结果。[结构评估方案](../../结构评估方案.md)

| 条件 | 供体槽位分配 | new12重组J | 整码重命名参照均值 | 参照2.5%–97.5%范围 |
|---|---|---:|---:|---:|
{structure_rows}

{structure_sources}

人工重组绝对成功率仍低。水／食物分配相对mean的四来源差均为正，食物／水分配为两正两负，不能把一个方向的优势推广到全部结构。高于整码重命名参照只支持固定供体操作中存在局部可复用的符号坐标，不能证明主体自然采用同一规则。参照分位范围也不是效果的总体置信区间。49个整码足以分别命名30布局，任务成功本身不排除整体记忆。全部分区、遮挡、两种分配和200个参考值见[结构结果](结构重组评估.md)与[原始结构数据](structural_assay.json)。

## 6. 操纵确实生效，但不能据此识别唯一机制

24个主体的V在首次更新都获得非零梯度，clip前范数范围为{pr['initial_contrast_range'][0]:.6f}–{pr['initial_contrast_range'][1]:.6f}，该梯度已经包含每人角色损失平均的1/2权重。终点V的Frobenius范数为{pr['final_contrast_range'][0]:.3f}–{pr['final_contrast_range'][1]:.3f}。因此不是新增参数一直未更新的空操作。

mean、attention、joint各57,600个人次更新的发送/接收梯度裁剪均未施加非单位系数；独立检查按实际float32含epsilon公式确认，不能用“新增参数触发共同裁剪”解释本批差异。不过有效参数和新的直接梯度路径仍是操纵组成，未因此证明纯信息恢复或因果中介。[过程汇总](process_summary.json)、[独立过程核查](process_summary_qa.json)

## 7. 研究含义与后续取舍

本轮把一个问题收窄了：从相同初始协议出发，允许两个资源角色分别读取，能改善熟悉布局上的共同通信学习；这种改善没有转成可靠的新组合表达。它支持把私人行动成功、熟悉约定的学习过程、自然新组合表达、人工片段复用分开检验。这里不是声称刚发现这种一般区分，而是对本项目具体模型和任务取得了匹配证据。

熟悉任务83.333%也仍有误差，因此尚未完全消除学习不足这一限制；不能把2.083%的new12表现称为模型组合能力的理论上限。与此同时，不能继续靠增加同图库接口变体或训练预算直到出现希望的分数。本次joint对照已完成，后续不再将它列为待执行候选，也不自动扩大joint/attention超参数网格。

下一步应收束这条读入诊断，先盘点并统一已经可用的通信基线、分清其信息权限和训练支持，再为一个明确可反驳的迁移问题固定独立材料确认。后续更复杂资源任务必须检验新的能力或生态压力，而不是把增加模块数量当作语言出现的证据。独立图库准备仍有缺口，尚未发生新的图像确认；本轮没有补齐这一门槛。

环境仍是有限资源定位的抽象，未模拟资源生产/消耗、长期生存或自由形成社会分工。地点、需求、角色与私人预测结构由实验设置给定。研究可以讨论给定这些非语言先验后，什么读入条件影响约定学习，不能据此证明分工是人类语言产生的必要条件、认知完全从零或一般句法已经形成。

## 8. 复核与可复现范围

本轮CPU单线程训练耗时约{receipt['seconds']:.1f}秒，无新DINO推理、私人训练或缓存前向。所有12组按固定预算完成；旧24组参照不计作新增训练。

[独立执行审计](audit_execution.json)通过{audit['checks']:,}项检查，[指标比较](comparison.json)包含16,380项比较，最大差2.22×10⁻¹⁶。覆盖全部144份新评价表、初态及终点各23,040发送世界的完整49码概率/贪心、24张完整接收表、全部28,800外部随机流、各检查点冻结参数；固定33101/p1第1及2101共两次成对更新的trace、损失、读出梯度范数、clip、参数和Adam逐位一致。没有逐步重训整条轨迹。私人缓存只验已有封存字节，未冒称本轮重新推理私人模型。

[结构核查](structural_assay_audit.json)只在固定33101/p1/d0一份新终点协议，用独立朴素循环复算960目标×两分配×identity及三双射，共7,680项；其余新结构表由通过开发检查的统一程序计算。这与全部神经终点和指标复核的覆盖不同。过程核查另有349项独立检查，不等于全程模型梯度重放。

主要数值见[独立分析](analysis.json)，完整来源与预算见[训练凭证](training_complete.json)，复现命令见[v25索引](../../README.md)。本轮全部工程复核通过，不增加来源独立性，也不证明论文新颖性或ICLR稿件已经完成。完整论文仍需明确中心贡献、独立确认、收敛的复现包和完整论证。
'''
    path=OUT/'资源角色分别读入与共同通信学习研究报告.md';path.write_text(report);print(json.dumps(dict(report=str(path),characters=len(report)),ensure_ascii=False))
if __name__=='__main__':main()
