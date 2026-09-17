"""Build the readable report from completed independent analysis outputs."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/attention_001'
def read(p):return json.loads(p.read_text())
def pct(x):return f'{100*x:.3f}%'
def pp(x):return f'{100*x:+.3f}'
def main():
    a=read(OUT/'analysis.json');st=read(OUT/'structural_assay.json');audit=read(OUT/'audit_execution.json');receipt=read(OUT/'training_complete.json');ex=read(OUT/'protocol_examples.json')
    assert a['formal'] and audit['passed'] and receipt['social_runs']==24
    g=a['aggregate'];score=lambda arm,group='new12',mask='pooled':g[arm]['scores'][group][mask]
    main_rows='\n'.join(f"| {title} | {pct(score('mean',group)[metric])} | {pct(score('attention',group)[metric])} | {pp(score('attention',group)[metric]-score('mean',group)[metric])} |" for title,group,metric in [
        ('社会未见12组合：联合J','new12','J'),('社会未见12组合：食物','new12','food'),('社会未见12组合：水','new12','water'),
        ('社会未见12组合：随机政策Q','new12','Q'),('社会已见18组合：联合J','old','J'),('共同30组合：联合J','common30','J')])
    source_rows='\n'.join(f"| {r['seed']} | {pct(r['mean'])} | {pct(r['attention'])} | {pp(r['difference'])} | {pp(r['auc_difference'])} |" for r in a['primary'])
    mask_rows='\n'.join(f"| {label} | {pct(score('mean',mask=mask)['J'])} | {pct(score('attention',mask=mask)['J'])} |" for label,mask in [('末帧仅食物','food_only'),('末帧仅水','water_only')])
    structure_rows=[]
    for arm,label in [('mean','均值'),('attention','注意力')]:
        for assignment,desc in [('food_water','第0槽食物／第1槽水'),('water_food','第0槽水／第1槽食物')]:
            z=st['aggregate'][arm][assignment]['new12']['pooled'];r=z['recoding'];q=r['quantiles']
            structure_rows.append(f"| {label} | {desc} | {pct(z['recombined_J'])} | {pct(r['mean'])} | {pct(q['0.025'])}–{pct(q['0.975'])} |")
    structure_rows='\n'.join(structure_rows)
    controls='\n'.join(f"| {label} | {pct(score(arm)['J'])} | {pct(score(arm)['blank_J'])} | {pct(score(arm)['shuffle_J'])} |" for arm,label in [('mean','均值'),('attention','注意力')])
    erows=[]
    for row in ex['rows']:
        erows.append(f"| {row['condition']} | {row['map_id']} | {row['social_exposure']} | {','.join(map(str,row['target_sites_1based']))} | `{row['display_message']}` | {','.join(map(str,row['receiver_sites_1based']))} | {'是' if row['joint_correct'] else '否'} |")
    erows='\n'.join(erows);positive=sum(r['difference']>0 for r in a['primary']);auc=sum(r['auc_difference']>0 for r in a['primary'])
    interpretation=(OUT/'interpretation.md').read_text()
    report=f'''# 私人预测的动态读取与共同符号结构：v24研究报告

本轮未获得稳定、可迁移的共同符号结构。完成的24次新双主体社会训练使用四个继承来源，两组都读取已经学会资源定位的私人行动头；社会未见12种资源组合上的联合成功率，均值组为 **{pct(score('mean')['J'])}**，注意力组为 **{pct(score('attention')['J'])}**，主差为 **{pp(a['primary_mean'])}个百分点**。四来源中{positive}个主差为正、一个为零，{auc}个学习曲线AUC差为正；注意力组在已训练组合上的成功率却更低，因此不能称整体通信改善。

这项比较检验了给定私人能力后的表达迁移。它将保留两个预测并动态读取与先做均值压缩比较，不能单独归因于抽象的“注意能力”；私人预测已有需求和地点结构，也不是认知完全从零。以下同时报告自然消息、消息干预和预先固定的符号重组检查。

## 1. 研究问题与近邻

v23表明，在有限测试上私人定位都已成功时，共同通信仍可能难以表达社会学习未见的资源组合。本轮问：把已经可用的私人行动预测分开保留，在每枚符号生成时读取，会怎样改变通信形成过程和迁移？

直接方法依据为 Ri、Ueda、Naradowsky（2023）的 [Emergent Communication with Attention](https://arxiv.org/abs/2305.10920)。其AT/NoAT发送者分别逐token查询多个输入向量或查询均值后的单向量。本轮保留这一对比，但输入由ConvNeXt图像patch改为私人行动预测，接收者也保持本任务的信息权限。初态、融合层和无input feeding是本地选择；未取得作者代码，不能称原样复现。

Feng、An、Lu（2024）的 [Learning Multi-Object Positional Relationships via Emergent Communication](https://ojs.aaai.org/index.php/AAAI/article/view/29685)采用另一种CNN—LSTM指称任务，不能把本轮模型称为其注意力架构。详见[近邻方法核查](../../近邻基线方法核查.md)。本轮补足一个已有方法启发的任务内基线，尚未构成新的注意力算法。

## 2. 固定环境、主体和训练

| 项目 | 本轮设置 |
|---|---|
| 视觉来源 | 官方冻结DINOv2 ViT-L/14，304,368,640参数；复用既有特征 |
| 私人主体 | v23 private_all终点，共24个私人头；四来源×三分区×两人 |
| 世界 | 六地点、食物和水各占不同地点，共30种布局 |
| 观察 | 第一帧两资源可见；第二帧只见一种，布局不动 |
| 发送输入 | 自己的冻结私人头输出两组6地点概率，各附2位角色标记 |
| 接收输入 | 同一条两枚离散符号消息；既有两需求行动分支，无图像或私人预测 |
| 社会训练／测试 | 仅old18训练；960测试世界涵盖30布局×16照片对×2遮挡 |
| 可训练部分 | 每人新sender73,191参数＋receiver6,364参数；其余冻结 |
| 预算 | 24次双主体训练，每次2400更新；共57,600对更新、29,491,200消息、58,982,400行动 |

冻结私人头在本有限测试集的两种遮挡、所有布局上均达到100%联合定位。私人学习见过30布局，所以“社会未见12组合”只表示通信学习没有暴露这些组合。它们不是整个主体从未经历的世界。44张训练／16张测试图片的旧池已用于多轮开发，尚无独立新图库确认。

两组共享Linear(8,96)+GELU逐预测编码。均值组先将两个编码取均值；注意力组保留两个编码。其后都使用LSTMCell和双线性注意力，每步由上一token产生query，再融合query/context生成下一个符号。两枚符号顺序未指定资源角色；词表7，共49条可能消息。SOS不在通信词表中。

同来源、分区和主体，两组初始参数逐位相同，训练世界与外部抽样随机流相同。收益为R=0.25(cF+cW)+0.5cFcW，REINFORCE常数基线0.5，Adam学习率0.0007，每角色clip2，前2100更新熵系数0.02、后300为0。所有条件固定预算，不因成绩延长。正式前两臂40步开发试跑和独立审计已通过。完整细节见[固定执行方案](../../固定执行方案.md)、[前置科学审查](../../前置科学审查.md)和[正式门禁](../../preflight_qa.json)。

均值组单key的softmax恒为1，双线性矩阵任务梯度为0；两组存储参数相同不意味着有效容量相同。附加角色位与非线性编码也意味着均值未必完全丢掉角色信息。实验操纵因素是上述整体结构差异，不能缩写成“只开启注意力”。

## 3. 自然通信表现与形成过程

J使用逐token贪心发信，再由接收者两分支贪心行动，只有两种资源均正确才成功。Q对全部49条消息及两个正确行动概率求和，是随机政策成功率，两者不能混用。每来源内先平均分区、双向通信与等权遮挡，再平均四来源。

| 指标 | 均值组 | 注意力组 | 注意力−均值，百分点 |
|---|---:|---:|---:|
{main_rows}

![四来源配对终点](figures/01_source_comparison.png)

| 来源 | 均值组new12 J | 注意力组new12 J | 终点差，百分点 | 归一AUC差，百分点 |
|---|---:|---:|---:|---:|
{source_rows}

只有四个继承来源，三个分区、两方向和许多世界不能当成更多独立重复。曲线在0/100/600/1200/2100/2400固定测量；AUC按这些点作梯形积分再除以2400，是所采样曲线的摘要。

![学习曲线](figures/02_learning_curves.png)

| new12遮挡条件 | 均值组J | 注意力组J |
|---|---:|---:|
{mask_rows}

v23历史all_old条件的new12 J为15.896%。新组与该结果相比，同时改变了私人头读出、发送编码与生成架构，故只作[历史参照](../../../redesign_v0.23/results/experience_001/私人状态经验与共同符号表达迁移研究报告.md)，不把跨轮差值归于某个单一能力。

## 4. 消息是否起作用，以及片段能否重组

| new12条件 | 自然消息J | 固定空码J | 完整消息错配期望J |
|---|---:|---:|---:|
{controls}

空码固定为整数[0,0]；错配从同一有限完整测试支持中的自然贪心消息独立取样，按完整码频率计算精确期望。它们检验当前协议对消息内容的依赖，不是另训练的无通信组。

符号重组只取社会old18的供体消息。针对同照片对、同遮挡的目标，一条供体匹配食物且水不同，另一条匹配水且食物不同。枚举供体配对，从两条消息各取指定槽位的符号，再查询固定接收器。两个槽位到资源的分配方向均报告，不按new12成绩择优。每个目标先均匀平均供体对，再目标等权。

200个固定全49码双射作为重编码参照：先重命名完整供体码，拆位拼接后再逆映射接收器。完整自然消息的行为保持不变，字符结构被改变。先逐双射进行来源层级汇总，再取分位数，避免平均分位数。

| 组别 | 供体槽位分配 | new12重组J | 重编码参照均值 | 参照2.5%–97.5%范围 |
|---|---|---:|---:|---:|
{structure_rows}

参照范围是200次人为整码重命名的分布，不是总体置信区间。人工重组的成功说明这些固定协议的符号槽具有一定可复用结构，是否成立须结合实际表值；它不证明自然发信就按相同规则生成，更不等于获得语法。49条完整码足以记住30种布局。详见[重组方法](../../符号重组评估方法.md)、[完整重组结果](结构重组评估.md)和[原始200参照](structural_assay.json)。

## 5. 一份固定协议的具体消息

以下固定源33101、分区1、方向0、首个照片对、末帧仅食物、前六地图ID；选择规则在查看正式效应前写入脚本，不挑最高分协议。地点为1–6，两个数字依次是食物、水。字符只是整数0–6的显示映射，不预设自然语言意义。

| 组别 | 地图ID | 社会经验 | 目标地点 | 消息 | 接收地点 | 联合正确 |
|---|---:|---|---|---|---|---|
{erows}

该小表只帮助查看具体约定，不能代替四来源完整统计。[固定选择规则与原始例子](protocol_examples.json)

## 6. 解释、限制和下一步

{interpretation}

该环境仍是资源定位与分工的抽象，未包含真实生产、资源消耗、生存风险或自由形成的社会角色。两种需求与六地点行动空间由研究者提供，私人头又提供可读任务预测。本轮支持的范围是给定这些非语言与任务先验后的离散共同约定，不能据此断言分工是人类语言产生的必要条件或已经模拟语言从零诞生。

## 7. 执行与复核

正式训练耗时约{receipt['seconds']:.1f}秒，CPU单线程，未新增DINO推理或私人训练。所有24组按固定预算完成，无中途成绩筛选。源代码与输入、检查点、优化器、随机流摘要和完整评价表均保存。

[独立执行审计](audit_execution.json)通过{audit['checks']:,}项检查；[指标比较](comparison.json)另记录数值差异。审计覆盖全部288份评价表、48个终点发送方向各960世界、48张完整49码接收表、24私人头完整测试预测、全部57,600次外部世界流，以及固定源33101/p1两臂第1与2101次成对更新的trace、损失、裁剪、参数和Adam。训练概念缓存只重算原首尾块，测试缓存全量重算；没有从头重训全轨迹，早期冻结时间表征原子运算复用原有冻结模块，并非重新独立实现。

[重组开发核查](../smoke_001/structural_assay_dev_audit.json)用朴素目标优先循环核对7,680个目标/分配/重编码结果，并用完美因子码等六个合成测试检查方法边界；正式又在固定源33101/p1/d0两臂重复有界朴素复核，共15,360项完全一致，见[正式重组核查](structural_assay_audit.json)；其覆盖与神经训练审计分开。预检构建曾在并行代码审查文件尚未落地时退出；等待文件后未改代码重试通过，正式训练此前未开始，见[启动记录](../../launch_history.json)。

主数字来自[独立分析](analysis.json)，训练凭证见[training_complete.json](training_complete.json)，复现入口见[v24索引](../../README.md)。这批工程核验保证保存结果可核查，不会增加独立来源数，也不证明论文新颖性或ICLR投稿条件已经满足。
'''
    (OUT/'私人预测动态读取与共同符号结构研究报告.md').write_text(report)
    print(json.dumps(dict(report=str(OUT/'私人预测动态读取与共同符号结构研究报告.md'),characters=len(report)),ensure_ascii=False))
if __name__=='__main__':main()
