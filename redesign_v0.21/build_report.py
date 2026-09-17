"""Build the v21 report from the completed independent analysis and interpretation."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/communication_001'
def read(p):return json.loads(p.read_text())
def pc(v):return f'{100*v:.3f}%'
def pp(v):return f'{100*v:+.3f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    a=read(OUT/'analysis.json');receipt=read(OUT/'training_complete.json');cmp=read(OUT/'comparison.json')
    assert a['formal'] and receipt['social_runs']==24 and cmp['passed']
    modes={r['mode']:r for r in a['aggregate']};labels={'immediate':'完整终态可见','delayed':'局部终态＋历史'}
    final={k:v['curve'][-1]['scores'] for k,v in modes.items()}
    mainrows=[]
    for k,f in final.items():
        mainrows.append([labels[k],*[pc(f[g]['J']) for g in ('old','added','sealed')],pc((f['added']['J']+f['sealed']['J'])/2),pc(f['common30']['J']),pc(f['common30']['Q'])])
    result=table(['训练观察条件','old J','added J','sealed J','未训练12图J','共同30 J','共同30 Q'],mainrows)
    history=table(['条件','移动资源正确','静止资源正确','两次静止都正确','两次双目标都正确','历史配对消息不同'],[[labels[k],*[pc(f['common30'][m]) for m in ('moved','stationary','history_pair_J','history_pair_both_joint_J','history_message_separation')]] for k,f in final.items()])
    info=table(['条件','静止信息CMI（bit）','同终态路线码一致','路线贪心码熵（bit）','同码且两次双正确'],[[labels[k],f"{f['common30']['static_message_cmi_bits']:.3f}",pc(f['common30']['route_code_agreement']),f"{f['common30']['route_code_entropy_bits']:.3f}",pc(f['common30']['route_samecode_bothcorrect'])] for k,f in final.items()])
    intervention=table(['条件','自然J','全局置换J','局部条件置换J','固定00 J','局部帧且历史清零J','交叉观察J'],[[labels[k],*[pc(f['common30'][m]) for m in ('J','global_shuffle_J','conditional_shuffle_J','constant00_J')],pc(modes[k]['extra']['erase']['common30']['J']),pc(modes[k]['extra']['cross_view']['common30']['J'])] for k,f in final.items()])
    crossed=[]
    for k,f in final.items():
        crossed.append([labels[k],labels[k],*[pc(f['common30'][m]) for m in ('moved','stationary','J')]])
        other='delayed' if k=='immediate' else 'immediate';x=modes[k]['extra']['cross_view']['common30']
        crossed.append([labels[k],labels[other],*[pc(x[m]) for m in ('moved','stationary','J')]])
    cross_table=table(['协议训练条件','评价观察','移动资源正确','静止资源正确','双目标J'],crossed)
    differences=table(['来源','主要共同30 J差','共同30 J AUC差','静止历史配对差','同终态路线一致差'],[[r['seed'],pp(r['endpoint']['common30']['J']),pp(r['auc']['common30']['J']),pp(r['endpoint']['common30']['history_pair_J']),pp(r['endpoint']['common30']['route_code_agreement'])] for r in a['contrasts']])
    figures='\n\n'.join(f'![{p.stem}]({p.resolve()})' for p in sorted((OUT/'figures').glob('*.png')))
    interpretation=(ROOT/'结果解释.md').read_text()
    report=f'''# 资源可见性与共同通信形成研究报告

2026年9月16日，v0.21。24次新的双主体社会训练已按固定预算完成，来自四个既有私人准备来源。唯一主要比较为2400步共同30终态的自然同消息贪心双目标成功率：局部终态＋历史减完整终态可见，差 **{pp(a['primary_mean'])} 个百分点**。本轮新训练离散通信；私人行动头未转入社会网络，v0.20失败的能力差验收保持不变。

## 本轮实际比较什么

资源场景有六地点，食物与水各处一地。先显示完整初态，再将一个资源移到原空地。完整条件再次显示完整终态；局部条件只显示移动资源的新位置，另一资源要从前一帧恢复。两条件使用相同底层事件、照片、两个角色的初始通信权重、外生抽样数和更新预算；两者的第二帧及经冻结编码器得到的状态不同。

全部24个v0.20 full私人终点按事前指定使用，不按表现筛人；它们在两条件的共同30私人J约92%，但概率和误差模式不完全相同。原官方冻结DINOv2 ViT-L/14的60行缓存、资源投影、slot_phi和两步GRU全部冻结。两步后作原无仿射LayerNorm。新发送接口每人43,319参数，接收接口6,364参数；不用LLM，不新增主干推理。该比较识别观察条件经当前编码器影响新通信学习的总作用，不是纯记忆能力的因果分离。

发送者只接触自己的视觉历史状态，不知接收需求；接收者只收到一条两token消息，再以两需求分支各作一次地点行动，不见场景、移动事件、对方状态或另一行动的反馈。词表7、消息长2，49个完整码，无预定词义。需求分支、六地点槽及资源规则由实验者提供；同一完整码记住一整张地图仍是可行解释。

四来源×三分区×两观察条件，共24运行，每运行两名独立主体互换发送/接收。每方向每步256世界，每条消息对应两次独立抽样行动。R=.25×两单成功之和+.5×双成功；两角色常数基线.5，REINFORCE角色损失先各求批均值，再对个人两角色取均值。Adam .0007；发送/接收分别范数裁剪2；熵系数前2100步.02、后300步0。完整固定2400更新，没有根据开发或正式成绩增加预算。

## 世界支持和主要结果

训练表为old→old的72种合法移动事件与22×22旧训练照片对的笛卡尔积，按行均匀抽样。added/sealed终态不进入训练。评价初态old18，终态覆盖30；144独特事件按old终态重复3次、其他2次，形成360权重行，配4×4旧test照片为5760行。重复只实现终态均衡，不增加独立样本。共同30含60%的熟悉终态，不能称纯未见泛化指标。三个分区、两方向均在四来源内平均，外层n=4。

J是同一自然贪心消息使两个需求的贪心行动都正确；Q对49种完整消息及两独立行动的原生概率精确积分。发送贪心按自回归顺序逐token选择，没有换成49码联合argmax。两者均与人工最优选码或接收覆盖分开。

{result}

{differences}

差值单位为百分点，均为局部终态＋历史减完整终态可见。AUC为0/100/300/600/1200/1800/2100/2400八点在0–2400上的归一化梯形面积，不选择最佳检查点。

## 消息是否传递了历史有关信息

合法历史配对固定移动资源、新地点及两张照片，静止资源地点不同。局部条件的当前画面相同；完整条件的完整终态画面不同。两张照片中一张当前不可见，因此这是分析者固定照片的条件，不表示key全都可被主体当前看到。对不同静止位置的历史对，分别要求两次静止目标都正确，或两次双目标都正确；单纯消息不同不算成功。

{history}

条件互信息仅用贪心完整码，描述固定评价支持中静止位置与消息的关系。路线指标固定终态和照片，去除重复权重行，只比较不同合法初态/移动资源路线；先在每个终态/照片组内计算，再宏平均。高码一致可能来自错误常量，高消息信息量也不保证伙伴正确利用。

{info}

## 冻结协议的消息与观察干预

全局与局部条件置换都在同一发送方向和观察条件内置换完整双token，保留双token内部关系。报告贪心消息/贪心行动的解析置换期望，不把它混成原生随机Q。供体分布先在共同30支持定义，各子组用同一干预；局部置换保持上述key内消息边际，破坏该组内消息与真实历史的对应。固定00是预定常量，非最佳无通信码。

{intervention}

固定同一协议再切换评价观察，资源分项如下。交叉观察本身事前已执行；将其用于解释结果后的资源偏向是解释性分析，不另立主要终点。

{cross_table}

只有固定消息而不见场景的接收者，共同30最大J为1/30；发送者仍看局部末帧时的分析上界为1/5，这两个参照不同。added/sealed单个matching中局部末帧足以唯一推出静止地点，所以单组高分不能证明历史表达。erase对两种已训练协议都使用局部末帧并清零h0。因此局部训练协议的正常与erase比较保持末帧相同；完整训练协议的正常与erase比较同时改变了末帧和历史，不能单独归于历史清零。清零及交叉观察可能超出对应社会训练分布，下降不能直接当作纯记忆中介效应。

{figures}

{interpretation}

## 执行与复核

正式完成{receipt['pair_updates']:,}次双主体更新、{receipt['messages']:,}条消息和{receipt['actions']:,}次接收行动；原私人准备只读取，新私人训练和DINO推理均为0。训练程序约{receipt['seconds']:.2f}秒，Torch CPU单线程；分析、审计与报告时间另计。

[执行审计](audit_execution.json)核对来源、冻结参数、缓存世界与全部外生抽样日志；按原批大小抽样重放缓存和评价前向，首步及2101步独立重建损失、分角色梯度裁剪与Adam。没有重做全部训练，也没有声称每个训练梯度或所有评价前向均双实现重放。

[独立分析](analysis.json)从全部{a['protocol_files']:,}份原始协议文件重写计数、概率、条件置换、历史配对和路线公式，不调用生产metrics。[比较记录](comparison.json)通过{cmp['comparisons']:,}项数值比较，最大绝对差{cmp['max_absolute_error']:.3g}。这些核验保证报告与实际执行相符，不能替代新颖性或独立确认。数据仍来自旧图库和反复开发的地图框架。

[固定执行方案](../../固定执行方案.md) · [前置审查](../../前置审查.md) · [设计取舍](../../设计取舍.md) · [近邻全文定位](../../近邻与主要比较.md) · [复现索引](../../README.md)
'''
    (OUT/'资源可见性与共同通信形成研究报告.md').write_text(report)
    print('report built')
if __name__=='__main__':main()
