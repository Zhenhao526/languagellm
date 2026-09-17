"""Report the complete temporal batch without promoting an engineering gate."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/temporal_001'
def read(n):return json.loads((OUT/n).read_text())
def pc(x):return f'{100*x:.3f}%'
def pp(x):return f'{100*x:+.3f}'

def main():
    a=read('analysis.json');done=read('training_complete.json');audit=read('audit_execution.json');comp=read('analysis_comparison.json')
    assert a['formal'] and a['person_fits']==48 and audit['passed'] and comp['passed']
    groups={r['arm']:r for r in a['aggregate']};ep={k:r['curve'][-1]['scores'] for k,r in groups.items()}
    gate=a['capability_gate'];assert gate['eligible_for_formal_decision']
    label={'full':'完整跨步反传','detach':'边界截断梯度'}
    lines=['# 资源移动后的私人记忆与更新研究报告','',
        '2026年9月16日；v0.20，固定方案下的正式开发批 temporal_001。','',
        f'48个私人主体已完成固定预算训练，来自四个新来源。共同30种终态组合上，当前只看见移动资源时，完整跨步反传与截断梯度组的贪心双目标成功率分别为 **{pc(ep["full"]["delayed"]["common30"]["J"])}、{pc(ep["detach"]["delayed"]["common30"]["J"])}**，唯一主差为 **{pp(a["primary_mean"])} 个百分点**。只看old测试组的整批能力验收{ "通过" if gate["passed"] else "未通过"}；它与主比较分开，不能当作显著性或一般语言能力门槛。','',
        '## 两步任务改变了什么','',
        '以前的私人任务在决策时能看到食物和水的位置。本轮先显示完整场景，随后一个资源移到原空地；即时条件再显示完整终态，延迟条件只显示移动资源的新位置。主体需要更新移动资源的位置，并从前一次观察保持另一资源的位置。例子如下；示意图中的词语和问号只面向报告读者，不是模型输入。','',
        f'![两步资源任务]({ROOT}/task_figure/00_temporal_task.png)','',
        '两组接收同样的两帧、照片、需求和外生行动随机数。两次GRU均真实承接状态：完整组让最终行动回报经由h0反向影响第一次观察的编码；截断组仅切断这条反向路径，前向历史仍保留，共享参数也仍从第二次观察学习。因此，两组比较的是跨时间信用分配准备的总作用，截断组并非“没有记忆”。','',
        '沿用官方冻结DINOv2 ViT-L/14原60行特征，没有LLM或新的视觉主干推理。四个新来源32101–32104先各做两人的200×64资源后果准备，再冻结project。后续仅训练原CampAgent的memory/slot_phi和全新私人行动头，共155,598参数；末状态统一作eps=1e−5的无仿射LayerNorm，不拟合缩放、不声称精确固定L2。地点槽、需求分支、资源规则与合法行动由实验者给定；这仍是受控非语言学习任务。','',
        '每来源运行三分区、两人、两种准备，合计48次拟合。每次2400更新，每步256事件复制为即时与延迟各256个单目标行动；仅所抽需求的0/1后果进入REINFORCE，常数基线0.5，Adam学习率0.0007、共同梯度范数上限2，熵系数前2100步0.02、后300步0。没有CE、语言标签或新通信学习。四来源内先平均分区和两人，再在来源间等权平均。','',
        '## 排除只看当前画面的捷径','',
        '训练初态和终态均只来自old18，共72种合法移动事件；added6和sealed6不进入训练。评价初态仍只用old18，终态覆盖全部30图。144种独特事件按old终态重复3次、其他终态重复2次，共360权重行，使每终态恰好12行；配原test池4×4照片对，共5760评价行。重复只用于精确权重，不增加独立样本数。','',
        '| 评价支持 | 仅局部末帧的最优双目标J | 同末帧、不同静止地点的贪心历史配对上界 |','|---|---:|---:|',
        '| old18 | 33.333% | 0% |','| 共同30 | 20.000% | 0% |','| added6或sealed6单独 | 100.000% | 没有可比较的不同静止地点对 |','',
        '单一matching中，移动资源类别和新地点已能反推出另一资源的地点。因此本轮主指标采用共同30；不能单独以sealed高分证明历史能力，也不能把不同子组最优规则拼成一个不接收组别的共同规则。','',
        '更具体的历史配对检查固定移动资源类别、新地点和两张照片，选取静止资源地点不同的两条合法历史，两次静止资源贪心行动都正确才算成功。共同30每个当前画面有五种静止地点、各六条权重行，跨类无序历史对360个；old为108个。仅末帧的确定性策略会给相同答案，配对成功上界为0；该零界不适用于独立随机行动。','',
        '## 完整终点与过程结果','',
        'J要求同一个私人主体分别应对两需求均选对地点，Q是两需求按各自原生概率独立行动时的双成功期望；它们都不是通信成功率。','',
        '| 准备 | 观察条件 | old J | added J | sealed J | 共同30 J | 共同30 Q |','|---|---|---:|---:|---:|---:|---:|']
    for arm in ('full','detach'):
        for mode in ('immediate','delayed'):
            e=ep[arm][mode];lines.append(f'| {label[arm]} | {"即时完整" if mode=="immediate" else "延迟局部"} | {pc(e["old"]["J"])} | {pc(e["added"]["J"])} | {pc(e["sealed"]["J"])} | {pc(e["common30"]["J"])} | {pc(e["common30"]["Q"])} |')
    lines += ['', '| 准备 | 延迟：移动资源准确率 | 延迟：静止资源准确率 | 同末帧历史配对 | 历史清零后的J | 历史清零后的配对 |','|---|---:|---:|---:|---:|---:|']
    for arm in ('full','detach'):
        e=ep[arm]['delayed']['common30'];z=groups[arm]['erase_scores']['common30']
        lines.append(f'| {label[arm]} | {pc(e["moved"])} | {pc(e["stationary"])} | {pc(e["history_pair_J"])} | {pc(z["J"])} | {pc(z["history_pair_J"])} |')
    lines += ['', '| 来源 | 主差：共同30延迟J | old延迟J差 | old即时J差 | 共同30延迟J AUC差 |','|---|---:|---:|---:|---:|']
    for r in a['contrasts']:
        lines.append(f'| {r["seed"]} | {pp(r["primary_common30_delayed_J"])} | {pp(r["old_delayed_J"])} | {pp(r["old_immediate_J"])} | {pp(r["common30_delayed_J_AUC"])} |')
    lines += ['', '表中差值单位为百分点，方向均为完整减截断。AUC是0/100/300/600/1200/1800/2100/2400固定八点的0–2400归一化梯形面积；没有按最佳成绩选择检查点。全部单目标、原始计数、概率、熵、NLL和并列率保留于[结构化分析](analysis.json)。','',
        f'![私人时间学习曲线]({OUT}/figures/01_temporal_learning.png)','',
        f'![同末帧历史配对表现]({OUT}/figures/02_paired_history_capability.png)','',
        '## 事前能力验收','',
        '此处只使用old测试组、整个批次判断是否形成清楚的能力对照。共同30主比较、added/sealed成绩不进入推进条件；不选删个人或来源，也不追加训练。','',
        '| 事前条件 | 实际值 | 是否满足 |','|---|---|---|']
    v=gate['values'];c=gate['criteria']
    entries=[('两臂即时J均≥90%',f'{pc(v["full_old_immediate_J"])} / {pc(v["detach_old_immediate_J"])}','both_immediate_at_least_090'),
        ('即时J绝对差≤5个百分点',f'{100*v["old_immediate_absolute_difference"]:.3f}个百分点','immediate_difference_at_most_005'),
        ('完整组延迟J≥80%',pc(v['full_old_delayed_J']),'full_delayed_at_least_080'),
        ('完整减截断的延迟J差≥10个百分点',f'{pp(v["old_delayed_difference"])}个百分点','delayed_difference_at_least_010'),
        ('四来源old延迟差均为正',' / '.join(pp(x) for x in v['old_delayed_source_differences']),'all_source_old_delayed_differences_positive')]
    for title,value,key in entries:lines.append(f'| {title} | {value} | {"是" if c[key] else "否"} |')
    lines += ['', (ROOT/'结果解释.md').read_text().strip(), '',
        '## 执行、复核与证据边界','',
        f'正式私人训练完成{done["private_updates"]:,}次更新、{done["selected_goal_actions"]:,}次单目标行动；前置资源准备另有{done["initial_resource_preparation_updates"]:,}次更新、{done["initial_resource_preparation_actions"]:,}次行动。正式程序约{done["seconds"]:.2f}秒，CPU单线程，审计和分析时间另计。全部48拟合终态齐后统一分析，源码与成功开发预检匹配。','',
        '[独立执行审计](audit_execution.json)核查完整源文件绑定、全部训练世界/随机身份摘要、评价世界表、冻结参数和Adam步数；每个主体的首步及第2101步独立重建损失、梯度与更新，评价前向逐文件核查原始首256与末128行。它没有重做全部训练或前置资源准备，也没有声称全部评价前向均被双实现重放。','',
        f'[原始结果复算](independent_recount.json)复用冻结环境指标函数，另行遍历与聚合；[独立分析](analysis.json)另写事件枚举、概率和历史配对公式，从全部816份原始评价复算。两者[交叉比较](analysis_comparison.json)通过{comp["comparisons"]:,}项，最大差{comp["max_absolute_error"]:.3g}。这是独立公式与另一遍历的相互核对，不冒称三套独立指标实现。初步开发、固定方案、各检查点及失败记录均保留。','',
        '资源移动只涉及一次事件，没有长时间遮挡、连续多事件、库存规划或真正社会分工。本轮尚未训练新共同协议，也没有证明某种私人能力足以产生组合语言。历史清零是冻结模型的信息干预，可能偏离训练状态分布；正常表现、同末帧合法历史配对和清零结果须一起解释。','',
        '本轮四个新初始化来源仍使用旧图库及已反复开发的地图框架，不是独立视觉确认。Kottur等已有记忆/词表限制与组合通信研究，Resnick等已有容量/信道比较；梯度截断本身不是新算法。对应原文及本轮可识别量见[近邻与因果边界](../../近邻与因果边界.md)。目前仍缺超出近邻的中心贡献、独立确认和完整投稿论证，不能据48次训练或验收通过宣布达到ICLR论文水平。','',
        '[固定方案](../../固定执行方案.md) · [环境与能力审查](../../环境与能力验收_独立审查.md) · [开发预检](../../preflight_qa.json) · [训练输入](invocation.json) · [完成凭证](training_complete.json) · [复现索引](../../README.md)','']
    (OUT/'资源移动后的私人记忆与更新研究报告.md').write_text('\n'.join(lines));print('report built')

if __name__=='__main__':main()
