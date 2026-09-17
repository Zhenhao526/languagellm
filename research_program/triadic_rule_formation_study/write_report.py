"""Write the narrative only after the independent audit and JSON summary pass."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
RUN = HERE / 'results/formation_001'
CONDITIONS = ('strict_PL_live', 'strict_PL_silent', 'reciprocal_PL_live', 'reciprocal_PL_silent')
LABELS = ('第三人等待／开放', '第三人等待／静默', '互选执行／开放', '互选执行／静默')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sp = RUN / 'summary_001/summary.json'
    ap = RUN / 'audit_execution_001/verification.json'
    s, a = read(sp), read(ap)
    assert s['audit_status'] == a['status'] == 'passed'
    assert read(sp.parent / 'receipt.json')['outputs_sha256'][str(sp)] == sha(sp)
    primary = s['primary']['statistics']
    assert primary == s['primary']['auxiliary_statistics']['native_Q']['centered_AUC']
    aux = s['primary']['auxiliary_statistics']
    rows = s['primary']['by_seed']
    runs = s['runs']
    e = {r['condition']: r['scoring']['native']['means'] for r in s['complete_endpoint_summaries']
         if r['partition'] == 'new_needs_and_layouts'}
    q = {(r['condition'], r['update']): r['scoring']['native']['means']['Q']
         for r in s['target_trajectory_summaries']}
    entropy = {(r['condition'], r['actor'], r['window']): r['means']['conditional_entropy_bits'][-1]
               for r in s['message_entropy_curves']}
    ari = {(r['condition'], r['actor'], r['window']): r['means']['mean_background_ARI'][-1]
           for r in s['message_transition_curves']}
    support = {}
    for c in CONDITIONS:
        support[c] = [sum(v > 0 for k, v in r['final']['new_needs_and_layouts']['native']['actual_pair_counts'].items()
                          if k != 'none') for r in runs if r['condition'] == c]
    assert support[CONDITIONS[0]] == support[CONDITIONS[1]] == [1] * 16
    assert min(support[CONDITIONS[2]]) == 2 and max(support[CONDITIONS[2]]) == 3
    constant_panels = [r['trajectory'][-1]['message_snapshot']['panels'][1] for r in runs
                       if r['condition'] == 'reciprocal_PL_silent']
    assert len(constant_panels) == 16
    assert all(p['actor'] == 0 and p['window'] == 1 and p['conditional_entropy_bits'] == 0
               and len(p['backgrounds']) == 36 and all(b['entropy_bits'] == 0 and b['constant_code']
               and b['observed_packet_count'] == 1 for b in p['backgrounds']) for p in constant_panels)
    def pp(value):
        return f'{100 * value:+.4f}'
    def stat(v):
        return f"{pp(v['mean'])} [{pp(v['ci95_lower'])}, {pp(v['ci95_upper'])}]"
    def table(headers, values):
        return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + '\n'.join(
            '| ' + ' | '.join(map(str, row)) + ' |' for row in values)
    endpoint_table = table(['条件', '需求响应Q（%）', 'Q_excess（百分点）', '完整成功（%）', '有效执行（%）', '平均收益'],
        [[label, f"{100*e[c]['Q']:.4f}", f"{100*e[c]['Q_excess']:.4f}", f"{100*e[c]['full_success_rate']:.4f}",
          f"{100*e[c]['physical_execution_rate']:.4f}", f"{e[c]['reward_mean']:.4f}"] for c, label in zip(CONDITIONS, LABELS)])
    trajectory_table = table(['更新数', *LABELS], [[t, *[f'{100*q[c,t]:.4f}' for c in CONDITIONS]] for t in (0, 100, 500, 1500, 3000, 6000)])
    auxiliary_table = table(['计分及响应量', '扣初始化的时间平均交互及近似95%区间（百分点）'],
        [[label, stat(aux[key]['centered_AUC'])] for key, label in (
            ('native_Q', '原生Q（唯一主量）'), ('common_reciprocal_Q', '共同互选Q（辅助）'),
            ('native_Q_excess', '原生Q_excess（辅助）'), ('common_reciprocal_Q_excess', '共同互选Q_excess（辅助）'))])
    panels = [(actor, window) for actor in range(3) for window in range(2)]
    entropy_table = table(['主体／窗口', *LABELS], [[f'{"ABC"[actor]}／W{window+1}',
        *[f'{entropy[c,actor,window]:.3f}' for c in CONDITIONS]] for actor, window in panels])
    ari_table = table(['主体／窗口', *LABELS], [[f'{"ABC"[actor]}／W{window+1}',
        *[f'{ari[c,actor,window]:.3f}' for c in CONDITIONS]] for actor, window in panels])
    values = [r['measures']['native_Q']['centered_AUC'] for r in rows]
    assert sum(v > 0 for v in values) == 16
    body = f'''# 执行规则影响需求响应，语言形成仍需内容证据

已完成16个配对初始化、64次训练及独立审计。**完整双留出的唯一主量为{pp(primary['mean'])}个百分点，社会层面近似95%区间为[{pp(primary['ci95_lower'])}, {pp(primary['ci95_upper'])}]，16个区组均为正。** 在本任务和固定学习预算下，“互选即可执行”相对“第三人必须等待”，使开放通信相对静默的需求响应优势在训练期间有更正的累计变化。这是规则与通信机会的交互，不是语言出现速度或语言水平。

完整双留出、原生结算下的末点执行对计数还提供了描述性补充：严格等待条件的32个政策都只实际使用一个固定执行对；互选开放条件的16个政策各使用2或3个执行对。这只描述实际执行，不等于所有提议或所有时点都固定。严格等待开放组的16组平均完整成功率高于互选开放组。合作成功、需求响应与消息稳定不能互相替代。

## 固定比较与主量

种子60101—60116，每个种子从相同初值独立训练四条件，固定6000次更新、batch256。每人只见自己的私人需求及公共布局；三人各有独立发送和行动网络，共9个小MLP。通信为两窗同步消息，每窗4个8进制符号；静默从训练开始阻断跨人投递，保留自身消息。对象／材料、长短、搭档和目的地保留，没有加工步骤、道路事件或新增Qwen调用。

Q考察只改变一人一项需求、正确执行搭档随之反转的原有全部配对边：两端实际执行搭档都正确才记1，不执行也计错。它不要求对象、长短和目的地全部正确。按每层的背景／边及九个主体×需求轴层等权，目标为完整双留出53,568世界。主体、世界与需求边都不是独立社会样本。

对每个初始化计算 `D(t)=(Q互选开放−Q互选静默)−(Q等待开放−Q等待静默)`，减去本组D(0)，在0／100／500／1500／3000／6000更新上按真实间距作梯形积分、除以6000。最后16组等权平均。主量是预算内累计变化，后两点合计占原梯形权重62.5%；不能定位约定首次形成时刻。[方案](plan.md)与[测量口径](measure_design.md)均在训练前固定。

## 行为结果：规则改变了学到的合作方式

以下均为完整双留出、16初始化等权均值；成功和执行按贪心消息及贪心动作结算。Q采用需求边权重，完整成功与有效执行采用全世界权重，不将两类数值当同一事件。

{endpoint_table}

末点互选条件的通信Q优势为{pp(e[CONDITIONS[2]]['Q']-e[CONDITIONS[3]]['Q'])}个百分点。严格等待两条件的Q均为0，并不表示没有通信作用：开放组完整成功平均高于静默组；通信可能改善固定执行对下的其他行动协调，尚不能确定是否涉及内容表达。完整双留出上，每个严格等待政策只有一个非零执行对计数，其余世界可能不执行；这里没有把不执行删掉，也不推断它在所有未测试世界中都固定搭档。

六个预定时点的原生Q（%）如下：

{trajectory_table}

互选开放组在已记录的500更新时点Q高于最终时点，整个过程不是单调上升；不能用早期最高点替代预定面积或最终表现。表中的群体平均也不表示每个社会都有同样的曲线。全部16个主读数范围为{pp(min(values))}至{pp(max(values))}个百分点；原始四格曲线、两条通信差及每组主量完整保留。

![完整双留出需求响应学习轨迹]({RUN}/figures_002/01_formation_trajectory.png)

## 共同结算和超额响应限制了什么解释

{auxiliary_table}

各区间均以16个配对初始化为单位；主量使用样本标准差与t15近似区间。辅助区间是未校正的探索性描述，没有以显著辅助项更换主量。Q的原生未扣初始面积均{pp(aux['native_Q']['raw_AUC']['mean'])}个百分点，初始化D均{pp(aux['native_Q']['baseline_DiD']['mean'])}个百分点；共同结算初始化D为0，源于配对初值及相同通信路由，不是学到的结果。

把同一批动作统一按互选规则结算后，主方向与量级保留，说明结果不只依赖评价时是否要求第三人等待。但训练规则还同时改变了可执行联合动作、部分奖励机会和优化过程，不能据此拆成纯粹的通信机制或证明分工是语言产生条件。

Q_excess是Q减去背景内全需求世界的解析打乱期望，保持实际执行搭档及不执行频数，没有另抽随机排列。其累计交互接近零、区间跨零；这既不是等价性结论，也不能把原Q的改善直接解释为更强需求语义。一般执行能力及伙伴使用频数的变化仍是重要替代解释。末点原生Q_excess交互均{pp(aux['native_Q_excess']['endpoint_DiD']['mean'])}个百分点，也未替换累计主量。

## 符号使用与稳定：保留六个面板

下表为末点四符号整包的条件熵，单位bits；先在36个公共背景内对全部需求世界计数，再背景等权，最后16初始化等权。两窗、三主体分别报告。

{entropy_table}

![六个主体／窗口的消息条件熵]({RUN}/figures_002/02_message_entropy.png)

下表为最后相邻时点3000→6000的背景内ARI均值，仍分别保留六个面板。

{ari_table}

熵描述用包的范围，ARI描述输入案例按整包分组的稳定性；后者对整包换名不敏感。静默条件也可稳定并使用多种自身符号，这属于内部离散计算。互选静默组A的W2末点条件熵为0，是给定背景下的常量码；常量分区即使ARI为1也不能当作丰富约定。不同主体的同一字符没有预设共同词义。完整六时点／五过渡、原包相等率、字符Hamming、全域与各背景ARI、常量标志及有效包数均保存在[完整JSON](results/formation_001/summary_001/summary.json)，没有合成总语言分数。

## 执行、验证与来源

训练前固定[机器计划](results/formation_001/plan.json)、65个源文件及[审计源码](audit_freeze_001.json)。生产13项组件测试、独立审计11项测试通过；增加测试或依赖更新后的重复不当作额外证据。正式主运行一次完成，用时{s['source_elapsed_seconds']:.4f}秒，384000次更新，98,304,000训练世界样本；384检查点、576唯一评价、66,686,976评价世界。最终双留出引用6000评价，64次引用不计新前向。训练与评价合计2,369,654,784模块样本。

[独立审计](results/formation_001/audit_execution_001/verification.json)一次完成，用时{a['elapsed_seconds']:.4f}秒，重算全部576评价、600,182,784模块样本，并核查全部384检查点、384000条日志、配对外部随机流和统计量。它没有重放训练梯度、优化器更新或训练中实际采样消息，不能称独立重训。最大数值误差见审计记录：`{json.dumps(a['max_errors'], ensure_ascii=False)}`。主运行和正式审计均无失败或重试。

[JSON汇总](results/formation_001/summary_001/receipt.json)与绘图不读取NPZ或模型。3项合成汇总测试通过；首次测试曾因macOS临时路径别名未统一而失败，仅修测试路径后保留失败记录，没有修改正式统计。两幅PNG另作实际目检，PDF与其同源但不声称单独栅格检查。

本轮补读了Wright等2026年[正式论文](https://academic.oup.com/jole/article/doi/10.1093/jole/lzag006/8733633)的出版社HTML设定、干预、结果和讨论。其通信成功与群体约定性区分、新成员／噪声／容量测试均是直接先行，不能列作本项目原创。PDF直接下载403，失败已记录，本轮0新增本地PDF。[阅读范围和创新性限制](literature/近期约定性研究与本轮定位.md)。

## 下一步与论文范围

本轮给出的是执行规则×通信机会对需求响应学习的受控证据。新初始化增加当前任务内的独立重复，但既有世界划分已多次用于开发，不能称全新确认基准。符号统计、更多种子和完整审计都不能单独建立ICLR级别创新性；词义适切、可复用表达及环境机制仍需证据。

结果读取前完成的[后续构念审查](next_construct_review.md)建议优先做固定搭档下的四选一内容检验：保持接收者观察不变，比较四种需求包能否分别引导对应的物资／属性与目的地行动。它比仅看跨期消息替换损伤更接近内容表达，但整包查表也可能通过，不是组合性证明。候选尚未正式冻结或执行，本轮没有追加模型调用。跨期发送—接收相配仍是另一后续候选，不能将不相配造成的损伤自动解释成共同词义。

本轮取得进展，持续研究目标保持进行中；语言诞生主张、中心机制和可投稿论文仍未完成。
'''
    out = HERE / '结果与下一步.md'
    with out.open('x') as f:
        f.write(body)
    receipt = dict(status='written_pending_independent_report_and_visual_review',
        inputs_sha256={str(p): sha(p) for p in (sp, ap, Path(__file__).resolve())},
        output_sha256=sha(out), neural_forwards=0, npz_reads=0)
    with (HERE / 'report_generation_001.json').open('x') as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    print(json.dumps(dict(report=str(out), sha256=sha(out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
