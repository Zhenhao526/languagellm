"""Build the v0.31 report from independently recomputed statistics."""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONDITIONS = ('fixed_partners', 'rotating_partners')
LABELS = {'fixed_partners': '固定伙伴', 'rotating_partners': '轮换伙伴'}


def read(path):
    return json.loads(Path(path).read_text())


def pct(value):
    return f'{100 * float(value):.3f}%'


def pp(value):
    return f'{100 * float(value):+.3f}'


def pair_mean(metric):
    return (metric['food_partner_pair_J'] + metric['water_partner_pair_J']) / 2


def link(path, label):
    return f'[{label}]({Path(path).resolve()})'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    out = args.out.resolve()
    analysis = read(out / 'analysis.json'); validation = read(out / 'raw_validation.json'); audit = read(out / 'audit_execution.json')
    complete = read(out / 'training_complete.json'); invocation = read(out / 'invocation.json'); preflight = read(ROOT / 'preflight_qa.json')
    assert analysis['formal'] and validation['passed'] and audit['passed'] and complete['status'] == 'complete'
    aggregate = analysis['aggregate']; primary = analysis['primary']
    fixed, rotating = (aggregate[c] for c in CONDITIONS)
    fixed_final = fixed['scores']; rotating_final = rotating['scores']
    fixed_target = fixed_final['target12']['pooled']; rotating_target = rotating_final['target12']['pooled']
    target_diff = rotating_target['J'] - fixed_target['J']
    pair_diff = pair_mean(rotating_target) - pair_mean(fixed_target)
    agreement_fixed = fixed['agreement'][-1]['agreement']; agreement_rotating = rotating['agreement'][-1]['agreement']
    agreement_diff = agreement_rotating['full_message_agreement'] - agreement_fixed['full_message_agreement']
    auc_diff = rotating['auc']['target12']['pooled']['J'] - fixed['auc']['target12']['pooled']['J']
    lines = [
        '# 固定伙伴与轮换伙伴中的共同符号形成', '',
        '2026-09-16 · v0.31 · 本地固定预算正式批次', '',
        '## 摘要', '',
        f'本轮把伙伴关系从评价对象推进为训练条件。在同一多伙伴资源任务、同一冻结视觉输入和同一消息容量下，四个主体分别接受固定伙伴或轮换伙伴训练。目标12终点双资源自然成功 J 从固定伙伴的 {pct(fixed_target["J"])} 提高到轮换伙伴的 {pct(rotating_target["J"])}，差值 {pp(target_diff)} 个百分点；四个初始化来源的终点差值全部为正。',
        f'更明显的变化出现在同私有类型主体的消息一致率：固定伙伴 {pct(agreement_fixed["full_message_agreement"])}，轮换伙伴 {pct(agreement_rotating["full_message_agreement"])}，差值 {pp(agreement_diff)} 个百分点。训练支持上的 J 分别为 {pct(fixed_final["train12"]["pooled"]["J"])} 与 {pct(rotating_final["train12"]["pooled"]["J"])}，说明轮换条件同时提高了跨主体协议复用和训练任务表现。目标12的两个伙伴同时正确率只有 {pct(pair_mean(fixed_target))} 与 {pct(pair_mean(rotating_target))}，仍远低于单边正确率。因此本轮支持“伙伴轮换促进共同符号复用”的有限结论，但不支持已经形成了稳定的多伙伴语法。',
        '', '## 要回答的问题', '',
        '本项目的核心问题是：哪些非语言能力已经足以支持共同符号的诞生，哪些能力会改变诞生过程和最终结构。v0.31只检验其中一个可控机制：当一个主体必须在多个潜在伙伴之间复用同一通信接口时，伙伴轮换是否会推动独立主体收敛到更相近的符号协议。',
        '“共同符号”在本轮有操作性定义：发送者在冻结视觉状态上产生离散双token消息，接收者从消息恢复食物和水位置；两个独立主体副本在相同私有视觉类型下产生完全相同的贪心双token消息，作为跨主体消息一致率。它不是自然语言，也不等于有意图的词义。',
        '', '## 实验条件', '',
        '| 项目 | 固定设置 |', '|---|---|',
        '| 主体 | 4个独立主体，私有视觉类型为 `[0,1,0,1]`；通信模块分别重新初始化；参数对象不共享 |',
        '| 感知 | 继承 v28 已准备的冻结视觉序列和私有定位能力；本轮不重新拟合私人接口、不做新的 DINO 前向 |',
        '| 通信 | 每次发送两个离散 token，词表为7，完整消息为49种；接收者输出食物和水位置 |',
        '| 目标 | 12条食物—水边的2-正则二部图；每个食物和水位置各有两个伙伴 |',
        '| 训练支持 | 与目标边不相交的12条2-正则训练边；两种伙伴条件使用完全相同的训练图 |',
        '| 固定伙伴 | 每步 `(0,1)` 与 `(2,3)`，双向通信 |',
        '| 轮换伙伴 | 偶数步同固定伙伴；奇数步 `(0,3)` 与 `(2,1)`，双向通信 |',
        '| 奖励 | `R=.25(c食物+c水)+.5*c食物*c水`，基线为0.5；Adam 0.0007；梯度裁剪2 |',
        '| 预算 | 每组2400个群体更新；每更新4个有向批次、960条消息和1920次资源动作 |',
        '| 矩阵 | 4个来源 × 3个坐标置换 × 2种条件 = 24组 |',
        '',
        '两种条件共享目标图、训练图、世界表、图片/遮罩采样和动作采样；fixture随机种子不含条件标识。因此正式比较只改变伙伴配对时序。固定和轮换条件都保存8个有序跨私有类型伙伴协议，以便分别观察主体之间的泛化。',
        '',
        f'![伙伴制度与目标图]({(out / "figures/01_partner_design.png").resolve()})', '',
        '## 预注册指标', '',
        '- 主指标是目标12上的自然贪心双资源成功 J，先在每个来源内平均面板、伙伴组合、方向和遮罩，再在四个来源之间取均值。',
        '- 伙伴对指标要求同一固定食物或水位置的两个目标边同时正确，分别计算食物伙伴对、水伙伴对及其均值。',
        '- 消息一致率比较同一私有类型的独立副本（主体0与2、主体1与3）在同一冻结视觉输入上的完整双token、token0和token1。',
        '- 训练12、目标12和留出18用于区分训练记忆、目标迁移和留出布局迁移。',
        '- FW/WF片段重组和199个完整49码双射只作为离线结构参照，不计作额外训练重复。',
        '', '## 主要结果', '',
        '| 来源 | 固定伙伴目标 J | 轮换伙伴目标 J | 轮换−固定（百分点） | AUC差（百分点） | 完整消息一致率：固定 | 完整消息一致率：轮换 |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in primary:
        lines.append(f'| {row["seed"]} | {pct(row["fixed"])} | {pct(row["rotating"])} | {pp(row["difference"])} | {pp(row["auc_difference"])} | {pct(row["full_message_agreement_fixed"])} | {pct(row["full_message_agreement_rotating"])} |')
    lines += [
        f'| 来源均值 | {pct(fixed_target["J"])} | {pct(rotating_target["J"])} | {pp(target_diff)} | {pp(auc_diff)} | {pct(agreement_fixed["full_message_agreement"])} | {pct(agreement_rotating["full_message_agreement"])} |', '',
        f'终点差值在4个来源中均为正（{", ".join(pp(row["difference"]) for row in primary)} 个百分点）；AUC差的来源值为 {", ".join(pp(row["auc_difference"]) for row in primary)} 个百分点，其中一个来源略低于0。这说明终点效应方向一致，但学习曲线的早期过程并非完全同质。',
        '', '### 学习曲线、伙伴对与迁移', '',
        '| 条件 | 训练12 J | 目标12 J | 目标食物单边 | 目标水单边 | 目标伙伴对均值 | 留出18 J |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for c in CONDITIONS:
        scores = aggregate[c]['scores']; t = scores['target12']['pooled']; train = scores['train12']['pooled']; held = scores['held18']['pooled']
        lines.append(f'| {LABELS[c]} | {pct(train["J"])} | {pct(t["J"])} | {pct(t["food"])} | {pct(t["water"])} | {pct(pair_mean(t))} | {pct(held["J"])} |')
    lines += ['', f'![正式结果与学习曲线]({(out / "figures/02_partner_outcomes.png").resolve()})', '',
        f'固定伙伴的目标12食物/水单边正确率为 {pct(fixed_target["food"])} / {pct(fixed_target["water"])}，轮换伙伴为 {pct(rotating_target["food"])} / {pct(rotating_target["water"])}。伙伴对均值只从 {pct(pair_mean(fixed_target))} 提高到 {pct(pair_mean(rotating_target))}，差值 {pp(pair_diff)} 个百分点，仍远低于单边正确率。这表示轮换主要带来了跨主体消息复用和单边策略的改善，多伙伴关系的同时区分仍未解决。',
        '', '### 8个有序伙伴协议', '',
        '| 有序协议 | 固定伙伴目标 J | 轮换伙伴目标 J |', '|---|---:|---:|']
    for key in sorted(aggregate['fixed_partners']['pair_endpoint_J']):
        lines.append(f'| {key.replace("_j", "→").replace("i", "")} | {pct(aggregate["fixed_partners"]["pair_endpoint_J"][key])} | {pct(aggregate["rotating_partners"]["pair_endpoint_J"][key])} |')
    fvals = list(aggregate['fixed_partners']['pair_endpoint_J'].values()); rvals = list(aggregate['rotating_partners']['pair_endpoint_J'].values())
    lines += [f'8个有序协议的均值从 {pct(sum(fvals) / len(fvals))} 提高到 {pct(sum(rvals) / len(rvals))}；固定伙伴的协议间范围为 {pct(min(fvals))}–{pct(max(fvals))}，轮换伙伴为 {pct(min(rvals))}–{pct(max(rvals))}。轮换条件下未训练伙伴也能使用相近协议，但目标图的两个伙伴同时正确仍很低。',
        '', '## 结构参照：片段重组与整码双射', '',
        '对每个目标边，在相同图片和遮罩下从训练边取两个同食物供体和两个同水供体，固定遍历四种供体组合。FW把食物供体的第一个token与水供体的第二个token拼接；WF反向。再对完整49码做199个固定双射并同步逆重编码接收表。', '',
        '| 条件 | 方向 | 原始重组 | 199双射参照均值 | 超出参照（百分点） | 上尾比例 |', '|---|---|---:|---:|---:|---:|']
    for c in CONDITIONS:
        nulls = aggregate[c]['recombination_null']['pooled']
        for orient in ('FW', 'WF'):
            z = nulls[f'recombine_{orient}_J']
            lines.append(f'| {LABELS[c]} | {orient} | {pct(z["observed"])} | {pct(z["null_mean"])} | {pp(z["excess"])} | {z["upper_tail_fraction"]:.3f} |')
    lines += [
        '', '两种条件的原始重组都高于整码双射参照，轮换伙伴的原始 FW/WF 分别为 '+pct(aggregate['rotating_partners']['recombination_null']['pooled']['recombine_FW_J']['observed'])+' / '+pct(aggregate['rotating_partners']['recombination_null']['pooled']['recombine_WF_J']['observed'])+'。这说明按资源真值挑选供体时可以提取局部对应，但不能证明主体在训练过程中自主形成了可组合的词法或语法；整码双射保持了整条消息行为，是描述性对照。',
        '', '## 审计与可复算性', '',
        f'正式批次包含 {complete["social_runs"]} 个社会训练、{complete["pair_updates"]:,} 个群体更新、{complete["messages"]:,} 条消息和 {complete["actions"]:,} 次动作；没有新的私人拟合和新的DINO前向。',
        f'独立 NumPy 分析从保存的原始 sender/receiver 表复算所有 J、Q、伙伴对和重组指标：{validation["checks"]:,} 项断言、{validation["scalar_comparisons"]:,} 个标量比较通过，最大绝对误差为 {validation["maximum_metric_absolute_difference"]}。执行审计通过 {audit["checks"]:,} 项检查，重放 {audit["coverage"]["endpoint_sender_worlds_replayed"]:,} 个终点 sender 世界和 {audit["coverage"]["endpoint_receiver_tables_replayed"]:,} 张49码 receiver 表，并抽查训练 fixture/trace。',
        f'正式运行由 {link(ROOT / "preflight_qa.json", "preflight_qa.json")} 同时绑定源文件与正式输入哈希；烟雾批包含 {preflight["smoke_protocol_tables"]} 张协议表，未按开发成绩筛选来源或条件。',
        '', '## 解释与边界', '',
        '本轮最强的证据是：在保持视觉输入、目标图、训练图、消息容量和预算不变时，伙伴轮换同时提高了目标 J 和同类型主体的消息一致率，而且终点差值在4个来源中同向。这与“伙伴轮换迫使协议对多个主体可复用”的机制解释一致。',
        '证据仍不足以称作语言诞生。目标12双资源 J 只有7.581%，伙伴对均值只有0.709%，说明主体仍大量使用单资源或不稳定映射。轮换条件提升了训练12 J 到98.785%，但目标12和留出18没有出现同等幅度的结构性迁移。',
        '四个主体只复用两种冻结私有编码，轮换是固定四主体中的配对操纵，不是自然人口、代际学习或文化演化。视觉前端和私人能力继承自v28；本轮检验的是通信接口在受控资源—位置任务中的形成，不是从无语言的感知系统重建人类生态。样本独立单位是4个初始化来源，面板、伙伴组合、方向、遮罩和图片是嵌套重复。',
        '', '## 下一步', '',
        '下一轮可沿着“从简单到复杂”的路线增加任务压力，但每次只添加一个明确操纵：先把同一资源的两个伙伴收益绑定到同一轮协作，再加入需要第三方转述的延迟任务，最后才引入新主体和代际替换。每一步都应保留多伙伴目标、固定/轮换伙伴对照、同信息量记忆对照和整码重命名对照。',
        '预先成功标准应同时要求：目标双资源 J 超过单资源策略基线；伙伴对显著高于单边成功率推导的偶然水平；消息一致率在新主体或新伙伴上保持；片段重组和整码双射出现可重复的行为差异。若条件差异在来源间改变方向，应报告异质性而不增加种子寻找有利结果。',
        '', '## 文件', '',
        '- '+link(ROOT / '固定执行方案.md', '固定执行方案')+'；'+link(ROOT / 'support_design.json', '实验设计')+'；'+link(ROOT / 'implementation_review.json', '实现审查')+'；'+link(ROOT / 'preflight_qa.json', '正式前置记录'),
        '- '+link(out / 'analysis.json', '独立统计')+'；'+link(out / 'raw_validation.json', '统计复核')+'；'+link(out / 'audit_execution.json', '执行审计')+'；'+link(out / 'training_complete.json', '训练完成与哈希'),
        '- '+link(out / 'figures/01_partner_design.png', '伙伴制度图')+'；'+link(out / 'figures/02_partner_outcomes.png', '结果图')+'；'+link(out / '实际消息示例.md', '实际消息样本'),
        '', '复现顺序：在项目根使用既有环境运行 `run_support.py --out` 指向一个不存在的目录；随后运行 `analyze_results.py`、`audit_results.py`、`plot_results.py`、`message_examples.py` 和 `build_report.py`，并将同一输出目录传给各脚本。正式输入路径以 invocation.json 为准。'
    ]
    path = out / '固定伙伴与轮换伙伴中的共同符号形成研究报告.md'
    path.write_text('\n'.join(lines) + '\n')
    print(path)


if __name__ == '__main__':
    main()
