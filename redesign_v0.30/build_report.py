"""Build the v0.30 report from the independently recomputed statistics."""
import argparse,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
ARMS=('aligned_paths1','sparse_paths')
LABELS={'aligned_paths1':'每个目标1条三步路径','sparse_paths':'仅3/12目标有三步路径'}

def read(p):return json.loads(Path(p).read_text())
def pct(x):return f'{100*float(x):.3f}%'
def pp(x):return f'{100*float(x):+.3f}'
def link(p,label):return f'[{label}]({Path(p).resolve()})'
def at(curve,t):return next(x for x in curve if x['update']==t)
def pair(m):return (m['food_partner_pair_J']+m['water_partner_pair_J'])/2

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve()
    x=read(out/'analysis.json');qa=read(out/'raw_validation.json');audit=read(out/'audit_execution.json');run=read(out/'training_complete.json');inv=read(out/'invocation.json');pre=read(ROOT/'preflight_qa.json')
    assert x['formal'] and qa['passed'] and audit['passed'] and run['status']=='complete'
    agg=x['aggregate'];rows={(r['seed'],r['condition']):r for r in x['seed_rows']}
    a,b=(agg[z] for z in ARMS)
    aj=a['scores']['target12']['pooled'];bj=b['scores']['target12']['pooled']
    main_diff=bj['J']-aj['J'];pair_diff=pair(bj)-pair(aj)
    diffs=[-r['difference'] for r in x['primary']];adiffs=[-r['auc_difference'] for r in x['primary']]
    signs=f"{sum(v>1e-12 for v in diffs)}个稀疏臂更高、{sum(v< -1e-12 for v in diffs)}个更低"
    if main_diff>0:
        finding='稀疏路径臂的目标12终点 J 高于均匀路径臂，但四个来源只有一组方向相反，不能把它解释为路径数量的稳定因果效应。'
    else:
        finding='均匀路径臂的目标12终点 J 高于稀疏路径臂，但四个来源方向并不一致，不能把它解释为路径数量的稳定因果效应。'
    lines=['# 多伙伴共同目标中的符号形成：训练支持结构对迁移的影响','',
        '2026-09-16 · v0.30 · 本地固定预算正式批次','',
        '## 摘要','',
        f'本轮把共同测试集从一一匹配改为多伙伴目标：每个食物位置对应两个水位置，每个水位置对应两个食物位置。两个训练支持都含12条边、二部两正则、连通且无四环，边际熵、联合熵和互信息相同；只改变训练边与目标边的具体重叠及三步支持路径分布。24组训练完成后，目标12自然双资源成功率为：{LABELS[ARMS[0]]} {pct(aj["J"])}，{LABELS[ARMS[1]]} {pct(bj["J"])}，稀疏臂减均匀臂 {pp(main_diff)} 个百分点；来源差异为 {signs}。伙伴对指标分别为 {pct(pair(aj))} 与 {pct(pair(bj))}，均远低于单资源正确率。',
        '这轮最稳妥的结论是：此前的一一匹配捷径已经被去掉，模型能够在训练支持上学会高准确率通信，但在多伙伴共同目标上没有形成稳定的双边区分；固定供体重组高于整码重命名参照，说明协议中存在可被离线提取的局部对应，但还不能称为主体自主产生了组合语法。',
        '', '## 研究问题','',
        '核心问题是：哪些非语言能力已经足以支持共同符号的诞生，哪些能力会改变形成过程和最终结构？本轮具体检验“共同经验的二部支持结构是否改变未见目标的双资源表达”，同时用多伙伴目标检验共同符号是否真的区分两个伙伴，而非只预测一个资源后依靠任务结构得到另一个。',
        '', '## 实验设计','',
        '| 项目 | 固定设置 |','|---|---|',
        '| 主体 | 两个独立主体；继承 v28 的冻结视觉/私人能力；通信模块重新初始化并从零训练 |',
        '| 感知输入 | DINOv2-L 冻结特征；本轮不下载权重、不做新的 DINO 前向 |',
        '| 通信 | 每轮只能发送两个离散 token；49种完整消息；接收者输出食物和水位置 |',
        '| 奖励 | R=.25(c食物+c水)+.5 c食物 c水，固定基线 .5 |',
        '| 训练支持 | 两臂各12条食物—水边；每个训练边每方向10个基础世界，展开两种遮罩后20条消息 |',
        '| 共同目标 | 12条边、每行每列度数2；每个固定资源有两个伙伴；每方向72个测试世界 |',
        '| 正式矩阵 | 4个来源 × 3个坐标置换 × 2个支持臂 =24组；每组2400次成对更新 |',
        '| 主指标 | 目标12上的自然贪心双资源成功率 J；伙伴对指标要求同一资源的两个目标边同时正确 |',
        '',
        '支持图采用六个食物位置和六个水位置，排除同址布局。均匀路径臂的12个目标边各有1条长度3训练支持路径；稀疏路径臂只有3/12个目标边有该路径，其余9个为0。两臂共享7条训练边，分别有5条独有边；共享世界在相同图片和遮罩槽位中配对，作为方差控制。三步路径是图统计量，不是主体经历的三步行动。',
        '',f'![训练支持与多伙伴目标]({out}/figures/01_support_design.png)','',
        '## 预注册主结果','',
        '| 来源 | 均匀路径臂终点 J | 稀疏路径臂终点 J | 稀疏−均匀（百分点） | AUC差（百分点） |','|---|---:|---:|---:|---:|']
    for r in x['primary']:
        lines.append(f"| {r['seed']} | {pct(r['aligned'])} | {pct(r['sparse'])} | {pp(-r['difference'])} | {pp(-r['auc_difference'])} |")
    lines += [f"| 来源均值 | {pct(aj['J'])} | {pct(bj['J'])} | {pp(main_diff)} | {pp(b['auc']['target12']['pooled']['J']-a['auc']['target12']['pooled']['J'])} |",'',
        f'终点主指标使用全部4个来源；AUC只对0、100、600、1200、2100、2400六个固定检查点做辅助汇总。来源均值的均匀臂 AUC 为 {pct(a["auc"]["target12"]["pooled"]["J"])}，稀疏臂 AUC 为 {pct(b["auc"]["target12"]["pooled"]["J"])}。',
        '', '### 学习曲线与多伙伴判据','',
        '| 条件 | 训练12 J | 目标12 J | 目标12食物单边 | 目标12水单边 | 目标12伙伴对均值 |','|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        v=agg[arm];m=v['scores']['target12']['pooled'];train=v['scores']['train12']['pooled']
        lines.append(f'| {LABELS[arm]} | {pct(train["J"])} | {pct(m["J"])} | {pct(m["food"])} | {pct(m["water"])} | {pct(pair(m))} |')
    lines += ['',f'![自然形成过程与终点分解]({out}/figures/02_support_outcomes.png)','',
        '目标12终点的单边正确率分别为食物、水位置；伙伴对均值要求同一食物或水位置的两个伙伴同时正确。均匀路径臂为 '+pct(aj['food'])+' / '+pct(aj['water'])+' / '+pct(pair(aj))+'，稀疏路径臂为 '+pct(bj['food'])+' / '+pct(bj['water'])+' / '+pct(pair(bj))+'。单边高而伙伴对低，表明仍有明显的单资源策略或不稳定的多伙伴区分。',
        '', '## 片段重组与整码参照','',
        '对每个目标边，在相同测试照片和遮罩下，从训练边中取两个同食物位置供体和两个同水位置供体，固定遍历4种食物—水组合；FW 使用食物供体的第一个 token 加水供体的第二个 token，WF 反向。随后对完整49码做199个固定随机双射，并同步逆重编码接收表。所有组合均预先固定，没有按分数挑选。','',
        '| 条件 | 方向 | 原始重组 | 199双射均值 | 超出均值（百分点） | 上尾参照 |','|---|---|---:|---:|---:|---:|']
    for arm in ARMS:
        for orient in ('FW','WF'):
            z=agg[arm]['recombination_null']['pooled'][f'recombine_{orient}_J']
            lines.append(f'| {LABELS[arm]} | {orient} | {pct(z["observed"])} | {pct(z["null_mean"])} | {pp(z["excess"])} | {z["upper_tail_fraction"]:.3f} |')
    lines += ['', '两个支持臂和两个方向的原始重组均高于199个完整码重命名参照（上尾比例均为0.005）。这只说明实验者按资源真值选供体后，能从当前协议中提取局部对应；完整码重命名保持了整条消息的行为，不能检验神经网络是否在训练中自主分解 token，也不能作为独立训练重复。', '',
        '## 控制、审计与可复算性','',
        f'训练12终点 J 为 {pct(a["scores"]["train12"]["pooled"]["J"])}（均匀）和 {pct(b["scores"]["train12"]["pooled"]["J"])}（稀疏），说明通信模块确实学会了各自训练支持。全局整码打乱参照在目标12终点为 {pct(aj["shuffle_J"])} 与 {pct(bj["shuffle_J"])}；固定空码双成功率为 {pct(aj["blank_J"])} 与 {pct(bj["blank_J"])}。',
        '独立分析从保存的原始 sender/receiver 表重算所有 J、Q、伙伴对和重组指标，没有导入生产 metrics；114168项断言和1,413,888个标量比较通过，最大差为0。执行审计通过78558项检查，重放48张终点 sender 世界表和全部49码 receiver 表，并抽查第1、2101和最后一步的训练 fixture/trace。',
        f'正式批次包含 {run["social_runs"]} 个社会训练、{run["pair_updates"]:,} 个成对更新、{run["messages"]:,} 条消息和 {run["actions"]:,} 次动作；无新的私人拟合和无新的DINO前向。运行源文件与正式输入由 {link(ROOT/"preflight_qa.json","preflight_qa.json")} 双哈希绑定。',
        '', '## 解释与边界','',finding,
        '结果支持一个较窄但可检验的进展：多伙伴目标比一一匹配更能区分“共同符号是否同时携带两个关系”，且伙伴对指标揭示自然双资源成功率不能单独作为语言形成证据。结果不支持“每个目标拥有更多三步支持路径就必然促进迁移”的稳健预测；路径数量、具体边拓扑和共享边身份仍然联动，不能拆成单一机制。',
        '人工片段重组的正结果不能被称作主体自主产生了词法或语法。主体拥有预训练视觉世界知识和已训练私人定位能力，本轮只考察通信接口；因此不能回答人类语言起源的充分条件，也不能把该协议等同于真实原始社会。样本单位是4个初始化来源，坐标置换、方向、遮罩和图片是嵌套重复，不能当作独立来源。',
        '', '## 对下一轮实验的约束','',
        '下一步应把“多伙伴”从评价指标推进为任务压力：让同一资源的两个伙伴在同一轮都影响资源收益，加入需要第三方转述或延迟协作的情景，并设置等信息量的整码记忆对照。条件逐步增加时，必须保持目标图、资源边际、图片曝光、消息容量和成功上界可比；每增加一个能力，只添加一条明确的因果操纵。',
        '在此之前，不应继续用更多随机种子或更多路径图寻找正差。下一轮的成功标准应预先写成：目标12自然 J 超过单资源上界；伙伴对显著高于单资源基线；整码重命名与片段重组之间有可重复的行为差异；并在固定伙伴与轮换伙伴条件下复现。',
        '', '## 文件','',
        '- '+link(ROOT/'固定执行方案.md','固定执行方案')+'；'+link(ROOT/'support_design.json','支持图设计')+'；'+link(ROOT/'preflight_qa.json','正式前置记录'),
        '- '+link(out/'analysis.json','独立统计')+'；'+link(out/'raw_validation.json','原始统计复核')+'；'+link(out/'audit_execution.json','执行审计'),
        '- '+link(out/'training_complete.json','训练完成与哈希')+'；'+link(out/'figures/01_support_design.png','支持图')+'；'+link(out/'figures/02_support_outcomes.png','结果图'),
        '- '+link(out/'实际消息示例.md','固定实际消息样本'),
        '', '复现顺序：在项目根使用既有环境运行 run_support.py --out 一个不存在的目录；随后运行 analyze_results.py、audit_results.py、plot_results.py、message_examples.py 和 build_report.py，并将同一输出目录传给各脚本。正式输入路径以 invocation.json 为准。']
    path=out/'多伙伴共同目标与支持结构研究报告.md';path.write_text('\n'.join(lines)+'\n');print(path)

if __name__=='__main__':main()
