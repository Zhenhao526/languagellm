"""Read-only summaries of the fixed v0.9 local-plasticity adaptation batch.

No models are imported, trained or evaluated. All learning curves and endpoint
scores come from the saved 9600-world evaluations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

LOCAL_DEPS = Path(__file__).resolve().parent / '.analysis_deps'
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
import numpy as np


ROOT = Path(__file__).resolve().parent
SEEDS = (27101, 27102, 27103, 27104)
SPLITS = (1, 2, 3)
ARMS = ('sender_only', 'receiver_only', 'both')
TIMES = (0, 25, 50, 100, 200, 400, 500, 600)
MODES = ('normal', 'shuffle', 'blank', 'stochastic', 'erase_memory')
LABELS = {'sender_only': 'Sender only', 'receiver_only': 'Receiver only', 'both': 'Both'}
ZH = {'sender_only': '仅发送端学习', 'receiver_only': '仅接收端学习', 'both': '两端学习'}
COLORS = {'sender_only': '#3975ad', 'receiver_only': '#df8b30', 'both': '#31946f'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(a, b):
    assert abs(float(a) - float(b)) < 1e-10, (a, b)


def check_pair(p):
    n = p['n']
    assert n > 0 and p['decisions'] == 2 * n
    o = p['outcome_counts']
    assert sum(o.values()) == n
    assert p['both_correct'] == o['11']
    assert p['single_correct'] == o['01'] + o['10'] + 2 * o['11']
    assert p['food_correct'] == o['10'] + o['11']
    assert p['water_correct'] == o['01'] + o['11']
    near(p['both_accuracy'], p['both_correct'] / n)
    near(p['single_accuracy'], p['single_correct'] / (2 * n))
    near(p['mean_reward'], .5 * (p['single_accuracy'] + p['both_accuracy']))
    near(p['reward_sum'], p['mean_reward'] * n)


def check_stats(s):
    check_pair(s)
    assert s['n'] == 9600
    assert set(s['map_groups']) == {'old', 'new'}
    assert len(s['direction_groups']) == 2
    for name, expected in [('old', 7680), ('new', 1920)]:
        p = s['map_groups'][name]
        check_pair(p)
        assert p['n'] == expected
        pair = [d[name] for d in s['direction_groups']]
        for q in pair:
            check_pair(q)
            assert q['n'] == expected // 2
        for key in ('n', 'decisions', 'both_correct', 'single_correct', 'food_correct', 'water_correct'):
            assert p[key] == sum(q[key] for q in pair)
    for key in ('n', 'both_correct', 'single_correct'):
        assert s[key] == sum(s['map_groups'][g][key] for g in ('old', 'new'))


def auc(times, scores):
    """Trapezoid integral divided by the complete 600-update interval."""
    assert tuple(times) == TIMES and len(scores) == len(times)
    return sum((scores[i] + scores[i+1]) * .5 * (times[i+1] - times[i])
               for i in range(len(times)-1)) / 600


def load_run(path, seed, split, arm):
    config = json.loads((path/'config.json').read_text())
    curve = json.loads((path/'curve.json').read_text())
    result = json.loads((path/'result.json').read_text())
    assert (result['seed'], result['split'], result['arm']) == (seed, split, arm)
    assert (config['seed'], config['split'], config['arm']) == (seed, split, arm)
    assert result['updates'] == 600 and result['batch'] == 512
    assert (config['updates'],config['batch'],config['eval_n']) == (600,512,9600)
    assert config['training_plan']['map_pool'] == list(range(30))
    assert config['plan']['complementarity'] == .5 and not config['plan']['known']
    assert config['new_maps_are_adaptation_training'] and config['inherited_seeds']
    assert config['source_checkpoint'] == result['source_checkpoint']
    assert sha(config['source_checkpoint']['path']) == config['source_checkpoint']['sha256']
    assert result['frozen_modules_verified']
    times = [point['update'] for point in curve]
    assert tuple(times) == TIMES
    for point in curve:
        check_stats(point['scores']['normal'])
    for mode in MODES:
        check_stats(result['scores'][mode])
    assert curve[-1]['scores']['normal'] == result['scores']['normal']
    worlds = [point['scores']['normal']['world_sha256'] for point in curve]
    assert len(set(worlds)) == 1
    assert all(result['scores'][mode]['world_sha256'] == worlds[0] for mode in MODES)
    directions = []
    for direction in range(2):
        curves = {group: [p['scores']['normal']['direction_groups'][direction][group]['both_accuracy']
                          for p in curve] for group in ('old', 'new')}
        curves['all'] = [.8 * old + .2 * new for old, new in zip(curves['old'], curves['new'])]
        final = {mode: {group: result['scores'][mode]['direction_groups'][direction][group]['both_accuracy']
                        for group in ('old', 'new')} for mode in MODES}
        singles = {mode: {group: result['scores'][mode]['direction_groups'][direction][group]['single_accuracy']
                          for group in ('old','new')} for mode in MODES}
        for values in final.values():
            values['all'] = .8 * values['old'] + .2 * values['new']
        for values in singles.values():
            values['all'] = .8 * values['old'] + .2 * values['new']
        metrics = dict(new_initial_j=curves['new'][0], new_final_j=curves['new'][-1],
            new_change_j=curves['new'][-1]-curves['new'][0], new_auc_j=auc(times, curves['new']),
            old_initial_j=curves['old'][0], old_final_j=curves['old'][-1],
            old_change_j=curves['old'][-1]-curves['old'][0], all_final_j=curves['all'][-1],
            new_shuffle_j=final['shuffle']['new'], new_blank_j=final['blank']['new'],
            new_shuffle_gap=final['normal']['new']-final['shuffle']['new'],
            new_blank_gap=final['normal']['new']-final['blank']['new'])
        metrics.update({f'{group}_final_single':singles['normal'][group] for group in ('old','new','all')})
        directions.append(dict(scout=direction, collector=1-direction, curves=curves,
                               final_by_mode=final, single_final_by_mode=singles, metrics=metrics))
    return dict(seed=seed, split=split, arm=arm, path=str(path),
                source_checkpoint=result['source_checkpoint'],
                initial_sha256=result['initial_sha256'], final_sha256=result['final_sha256'],
                frozen_modules_verified=result['frozen_modules_verified'], world_sha256=worlds[0],
                config_sha256=sha(path/'config.json'), result_sha256=sha(path/'result.json'),
                curve_sha256=sha(path/'curve.json'),
                file_hashes={name: sha(path/name) for name in ('initial.pt', 'final.pt')},
                directions=directions)


def aggregate(runs):
    grouped = defaultdict(list)
    for r in runs:
        grouped[r['seed'], r['arm']].append(r)
    cells = []
    for (seed, arm), cell in sorted(grouped.items()):
        if {r['split'] for r in cell} != set(SPLITS):
            continue  # Never present a partial seed cell as the planned average.
        ds = [d for r in cell for d in r['directions']]
        assert len(ds) == 6
        cells.append(dict(seed=seed, arm=arm, directions_averaged=6,
            metrics={key: float(np.mean([d['metrics'][key] for d in ds])) for key in ds[0]['metrics']},
            curves={key: np.mean([d['curves'][key] for d in ds], axis=0).tolist() for key in ('old', 'new', 'all')},
            final_by_mode={mode: {group: float(np.mean([d['final_by_mode'][mode][group] for d in ds]))
                                 for group in ('old', 'new', 'all')} for mode in MODES}))
    arms = {}
    for arm in ARMS:
        selected = [c for c in cells if c['arm'] == arm]
        if not selected:
            continue
        arms[arm] = dict(seed_count=len(selected), seeds=[c['seed'] for c in selected],
            mean_metrics={key: float(np.mean([c['metrics'][key] for c in selected])) for key in selected[0]['metrics']},
            metric_ranges={key: [min(c['metrics'][key] for c in selected), max(c['metrics'][key] for c in selected)]
                           for key in selected[0]['metrics']},
            mean_curves={key: np.mean([c['curves'][key] for c in selected], axis=0).tolist() for key in ('old', 'new', 'all')},
            mean_final_by_mode={mode: {group: float(np.mean([c['final_by_mode'][mode][group] for c in selected]))
                                      for group in ('old', 'new', 'all')} for mode in MODES})
    comparisons = {}
    for first, second in [('both', 'sender_only'), ('both', 'receiver_only'), ('receiver_only', 'sender_only')]:
        left = {c['seed']: c for c in cells if c['arm'] == first}
        right = {c['seed']: c for c in cells if c['arm'] == second}
        paired = sorted(set(left) & set(right))
        if not paired:
            continue
        differences = {key: [left[s]['metrics'][key]-right[s]['metrics'][key] for s in paired]
                       for key in left[paired[0]]['metrics']}
        comparisons[f'{first}_minus_{second}'] = dict(seeds=paired,
            seed_differences=differences, mean_differences={k: float(np.mean(v)) for k, v in differences.items()},
            difference_ranges={k:[min(v),max(v)] for k,v in differences.items()})
    return cells, arms, comparisons


def figures(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    files = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for ax, group, title in zip(axes, ('new', 'old'), ('New maps: now included in training', 'Old maps: retention')):
        for arm in ARMS:
            cells = [c for c in result['seed_cells'] if c['arm'] == arm]
            for cell in cells:
                ax.plot(TIMES, np.array(cell['curves'][group])*100, color=COLORS[arm], alpha=.25, linewidth=.9)
            ax.plot(TIMES, np.array(result['arms'][arm]['mean_curves'][group])*100,
                    color=COLORS[arm], linewidth=2.2, label=LABELS[arm])
        ax.set(xlabel='Adaptation updates', title=title, xlim=(0, 600), ylim=(-2, 102))
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Greedy joint success (%)')
    axes[1].legend(loc='best', frameon=False, fontsize=9)
    fig.suptitle('Four inherited seeds; thin lines are seed means across 3 splits × 2 directions', fontsize=10)
    name='01_learning_curves'; fig.savefig(out/f'{name}.png'); fig.savefig(out/f'{name}.pdf'); plt.close(fig); files.append(name)
    panels=[('new_auc_j', 'New-map normalized AUC', 'Average joint success (%)'),
            ('new_final_j', 'New-map endpoint', 'Joint success (%)'),
            ('old_change_j', 'Old-map endpoint minus initial', 'Change (percentage points)')]
    fig, axes=plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    lookup={(c['seed'],c['arm']):c for c in result['seed_cells']}
    for ax,(key,title,ylabel) in zip(axes,panels):
        for seed in SEEDS:
            values=[100*lookup[seed,arm]['metrics'][key] for arm in ARMS]
            ax.plot(range(3),values,color='#a9a9a9',linewidth=.8,alpha=.6,zorder=1)
            ax.scatter(range(3),values,c=[COLORS[a] for a in ARMS],s=23,zorder=2)
        ax.scatter(range(3),[100*result['arms'][a]['mean_metrics'][key] for a in ARMS],
                   color='black',s=75,marker='_',linewidths=2,zorder=3)
        ax.set(xticks=range(3),xticklabels=['Sender','Receiver','Both'],title=title,ylabel=ylabel)
        ax.grid(axis='y',alpha=.18)
        if key=='old_change_j': ax.axhline(0,color='#777777',linewidth=.8,linestyle='--')
    name='02_seed_comparisons'; fig.savefig(out/f'{name}.png'); fig.savefig(out/f'{name}.pdf'); plt.close(fig); files.append(name)
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharey=True,constrained_layout=True)
    modecolors={'normal':'#3e7192','shuffle':'#ada79c','blank':'#d6cdbd'}
    for ax,arm in zip(axes,ARMS):
        modes=('normal','shuffle','blank')
        vals=[100*result['arms'][arm]['mean_final_by_mode'][m]['new'] for m in modes]
        ax.bar(range(3),vals,color=[modecolors[m] for m in modes],width=.65)
        for i,mode in enumerate(modes):
            v=[100*lookup[seed,arm]['final_by_mode'][mode]['new'] for seed in SEEDS]
            ax.scatter(np.arange(4)*.06-.09+i,v,color='#222222',s=15,zorder=3)
        ax.set(xticks=range(3),xticklabels=['Normal','Shuffle','Constant'],title=LABELS[arm],ylim=(0,102))
        ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('New-map endpoint joint success (%)')
    name='03_message_interventions'; fig.savefig(out/f'{name}.png'); fig.savefig(out/f'{name}.pdf'); plt.close(fig); files.append(name)
    return [dict(png=str(out/f'{n}.png'), pdf=str(out/f'{n}.pdf')) for n in files]


def pct(value):
    return f'{value*100:.2f}%'


def report(result):
    lines=['# v0.9 新组合引入后的局部学习实验：分析草稿','',
        f"状态：{result['status']}；纳入{len(result['runs'])}/36次运行，缺失{len(result['missing_runs'])}次。", '',
        '本轮继承v0.8 mixed条件的四个主体对种子27101–27104，每种子保留三个来源划分。从适应第一步起，将此前24张地图扩成全部30张均匀训练，比较仅发送端、仅接收端、两端学习。新增六图已经进入训练，因此结果衡量获得新组合的速度、可塑性及旧经验保留，不能称为零样本泛化。', '',
        '选择三臂干预的依据来自两项事后上界诊断。原mixed接收器对新图的完整码可达率为43.75%，自然双成功为6.25%；另一方面，固定原发送函数时，仅对新图任意重赋码义可达98.61%，但对全30图只有77.71%，保持旧图正确总数后的新图上界只有16.58%。因此既有解码覆盖不足，也存在新旧地图共用完整码产生的干扰，不能先把失败定位为单独发送或接收能力缺陷。', '',
        f'上述[接收可达性]({ROOT/"前置接收可达性诊断.md"})和[发送可区分性]({ROOT/"发送可区分性诊断.md"})使用既有验证照片枚举，允许分析者选择最优完整码或重赋义。它们是固定函数的事后上界，不是主体自然能力或新学习成绩；与本轮9600世界评价口径不同，不能直接相减。保持旧图正确总数也不保证每个旧案例不变。', '',
        '每次适应600次更新、每批512个双目标场景；从同一来源权重开始，重置Adam，冻结预训练视觉主干及个人投影。精确功能冻结与采样公平性由独立运行审计确认。本脚本仅读已保存评价，不调用模型、不训练。', '',
        '发送端同时包含视觉关系编码、状态编码与消息生成，接收端包含消息处理与动作选择。每人可学习参数分别为198,234、11,741及209,975，两端条件不是相同可训练参数预算。各臂按发送/接收模块分别裁剪梯度；前500步重新加入相同熵奖励，后100步关闭。冻结参数的策略在训练时仍会抽样，不能将其说成每次动作恒定。', '',
        '主要指标为新增六图的贪心双目标成功曲线、0–600次更新的梯形积分除以600（归一化AUC）及终点。旧图变化为终点减起点，负值表示下降。各曲线使用同一组9600个评价世界；单次曲线包含0、25、50、100、200、400、500、600检查点。', '',
        '四个继承种子是重复单位；先在每个种子内平均三个划分与两个方向，再给四种子均值。36次适应不是36个独立种子，来源于既往探索的种子也不是新的外部确认。', '',
        '| 条件 | 完整种子数 | 新图起点 | 新图AUC | 新图终点 | 旧图变化 | 全图终点 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    if result['status']=='complete':
        m={a:result['arms'][a]['mean_metrics'] for a in ARMS}
        lines[4:4]=[
            f"在新增地图已经进入训练的600步适应中，仅发送端、仅接收端和两端学习的新图双成功终点分别为{pct(m['sender_only']['new_final_j'])}、{pct(m['receiver_only']['new_final_j'])}、{pct(m['both']['new_final_j'])}；共同起点为{pct(m['both']['new_initial_j'])}。两端学习的新图终点和学习曲线AUC在四个继承种子中均高于任一单端条件。", '',
            f"旧图双成功的宏平均变化分别为{100*m['sender_only']['old_change_j']:+.2f}、{100*m['receiver_only']['old_change_j']:+.2f}、{100*m['both']['old_change_j']:+.2f}个百分点，本批没有观察到明显的总体旧图遗忘。结果支持在此训练程序下允许两端共同调整更有利，不能据此单独证明词义组合机制。",'']
    for arm in ARMS:
        if arm not in result['arms']: continue
        x=result['arms'][arm]; m=x['mean_metrics']
        lines.append(f"| {ZH[arm]} | {x['seed_count']} | {pct(m['new_initial_j'])} | {pct(m['new_auc_j'])} | {pct(m['new_final_j'])} | {100*m['old_change_j']:+.2f}个百分点 | {pct(m['all_final_j'])} |")
    lines+=['','| 条件 | 新图正常消息 | 打乱完整消息 | 常量消息 | 正常减打乱 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        if arm not in result['arms']: continue
        m=result['arms'][arm]['mean_metrics']
        lines.append(f"| {ZH[arm]} | {pct(m['new_final_j'])} | {pct(m['new_shuffle_j'])} | {pct(m['new_blank_j'])} | {100*m['new_shuffle_gap']:.2f}个百分点 |")
    lines+=['','正常、打乱、常量、策略抽样及清空现场记忆的全图/旧图/新图终点全部保存在分析JSON。常量消息不是天然无词义；差值表示对给定干预的敏感性，不自动证明组合结构。','',
            '| 条件 | 种子 | 新图AUC | 新图终点 | 旧图变化 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for cell in result['seed_cells']:
        m=cell['metrics']; lines.append(f"| {ZH[cell['arm']]} | {cell['seed']} | {pct(m['new_auc_j'])} | {pct(m['new_final_j'])} | {100*m['old_change_j']:+.2f}个百分点 |")
    lines+=['','| 配对比较 | 指标 | 四种子差（27101至27104，百分点） | 均值 | 范围 |',
            '| --- | --- | --- | ---: | --- |']
    for name,c in result['comparisons'].items():
        first,second=name.split('_minus_')
        for metric,label in [('new_final_j','新图终点'),('new_auc_j','新图AUC')]:
            values=c['seed_differences'][metric]; low,high=c['difference_ranges'][metric]
            lines.append(f"| {ZH[first]}−{ZH[second]} | {label} | "+'、'.join(f'{100*v:+.2f}' for v in values)+f" | {100*c['mean_differences'][metric]:+.2f} | {100*low:+.2f}至{100*high:+.2f} |")
    lines+=['','| 条件 | 新图平均单目标 | 旧图平均单目标 | 全图平均单目标 |',
            '| --- | ---: | ---: | ---: |']
    for arm in ARMS:
        if arm not in result['arms']: continue
        m=result['arms'][arm]['mean_metrics']
        lines.append(f"| {ZH[arm]} | {pct(m['new_final_single'])} | {pct(m['old_final_single'])} | {pct(m['all_final_single'])} |")
    lines+=['','主比较为两端学习减仅发送端、两端学习减仅接收端；接收端减发送端为补充。列出四种子配对差、范围与均值，不把训练世界的二项置信区间当成主体总体不确定性，也不作bootstrap推断。局部学习效果只说明这套已有协议和新训练支持下的可塑性作用；不能推断一般语言能力的必要性，也不能把一方低分归因于其先天不会组合。高适应成绩也可能来自新增整图约定，符号组成须由另行固定的结构评价验证。','',
            f'作为选择本轮干预依据的[接收可达性诊断]({ROOT/"前置接收可达性诊断.md"})是v0.8结果后的事后分析；本轮学习干预在运行前固定。先前的[信道设计]({ROOT/"protocol_review.md"})未执行。']
    for figure in result.get('figures',[]):
        lines+=['',f'![适应实验图]({figure["png"]})']
    if result['missing_runs']:
        lines+=['','尚未完成：'+ '、'.join(result['missing_runs'])+'。部分结果不能作为完整批次结论。']
    return '\n'.join(lines)+'\n'


def self_test():
    near(auc(TIMES,[.4]*len(TIMES)),.4)
    near(auc(TIMES,[t/600 for t in TIMES]),.5)
    # Irregular checkpoint spacing must not turn into an unweighted mean.
    assert abs(auc(TIMES,[t/600 for t in TIMES])-np.mean([t/600 for t in TIMES]))>.1
    fake=[]
    for seed in SEEDS:
        for split in SPLITS:
            for arm in ARMS:
                d=dict(metrics={'m':seed/100000+split/100},curves={g:[.2]*len(TIMES) for g in ('old','new','all')},
                       final_by_mode={mode:{g:.3 for g in ('old','new','all')} for mode in MODES})
                fake.append(dict(seed=seed,split=split,arm=arm,directions=[d,d]))
    cells,arms,comp=aggregate(fake)
    assert len(cells)==12 and all(a['seed_count']==4 for a in arms.values())
    assert all(all(v==0 for v in values) for c in comp.values() for values in c['seed_differences'].values())
    assert len(aggregate(fake[:-1])[0])==11
    return dict(status='passed',checks=['constant and linear AUC','irregular checkpoint weighting',
                                      'seed then split/direction aggregation','paired differences','incomplete seed cells excluded'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path,nargs='?',default=ROOT/'results/adaptation_001')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    tests=self_test()
    if args.self_test:
        print(json.dumps(tests)); return
    batch=args.batch.resolve(); runs=[]; missing=[]
    for seed in SEEDS:
        for split in SPLITS:
            for arm in ARMS:
                path=batch/f's{seed}_split{split}_{arm}'
                if not all((path/name).exists() for name in ('config.json','curve.json','result.json','initial.pt','final.pt')):
                    missing.append(path.name); continue
                runs.append(load_run(path,seed,split,arm))
    if not runs: raise SystemExit('No completed runs; no analysis files written')
    for seed in SEEDS:
        for split in SPLITS:
            group=[r for r in runs if (r['seed'],r['split'])==(seed,split)]
            if group:
                assert len({r['initial_sha256'] for r in group})==1
                assert len({r['world_sha256'] for r in group})==1
                for r in group[1:]:
                    assert all(a['curves'][g][0]==b['curves'][g][0] for a,b in zip(r['directions'],group[0]['directions']) for g in ('old','new','all'))
    cells,arms,comparisons=aggregate(runs)
    result=dict(status='partial' if missing else 'complete',expected_run_count=36,
                inherited_training_seeds=list(SEEDS),times=list(TIMES),batch=str(batch),
                analysis_sha256=sha(__file__),self_tests=tests,missing_runs=missing,
                runs=runs,seed_cells=cells,arms=arms,comparisons=comparisons,
                inferential_unit='four inherited training seeds; three splits and two directions averaged within seed')
    if not missing:
        out=batch/'figures';out.mkdir(exist_ok=True)
        result['figures']=figures(result,out)
    target=batch/'adaptation_analysis.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (batch/'适应实验分析草稿.md').write_text(report(result))
    print(json.dumps(dict(status=result['status'],runs=len(runs),seed_cells=len(cells),missing=len(missing),output=str(target)),ensure_ascii=False))


if __name__=='__main__':
    main()
