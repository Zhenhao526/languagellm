"""Read-only analysis of all 36 fixed historical-partner anchoring runs.

Only saved evaluations/logs are read. No model imports, training or inference.
Incomplete batches are inventoried by filename without reading result content.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'redesign_v0.10'))
import analyze_generalization as helper
import numpy as np

SEEDS = (29101, 29102, 29103, 29104)
PARTITIONS = (1, 2, 3)
ARMS = ('current', 'anchor', 'release')
VARIANTS = ('current', 'current_to_old', 'old_to_current')
GROUPS = helper.GROUPS
ALL_GROUPS = helper.ALL_GROUPS
MODES = helper.MODES
TIMES = (0, 25, 50, 100, 200, 300, 325, 350, 400, 500, 600)
POST_TIMES = tuple(t for t in TIMES if t >= 300)
PRIMARY = 'current_added_postrelease_auc_j'
SECONDARY = 'current_added_final_j'
ZH = dict(current='当前伙伴', anchor='持续锚定', release='锚定后解除')
LABELS = dict(current='Current partner', anchor='Anchor', release='Release after 300')
COLORS = dict(current='#3975ad', anchor='#c37d2b', release='#26836b')
CONNECTION_LABELS = dict(current='Current sender → current receiver',
    current_to_old='Current sender → old receiver', old_to_current='Old sender → current receiver')
REQUIRED = ('config.json', 'curve.json', 'result.json', 'training.jsonl', 'initial.pt', 'reference.pt',
    'final.pt', 'initial_optimizer.pt', 'final_optimizer.pt',
    *(f'checkpoint_{t:04d}.pt' for t in TIMES), *(f'optimizer_{t:04d}.pt' for t in TIMES),
    *(f'final_current_{m}.npz' for m in MODES),
    'final_current_to_old_normal.npz', 'final_old_to_current_normal.npz')
sha, near = helper.sha, helper.near


def read(path):
    return json.loads(Path(path).read_text())


def inventory(batch):
    entries, missing = [], []
    for seed, p, arm in itertools.product(SEEDS, PARTITIONS, ARMS):
        path = batch/f's{seed}_p{p}_{arm}'
        absent = [n for n in REQUIRED if not (path/n).is_file()]
        entries.append((path, seed, p, arm))
        if absent:
            missing.append(dict(run=path.name, missing_files=absent))
    return entries, missing


def auc_after_release(times, values):
    assert tuple(times) == TIMES and len(values) == len(TIMES)
    selected = [(t, v) for t, v in zip(times, values) if t >= 300]
    assert tuple(t for t, _ in selected) == POST_TIMES
    return sum((a+b)*.5*(t1-t0) for (t0, a), (t1, b) in zip(selected, selected[1:]))/300


def load_run(path, seed, p, arm):
    cfg, curve, result = [read(path/n) for n in ('config.json', 'curve.json', 'result.json')]
    for obj in (cfg, result):
        assert (obj['seed'], obj['partition'], obj['arm'], obj['updates']) == (seed, p, arm, 600)
        assert obj['sealed_never_trained']
    assert (cfg['contexts_per_update'], cfg['communications_per_update'], cfg['actions_per_update'],
            cfg['eval_n'], cfg['release_after'], cfg['entropy_off_after']) == (512, 768, 1536, 9600, 300, 500)
    assert tuple(cfg['checkpoints']) == TIMES
    assert cfg['optimizer'] == 'fresh Adam' and cfg['learning_rate'] == .0007 and cfg['entropy_coefficient'] == .02
    assert cfg['rng_namespace'] == 11011 and cfg['inherited_v10_sources'] and cfg['old_photos_development']
    assert cfg['role_loss_weights'] == dict(old_s_sender=.25, old_r_receiver=.25, new_sender=.25, new_receiver=.25)
    assert cfg['map_groups'] == helper.map_partition(p)
    plan = cfg['plan']
    assert plan['complementarity'] == .5 and plan['representation'] == 'identity'
    assert (plan['vocab'], plan['length'], plan['known'], plan['blocked']) == (7, 2, False, False)
    source = cfg['source_checkpoint']
    expected_source = ROOT.parent/f'redesign_v0.10/results/generalization_001/s{seed}_p{p}_base/final.pt'
    assert Path(source['path']).resolve() == expected_source.resolve()
    for key in ('source_checkpoint', 'source_config', 'prepared_source'):
        assert sha(cfg[key]['path']) == cfg[key]['sha256']
    source_result = read(expected_source.parent/'result.json')
    assert cfg['initial_sha256'] == cfg['reference_state_sha256'] == source_result['final_sha256']
    assert result['initial_sha256'] == result['reference_state_sha256'] == cfg['initial_sha256']
    assert result['frozen_modules_verified']
    assert tuple(point['update'] for point in curve) == TIMES
    worlds = set()
    for point in curve:
        assert point['reference_state_sha256'] == cfg['initial_sha256']
        assert set(point['scores']) == set(VARIANTS)
        for variant in VARIANTS:
            assert set(point['scores'][variant]) == {'normal'}
            stats = point['scores'][variant]['normal']
            helper.check_stats(stats)
            worlds.add(stats['world_sha256'])
    assert curve[0]['current_sha256'] == cfg['initial_sha256']
    assert curve[-1]['current_sha256'] == result['final_sha256']
    assert set(result['scores']) == set(VARIANTS)
    for variant in VARIANTS:
        modes = MODES if variant == 'current' else ('normal',)
        assert set(result['scores'][variant]) == set(modes)
        for mode in modes:
            helper.check_stats(result['scores'][variant][mode])
            worlds.add(result['scores'][variant][mode]['world_sha256'])
        assert result['scores'][variant]['normal'] == curve[-1]['scores'][variant]['normal']
    assert len(worlds) == 1
    assert all(curve[0]['scores'][v] == curve[0]['scores']['current'] for v in VARIANTS)
    digests = {q: hashlib.sha256() for q in ('old_s', 'old_r', 'new')}
    prefix = hashlib.sha256()
    with (path/'training.jsonl').open() as stream:
        for step, line in enumerate(stream, 1):
            row = json.loads(line)
            assert row['update'] == step
            assert row['anchored'] == (arm == 'anchor' or (arm == 'release' and step <= 300))
            assert row['entropy_weight'] == (.02 if step <= 500 else 0.)
            assert set(row['queries']) == {'old_s', 'old_r', 'new'}
            for tag, stats in row['queries'].items():
                helper.check_pair(stats)
                assert stats['n'] == 256
                supported = 'added' if tag == 'new' else 'old'
                for group in GROUPS:
                    assert stats['map_groups'][group]['n'] == (256 if group == supported else 0)
                digests[tag].update((stats['world_sha256']+'\n').encode())
            assert row['queries']['old_s']['world_sha256'] == row['queries']['old_r']['world_sha256']
            if step <= 300:
                prefix.update(line.encode())
    assert step == 600
    directions = []
    for d in range(2):
        curves, final, singles, rewards, metrics = {}, {}, {}, {}, {}
        for variant in VARIANTS:
            curves[variant] = {g: [x['scores'][variant]['normal']['direction_groups'][d][g]['both_accuracy']
                                  for x in curve] for g in GROUPS}
            curves[variant]['all'] = [helper.weighted(dict(zip(GROUPS, xs)))
                for xs in zip(*(curves[variant][g] for g in GROUPS))]
            modes = MODES if variant == 'current' else ('normal',)
            final[variant], singles[variant], rewards[variant] = {}, {}, {}
            for mode in modes:
                for target, key in ((final, 'both_accuracy'), (singles, 'single_accuracy'), (rewards, 'mean_reward')):
                    values = {g: result['scores'][variant][mode]['direction_groups'][d][g][key] for g in GROUPS}
                    values['all'] = helper.weighted(values)
                    target[variant][mode] = values
            for g in ALL_GROUPS:
                c = curves[variant][g]
                prefix_key = f'{variant}_{g}'
                metrics.update({f'{prefix_key}_initial_j': c[0], f'{prefix_key}_at_release_j': c[TIMES.index(300)],
                    f'{prefix_key}_final_j': c[-1], f'{prefix_key}_change_j': c[-1]-c[0],
                    f'{prefix_key}_postrelease_auc_j': auc_after_release(TIMES, c),
                    f'{prefix_key}_final_single': singles[variant]['normal'][g],
                    f'{prefix_key}_final_reward': rewards[variant]['normal'][g]})
        directions.append(dict(scout=d, collector=1-d, curves=curves, final_by_mode=final,
            final_single_by_mode=singles, final_reward_by_mode=rewards, metrics=metrics))
    return dict(seed=seed, partition=p, arm=arm, path=str(path), source_checkpoint=source,
        initial_sha256=cfg['initial_sha256'], reference_state_sha256=result['reference_state_sha256'],
        final_sha256=result['final_sha256'], world_sha256=worlds.pop(), source_hashes=cfg['source_hashes'],
        parameter_partition=cfg['parameter_partition'], directions=directions,
        training_world_sequence_sha256={q: h.hexdigest() for q, h in digests.items()},
        training_prefix_300_sha256=prefix.hexdigest(),
        pre_release_curve=[point for point in curve if point['update'] <= 300],
        file_hashes={n: sha(path/n) for n in REQUIRED})


def aggregate(runs):
    grouped = defaultdict(list)
    for r in runs:
        grouped[r['seed'], r['arm']].append(r)
    cells = []
    for (seed, arm), rs in sorted(grouped.items()):
        assert sorted(r['partition'] for r in rs) == list(PARTITIONS)
        ds = [d for r in rs for d in r['directions']]
        assert len(ds) == 6
        nested = {}
        for key in ('final_by_mode', 'final_single_by_mode', 'final_reward_by_mode'):
            nested[key] = {v: {m: {g: float(np.mean([d[key][v][m][g] for d in ds])) for g in ALL_GROUPS}
                              for m in ds[0][key][v]} for v in VARIANTS}
        cells.append(dict(seed=seed, arm=arm, partitions_averaged=3, directions_averaged=6,
            metrics={k: float(np.mean([d['metrics'][k] for d in ds])) for k in ds[0]['metrics']},
            curves={v: {g: np.mean([d['curves'][v][g] for d in ds], axis=0).tolist()
                        for g in ALL_GROUPS} for v in VARIANTS}, **nested))
    arms = {}
    for arm in ARMS:
        cs = [c for c in cells if c['arm'] == arm]
        assert [c['seed'] for c in cs] == list(SEEDS)
        nested = {f'mean_{key}': {v: {m: {g: float(np.mean([c[key][v][m][g] for c in cs])) for g in ALL_GROUPS}
                                    for m in cs[0][key][v]} for v in VARIANTS}
                  for key in ('final_by_mode', 'final_single_by_mode', 'final_reward_by_mode')}
        arms[arm] = dict(seed_count=4, seeds=list(SEEDS),
            mean_metrics={k: float(np.mean([c['metrics'][k] for c in cs])) for k in cs[0]['metrics']},
            metric_ranges={k: [min(c['metrics'][k] for c in cs), max(c['metrics'][k] for c in cs)]
                           for k in cs[0]['metrics']},
            mean_curves={v: {g: np.mean([c['curves'][v][g] for c in cs], axis=0).tolist()
                            for g in ALL_GROUPS} for v in VARIANTS}, **nested)
    lookup = {(c['seed'], c['arm']): c for c in cells}
    comparisons = {}
    for left, right in (('release', 'anchor'), ('current', 'anchor')):
        diff = {k: [lookup[s, left]['metrics'][k]-lookup[s, right]['metrics'][k] for s in SEEDS]
                for k in cells[0]['metrics']}
        comparisons[f'{left}_minus_{right}'] = dict(seeds=list(SEEDS), seed_differences=diff,
            mean_differences={k: float(np.mean(x)) for k, x in diff.items()},
            difference_ranges={k: [min(x), max(x)] for k, x in diff.items()})
    return cells, arms, comparisons


def figures(r, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    saved = []
    def save(fig, name):
        fig.savefig(out/f'{name}.png'); fig.savefig(out/f'{name}.pdf'); plt.close(fig)
        saved.append(dict(png=str(out/f'{name}.png'), pdf=str(out/f'{name}.pdf')))
    def curveplot(ax, variant, group):
        for arm in ARMS:
            for cell in [c for c in r['seed_cells'] if c['arm'] == arm]:
                ax.plot(TIMES, 100*np.array(cell['curves'][variant][group]), color=COLORS[arm], alpha=.22, linewidth=.8)
            ax.plot(TIMES, 100*np.array(r['arms'][arm]['mean_curves'][variant][group]), color=COLORS[arm],
                    linewidth=2.2, label=LABELS[arm])
        ax.axvline(300, color='#777777', linestyle='--', linewidth=1)
        ax.set(xlabel='Continuation updates', xlim=(0, 600), ylim=(-2, 102))
        ax.grid(axis='y', alpha=.18)
    lookup = {(c['seed'], c['arm']): c for c in r['seed_cells']}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    curveplot(axes[0], 'current', 'added')
    axes[0].set(title='Added maps: current partners', ylabel='Greedy joint success (%)')
    axes[0].legend(frameon=False, fontsize=9)
    for seed in SEEDS:
        vals = [100*lookup[seed, arm]['metrics'][PRIMARY] for arm in ('anchor', 'release')]
        axes[1].plot(range(2), vals, color='#aaaaaa', linewidth=.9)
        axes[1].scatter(range(2), vals, c=[COLORS[a] for a in ('anchor', 'release')], s=28, zorder=3)
    axes[1].scatter(range(2), [100*r['arms'][a]['mean_metrics'][PRIMARY] for a in ('anchor', 'release')],
        color='black', marker='_', linewidths=2.4, s=100, zorder=4)
    axes[1].set(xticks=range(2), xticklabels=['Anchor', 'Release'], ylim=(-2, 102),
                title='Primary: added-map AUC, updates 300–600', ylabel='Average joint success (%)')
    axes[1].grid(axis='y', alpha=.18)
    fig.suptitle('Four inherited seeds; seed means average 3 partitions × 2 directions. Dashed line: release.', fontsize=10)
    save(fig, '01_added_learning_and_release_auc')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, constrained_layout=True)
    for ax, variant in zip(axes, VARIANTS):
        curveplot(ax, variant, 'old')
        ax.set_title(CONNECTION_LABELS[variant], fontsize=10)
    axes[0].set_ylabel('Old-map greedy joint success (%)')
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle('Current-task retention and two directions of compatibility with the historical partner', fontsize=11)
    save(fig, '02_old_task_and_historical_compatibility')
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True, constrained_layout=True)
    for ax, group, title in zip(axes, GROUPS, ('Old 18 maps', 'Added 6 maps: trained', 'Sealed 6 maps: never trained')):
        ax.bar(range(3), [100*r['arms'][a]['mean_metrics'][f'current_{group}_final_j'] for a in ARMS],
               color=[COLORS[a] for a in ARMS], alpha=.85, width=.65)
        for i, arm in enumerate(ARMS):
            ax.scatter(i+np.array([-.12, -.04, .04, .12]),
                [100*lookup[s, arm]['metrics'][f'current_{group}_final_j'] for s in SEEDS], color='#222222', s=18, zorder=3)
        ax.axhline(100*r['arms']['current']['mean_metrics'][f'current_{group}_initial_j'],
                   color='#333333', linestyle='--', linewidth=1, label='Shared start')
        ax.set(xticks=range(3), xticklabels=['Current', 'Anchor', 'Release'], title=title, ylim=(0, 102))
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Current-partner endpoint joint success (%)')
    axes[2].legend(frameon=False, fontsize=9)
    fig.suptitle('Bars: four-seed means; dots: all seed means; dashed lines: common start', fontsize=10)
    save(fig, '03_current_endpoint_supports')
    return saved


def pct(x):
    return f'{100*x:.2f}%'


def report(r):
    main = r['comparisons']['release_minus_anchor']
    lo, hi = main['difference_ranges'][PRIMARY]
    lines = ['# v0.11 旧伙伴锚定与解除：功能分析草稿', '',
        f"36次继续学习全部完成。主比较中，解除相对持续锚定的新增图300–600步归一化AUC平均差为{100*main['mean_differences'][PRIMARY]:+.2f}个百分点，四个继承种子的差范围为{100*lo:+.3f}至{100*hi:+.3f}个百分点。该指标评价已加入训练的新用途学习，不是未训练组合泛化。", '',
        '三臂均能学习较多新增图，持续锚定与当前伙伴的终点差较小；解除没有带来大幅跃升。主指标中前三个种子为正，最后一个为约−0.003个百分点的微小负值，不能写成四种子全部提高。旧跨伙伴成绩整体仍高，两个跨伙伴方向上的宏平均变化也较小；这些结果不支持旧伙伴要求是本批主要学习瓶颈的强解释。', '',
        '本轮继承v0.10全部12个形成来源，四个种子29101–29104各三个平衡分区，不按成绩或上界筛选。每个来源生成当前伙伴、持续锚定、锚定后解除三臂，各600次更新；预训练视觉主干、私人投影和旧伙伴副本固定，三臂的可训练发送／接收模块与初始参数相同。独立重复仍为四个继承主体对，每种子先平均三个分区与两个方向，36次运行不能当作36种子。', '',
        '每次更新包含256个旧世界上下文和256个新增世界上下文；旧世界分别执行发送端查询与接收端查询，各只更新指定当前模块，新增世界更新两端。实际每步768次通信、1536次资源行动。三臂实际世界与角色查询预算相同；所有查询使用本次更新开始的权重，收齐损失后再更新。每人四类损失按0.25等权平均，重复旧查询没有额外加倍旧侧角色损失。', '',
        '持续锚定在旧世界让当前发送者与旧接收者配对、旧发送者与当前接收者配对。解除臂在第300次更新完成后改用当前伙伴，第一次解除后的更新为301。解除不重置Adam，不改变奖励、学习率、数据支持或预算；探索系数另在500步结束后关闭。锚定与解除的0–300步状态、日志及评价完全相同。', '',
        '唯一主指标是解除减持续锚定的新增图当前伙伴J-AUC，使用300、325、350、400、500、600六点梯形积分再除以300。预定次比较为当前伙伴减持续锚定的新增图600步终点。AUC是后半程平均成功水平，不是未加权检查点均值或单独的斜率。共同300步起点的变化差不另算独立发现。', '',
        '每个检查点分别评价当前伙伴Sθ→Rθ、当前发送者到旧接收者Sθ→R₀、旧发送者到当前接收者S₀→Rθ。三类连接均用同一组9600世界（old5760、added1920、sealed1920），同一来源三臂评价世界相同。两种旧跨伙伴连接各自报告，不能用平均值隐藏单端不兼容。', '',
        '| 条件 | 新增图300步 | 新增图后半程AUC | 新增图终点 | 当前伙伴旧图终点 | 未训练图终点 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        m = r['arms'][arm]['mean_metrics']
        lines.append(f"| {ZH[arm]} | {pct(m['current_added_at_release_j'])} | {pct(m[PRIMARY])} | {pct(m[SECONDARY])} | {pct(m['current_old_final_j'])} | {pct(m['current_sealed_final_j'])} |")
    lines += ['', '| 预定比较 | 指标 | 四种子差（29101至29104，百分点） | 均值 | 范围 |',
              '| --- | --- | --- | ---: | --- |']
    for name, key, label in [('release_minus_anchor', PRIMARY, '主：300–600新增AUC'),
                             ('current_minus_anchor', SECONDARY, '次：新增图终点')]:
        c = r['comparisons'][name]; left, right = name.split('_minus_')
        vals = '、'.join(f'{100*x:+.3f}' for x in c['seed_differences'][key]); lo, hi = c['difference_ranges'][key]
        lines.append(f"| {ZH[left]}−{ZH[right]} | {label} | {vals} | {100*c['mean_differences'][key]:+.2f} | {100*lo:+.3f}至{100*hi:+.3f} |")
    lines += ['', '| 条件 | 旧图连接 | 起点 | 300步 | 终点 | 终点减起点 |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    connections = dict(current='当前→当前', current_to_old='当前发送→旧接收', old_to_current='旧发送→当前接收')
    for arm in ARMS:
        m = r['arms'][arm]['mean_metrics']
        for v in VARIANTS:
            prefix = f'{v}_old'
            lines.append(f"| {ZH[arm]} | {connections[v]} | {pct(m[prefix+'_initial_j'])} | {pct(m[prefix+'_at_release_j'])} | {pct(m[prefix+'_final_j'])} | {100*m[prefix+'_change_j']:+.2f}个百分点 |")
    lines += ['', '| 条件 | 种子 | 新增后半程AUC | 新增终点 | 当前旧图终点 | 当前发送→旧接收旧图 | 旧发送→当前接收旧图 | sealed终点 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for c in r['seed_cells']:
        m = c['metrics']; keys = (PRIMARY, SECONDARY, 'current_old_final_j', 'current_to_old_old_final_j',
                                  'old_to_current_old_final_j', 'current_sealed_final_j')
        lines.append(f"| {ZH[c['arm']]} | {c['seed']} | "+' | '.join(pct(m[k]) for k in keys)+' |')
    lines += ['', '| 条件 | 当前伙伴全30图：正常 | 打乱完整消息 | 常量码 | 随机策略 | 清空发送场景表示 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        modes = r['arms'][arm]['mean_final_by_mode']['current']
        lines.append(f"| {ZH[arm]} | "+' | '.join(pct(modes[m]['all']) for m in MODES)+' |')
    lines += ['', '正常、打乱、常量、随机和清空发送场景表示的当前伙伴全图及三个支持集成绩、各自单目标准确率和收益全部保存在JSON。通信干预说明对相应信息路径的敏感性，不直接证明组合结构；常量码可能已有用途。跨旧伙伴条件只做正常贪心评价。', '',
        '如果锚定维持跨旧伙伴互通、限制新增学习，而且解除后新增学习恢复，同时当前伙伴旧任务保留，可以支持兼容性要求与当前任务保持是不同约束。若互通没有保持、旧任务整体崩溃、主差方向混合或解除未改善，应保留这些反证，不能仅凭一个较高终点定位机制。不事后选择“相近”门槛，也不要求当前、解除、锚定呈固定单调排名。', '',
        '锚定改变伙伴非平稳性、奖励分布和协同更新机会，主比较识别这些改变的总效应；不能单独把DP码冲突认定为因果中介。两端可变时码频数本身会改变，旧的静态oracle不约束整个训练过程。旧任务总成功保持不意味着每个旧动作或约定均保持。', '',
        '旧图与新增图均训练，sealed从未进入训练；若仅新增图改善而sealed低，只能称新用途适应。旧副本是实验者保存的历史行为，不是新人或代际传承。49码足够整图查表，本功能分析没有执行片段替换、拼接或重编码，不能据成功率推断一般语法。', '',
        '本批是四个继承来源的开发机制实验，照片和任务沿用既有环境。全部预算完成后才统一读取成绩，没有根据sealed结果调参；不作世界级二项推断或追加bootstrap。当前报告不宣称独立视觉确认、ICLR证据门槛已完成或人类语言起源得到充分解释。', '',
        f'测量定义和否证边界见[事前测量细则]({ROOT/"测量与解释边界.md"})。执行范围及来源核对记录在同目录分析JSON。']
    for fig in r['figures']:
        lines += ['', f'![旧伙伴锚定实验]({fig["png"]})']
    return '\n'.join(lines)+'\n'


def self_test():
    near(auc_after_release(TIMES, [.4]*len(TIMES)), .4)
    near(auc_after_release(TIMES, [(t-300)/300 for t in TIMES]), .5)
    values = [100 if t < 300 else .2 for t in TIMES]
    near(auc_after_release(TIMES, values), .2)
    assert abs(auc_after_release(TIMES, [(t-300)/300 for t in TIMES])-np.mean([(t-300)/300 for t in POST_TIMES])) > .1
    fake = []
    for seed, p, arm in itertools.product(SEEDS, PARTITIONS, ARMS):
        nested = {v: {m: {g: .3 for g in ALL_GROUPS} for m in (MODES if v == 'current' else ('normal',))} for v in VARIANTS}
        d = dict(metrics={'m': seed/100000+p/100}, curves={v: {g: [.2]*len(TIMES) for g in ALL_GROUPS} for v in VARIANTS},
                 final_by_mode=nested, final_single_by_mode=nested, final_reward_by_mode=nested)
        fake.append(dict(seed=seed, partition=p, arm=arm, directions=[d, d]))
    cells, arms, comp = aggregate(fake)
    assert len(cells) == 12 and all(a['seed_count'] == 4 for a in arms.values())
    assert all(x == 0 for c in comp.values() for xs in c['seed_differences'].values() for x in xs)
    try:
        aggregate(fake[:-1]); raise RuntimeError('Incomplete seed cell accepted')
    except AssertionError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        batch = Path(tmp)
        entries, missing = inventory(batch); assert len(entries) == len(missing) == 36
        for path, *_ in entries:
            path.mkdir()
            for name in REQUIRED:
                (path/name).write_text('Not valid JSON or evaluation data')
        assert not inventory(batch)[1]
        (entries[0][0]/'result.json').unlink(); assert len(inventory(batch)[1]) == 1
    return dict(status='passed', checks=['Constant/linear post-release AUC', 'Pre-release points excluded',
        'Irregular checkpoint weighting', 'Nested three-connection seed aggregation', 'Paired differences',
        'Incomplete cells rejected', 'Filename-only 36-run inventory gate'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', nargs='?', type=Path, default=ROOT/'results/anchoring_001')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(); tests = self_test()
    if args.self_test:
        print(json.dumps(tests)); return
    batch = args.batch.resolve(); entries, missing = inventory(batch)
    if missing:
        print(json.dumps(dict(status='pending', completed_file_sets=36-len(missing), expected=36,
            evaluation_content_read=False, analysis_files_written=False, missing_runs=missing), ensure_ascii=False)); return
    runs = [load_run(*entry) for entry in entries]
    for seed, p in itertools.product(SEEDS, PARTITIONS):
        cell = {r['arm']: r for r in runs if (r['seed'], r['partition']) == (seed, p)}
        assert set(cell) == set(ARMS)
        assert len({r['initial_sha256'] for r in cell.values()}) == 1
        assert len({r['world_sha256'] for r in cell.values()}) == 1
        assert all(r['pre_release_curve'][0] == cell['current']['pre_release_curve'][0] for r in cell.values())
        assert cell['anchor']['pre_release_curve'] == cell['release']['pre_release_curve']
        assert cell['anchor']['training_prefix_300_sha256'] == cell['release']['training_prefix_300_sha256']
        for step in (t for t in TIMES if t <= 300):
            for kind in ('checkpoint', 'optimizer'):
                name = f'{kind}_{step:04d}.pt'
                assert cell['anchor']['file_hashes'][name] == cell['release']['file_hashes'][name]
        for name in ('initial.pt', 'reference.pt', 'initial_optimizer.pt'):
            assert len({r['file_hashes'][name] for r in cell.values()}) == 1
        assert all(r['parameter_partition'] == cell['current']['parameter_partition'] for r in cell.values())
        for tag in ('old_s', 'old_r', 'new'):
            assert len({r['training_world_sequence_sha256'][tag] for r in cell.values()}) == 1
    source_hashes = runs[0]['source_hashes']
    assert all(r['source_hashes'] == source_hashes for r in runs)
    assert all(sha(path) == digest for path, digest in source_hashes.items())
    cells, arms, comparisons = aggregate(runs)
    result = dict(status='complete', expected_run_count=36, inherited_training_seeds=list(SEEDS),
        times=list(TIMES), postrelease_times=list(POST_TIMES), batch=str(batch),
        analysis_sha256=sha(__file__), analysis_helper_sha256=sha(helper.__file__), self_tests=tests,
        primary=dict(comparison='release_minus_anchor', metric=PRIMARY, interval=[300, 600], normalization=300),
        secondary=dict(comparison='current_minus_anchor', metric=SECONDARY),
        inferential_unit='Four inherited v0.10 source seeds; average three partitions and two directions within each seed',
        inventory_gate='All 36 required file sets existed before any evaluation contents were loaded',
        runs=runs, seed_cells=cells, arms=arms, comparisons=comparisons, source_hashes=source_hashes,
        audit=dict(run_count=36, source_checkpoints=12, inherited_seed_count=4, seed_cells=12,
            current_directions=72, total_training_updates=21600, total_training_contexts=11059200,
            total_training_communications=16588800, total_training_resource_actions=33177600,
            sealed_training_exposure=0, common_initial_state=True, common_evaluation_worlds=True,
            common_training_worlds=True, identical_anchor_release_through_300=True,
            identical_anchor_release_optimizers_through_300=True,
            parameter_partition_equal=True, source_hashes_verified=True))
    out = batch/'figures'; out.mkdir(exist_ok=True)
    result['figures'] = figures(result, out)
    target = batch/'anchoring_analysis.json'; target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (batch/'锚定实验分析草稿.md').write_text(report(result))
    c = comparisons['release_minus_anchor']
    print(json.dumps(dict(status='complete', output=str(target), audit=result['audit'],
        primary_mean_difference=c['mean_differences'][PRIMARY], primary_seed_differences=c['seed_differences'][PRIMARY]), ensure_ascii=False))


if __name__ == '__main__':
    main()
