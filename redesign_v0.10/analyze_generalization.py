"""Read-only analysis of the fixed v0.10 18-to-24-map generalization batch.

Before loading any evaluation content, require all 12 bases and 48 continuations.
No model imports, inference, training, checkpoint selection or protocol probes.
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
LOCAL_DEPS = ROOT.parent / 'redesign_v0.9/.analysis_deps'
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
import numpy as np

SEEDS = (29101, 29102, 29103, 29104)
PARTITIONS = (1, 2, 3)
ARMS = ('expand_both', 'stay_old_both', 'expand_sender', 'expand_receiver')
TIMES = (0, 25, 50, 100, 200, 400, 500, 600)
BASE_TIMES = (0, 100, 300, 600, 1200, 1800, 2100, 2400)
MODES = ('normal', 'shuffle', 'blank', 'stochastic', 'erase_memory')
GROUPS = ('old', 'added', 'sealed')
ALL_GROUPS = (*GROUPS, 'all')
WEIGHTS = dict(old=.6, added=.2, sealed=.2)
COUNTS = dict(old=5760, added=1920, sealed=1920)
LABELS = dict(expand_both='Expand: both', stay_old_both='Stay old: both',
              expand_sender='Expand: sender', expand_receiver='Expand: receiver')
ZH = dict(expand_both='扩展／两端学习', stay_old_both='只练旧图／两端学习',
          expand_sender='扩展／仅发送端', expand_receiver='扩展／仅接收端')
COLORS = dict(expand_both='#26836b', stay_old_both='#777777',
              expand_sender='#3975ad', expand_receiver='#ce812c')
COMPARISONS = (('expand_both', 'stay_old_both'), ('expand_both', 'expand_sender'),
               ('expand_both', 'expand_receiver'))
REQUIRED = ('config.json', 'curve.json', 'result.json', 'initial.pt', 'final.pt',
            'training.jsonl', *(f'final_{m}.npz' for m in MODES))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def near(a, b):
    assert abs(float(a)-float(b)) < 1e-10, (a, b)


def inventory(batch):
    """Stat filenames only. Incomplete formal results are never partially read."""
    entries, missing = [], []
    for seed, partition, arm in itertools.product(SEEDS, PARTITIONS, ('base', *ARMS)):
        path = batch/f's{seed}_p{partition}_{arm}'
        absent = [name for name in REQUIRED if not (path/name).is_file()]
        entries.append((path, seed, partition, arm))
        if absent:
            missing.append(dict(run=path.name, missing_files=absent))
    return entries, missing


def map_partition(p):
    maps = list(itertools.permutations(range(6), 2))
    matchings = (((0, 1), (2, 3), (4, 5)), ((0, 2), (1, 4), (3, 5)),
                 ((0, 3), (1, 5), (2, 4)))
    def ids(edges):
        pairs = {x for edge in edges for x in (edge, edge[::-1])}
        return sorted(i for i, pair in enumerate(maps) if pair in pairs)
    added = ids(matchings[p-1])
    sealed = ids(matchings[p % 3])
    return dict(old=sorted(set(range(30))-set(added)-set(sealed)), added=added, sealed=sealed)


def check_pair(p):
    n = p['n']
    assert isinstance(n, int) and n > 0 and p['decisions'] == 2*n
    o = p['outcome_counts']
    assert set(o) == {'00', '01', '10', '11'}
    assert all(isinstance(x, int) and x >= 0 for x in o.values()) and sum(o.values()) == n
    assert p['both_correct'] == o['11']
    assert p['single_correct'] == o['01']+o['10']+2*o['11']
    assert p['food_correct'] == o['10']+o['11']
    assert p['water_correct'] == o['01']+o['11']
    near(p['both_accuracy'], p['both_correct']/n)
    near(p['single_accuracy'], p['single_correct']/(2*n))
    near(p['mean_reward'], .5*(p['single_accuracy']+p['both_accuracy']))
    near(p['reward_sum'], p['mean_reward']*n)


def check_stats(s):
    check_pair(s)
    assert s['n'] == 9600 and s['sealed_maps_never_in_this_social_training']
    assert set(s['map_groups']) == set(GROUPS) and len(s['direction_groups']) == 2
    sumkeys = ('n', 'decisions', 'both_correct', 'single_correct', 'food_correct', 'water_correct')
    for group in GROUPS:
        p = s['map_groups'][group]
        check_pair(p)
        assert p['n'] == COUNTS[group]
        pair = [d[group] for d in s['direction_groups']]
        for q in pair:
            check_pair(q)
            assert q['n'] == COUNTS[group]//2
        for key in sumkeys:
            assert p[key] == sum(q[key] for q in pair)
    for key in sumkeys:
        assert s[key] == sum(s['map_groups'][g][key] for g in GROUPS)
    for key in ('single_accuracy', 'both_accuracy'):
        near(s[key], sum(WEIGHTS[g]*s['map_groups'][g][key] for g in GROUPS))


def auc(times, scores):
    """Trapezoid area divided by duration; retain irregular time spacing."""
    assert tuple(times) in (TIMES, BASE_TIMES) and len(scores) == len(times)
    return sum((a+b)*.5*(t1-t0) for a, b, t0, t1 in
               zip(scores[:-1], scores[1:], times[:-1], times[1:]))/(times[-1]-times[0])


def weighted(values):
    return sum(WEIGHTS[g]*values[g] for g in GROUPS)


def load_run(path, seed, partition, arm):
    cfg, curve, result = (read_json(path/name) for name in ('config.json', 'curve.json', 'result.json'))
    expected_times = BASE_TIMES if arm == 'base' else TIMES
    updates = expected_times[-1]
    for obj in (cfg, result):
        assert (obj['seed'], obj['partition'], obj['arm']) == (seed, partition, arm)
        assert (obj['updates'], obj['batch']) == (updates, 512)
        assert obj['sealed_never_trained']
    assert cfg['eval_n'] == 9600 and tuple(cfg['checkpoints']) == expected_times
    groups = map_partition(partition)
    assert cfg['map_groups'] == groups
    pool = groups['old'] if arm in ('base', 'stay_old_both') else sorted(groups['old']+groups['added'])
    assert cfg['training_pool'] == cfg['training_plan']['map_pool'] == pool
    assert cfg['training_plan']['allowed_sites'] == list(range(6))
    assert cfg['plan']['complementarity'] == .5 and not cfg['plan']['known'] and not cfg['plan']['blocked']
    assert cfg['plan']['representation'] == 'identity'
    assert (cfg['plan']['vocab'], cfg['plan']['length']) == (7, 2)
    assert cfg['learning_rate'] == .0007 and cfg['optimizer'] == 'fresh Adam'
    assert cfg['entropy_coefficient'] == .02 and cfg['entropy_off_after'] == (2100 if arm == 'base' else 500)
    assert cfg['rng_namespace'] == 10010 and cfg['training_purpose'] == (1 if arm == 'base' else 2)
    assert cfg['initial_sha256'] == result['initial_sha256']
    assert cfg['source_checkpoint'] == result['source_checkpoint']
    if arm == 'base':
        assert result['source_checkpoint'] is None
    else:
        source = result['source_checkpoint']
        assert Path(source['path']).resolve() == (path.parent/f's{seed}_p{partition}_base/final.pt').resolve()
        assert sha(source['path']) == source['sha256']
    assert sha(cfg['prepared_source']['path']) == cfg['prepared_source']['sha256']
    assert result['frozen_modules_verified']
    assert tuple(point['update'] for point in curve) == expected_times
    for point in curve:
        assert point['frozen_modules_verified']
        check_stats(point['scores']['normal'])
    assert set(result['scores']) == set(MODES)
    for mode in MODES:
        check_stats(result['scores'][mode])
    assert curve[-1]['scores']['normal'] == result['scores']['normal']
    worlds = [point['scores']['normal']['world_sha256'] for point in curve]
    assert len(set(worlds)) == 1
    assert all(result['scores'][mode]['world_sha256'] == worlds[0] for mode in MODES)
    exposure = dict.fromkeys(GROUPS, 0)
    train_worlds, train_rng = hashlib.sha256(), hashlib.sha256()
    with (path/'training.jsonl').open() as stream:
        for step, line in enumerate(stream, 1):
            row = json.loads(line)
            assert row['update'] == step and row['exposure']['sealed'] == 0
            assert sum(row['exposure'].values()) == 512
            assert all(row['exposure'][g] >= 0 for g in GROUPS)
            assert row['entropy_weight'] == (.02 if step <= cfg['entropy_off_after'] else 0.)
            if arm in ('base', 'stay_old_both'):
                assert row['exposure']['added'] == 0
            train_worlds.update((row['world_sha256']+'\n').encode())
            train_rng.update((str(row['rng_seed'])+'\n').encode())
            for g in GROUPS:
                exposure[g] += row['exposure'][g]
    assert step == updates and sum(exposure.values()) == updates*512
    directions = []
    for direction in range(2):
        curves = {g: [p['scores']['normal']['direction_groups'][direction][g]['both_accuracy']
                      for p in curve] for g in GROUPS}
        curves['all'] = [weighted(dict(zip(GROUPS, x))) for x in zip(*(curves[g] for g in GROUPS))]
        final = {m: {g: result['scores'][m]['direction_groups'][direction][g]['both_accuracy']
                     for g in GROUPS} for m in MODES}
        singles = {m: {g: result['scores'][m]['direction_groups'][direction][g]['single_accuracy']
                       for g in GROUPS} for m in MODES}
        for values in [*final.values(), *singles.values()]:
            values['all'] = weighted(values)
        metrics = {}
        for g in ALL_GROUPS:
            metrics.update({f'{g}_initial_j': curves[g][0], f'{g}_final_j': curves[g][-1],
                            f'{g}_change_j': curves[g][-1]-curves[g][0],
                            f'{g}_auc_j': auc(expected_times, curves[g]),
                            f'{g}_final_single': singles['normal'][g],
                            f'{g}_shuffle_j': final['shuffle'][g], f'{g}_blank_j': final['blank'][g],
                            f'{g}_shuffle_gap': final['normal'][g]-final['shuffle'][g],
                            f'{g}_blank_gap': final['normal'][g]-final['blank'][g]})
        directions.append(dict(scout=direction, collector=1-direction, curves=curves,
                               final_by_mode=final, single_final_by_mode=singles, metrics=metrics))
    return dict(seed=seed, partition=partition, arm=arm, path=str(path), times=list(expected_times),
                source_checkpoint=result['source_checkpoint'], initial_sha256=result['initial_sha256'],
                final_sha256=result['final_sha256'], world_sha256=worlds[0],
                evaluation_seed=cfg['evaluation_seed'], training_exposure=exposure,
                training_world_sequence_sha256=train_worlds.hexdigest(),
                training_rng_sequence_sha256=train_rng.hexdigest(),
                frozen_modules_verified=True, sealed_never_trained=True, source_hashes=cfg['source_hashes'],
                file_hashes={name: sha(path/name) for name in REQUIRED}, directions=directions,
                initial_stats=curve[0]['scores']['normal'], final_stats=result['scores']['normal'])


def aggregate(runs, arms=ARMS):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run['seed'], run['arm']].append(run)
    cells = []
    for (seed, arm), cell in sorted(grouped.items()):
        assert arm in arms
        assert sorted(r['partition'] for r in cell) == list(PARTITIONS)
        ds = [d for r in cell for d in r['directions']]
        assert len(ds) == 6
        cells.append(dict(seed=seed, arm=arm, partitions_averaged=3, directions_averaged=6,
            metrics={k: float(np.mean([d['metrics'][k] for d in ds])) for k in ds[0]['metrics']},
            curves={g: np.mean([d['curves'][g] for d in ds], axis=0).tolist() for g in ALL_GROUPS},
            final_by_mode={m: {g: float(np.mean([d['final_by_mode'][m][g] for d in ds]))
                               for g in ALL_GROUPS} for m in MODES}))
    summaries = {}
    for arm in arms:
        selected = [c for c in cells if c['arm'] == arm]
        assert [c['seed'] for c in selected] == list(SEEDS)
        summaries[arm] = dict(seed_count=4, seeds=list(SEEDS),
            mean_metrics={k: float(np.mean([c['metrics'][k] for c in selected])) for k in selected[0]['metrics']},
            metric_ranges={k: [min(c['metrics'][k] for c in selected), max(c['metrics'][k] for c in selected)]
                           for k in selected[0]['metrics']},
            mean_curves={g: np.mean([c['curves'][g] for c in selected], axis=0).tolist() for g in ALL_GROUPS},
            mean_final_by_mode={m: {g: float(np.mean([c['final_by_mode'][m][g] for c in selected]))
                                    for g in ALL_GROUPS} for m in MODES})
    comparisons = {}
    for first, second in COMPARISONS:
        if first not in arms or second not in arms:
            continue
        lookup = {(c['seed'], c['arm']): c for c in cells}
        differences = {k: [lookup[s, first]['metrics'][k]-lookup[s, second]['metrics'][k] for s in SEEDS]
                       for k in cells[0]['metrics']}
        comparisons[f'{first}_minus_{second}'] = dict(seeds=list(SEEDS), seed_differences=differences,
            mean_differences={k: float(np.mean(v)) for k, v in differences.items()},
            difference_ranges={k: [min(v), max(v)] for k, v in differences.items()})
    return cells, summaries, comparisons


def figures(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    files = []
    def save(fig, name):
        fig.savefig(out/f'{name}.png'); fig.savefig(out/f'{name}.pdf'); plt.close(fig)
        files.append(dict(png=str(out/f'{name}.png'), pdf=str(out/f'{name}.pdf')))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)
    for ax, group, title in zip(axes, ('added', 'sealed'),
                                ('Added maps: trained only by expansion arms', 'Sealed maps: never in social training')):
        for arm in ARMS:
            for cell in [c for c in result['seed_cells'] if c['arm'] == arm]:
                ax.plot(TIMES, 100*np.array(cell['curves'][group]), color=COLORS[arm], alpha=.23, linewidth=.8)
            ax.plot(TIMES, 100*np.array(result['arms'][arm]['mean_curves'][group]),
                    color=COLORS[arm], linewidth=2.2, label=LABELS[arm])
        ax.set(xlabel='Continuation updates', title=title, xlim=(0, 600), ylim=(-2, 102))
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Greedy joint success (%)')
    axes[1].legend(frameon=False, fontsize=9, loc='upper left')
    fig.suptitle('Fresh seeds n = 4; thin lines average 3 partitions × 2 directions within each seed', fontsize=10)
    save(fig, '01_added_sealed_learning')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    lookup = {(c['seed'], c['arm']): c for c in result['seed_cells']}
    for ax, key, title in zip(axes, ('sealed_final_j', 'sealed_auc_j'),
                              ('Primary: sealed-map endpoint', 'Secondary: sealed-map normalized AUC')):
        for seed in SEEDS:
            values = [100*lookup[seed, a]['metrics'][key] for a in ARMS]
            ax.plot(range(4), values, color='#aaaaaa', alpha=.65, linewidth=.8)
            ax.scatter(range(4), values, c=[COLORS[a] for a in ARMS], s=28, zorder=3)
        ax.scatter(range(4), [100*result['arms'][a]['mean_metrics'][key] for a in ARMS],
                   color='black', marker='_', s=110, linewidths=2.4, zorder=4)
        ax.set(xticks=range(4), xticklabels=['Expand\nboth', 'Stay old\nboth', 'Expand\nsender', 'Expand\nreceiver'],
               ylabel='Joint success (%)', title=title, ylim=(-.3, 10))
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('Each line is one fresh seed; black marks are four-seed means\n'
                 'Expanded 0–10% scale; Figures 1 and 3 show the full 0–100% scale', fontsize=10)
    save(fig, '02_sealed_seed_comparisons')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, constrained_layout=True)
    for ax, group, title in zip(axes, GROUPS, ('Old 18 maps', 'Added 6 maps', 'Sealed 6 maps')):
        vals = [100*result['arms'][a]['mean_metrics'][f'{group}_final_j'] for a in ARMS]
        ax.bar(range(4), vals, color=[COLORS[a] for a in ARMS], width=.66, alpha=.85)
        for i, arm in enumerate(ARMS):
            vals = [100*lookup[s, arm]['metrics'][f'{group}_final_j'] for s in SEEDS]
            ax.scatter(i+np.array([-.12, -.04, .04, .12]), vals, color='#222222', s=17, zorder=3)
        baseline = 100*result['arms']['expand_both']['mean_metrics'][f'{group}_initial_j']
        ax.axhline(baseline, color='#222222', linestyle='--', linewidth=1, label='Shared start')
        ax.set(xticks=range(4), xticklabels=['Expand\nboth', 'Stay old\nboth', 'Expand\nsender', 'Expand\nreceiver'],
               title=title, ylim=(0, 102))
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Endpoint greedy joint success (%)')
    axes[2].legend(frameon=False, fontsize=9)
    fig.suptitle('Bars: four-seed means; dots: seed means; dashed lines: common continuation start', fontsize=10)
    save(fig, '03_endpoint_supports')
    return files


def pct(x):
    return f'{100*x:.2f}%'


def report(r):
    main = r['comparisons']['expand_both_minus_stay_old_both']
    effect = main['mean_differences']['sealed_final_j']
    low, high = main['difference_ranges']['sealed_final_j']
    lines = ['# v0.10 永久保留组合实验：功能分析草稿', '',
        f"已完成12次旧图训练和48次继续学习，共60次训练。主要比较中，扩展且两端学习相对只练旧图且两端学习，永久保留六图的双目标成功终点平均相差{100*effect:+.2f}个百分点；四个新种子的配对差范围为{100*low:+.2f}至{100*high:+.2f}个百分点。完整个体差列于下表，不用评价世界数扩大独立重复数。", '',
        f"扩展且两端学习的新增图终点达到{pct(r['arms']['expand_both']['mean_metrics']['added_final_j'])}，永久保留图终点仍仅{pct(r['arms']['expand_both']['mean_metrics']['sealed_final_j'])}。本批显示明显的新经验适应与有限的未训练组合表现；主要比较的种子差并非全部为正，尚不支持获得新经验带来了稳定的组合迁移。", '',
        '本轮从四个新种子29101–29104建立独立主体对。每个种子包含三个平衡坐标划分，各自在18张旧地图上训练2400步，再从同一来源克隆四个继续学习条件，每个条件600步、每批512个场景。预训练视觉主干及个人投影固定；使用相同的地点槽输入、两个七选一符号和混合回报λ=0.5。每个场景只发一次消息，接收者各为食物、水执行一次独立查询，不读取另一查询结果。', '',
        '扩展条件均匀训练原18图与新增6图；只练旧图条件继续训练18图。另6张永久保留图在来源训练及继续学习阶段均未出现。新增图在扩展条件已被训练，新增图成绩是适应；永久保留图成绩才是本轮尚未训练组合的泛化。只练旧图条件的新增六图也未被训练。', '',
        '三个划分循环交换三个不交匹配中的新增／永久保留角色，各个地点在每一资源类别中的边际频数相同；三个划分是同构坐标配置。先在种子内平均三个划分与两个通信方向，再汇总四个新种子。60次训练、48次继续学习或24个方向都不等于独立种子数。', '',
        '主要结果为永久保留图600步终点的“扩展／两端学习−只练旧图／两端学习”。归一化AUC为0–600步曲线的梯形积分除以600，是次级过程指标；检查点为0、25、50、100、200、400、500、600。共同起点相等，因此两条件变化量之差与终点差相同，不作为另一独立发现。双方相对单端学习的差为次要机制比较。', '',
        '所有预算完成后才由本脚本统一读取永久保留图成绩，不依照该成绩挑选检查点、增加预算或改参数。每次运行固定使用9600个评价世界（旧图5760、新增1920、永久保留1920），同一种子与划分的全部条件共享评价世界。固定训练池及训练日志中永久保留图暴露为零的核查已通过。脚本仅读已有记录，不执行模型推理。', '',
        '| 条件 | 旧图终点 | 新增图起点 | 新增图终点 | 保留图起点 | 保留图终点 | 保留图AUC | 旧图变化 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        m = r['arms'][arm]['mean_metrics']
        lines.append(f"| {ZH[arm]} | {pct(m['old_final_j'])} | {pct(m['added_initial_j'])} | {pct(m['added_final_j'])} | {pct(m['sealed_initial_j'])} | {pct(m['sealed_final_j'])} | {pct(m['sealed_auc_j'])} | {100*m['old_change_j']:+.2f}个百分点 |")
    lines += ['', '| 配对比较 | 保留图指标 | 四种子差（29101至29104，百分点） | 平均差 | 范围 |',
              '| --- | --- | --- | ---: | --- |']
    for name, c in r['comparisons'].items():
        first, second = name.split('_minus_')
        for key, label in [('sealed_final_j', '终点'), ('sealed_auc_j', 'AUC')]:
            values = '、'.join(f'{100*v:+.2f}' for v in c['seed_differences'][key])
            lo, hi = c['difference_ranges'][key]
            lines.append(f"| {ZH[first]}−{ZH[second]} | {label} | {values} | {100*c['mean_differences'][key]:+.2f} | {100*lo:+.2f}至{100*hi:+.2f} |")
    lines += ['', '| 条件 | 种子 | 新增图终点 | 保留图终点 | 保留图AUC | 旧图变化 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for c in r['seed_cells']:
        m = c['metrics']
        lines.append(f"| {ZH[c['arm']]} | {c['seed']} | {pct(m['added_final_j'])} | {pct(m['sealed_final_j'])} | {pct(m['sealed_auc_j'])} | {100*m['old_change_j']:+.2f}个百分点 |")
    lines += ['', '| 条件 | 保留图正常消息 | 打乱完整消息 | 常量消息 | 策略抽样 | 清空记忆 | 全图正常终点 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        a = r['arms'][arm]
        vals = [pct(a['mean_final_by_mode'][m]['sealed']) for m in MODES]
        lines.append(f"| {ZH[arm]} | "+' | '.join(vals)+f" | {pct(a['mean_metrics']['all_final_j'])} |")
    lines += ['', '正常与打乱／常量消息的差表示对相应通信干预的敏感性。常量码可能已有词义，打乱也会偶然命中；这些分数不直接证明符号组合结构。单目标准确率及所有支持集的五种评价模式一并存入JSON。', '',
        '在相同交互预算下，扩展条件以新增经验替换25%的旧图经验。因此主要比较包含新经验内容与旧图暴露比例的改变，不能将其解释为纯粹的新颖性效应。三种扩展条件共享训练世界；只练旧图条件的地图支撑不同，其训练世界不能声称完全相同。两端学习与单端学习的可训练参数数目不同，模块效果限于本实现的局部可塑性。', '',
        '本轮能检验获得一组新经验是否帮助主体自然处理另一组未训练组合。它不直接检验完整语法、真实原始社会生存、语言从无到有的全部条件，也不证明一般语言能力的必要性。视觉照片与任务形式来自既往开发环境；这些组合对本轮新主体未经社会训练，但不属于研究者从未接触过的外部数据。', '',
        '未加入从头直接训练24图的条件，因此不主张本方案优于直接训练。若要检验训练次序，应另做相同输入多重集的顺序对照。本轮未新增片段替换或人工拼接实验；已固定的协议探针只区分自然使用、接收覆盖、连续成功概率及相应上界。不能仅凭成功率或覆盖率上升宣称协议更有组合性。', '',
        '分析不计算基于世界数的二项置信区间，也不对四个种子作bootstrap推断；保留全部种子的配对方向与范围。此前v0.9的新增图适应是本轮设计依据，不能与本轮永久保留图成绩直接当作同一批数据比较。']
    lines += ['', '图1与图3保留0–100%完整尺度；图2为便于查看四种子的低分差异，将纵轴展开至0–10%，不增加或改变任何统计指标。']
    for fig in r['figures']:
        lines += ['', f'![永久保留组合实验]({fig["png"]})']
    return '\n'.join(lines)+'\n'


def self_test():
    for times in (TIMES, BASE_TIMES):
        near(auc(times, [.4]*len(times)), .4)
        near(auc(times, [t/times[-1] for t in times]), .5)
    assert abs(auc(TIMES, [t/600 for t in TIMES])-np.mean([t/600 for t in TIMES])) > .1
    near(weighted(dict(old=1, added=0, sealed=0)), .6)
    seen_added, seen_sealed = [], []
    maps = np.array(list(itertools.permutations(range(6), 2)))
    for p in PARTITIONS:
        groups = map_partition(p)
        assert [len(groups[g]) for g in GROUPS] == [18, 6, 6]
        assert sorted(sum(groups.values(), [])) == list(range(30))
        for pool in groups.values():
            for coordinate in range(2):
                assert np.array_equal(np.bincount(maps[pool, coordinate], minlength=6), np.full(6, len(pool)//6))
        seen_added.append(groups['added']); seen_sealed.append(groups['sealed'])
    assert sorted(seen_added) == sorted(seen_sealed)
    fake = []
    for seed, p, arm in itertools.product(SEEDS, PARTITIONS, ARMS):
        d = dict(metrics={'m': seed/100000+p/100}, curves={g: [.2]*len(TIMES) for g in ALL_GROUPS},
                 final_by_mode={m: {g: .3 for g in ALL_GROUPS} for m in MODES})
        fake.append(dict(seed=seed, partition=p, arm=arm, directions=[d, d]))
    cells, arms, comparisons = aggregate(fake)
    assert len(cells) == 16 and all(a['seed_count'] == 4 for a in arms.values())
    assert all(v == 0 for c in comparisons.values() for vv in c['seed_differences'].values() for v in vv)
    try:
        aggregate(fake[:-1])
        raise RuntimeError('incomplete seed cell accepted')
    except AssertionError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        batch = Path(tmp)
        for path, *_ in inventory(batch)[0]:
            path.mkdir()
            # Contents intentionally invalid: an inventory must never parse them.
            for name in REQUIRED:
                (path/name).write_text('NOT EVALUATION DATA')
        assert not inventory(batch)[1]
        first = inventory(batch)[0][0][0]
        (first/'result.json').unlink()
        assert len(inventory(batch)[1]) == 1
    return dict(status='passed', checks=['constant and linear AUC for both time grids',
        'irregular checkpoint weighting', '18/6/6 weighting', 'cyclic balanced partitions',
        'seed-first aggregation', 'paired differences', 'incomplete cell rejected',
        'all-60 filename-only inventory guard'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', nargs='?', type=Path, default=ROOT/'results/generalization_001')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    tests = self_test()
    if args.self_test:
        print(json.dumps(tests)); return
    batch = args.batch.resolve()
    entries, missing = inventory(batch)
    if missing:
        print(json.dumps(dict(status='pending', complete_file_sets=60-len(missing), expected_run_count=60,
            evaluation_content_read=False, analysis_files_written=False, missing_runs=missing), ensure_ascii=False))
        return
    runs = [load_run(*entry) for entry in entries]
    bases = [r for r in runs if r['arm'] == 'base']
    adaptation = [r for r in runs if r['arm'] != 'base']
    assert len(bases) == 12 and len(adaptation) == 48
    for seed, p in itertools.product(SEEDS, PARTITIONS):
        base = next(r for r in bases if (r['seed'], r['partition']) == (seed, p))
        cell = [r for r in adaptation if (r['seed'], r['partition']) == (seed, p)]
        assert len(cell) == 4
        assert all(r['initial_sha256'] == base['final_sha256'] for r in cell)
        assert len({r['world_sha256'] for r in [base, *cell]}) == 1
        assert len({r['evaluation_seed'] for r in [base, *cell]}) == 1
        assert all(r['initial_stats'] == base['final_stats'] for r in cell)
        expansion = [r for r in cell if r['arm'] != 'stay_old_both']
        assert len({r['training_world_sequence_sha256'] for r in expansion}) == 1
        assert len({r['training_rng_sequence_sha256'] for r in cell}) == 1
    source_hashes = runs[0]['source_hashes']
    assert all(r['source_hashes'] == source_hashes for r in runs)
    assert all(sha(path) == digest for path, digest in source_hashes.items())
    cells, arms, comparisons = aggregate(adaptation)
    base_cells, base_summary, _ = aggregate(bases, ('base',))
    result = dict(status='complete', expected_run_count=60, fresh_training_seeds=list(SEEDS),
        base_times=list(BASE_TIMES), continuation_times=list(TIMES), batch=str(batch), analysis_sha256=sha(__file__),
        self_tests=tests, inventory_gate='All 60 required file sets existed before any result content was loaded',
        inferential_unit='Four fresh training seeds; average three partitions and two directions within seed',
        primary_comparison='expand_both_minus_stay_old_both: sealed_final_j',
        secondary_process_measure='sealed_auc_j; trapezoid area / 600',
        base_runs=bases, adaptation_runs=adaptation, base_seed_cells=base_cells,
        base_summary=base_summary['base'], seed_cells=cells, arms=arms, comparisons=comparisons,
        source_hashes=source_hashes,
        audit=dict(run_count=60, base_count=12, adaptation_count=48, fresh_seed_count=4,
            seed_cells=16, adaptation_directions=96, total_training_updates=57600,
            total_training_worlds=sum(sum(r['training_exposure'].values()) for r in runs),
            sealed_training_worlds=sum(r['training_exposure']['sealed'] for r in runs),
            common_base_state_verified=True, common_evaluation_worlds_verified=True,
            common_expansion_training_worlds_verified=True,
            support_counts_verified=True, source_hashes_verified=True))
    assert result['audit']['total_training_worlds'] == 29491200
    out = batch/'figures'; out.mkdir(exist_ok=True)
    result['figures'] = figures(result, out)
    target = batch/'generalization_analysis.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (batch/'泛化分析草稿.md').write_text(report(result))
    main_effect = comparisons['expand_both_minus_stay_old_both']
    print(json.dumps(dict(status='complete', output=str(target), audit=result['audit'],
        sealed_endpoint_mean_difference=main_effect['mean_differences']['sealed_final_j'],
        sealed_endpoint_seed_differences=main_effect['seed_differences']['sealed_final_j']), ensure_ascii=False))


if __name__ == '__main__':
    main()
