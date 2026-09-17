"""Population-paired plasticity contrasts; confirmation and exploration separate."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_contact import edge_checks, drift_metrics
from analyze_partners import success, gap, save

ROOT = Path(__file__).resolve().parent
CONDITIONS = ['behavior_frozen', 'sender_only', 'listener_only', 'both_learn', 'all_plastic']
NAMES = {'behavior_frozen': '行为固定', 'sender_only': '仅发送学习', 'listener_only': '仅接收学习',
         'both_learn': '发送与接收学习', 'all_plastic': '含视觉投影均可学习'}
COLORS = {'behavior_frozen': '#8b9098', 'sender_only': '#c17837', 'listener_only': '#2677a8',
          'both_learn': '#008978', 'all_plastic': '#8b5aab'}
GROUP_NAMES = {'cross': '原固定组间四对', 'original': '原搭档两对', 'all': '全部六对'}
CONTRASTS = {'listener_minus_frozen': {'listener_only': 1, 'behavior_frozen': -1},
             'sender_minus_frozen': {'sender_only': 1, 'behavior_frozen': -1},
             'both_minus_listener': {'both_learn': 1, 'listener_only': -1},
             'both_minus_sender': {'both_learn': 1, 'sender_only': -1},
             'interaction': {'both_learn': 1, 'sender_only': -1, 'listener_only': -1, 'behavior_frozen': 1},
             'projection_plasticity': {'all_plastic': 1, 'both_learn': -1}}
CONTRAST_NAMES = {'listener_minus_frozen': '仅接收 − 行为固定', 'sender_minus_frozen': '仅发送 − 行为固定',
                  'both_minus_listener': '双侧 − 仅接收', 'both_minus_sender': '双侧 − 仅发送',
                  'interaction': '双侧 − 仅发送 − 仅接收 + 固定', 'projection_plasticity': '投影也可变 − 双侧学习'}


def read(path):
    return json.loads(Path(path).read_text())


def estimate(values):
    values = np.asarray(values, dtype=np.float64)
    bootstrap = np.random.default_rng(6000914).integers(len(values), size=(10000, len(values)))
    interval = np.percentile(values[bootstrap].mean(1), [2.5, 97.5])
    return {'n_populations': len(values), 'mean': float(values.mean()), 'sd': float(values.std(ddof=1)) if len(values) > 1 else None,
            'min': float(values.min()), 'max': float(values.max()), 'population_values': values.tolist(),
            'paired_seed_bootstrap_95_percentile_interval': interval.tolist()}


def metric(evaluation, group, name):
    return gap(evaluation, group) if name == 'shuffle_gap' else success(evaluation, group, name)


def analyze(folder):
    config, completion = read(folder / 'config.json'), read(folder / 'completed.json')
    assert completion['status'] == 'completed' and completion['runs'] == 47
    runs = {}
    for seed in config['seeds']:
        for condition in config['conditions']:
            path = folder / f'{condition}_s{seed}'
            runs[seed, condition] = {'path': str(path), 'result': read(path / 'result.json'), 'curve': read(path / 'learning_curve.json')}
    for reference in read(folder / 'all_plastic_references.json'):
        path = Path(reference['path'])
        runs[reference['seed'], 'all_plastic'] = {'path': str(path), 'result': read(path / 'result.json'),
                                               'curve': read(path / 'learning_curve.json'), 'reused': reference['reused']}
    bounds = {seed: read(folder / f'frozen_function_bounds_s{seed}.json') for seed in config['seeds']}
    cohorts = {'confirmatory': config['confirmatory_seeds'], 'exploratory': config['exploratory_seeds'], 'combined_descriptive': config['seeds']}
    metrics = ['normal', 'shuffle', 'blank', 'stochastic', 'shuffle_gap']
    summaries = {}
    for name, seeds in cohorts.items():
        values = {}
        for condition in CONDITIONS:
            values[condition] = {group: {m: estimate([metric(runs[s, condition]['result']['evaluation'], group, m) for s in seeds])
                                        for m in metrics} for group in GROUP_NAMES}
            counts = []
            for seed in seeds:
                checked = [edge_checks(p, config['engineering_edge_criteria'])
                           for p in runs[seed, condition]['result']['evaluation']['pairs'] if p['group'] == 'cross']
                counts.append(sum(edge['checks']['all_pass'] for edge in checked))
            values[condition]['communication_criteria'] = {
                'populations_all_four_cross_pairs_pass': sum(count == 4 for count in counts),
                'population_count': len(seeds), 'cross_pairs_passed_total': sum(counts),
                'cross_pairs_total': len(seeds) * 4, 'cross_pairs_passed_by_population': counts,
                'note': 'Population and edge counts are descriptive; the edges are not independent replicates.'}
        contrasts = {key: {group: {m: estimate([sum(weight * metric(runs[s, condition]['result']['evaluation'], group, m)
                                                    for condition, weight in weights.items()) for s in seeds])
                                  for m in ('normal', 'shuffle_gap', 'stochastic')}
                          for group in GROUP_NAMES} for key, weights in CONTRASTS.items()}
        summaries[name] = {'seeds': seeds, 'conditions': values, 'paired_contrasts': contrasts}
    individual = []
    for seed in config['seeds']:
        for condition in CONDITIONS:
            item = runs[seed, condition]
            result, curve = item['result'], item['curve']
            edges = [edge_checks(p, config['engineering_edge_criteria']) for p in result['evaluation']['pairs']]
            origin = read(Path(item['path']) / 'origin_result.json')
            trajectory = []
            for row in curve:
                checks = [edge_checks(p, config['engineering_edge_criteria']) for p in row['evaluation']['pairs']]
                trajectory.append({'update': row['update'], 'cross_pass_count': sum(e['checks']['all_pass'] for e in checks if e['group'] == 'cross'),
                                   'groups': {g: {'normal': success(row['evaluation'], g), 'shuffle_gap': gap(row['evaluation'], g)} for g in GROUP_NAMES}})
            individual.append({'seed': seed, 'cohort': 'confirmatory' if seed in config['confirmatory_seeds'] else 'exploratory',
                               'condition': condition, 'path': item['path'], 'reused': item.get('reused', False),
                               'groups': {g: {m: metric(result['evaluation'], g, m) for m in metrics} for g in GROUP_NAMES},
                               'change_from_origin': {g: success(result['evaluation'], g) - success(origin['evaluation'], g) for g in GROUP_NAMES},
                               'edges': edges, 'cross_pass_count': sum(e['checks']['all_pass'] for e in edges if e['group'] == 'cross'),
                               'first_checkpoint_all_cross_pass': next((r['update'] for r in trajectory if r['cross_pass_count'] == 4), None),
                               'trajectory': trajectory, 'drift': drift_metrics(result['protocol_drift'])})
    report = {'schema_version': 1, 'primary_cohort': 'confirmatory', 'statistics': {'unit': 'independently initialized population seed',
              'bootstrap_resamples': 10000, 'bootstrap_seed': 6000914, 'interval': 'paired-seed percentile bootstrap 95%; descriptive intervals, no multiplicity-adjusted claims',
              'exploration': 'Original three populations informed design and are not new confirmation; combined ten are descriptive.'},
              'cohorts': summaries, 'runs': individual,
              'frozen_function_bounds': {str(seed): {k: v for k, v in bound.items() if k in ('fixed_sender', 'fixed_listener', 'origin_sha256', 'scope')}
                                         for seed, bound in bounds.items()},
              'limits': ['The factorial fixes shared projection in all four cells; all_plastic is a separate representation-plasticity comparison.',
                         'Frozen-function bounds are expectations for the finite image distribution; finite random evaluation scores can fluctuate around them.',
                         'Failed one-sided learning can reflect incompatible frozen conventions; it does not establish a general inability to learn.',
                         'Interventions concern current-resource content; shuffling can retain sender identity cues through symbol marginals.']}
    (folder / 'mechanism_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    plot(folder, config, runs, report)
    lines = ['# 机制实验结果汇总', '', '主分析为预先固定的7个新群体；原3个群体列为探索，合计10个群体仅作描述。', '',
             '| 条件 | 原组间四对成功率 | 正常−打乱 | 原搭档成功率 | 全六对成功率 | 全四边达标群体 | 达标交叉边 |', '|---|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        x = summaries['confirmatory']['conditions'][condition]
        checks = x['communication_criteria']
        lines.append(f"| {NAMES[condition]} | {x['cross']['normal']['mean']:.2%} | {100*x['cross']['shuffle_gap']['mean']:.2f}个百分点 | {x['original']['normal']['mean']:.2%} | {x['all']['normal']['mean']:.2%} | {checks['populations_all_four_cross_pairs_pass']}/7 | {checks['cross_pairs_passed_total']}/28 |")
    lines += ['', '配对差值及95%群体bootstrap区间：', '', '| 对照 | 交叉成功率差（百分点） | 95%区间 |', '|---|---:|---:|']
    for contrast in CONTRASTS:
        x = summaries['confirmatory']['paired_contrasts'][contrast]['cross']['normal']
        lo, hi = x['paired_seed_bootstrap_95_percentile_interval']
        lines.append(f"| {CONTRAST_NAMES[contrast]} | {100*x['mean']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ['', '上述区间以群体为单位，不把配对、主体或测试案例当作独立重复。机制差异仅解释当前任务、模型、既有协议及600次更新预算。', '',
              '冻结sender或actor的四条件同时固定共用视觉投影；价值基线均可学习。固定行为组的条件策略不变，但继续按策略概率抽样。', '',
              '固定函数上界通过枚举现有测试照片的全部有放回图对与公开时钟计算，仅约束当前图片分布；它不进入训练，也不保证当前网络能达到。', '',
              '完整逐种子、逐边、双方向、检查点门槛、协议漂移、两类上界及探索/确认/合并统计见 mechanism_summary.json。']
    (folder / '机制实验_结果汇总.md').write_text('\n'.join(lines) + '\n')
    return report


def plot(folder, config, runs, report):
    plt.rcParams.update({'font.family': 'PingFang SC', 'axes.unicode_minus': False, 'font.size': 10})
    seeds = config['confirmatory_seeds']
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    for ax, group in zip(axes, ('cross', 'original', 'all')):
        for x, condition in enumerate(CONDITIONS):
            e = report['cohorts']['confirmatory']['conditions'][condition][group]['normal']
            lo, hi = e['paired_seed_bootstrap_95_percentile_interval']
            ax.scatter(x + np.linspace(-.13, .13, len(seeds)), 100 * np.asarray(e['population_values']), s=16, alpha=.6, color=COLORS[condition])
            ax.errorbar(x, e['mean'] * 100, yerr=np.asarray([[e['mean'] - lo], [hi - e['mean']]]) * 100,
                        fmt='o', color=COLORS[condition], capsize=4, lw=2)
        ax.set_xticks(range(5), [NAMES[c].replace('发送与接收学习', '双侧学习').replace('含视觉投影均可学习', '含投影均学习') for c in CONDITIONS], rotation=30, ha='right')
        ax.set_ylim(35, 105)
        ax.set_title(GROUP_NAMES[group])
        ax.set_ylabel('正常消息成功率（%）')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('7个预定确认群体：点为各群体；大点和误差线为均值及群体bootstrap 95%区间')
    save(fig, folder, 'mechanism_confirmatory_outcomes')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for col, group in enumerate(('cross', 'original')):
        for row, measure in enumerate(('normal', 'shuffle_gap')):
            ax = axes[row, col]
            for condition in CONDITIONS:
                curves = [[metric(entry['evaluation'], group, measure) for entry in runs[s, condition]['curve']] for s in seeds]
                x = [entry['update'] for entry in runs[seeds[0], condition]['curve']]
                for y in curves:
                    ax.plot(x, 100 * np.asarray(y), color=COLORS[condition], alpha=.17, lw=.7)
                ax.plot(x, 100 * np.mean(curves, axis=0), 'o-', color=COLORS[condition], ms=3, label=NAMES[condition])
            ax.set_title(GROUP_NAMES[group])
            ax.set_xlabel('接触后新增更新')
            ax.set_ylabel('成功率（%）' if measure == 'normal' else '正常−打乱（百分点）')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('确认群体学习过程：全部条件按策略概率训练，无额外熵奖励')
    save(fig, folder, 'mechanism_confirmatory_process')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, measure in zip(axes, ('normal', 'shuffle_gap')):
        for i, key in enumerate(CONTRASTS):
            e = report['cohorts']['confirmatory']['paired_contrasts'][key]['cross'][measure]
            lo, hi = e['paired_seed_bootstrap_95_percentile_interval']
            ax.scatter(100 * np.asarray(e['population_values']), i + np.linspace(-.1, .1, len(seeds)), s=15, alpha=.4, color='#317586')
            ax.errorbar(e['mean'] * 100, i, xerr=np.asarray([[e['mean'] - lo], [hi - e['mean']]]) * 100, fmt='o', color='#174657', capsize=4)
        ax.axvline(0, color='#888', lw=1)
        ax.set_yticks(range(len(CONTRASTS)), [CONTRAST_NAMES[k] for k in CONTRASTS])
        ax.invert_yaxis()
        ax.set_xlabel('交叉成功率处理差（百分点）' if measure == 'normal' else '交叉打乱落差处理差（百分点）')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('7个确认群体的配对差值：均值及群体bootstrap 95%区间')
    save(fig, folder, 'mechanism_paired_contrasts')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, condition, bound_key, title in zip(axes, ('listener_only', 'sender_only'), ('fixed_sender', 'fixed_listener'),
                                             ('仅接收学习：发送函数固定', '仅发送学习：接收函数固定')):
        for i, seed in enumerate(seeds):
            observed = success(runs[seed, condition]['result']['evaluation'], 'cross')
            upper = report['frozen_function_bounds'][str(seed)][bound_key]['normal']['cross']['full_success_upper_bound']
            ax.plot([i, i], [observed * 100, upper * 100], color='#aaa', lw=1)
            ax.scatter(i, upper * 100, marker='_', s=180, color='#333', label='固定函数的乐观期望上界' if i == 0 else None)
            ax.scatter(i, observed * 100, color=COLORS[condition], s=40, label='正常评估成功率' if i == 0 else None)
        ax.set_xticks(range(len(seeds)), [str(s) for s in seeds])
        ax.set_xlabel('独立群体种子')
        ax.set_ylabel('原组间四对成功率（%）')
        ax.set_ylim(35, 105)
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle('固定函数的结构性限制：精确枚举有限测试图片；其余场景乐观按全成功计')
    save(fig, folder, 'mechanism_frozen_function_bounds')
    # A portrait companion remains legible when inserted into a Word/PDF page.
    with plt.rc_context({'font.size': 12}):
        fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.6), constrained_layout=True)
        for i, condition in enumerate(CONDITIONS):
            e = report['cohorts']['confirmatory']['conditions'][condition]['cross']['normal']
            lo, hi = e['paired_seed_bootstrap_95_percentile_interval']
            axes[0].scatter(100 * np.asarray(e['population_values']), i + np.linspace(-.12, .12, len(seeds)),
                            color=COLORS[condition], alpha=.45, s=24)
            axes[0].errorbar(e['mean'] * 100, i, xerr=np.asarray([[e['mean'] - lo], [hi - e['mean']]]) * 100,
                             fmt='o', color=COLORS[condition], capsize=4, lw=2)
        axes[0].set_yticks(range(5), [NAMES[c] for c in CONDITIONS])
        axes[0].invert_yaxis()
        axes[0].set_xlim(35, 105)
        axes[0].set_xlabel('原组间四对正常成功率（%）')
        axes[0].set_title('A  七个确认群体的终点表现', loc='left')
        short = ['仅接收 − 行为固定', '仅发送 − 行为固定', '双侧 − 仅接收', '双侧 − 仅发送',
                 '发送×接收交互', '投影也可变 − 双侧']
        for i, key in enumerate(CONTRASTS):
            e = report['cohorts']['confirmatory']['paired_contrasts'][key]['cross']['normal']
            lo, hi = e['paired_seed_bootstrap_95_percentile_interval']
            axes[1].scatter(100 * np.asarray(e['population_values']), i + np.linspace(-.12, .12, len(seeds)),
                            color='#317586', alpha=.45, s=24)
            axes[1].errorbar(e['mean'] * 100, i, xerr=np.asarray([[e['mean'] - lo], [hi - e['mean']]]) * 100,
                             fmt='o', color='#174657', capsize=4, lw=2)
        axes[1].set_yticks(range(len(CONTRASTS)), short)
        axes[1].invert_yaxis()
        axes[1].axvline(0, color='#888', lw=1)
        axes[1].set_xlabel('同一群体内的成功率差（百分点）')
        axes[1].set_title('B  配对差值', loc='left')
        for ax in axes:
            ax.grid(axis='x', alpha=.2)
        fig.suptitle('固定视觉投影的机制对照\n细点：各群体；大点与误差线：均值、群体bootstrap 95%区间', fontsize=12)
        save(fig, folder, 'mechanism_report_compact')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=ROOT / 'results/mechanism_001')
    args = parser.parse_args()
    result = analyze(args.directory)
    print(json.dumps({'populations': len(result['cohorts']['combined_descriptive']['seeds']), 'primary_n': 7}))
