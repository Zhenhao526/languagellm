"""Aggregate only complete new-seed runs; seeds are the statistical unit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
METRIC = 'mean_reward_per_step'
NAMES = {'course_communication': '课程＋通信', 'direct_communication': '直接完整任务＋通信',
         'course_blocked': '课程＋恒定接收0', 'course_substitutable': '课程＋资源可替代'}
COLORS = {'course_communication': '#007F79', 'direct_communication': '#A76132',
          'course_blocked': '#68798C', 'course_substitutable': '#985A91'}
MODES = ('normal', 'shuffle', 'blank', 'stochastic', 'stochastic_blank')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def aggregate(values):
    a = np.asarray(list(values), float)
    return {'n': len(a), 'mean': float(a.mean()), 'std': float(a.std(ddof=1)),
            'min': float(a.min()), 'max': float(a.max()), 'by_seed_order': a.tolist()}


def paired_estimate(differences, indices):
    a = np.asarray(differences, float)
    means = a[indices].mean(axis=1)
    return {**aggregate(a), 'paired_bootstrap_percentile_95_interval': np.quantile(means, [.025, .975]).tolist(),
            'positive_seeds': int((a > 0).sum()), 'zero_seeds': int((a == 0).sum()),
            'negative_seeds': int((a < 0).sum())}


def direction_gap(record, sender):
    values = []
    for mode in ('normal', 'shuffle'):
        values.append(next(d['success'] for d in record['full'][mode]['by_direction']
                           if d['restricted_sender'] == sender))
    return values[0] - values[1]


def passes(record, criteria):
    return (record['curriculum']['normal'][METRIC] >= criteria['course_success'] and
            record['full']['normal'][METRIC] >= criteria['full_success'] and
            record['full']['normal'][METRIC] - record['full']['shuffle'][METRIC] >= criteria['full_shuffle_gap'] and
            all(direction_gap(record, i) >= criteria['direction_shuffle_gap'] for i in range(2)))


def load(folder):
    config, completed = read(folder / 'config.json'), read(folder / 'completed.json')
    expected = len(config['conditions']) * len(config['seeds'])
    assert completed['status'] == 'completed' and completed['runs'] == expected
    runs = {}
    for c in config['conditions']:
        for seed in config['seeds']:
            path = folder / f'{c}_s{seed}'
            result, curve = read(path / 'result.json'), read(path / 'learning_curve.json')
            assert result['condition'] == c and result['seed'] == seed and result['updates'] == config['updates']
            assert [r['update'] for r in curve] == config['checkpoints']
            runs[c, seed] = {'path': path, 'result': result, 'curve': curve}
    for seed in config['seeds']:
        assert len({runs[c, seed]['result']['initial_parameter_sha256'] for c in config['conditions']}) == 1
        for task in ('curriculum', 'full'):
            assert len({runs[c, seed]['result'][task][m]['external_cases_sha256']
                        for c in config['conditions'] for m in MODES}) == 1
    return config, completed, runs


def summarize(config, completed, runs):
    seeds = config['seeds']
    indices = np.random.default_rng(6000914).integers(len(seeds), size=(10000, len(seeds)))
    result = {'seeds': seeds, 'completed': completed,
              'statistical_unit': 'one independent two-agent training seed; paired conditions within seed',
              'bootstrap': {'seed': 6000914, 'resamples': 10000, 'method': 'paired-seed percentile bootstrap; descriptive uncertainty, not new-photo external validity'},
              'conditions': {}, 'primary_comparisons': {}, 'runs': [],
              'scope': 'new training seeds; existing photo holdout reused; substitutable native success is structurally 1 and never ranked against complementary-task success'}
    for c in config['conditions']:
        fs = [runs[c, s]['result'] for s in seeds]
        result['conditions'][c] = {
            'native_full_greedy': aggregate(f['native_task']['full']['greedy_success'] for f in fs),
            'native_full_stochastic': aggregate(f['native_task']['full']['stochastic_success'] for f in fs),
            'common_balanced_full': {m: aggregate(f['full'][m][METRIC] for f in fs) for m in MODES},
            'normal_minus_shuffle': aggregate(f['full']['normal'][METRIC] - f['full']['shuffle'][METRIC] for f in fs),
            'direction_gaps': {str(i): aggregate(direction_gap(f, i) for f in fs) for i in range(2)},
            'balanced_diagnostic_criteria_pass_seeds': [s for s in seeds if passes(runs[c, s]['result'], config['engineering_criteria'])],
        }
        for seed in seeds:
            f, curve = runs[c, seed]['result'], runs[c, seed]['curve']
            passed_updates = [r['update'] for r in curve if passes(r, config['engineering_criteria'])]
            x = np.array([r['update'] for r in curve])
            gaps = np.array([r['full']['normal'][METRIC] - r['full']['shuffle'][METRIC] for r in curve])
            entry = {'condition': c, 'seed': seed,
                     'native_full': f['native_task']['full'],
                     'common_balanced_full': {m: f['full'][m][METRIC] for m in MODES},
                     'full_direction_gaps': [direction_gap(f, i) for i in range(2)],
                     'balanced_diagnostic_criteria_pass': passes(f, config['engineering_criteria']),
                     'passed_checkpoints': passed_updates, 'first_observed_passing_checkpoint': min(passed_updates) if passed_updates else None,
                     'not_observed_passing_within_budget': not bool(passed_updates),
                     'gap_curve_normalized_trapezoid_area': float(np.sum(np.diff(x) * (gaps[:-1] + gaps[1:]) * .5) / config['updates']),
                     'raw_sender_counts_by_private_context': f['full']['normal']['symbols_by_local_resource_set'],
                     'sender_probabilities_by_private_context': f['full']['normal']['mean_symbol_probabilities_by_local_resource_set'],
                     'receiver_used_symbol_sensitivity': [d['observed_symbol_sensitivity'] for d in f['intervention']['directions']],
                     'interpretation': 'formation timing is discrete and interval/censor limited; symbols are raw IDs, not translations; blocked open-channel diagnostics are counterfactual'}
            result['runs'].append(entry)
    course = [runs['course_communication', s]['result'] for s in seeds]
    direct = [runs['direct_communication', s]['result'] for s in seeds]
    blocked = [runs['course_blocked', s]['result'] for s in seeds]
    result['primary_comparisons']['course_minus_direct_full_success'] = paired_estimate(
        [a['full']['normal'][METRIC] - b['full']['normal'][METRIC] for a, b in zip(course, direct)], indices)
    result['primary_comparisons']['course_minus_blocked_native_full_success'] = paired_estimate(
        [a['full']['normal'][METRIC] - b['native_task']['full']['greedy_success'] for a, b in zip(course, blocked)], indices)
    result['secondary_comparisons'] = {
        'course_minus_direct_shuffle_gap': paired_estimate(
            [a['full']['normal'][METRIC] - a['full']['shuffle'][METRIC] - b['full']['normal'][METRIC] + b['full']['shuffle'][METRIC]
             for a, b in zip(course, direct)], indices),
        'course_minus_direct_stochastic_success': paired_estimate(
            [a['full']['stochastic'][METRIC] - b['full']['stochastic'][METRIC] for a, b in zip(course, direct)], indices)}
    result['initialization_provenance_correction'] = {
        'overlapping_seed': 1202,
        'prior_checkpoint': 'results/partners_001/prepared_s202.pt agents 2 and 3',
        'overlap_scope': 'prepared non-social weights identical; no previous social protocol loaded; fresh social training streams',
        'within_batch_independent_seeds': 10,
        'interpretation': '10 paired seeds remain primary; do not describe all initial preparations as unseen in earlier project runs or pool them as fully independent of the historical source population',
        'sensitivity_status': 'supplement added after initialization provenance discovery; not an originally preregistered primary analysis',
    }
    keep = np.array([s != 1202 for s in seeds])
    kept_seeds = [s for s in seeds if s != 1202]
    sensitivity_indices = np.random.default_rng(6000914).integers(len(kept_seeds), size=(10000, len(kept_seeds)))
    result['sensitivity_excluding_preparation_overlap_1202'] = {
        'seeds': kept_seeds, 'comparisons': {
            name: paired_estimate(np.asarray(estimate['by_seed_order'])[keep], sensitivity_indices)
            for name, estimate in result['primary_comparisons'].items()}}
    return result


def save(fig, folder, name):
    fig.savefig(folder / f'{name}.png', dpi=180)
    fig.savefig(folder / f'{name}.pdf')
    plt.close(fig)


def figures(folder, config, runs, summary):
    plt.rcParams.update({'font.sans-serif': ['PingFang SC', 'Arial Unicode MS', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 11})
    x = config['checkpoints']
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.7), constrained_layout=True)
    for ax, c in zip(axes.flat, config['conditions']):
        for mode, line, label in [('normal', '-', '资源平衡：最大概率'), ('shuffle', '--', '资源平衡：打乱消息')]:
            key = 'blank' if c == 'course_blocked' else mode
            values = np.array([[r['full'][key][METRIC] for r in runs[c, s]['curve']] for s in config['seeds']])
            for y in values:
                ax.plot(x, 100*y, line, color=COLORS[c], lw=.65, alpha=.16)
            ax.plot(x, 100*values.mean(0), line, color=COLORS[c], lw=2.4,
                    label='恒定接收0的原任务' if c == 'course_blocked' else label)
            if c == 'course_blocked':
                break
        if c == 'course_substitutable':
            ax.axhline(100, color='#777777', lw=1.3, ls=':', label='可替代原任务：恒为100%')
        ax.axhline(100*5/7, color='#b4b4b4', lw=.8, ls=':')
        for boundary in (600, 900):
            ax.axvline(boundary, color='#bdbdbd', ls=':', lw=.8)
        ax.set(title=NAMES[c], xlabel='社会训练更新', ylabel='完整场景成功率（%）', ylim=(25, 103), xlim=(0, 1200))
        ax.legend(loc='lower right', fontsize=9)
    fig.suptitle('十个群体：形成过程与边界对照\n细线为10个独立群体；粗线为均值；600/900处为预定阶段边界', fontsize=15)
    save(fig, folder, 'formation_confirm_process')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    comparable = config['conditions'][:3]
    for s in config['seeds']:
        values = [runs[c, s]['result']['native_task']['full']['greedy_success']*100 for c in comparable]
        axes[0].plot(range(3), values, color='#b7b7b7', lw=.7, alpha=.6)
    for i, c in enumerate(comparable):
        values = [runs[c, s]['result']['native_task']['full']['greedy_success']*100 for s in config['seeds']]
        axes[0].scatter(np.full(len(values), i), values, color=COLORS[c], s=25, zorder=3)
        axes[0].plot([i-.13, i+.13], [np.mean(values)]*2, color='black', lw=3)
    axes[0].set(xticks=range(3), xticklabels=['课程通信', '直接完整', '课程恒0'], ylabel='完整资源互补任务成功率（%）', ylim=(25, 102), title='相同起点与预算；每条线连接同一种子')
    keys = list(summary['primary_comparisons'])
    for i, key in enumerate(keys):
        estimate = summary['primary_comparisons'][key]
        vals = np.array(estimate['by_seed_order'])*100
        low, high = np.array(estimate['paired_bootstrap_percentile_95_interval'])*100
        mean = estimate['mean']*100
        axes[1].scatter(np.full(len(vals), i)-.09, vals, color='#6c8a96', s=20, alpha=.7)
        axes[1].errorbar(i+.1, mean, yerr=[[mean-low], [high-mean]], fmt='o', color='#124f4b', capsize=5, lw=2)
    axes[1].axhline(0, color='#888888', lw=1)
    axes[1].set(xticks=range(2), xticklabels=['课程 − 直接完整', '课程 − 恒0原任务'], ylabel='配对成功率差（百分点）', title='10群体配对差；95%种子重抽样区间')
    save(fig, folder, 'formation_confirm_paired_effects')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for i, c in enumerate(config['conditions']):
        selected = [r for r in summary['runs'] if r['condition'] == c]
        for sender, offset in ((0, -.11), (1, .11)):
            values = np.array([r['full_direction_gaps'][sender] for r in selected])*100
            axes[0].scatter(np.full(len(values), i)+offset, values, color=COLORS[c], marker='o' if sender == 0 else '^', s=25, alpha=.7)
        gaps = np.array([r['common_balanced_full']['normal'] - r['common_balanced_full']['shuffle'] for r in selected])*100
        axes[1].scatter(np.full(len(gaps), i), gaps, color=COLORS[c], s=30)
    for ax in axes:
        ax.axhline(0, color='#777777', lw=.9)
        ax.axhline(10, color='#aaaaaa', lw=.8, ls=':')
        ax.set(xticks=range(4), xticklabels=['课程通信', '直接完整', '课程恒0', '资源可替代'], ylabel='正常−打乱（百分点）')
    axes[0].set_title('两个受限发送方向：圆点 A→B，三角 B→A')
    axes[1].set_title('共同资源平衡诊断：每点一个群体')
    fig.suptitle('消息的功能作用；恒0组图中为评估解禁通道的诊断，不是原任务成绩', fontsize=13)
    save(fig, folder, 'formation_confirm_message_effects')


def report(folder, config, summary):
    pct = lambda x: f'{100*x:.2f}%'
    effect = lambda x: f'{100*x:.2f}'
    lines = ['# 形成确认与边界对照', '',
             '固定1201–1210十个训练种子、四条件和每组1,200次更新，完成40组。'
             '条件内群体独立训练，条件间按同一种子配对；全部结果保留。', '',
             '| 互补资源任务条件 | 原任务最大概率成功率 | 原任务按策略采样成功率 |',
             '| --- | ---: | ---: |']
    for c in config['conditions'][:3]:
        v = summary['conditions'][c]
        lines.append(f"| {NAMES[c]} | {pct(v['native_full_greedy']['mean'])} | {pct(v['native_full_stochastic']['mean'])} |")
    lines += ['', '恒0条件在训练和原任务评估中均收到0；按策略采样时也不开放通道。'
              '其“正常消息”结果只属于额外解禁通道后的共同诊断，不用于以上原任务比较。', '',
              '| 预定主要配对比较 | 平均差（百分点） | 95%种子重抽样区间 | 正／负方向群体数 |',
              '| --- | ---: | ---: | ---: |']
    for name, label in [('course_minus_direct_full_success', '课程通信 − 直接完整任务通信'),
                        ('course_minus_blocked_native_full_success', '课程通信 − 课程恒0原任务')]:
        d = summary['primary_comparisons'][name]
        lo, hi = d['paired_bootstrap_percentile_95_interval']
        lines.append(f"| {label} | {effect(d['mean'])} | [{effect(lo)}, {effect(hi)}] | {d['positive_seeds']}／{d['negative_seeds']} |")
    lines += ['', '区间以十个配对训练种子为重抽样单位，10,000次重抽样，固定随机种子6000914。'
              '它反映当前种子样本中的估计不确定性，不包括其他模型、环境或新增照片带来的外部不确定性。', '',
              '![配对主结果](' + str(folder.resolve()/'formation_confirm_paired_effects.png') + ')', '',
              '## 通信作用与形成轨迹', '',
              '| 可通信的互补资源条件 | 完整任务正常−打乱 | 双方向等工程条件全部通过的种子 |',
              '| --- | ---: | --- |']
    for c in config['conditions'][:2]:
        d = summary['conditions'][c]
        passed = d['balanced_diagnostic_criteria_pass_seeds']
        lines.append(f"| {NAMES[c]} | {effect(d['normal_minus_shuffle']['mean'])}个百分点 | {len(passed)}/10：{', '.join(map(str, passed)) or '无'} |")
    lines += ['', '工程标准沿用课程90%、完整任务85%、完整正常−打乱10个百分点，且两个受限发送方向各有10个百分点消息落差。'
              '门槛用于描述当前任务中的协调与消息作用，不定义语言诞生；首次通过只指预定检查点，不是精确形成时间。'
              '全部五编号的固定观察干预、原始发送频数和双方向结果保存在汇总JSON及各运行result.json。', '',
              '1209曾在第650次通过整套门槛，终点课程表现降至87.84%；它的完整任务仍为90.78%，'
              '两个受限发送方向仍有明显消息收益。未通过课程门槛不能改写成“未形成通信”。'
              '[过程后退与个人编码案例](' + str(folder.resolve()/'形成确认_过程与编号案例.md') + ')。', '',
              '![形成过程](' + str(folder.resolve()/'formation_confirm_process.png') + ')', '',
              '![双方向消息作用](' + str(folder.resolve()/'formation_confirm_message_effects.png') + ')', '',
              '## 完全可替代的边界条件', '']
    s = summary['conditions']['course_substitutable']
    lines += ['该条件每天任意两次采集都满足原任务需求，原生成功率按构造恒为100%。'
              f"同一终点策略在共同的互补资源诊断中，正常成功率为{pct(s['common_balanced_full']['normal']['mean'])}，"
              f"正常−打乱为{effect(s['normal_minus_shuffle']['mean'])}个百分点。"
              '前者不是原任务失败率，后者不能直接说明这些主体缺乏建立交流的能力。'
              '该条件仅说明移除收益中的协调需求后的一个边界，不检验连续生态效应或相变。', '',
              '四条件均有相同的200×64次个体资源后果练习，包括互补资源需求。'
              '可替代条件只在社会阶段改变资源后果，不代表从未经历过资源区分。', '',
              '## 准备起点溯源与敏感性', '',
              '运行期间的来源核查发现：种子1202的两主体准备权重与旧partners_001种子202的C/D逐tensor相同。'
              '它们没有继承旧社会协议，本轮使用新的社会训练随机流；十个本轮群体之间仍分别初始化、独立训练。'
              '因此不能说所有十个准备起点在项目历史中都首次出现，也不能将其与旧来源群体作为完全独立来源合并。'
              '不修改种子或删去主结果，另作排除1202的九种子敏感性；此项是发现来源重用后的补充分析。', '']
    for name, label in [('course_minus_direct_full_success', '课程−直接完整'),
                        ('course_minus_blocked_native_full_success', '课程−恒0原任务')]:
        d = summary['sensitivity_excluding_preparation_overlap_1202']['comparisons'][name]
        lo, hi = d['paired_bootstrap_percentile_95_interval']
        lines.append(f"- {label}：{effect(d['mean'])}个百分点，95%区间[{effect(lo)}, {effect(hi)}]。")
    lines += ['', '## 方法与范围', '',
              '官方DINOv2-L权重冻结，照片特征缓存；每主体独立的83,527参数接口。每次社会更新含1,024次双人采集，'
              '课程条件前600次只含一方受限场景，此后均为完整14场景；直接完整条件全程14场景。'
              '所有条件前900次行动熵系数0.05，此后为0，无发信辅助。恒0条件仍抽样发送并使用相同损失公式，仅改变交付消息。', '',
              '每检查点每任务每模式2,048个案例，终点每项8,192个案例，固定观察干预4,096个案例；'
              '同一种子的外部评估案例相同。两资源场景和照片实例是研究者生成并评分的任务变量，不作为类别标签进入策略。', '',
              '沿用44张训练照片和16张留出照片，没有新照片确认集。冻结视觉模型本身的贡献、最小非语言能力、'
              '自发分工、长期记忆、组合结构及人类语言起源均不由本批直接检验。', '',
              '[机器可读汇总](' + str(folder.resolve()/'formation_confirm_summary.json') + ') · '
              '[全部原始结果](' + str(folder.resolve()/'results.json') + ') · '
              '[只读重放核查](' + str(folder.resolve()/'形成确认_只读核查.md') + ')', '']
    (folder/'形成确认_结果.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='?', type=Path, default=ROOT / 'results/formation_confirm_001')
    args = parser.parse_args()
    config, completed, runs = load(args.directory)
    summary = summarize(config, completed, runs)
    provenance = args.directory / 'analysis_source'
    provenance.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__), provenance / Path(__file__).name)
    hash_path = provenance / 'source_hashes.json'
    source_hashes = read(hash_path) if hash_path.exists() else {}
    source_hashes[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(hash_path, source_hashes)
    write(args.directory / 'formation_confirm_summary.json', summary)
    figures(args.directory, config, runs, summary)
    report(args.directory, config, summary)
    print(json.dumps({'primary': summary['primary_comparisons'],
                      'conditions': {c: {'native': d['native_full_greedy']['mean'],
                                         'normal': d['common_balanced_full']['normal']['mean'],
                                         'shuffle_gap': d['normal_minus_shuffle']['mean'],
                                         'passed': d['balanced_diagnostic_criteria_pass_seeds']}
                                     for c, d in summary['conditions'].items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
