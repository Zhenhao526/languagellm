"""Create compact scientific figures from the JSON execution and summary."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import design

T7_975 = 2.364624251

# The macOS runtime has a Chinese system font, while Matplotlib's default
# DejaVu Sans does not contain the labels used in these figures.
plt.rcParams['font.sans-serif'] = ['STHeiti', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _by_run(runs):
    return {(r['seed'], r['rule'], r['information'], r['schedule'], bool(r['live'])): r
            for r in runs}


def _trajectory(run, key):
    return np.asarray([row['target_trajectory'][key]
                       for row in sorted(run['trajectory'], key=lambda r: int(r['update']))], dtype=float)


def _final(run, key):
    return float(run['final']['new_layouts'][key])


def _mean_ci(values):
    x = np.asarray(values, dtype=float)
    mean = x.mean(axis=0)
    se = x.std(axis=0, ddof=1) / math.sqrt(len(x))
    return mean, T7_975 * se


def plot_trajectory(by, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True)
    for ax, info in zip(axes, design.INFORMATIONS):
        for rule, color in (('strict', '#2f5597'), ('reciprocal', '#c0504d')):
            values = []
            for seed in design.SEEDS:
                schedule_gains = []
                for schedule in design.SCHEDULES:
                    schedule_gains.append(_trajectory(by[seed, rule, info, schedule, True],
                                                     'target_pair_legal_rate')
                                          - _trajectory(by[seed, rule, info, schedule, False],
                                                        'target_pair_legal_rate'))
                values.append(np.mean(schedule_gains, axis=0) * 100)
            mean, half = _mean_ci(values)
            ax.plot(design.UPDATES, mean, marker='o', linewidth=2, color=color, label=rule)
            ax.fill_between(design.UPDATES, mean - half, mean + half, color=color, alpha=.16)
        ax.axhline(0, color='0.35', linewidth=.8)
        ax.set_title(info)
        ax.set_xlabel('更新次数')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('live − silent：目标搭档合法率（百分点）')
    axes[1].legend(frameon=False)
    fig.suptitle('规则与信息条件下的通信增益轨迹', y=1.02)
    fig.tight_layout()
    _save(fig, out, 'target_pair_gain_trajectories')


def plot_decomposition(by, out):
    metrics = [('physical_execution_rate', '物理执行'),
               ('conditional_q_rate', '条件 Q'),
               ('target_pair_legal_rate', '目标搭档'),
               ('proposal_legal_rate', '提案合法')]
    labels = []
    values = []
    for rule in design.RULES:
        for info in design.INFORMATIONS:
            labels.append(f'{rule}\n{info}')
            row = []
            for key, _ in metrics:
                gains = []
                for seed in design.SEEDS:
                    g = []
                    for schedule in design.SCHEDULES:
                        g.append(_final(by[seed, rule, info, schedule, True], key)
                                 - _final(by[seed, rule, info, schedule, False], key))
                    gains.append(np.mean(g) * 100)
                row.append(np.mean(gains))
            values.append(row)
    values = np.asarray(values)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(labels)); width = .18
    for i, (_, label) in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, values[:, i], width, label=label)
    ax.axhline(0, color='0.35', linewidth=.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel('live − silent 终点差（百分点）')
    ax.set_title('通信增益的执行—选择分解（新布局终点）')
    ax.legend(ncol=4, frameon=False, fontsize=9)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    _save(fig, out, 'execution_vs_selection_decomposition')


def plot_rule_interaction(by, out):
    keys = [('target_pair_legal_rate', '目标搭档合法率'),
            ('q_rate', '团队 Q'), ('physical_execution_rate', '物理执行率'),
            ('ignored_proposal_rate', '被忽略提案率')]
    labels = [f'{info}\n{schedule}' for info in design.INFORMATIONS for schedule in design.SCHEDULES]
    values = []
    errors = []
    for key, _ in keys:
        row, err = [], []
        for info in design.INFORMATIONS:
            for schedule in design.SCHEDULES:
                vals = []
                for seed in design.SEEDS:
                    rec = (_final(by[seed, 'reciprocal', info, schedule, True], key)
                           - _final(by[seed, 'reciprocal', info, schedule, False], key))
                    strict = (_final(by[seed, 'strict', info, schedule, True], key)
                              - _final(by[seed, 'strict', info, schedule, False], key))
                    vals.append((rec - strict) * 100)
                row.append(np.mean(vals)); err.append(T7_975 * np.std(vals, ddof=1) / math.sqrt(len(vals)))
        values.append(row); errors.append(err)
    values = np.asarray(values); errors = np.asarray(errors)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(labels)); width = .18
    for i, (_, label) in enumerate(keys):
        ax.bar(x + (i - 1.5) * width, values[i], width, yerr=errors[i], capsize=2,
               label=label)
    ax.axhline(0, color='0.35', linewidth=.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel('（reciprocal − strict）通信增益（百分点）')
    ax.set_title('结算规则对通信效应的交互')
    ax.legend(ncol=4, frameon=False, fontsize=9)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    _save(fig, out, 'rule_communication_interaction')


def plot_information_interaction(by, out):
    keys = [('target_pair_legal_rate', '目标搭档合法率'),
            ('q_rate', '团队 Q'), ('physical_execution_rate', '物理执行率')]
    labels = [f'{rule}\n{schedule}' for rule in design.RULES for schedule in design.SCHEDULES]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    x = np.arange(len(labels)); width = .23
    for i, (key, label) in enumerate(keys):
        vals, errs = [], []
        for rule in design.RULES:
            for schedule in design.SCHEDULES:
                d = []
                for seed in design.SEEDS:
                    pl = (_final(by[seed, rule, 'PL', schedule, True], key)
                          - _final(by[seed, rule, 'PL', schedule, False], key))
                    fi = (_final(by[seed, rule, 'FI', schedule, True], key)
                          - _final(by[seed, rule, 'FI', schedule, False], key))
                    d.append((pl - fi) * 100)
                vals.append(np.mean(d)); errs.append(T7_975 * np.std(d, ddof=1) / math.sqrt(len(d)))
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=2, label=label)
    ax.axhline(0, color='0.35', linewidth=.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel('（PL − FI）通信增益（百分点）')
    ax.set_title('信息可见性对通信效应的交互')
    ax.legend(ncol=3, frameon=False, fontsize=9)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    _save(fig, out, 'information_communication_interaction')


def plot_absolute(by, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = [f'{rule}\n{info}' for rule in design.RULES for info in design.INFORMATIONS]
    x = np.arange(len(labels)); width = .19
    for ax, key, ylabel, title in (
            (axes[0], 'q_rate', 'Q（百分点）', '团队 Q'),
            (axes[1], 'physical_execution_rate', '物理执行率（百分点）', '物理执行')):
        for j, live in enumerate((True, False)):
            vals, errs = [], []
            for rule in design.RULES:
                for info in design.INFORMATIONS:
                    v = [_final(by[seed, rule, info, 'static', live], key) * 100
                         for seed in design.SEEDS]
                    vals.append(np.mean(v)); errs.append(T7_975 * np.std(v, ddof=1) / math.sqrt(len(v)))
            ax.bar(x + (j - .5) * width, vals, width, yerr=errs, capsize=2,
                   label='live' if live else 'silent')
        ax.set_xticks(x, labels); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(frameon=False); ax.grid(axis='y', alpha=.2)
    fig.suptitle('新布局终点的绝对表现', y=1.02)
    fig.tight_layout()
    _save(fig, out, 'absolute_q_and_execution')


def _save(fig, out, stem):
    fig.savefig(out / f'{stem}.png', dpi=180, bbox_inches='tight')
    fig.savefig(out / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite figure output')
    output.mkdir(parents=False)
    data = json.loads((source / 'execution' / 'results.json').read_text())
    by = _by_run(data['runs'])
    plot_trajectory(by, output); plot_decomposition(by, output)
    plot_rule_interaction(by, output); plot_information_interaction(by, output)
    plot_absolute(by, output)
    pngs = sorted(p.name for p in output.glob('*.png'))
    receipt = dict(status='created', source=str(source), pngs=pngs,
                   pdfs=sorted(p.name for p in output.glob('*.pdf')),
                   visual_review='pending', model_forwards=0)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    main(args.source, args.output)
