"""Publication-style plots for the global-scale control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import design


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(source):
    data = json.loads((Path(source) / 'execution' / 'results.json').read_text())
    return data['runs']


def trajectory(run, key):
    return np.asarray([r['target_trajectory'][key]
                       for r in sorted(run['trajectory'], key=lambda x: int(x['update']))])


def final(run, key):
    return float(run['final']['new_layouts'][key])


def figure_dir(output):
    p = Path(output); p.mkdir(parents=False, exist_ok=False); return p


def save(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / f'{name}.png', dpi=220, bbox_inches='tight')
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def main(source, output):
    runs = load(source)
    directory = figure_dir(output)
    by = {(int(r['seed']), r['rule'], r['information'], r['schedule'], bool(r['live'])): r
          for r in runs}
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['STHeiti', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10})
    colors = {'c0': '#4c78a8', 'c75': '#e45756', 'global25': '#59a14f'}
    labels = {'q_rate': 'Q', 'physical_execution_rate': 'physical execution',
              'target_pair_legal_rate': 'target-pair legality',
              'conflict_world_rate': 'conflict worlds'}

    # Average live-minus-silent endpoints by rule, information and schedule.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, info in zip(axes, design.INFORMATIONS):
        x = np.arange(len(design.RULES)); width = .18
        for j, key in enumerate(('q_rate', 'physical_execution_rate', 'target_pair_legal_rate', 'conflict_world_rate')):
            values = []
            for rule in design.RULES:
                seed_values = []
                for seed in design.SEEDS:
                    sched_values = []
                    for sched in design.SCHEDULES:
                        live = by[seed, rule, info, sched, True]
                        silent = by[seed, rule, info, sched, False]
                        sched_values.append(final(live, key) - final(silent, key))
                    seed_values.append(np.mean(sched_values))
                values.append(np.mean(seed_values) * 100)
            # Keep one subplot per information; bars are metrics, color encodes metric.
            ax.bar(j - 1.5 * width + np.arange(1) * 0, values[0:1], width=width, color='#4c78a8')
            ax.bar(j - .5 * width, values[1:2], width=width, color='#e45756')
            ax.bar(j + .5 * width, values[2:3], width=width, color='#59a14f')
        # redraw as grouped metrics with explicit values (the loop above stores one bar per metric)
        ax.clear()
        metric_values = {}
        for key in labels:
            metric_values[key] = [100 * np.mean([
                np.mean([final(by[seed, rule, info, sched, True], key) -
                         final(by[seed, rule, info, sched, False], key)
                         for sched in design.SCHEDULES]) for seed in design.SEEDS])
                                  for rule in design.RULES]
        xpos = np.arange(len(labels)); width = .24
        for j, rule in enumerate(design.RULES):
            ax.bar(xpos + (j - 1) * width, [metric_values[k][j] for k in labels], width,
                   label=rule, color=colors[rule])
        ax.axhline(0, color='black', linewidth=.7)
        ax.set_xticks(xpos, [labels[k] for k in labels], rotation=25, ha='right')
        ax.set_title(info)
        ax.grid(axis='y', alpha=.25)
        if info == 'PL': ax.set_ylabel('live − silent (percentage points)')
        ax.legend(frameon=False, fontsize=8)
    save(fig, directory, 'endpoint_control_contrasts')

    # The predicted global null versus the structural c75 contrast.
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    metrics = ('q_rate', 'physical_execution_rate', 'target_pair_legal_rate', 'conflict_world_rate')
    xpos = np.arange(len(metrics)); width = .36
    for j, info in enumerate(design.INFORMATIONS):
        vals = []
        for key in metrics:
            values = []
            for seed in design.SEEDS:
                values.append(np.mean([
                    (final(by[seed, 'c75', info, sched, True], key) - final(by[seed, 'c75', info, sched, False], key)) -
                    (final(by[seed, 'global25', info, sched, True], key) - final(by[seed, 'global25', info, sched, False], key))
                    for sched in design.SCHEDULES]))
            vals.append(np.mean(values) * 100)
        ax.bar(xpos + (j - .5) * width, vals, width, label=info,
               color=('#e45756' if j == 0 else '#7f7f7f'))
    ax.axhline(0, color='black', linewidth=.7)
    ax.set_xticks(xpos, ['Q', 'physical', 'target pair', 'conflict'])
    ax.set_ylabel('c75 − global25 (percentage points)')
    ax.set_title('Conflict-specific effect after global-scale control')
    ax.grid(axis='y', alpha=.25); ax.legend(frameon=False)
    save(fig, directory, 'conflict_vs_global_contrast')

    # Trajectory overlay for one representative seed; all other seeds remain in JSON.
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.5), sharex=True)
    updates = np.asarray(design.UPDATES)
    for rule in design.RULES:
        for live, style in ((True, '-'), (False, '--')):
            vals = np.mean([trajectory(by[seed, rule, 'PL', 'static', live], 'target_pair_legal_rate')
                            for seed in design.SEEDS], axis=0) * 100
            axes[0].plot(updates, vals, style, color=colors[rule], label=f'{rule} ' + ('live' if live else 'silent'))
            vals = np.mean([trajectory(by[seed, rule, 'PL', 'static', live], 'physical_execution_rate')
                            for seed in design.SEEDS], axis=0) * 100
            axes[1].plot(updates, vals, style, color=colors[rule], label=f'{rule} ' + ('live' if live else 'silent'))
    axes[0].set_title('PL target-pair legality'); axes[1].set_title('PL physical execution')
    for ax in axes:
        ax.set_xlabel('update'); ax.set_ylabel('%'); ax.grid(alpha=.25)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    save(fig, directory, 'control_trajectories')
    receipt = {'status': 'passed', 'visual_review': 'pending',
               'figures': sorted(p.name for p in directory.iterdir()),
               'source': str(Path(source).resolve())}
    (directory / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.source, args.output)
