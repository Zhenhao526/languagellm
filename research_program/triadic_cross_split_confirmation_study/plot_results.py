"""Plots for the JSON-only summary of the bidirectional confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


UPDATES = np.array([0, 100, 500, 1500, 3000, 6000], dtype=float)
POSITIONS = np.arange(len(UPDATES), dtype=float)


def load(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def pp(value):
    return 100.0 * np.asarray(value, dtype=float)


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial Unicode MS', 'DejaVu Sans'],
        'figure.dpi': 140,
        'savefig.dpi': 220,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
    })


def save(fig, base):
    fig.savefig(base.with_suffix('.png'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_trajectories(summary, base):
    rows = summary['primary']['by_seed']
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    colors = {'strict': '#4472C4', 'reciprocal': '#C55A11'}
    for ax, measure, title in zip(axes, ('Q', 'partner'), ('cross-split Q', 'destination partner rate')):
        for rule in ('strict', 'reciprocal'):
            curves = []
            for row in rows:
                # Q is symmetric under edge reversal because it is an AND of
                # the two endpoint successes. Use train→heldout as the
                # representative direction and state this in the caption.
                item = row['directions']['train_to_heldout'][measure][rule]
                curves.append(np.asarray(item['interaction'], dtype=float))
            curves = np.asarray(curves)
            mean = curves.mean(axis=0)
            se = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
            ax.plot(POSITIONS, pp(mean), color=colors[rule], marker='o', linewidth=1.8, label=rule)
            ax.fill_between(POSITIONS, pp(mean - 2.13145 * se), pp(mean + 2.13145 * se), color=colors[rule], alpha=0.16)
        ax.axhline(0, color='#333333', linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(POSITIONS, [str(int(x)) for x in UPDATES])
        ax.set_xlabel('training updates')
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel('payoff × communication interaction (percentage points)')
    fig.suptitle('Bidirectional cross-split interaction over training\n(shaded bands: approximate t15 pointwise bands across seeds)', y=1.03)
    fig.tight_layout()
    save(fig, base)


def plot_summary(summary, base):
    primary = summary['primary']
    partner = summary['partner_secondary']
    labels = ['Q AUC', 'partner AUC', 'Q endpoint', 'partner endpoint']
    values = np.array([
        pp(primary['mean_centered_AUC']),
        pp(partner['mean_centered_AUC']),
        pp(summary['endpoint']['q_interaction']),
        pp(summary['endpoint']['partner_interaction']),
    ])
    lows = np.array([
        pp(primary['statistics']['ci95_lower']),
        pp(partner['statistics']['ci95_lower']),
        pp(summary['endpoint']['q_statistics']['ci95_lower']),
        pp(summary['endpoint']['partner_statistics']['ci95_lower']),
    ])
    highs = np.array([
        pp(primary['statistics']['ci95_upper']),
        pp(partner['statistics']['ci95_upper']),
        pp(summary['endpoint']['q_statistics']['ci95_upper']),
        pp(summary['endpoint']['partner_statistics']['ci95_upper']),
    ])
    errors = np.vstack((values - lows, highs - values))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(4)
    ax.bar(x, values, yerr=errors, capsize=4, color=['#4472C4', '#70AD47', '#9DC3E6', '#A9D18E'], edgecolor='#333333', linewidth=0.6)
    ax.axhline(0, color='#333333', linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel('interaction (percentage points)\nmean ± approximate t15 95% CI')
    ax.grid(axis='y', alpha=0.25)
    for i, value in enumerate(values):
        ax.text(i, value + 1.0, f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_title('Bidirectional cross-split confirmation summary')
    fig.tight_layout()
    save(fig, base)


def plot_seeds(summary, base):
    rows = sorted(summary['primary']['by_seed'], key=lambda row: row['seed'])
    x = np.array([pp(row['q_centered_AUC']) for row in rows])
    y = np.arange(len(rows))
    strict = np.array([pp(np.mean([row['directions'][d]['Q']['strict']['centered_AUC'] for d in ('train_to_heldout', 'heldout_to_train')])) for row in rows])
    reciprocal = np.array([pp(np.mean([row['directions'][d]['Q']['reciprocal']['centered_AUC'] for d in ('train_to_heldout', 'heldout_to_train')])) for row in rows])
    mean = pp(summary['primary']['statistics']['mean'])
    lo = pp(summary['primary']['statistics']['ci95_lower'])
    hi = pp(summary['primary']['statistics']['ci95_upper'])
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.axvspan(lo, hi, color='#D9EAD3', alpha=0.75, label='mean CI')
    ax.axvline(mean, color='#38761D', linewidth=1.4, label=f'mean {mean:.1f} pp')
    ax.axvline(0, color='#333333', linewidth=0.8)
    ax.scatter(strict, y, color='#4472C4', s=28, label='strict')
    ax.scatter(reciprocal, y, color='#C55A11', marker='s', s=28, label='reciprocal')
    ax.scatter(x, y, color='#222222', marker='|', s=120, linewidths=1.5, label='seed mean')
    ax.set_yticks(y, [str(row['seed']) for row in rows])
    ax.set_xlabel('cross-split Q time AUC (percentage points)')
    ax.set_ylabel('independent seed')
    ax.grid(axis='x', alpha=0.25)
    ax.set_title('Q interaction is positive in all 16 independent seed blocks')
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout()
    save(fig, base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    data = load(args.results)
    summary = data['primary']
    plot_trajectories(summary, args.output / 'cross_split_trajectories')
    plot_summary(summary, args.output / 'interaction_summary')
    plot_seeds(summary, args.output / 'q_seed_forest')
    receipt = {
        'status': 'generated',
        'results': str(args.results),
        'outputs': ['cross_split_trajectories.png', 'cross_split_trajectories.pdf', 'interaction_summary.png', 'interaction_summary.pdf', 'q_seed_forest.png', 'q_seed_forest.pdf'],
        'model_forwards': 0,
        'optimizer_updates': 0,
    }
    (args.output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
