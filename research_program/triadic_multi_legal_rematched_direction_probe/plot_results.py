"""Plots for the posthoc four-arm directional probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import metrics


def plot(results, output):
    output = Path(output); output.mkdir(parents=False, exist_ok=False)
    rows = results['rows']; seeds = sorted({row['seed'] for row in rows})
    by = {(row['seed'], row['schedule'], row['live']): row for row in rows}
    effects = []
    for seed in seeds:
        cells = {(schedule, live): float(np.mean([x['plan_transfer'] for x in by[seed, schedule, live]['by_sender']]))
                 for schedule in ('static', 'rematched') for live in (False, True)}
        effects.append((cells['rematched', True] - cells['rematched', False]) -
                       (cells['static', True] - cells['static', False]))
    effects = np.asarray(effects) * 100
    fig, ax = plt.subplots(figsize=(7.4, 4.5)); y = np.arange(len(effects)); ax.axvline(0, color='0.35', lw=.8)
    ax.scatter(effects, y, color='#386cb0'); ax.axvline(effects.mean(), color='#d95f02', lw=1.6, label=f'mean {effects.mean():.2f}')
    ax.set_yticks(y, [str(seed) for seed in seeds]); ax.set(xlabel='directional plan-transfer interaction (pp)', title='Rematching × communication directional probe'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / 'directional_interaction_forest.png', dpi=180); fig.savefig(output / 'directional_interaction_forest.pdf'); plt.close(fig)
    labels = ['static\nsilent', 'static\nlive', 'rematched\nsilent', 'rematched\nlive']
    cells = [float(np.mean([x['plan_transfer'] for row in rows if row['schedule'] == schedule and row['live'] is live for x in row['by_sender']])) * 100
             for schedule, live in (('static', False), ('static', True), ('rematched', False), ('rematched', True))]
    fig, ax = plt.subplots(figsize=(6.8, 4.2)); ax.bar(np.arange(4), cells, color=['#4c78a8', '#9ecae1', '#f58518', '#ffbf79'])
    ax.set_xticks(np.arange(4), labels); ax.set_ylabel('signed plan transfer (pp)'); ax.set_title('Natural directional transfer by condition')
    fig.tight_layout(); fig.savefig(output / 'directional_condition_bars.png', dpi=180); fig.savefig(output / 'directional_condition_bars.pdf'); plt.close(fig)
    receipt = dict(status='generated', visual_review='pending', source=str(results.get('source', 'execution/results.json')),
                   figures=[p.name for p in output.glob('*.png')])
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--results', required=True); parser.add_argument('--output', required=True); args = parser.parse_args()
    plot(json.loads(Path(args.results).read_text()), args.output)
