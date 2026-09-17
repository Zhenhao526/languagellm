"""Plots for the exploratory neutral-action pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import design


def plot(results, output):
    output = Path(output); output.mkdir(parents=False, exist_ok=False)
    runs = results['runs']; by = {(r['seed'], r['schedule'], float(r['cancel_reward']), bool(r['live'])): r for r in runs}
    seeds = sorted(design.SEEDS); updates = np.asarray(design.UPDATES, dtype=float)
    colors = {0.0: '#386cb0', 0.1: '#d95f02'}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for cancel in design.CANCEL_REWARDS:
        curves = []
        for seed in seeds:
            values = []
            for i in range(len(updates)):
                live = np.mean([by[seed, schedule, float(cancel), True]['trajectory'][i]['target_trajectory']['conditional_q_rate'] for schedule in design.SCHEDULES])
                silent = np.mean([by[seed, schedule, float(cancel), False]['trajectory'][i]['target_trajectory']['conditional_q_rate'] for schedule in design.SCHEDULES])
                values.append(live - silent)
            curves.append(values)
        curves = np.asarray(curves) * 100
        mean = curves.mean(axis=0); se = curves.std(axis=0, ddof=1) / np.sqrt(len(seeds))
        ax.plot(updates, mean, marker='o', color=colors[float(cancel)], label=f'cancel={cancel:.2f}')
        ax.fill_between(updates, mean - 2.0 * se, mean + 2.0 * se, color=colors[float(cancel)], alpha=.16)
    ax.axhline(0, color='0.35', lw=.8); ax.set(xlabel='updates', ylabel='live − silent conditional Q (pp)', title='Communication effect with an explicit neutral outcome'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / 'conditional_q_by_cancel.png', dpi=180); fig.savefig(output / 'conditional_q_by_cancel.pdf'); plt.close(fig)

    labels = []; physical = []; neutral = []
    for schedule in design.SCHEDULES:
        for cancel in design.CANCEL_REWARDS:
            for live in (False, True):
                vals = [by[seed, schedule, float(cancel), live]['final']['new_layouts'] for seed in seeds]
                labels.append(f'{schedule}\n{("live" if live else "silent")}\ncancel={cancel:.2f}')
                physical.append(100 * np.mean([v['physical_execution_rate'] for v in vals]))
                neutral.append(100 * np.mean([v['cancel_protocol_rate'] for v in vals]))
    x = np.arange(len(labels)); width = .36
    fig, ax = plt.subplots(figsize=(10.2, 4.8)); ax.bar(x - width / 2, physical, width, label='physical execution', color='#9ecae1'); ax.bar(x + width / 2, neutral, width, label='all-cancel protocol', color='#fdae6b')
    ax.set_xticks(x, labels); ax.set_ylabel('final new-layout rate (%)'); ax.set_title('Execution and explicit neutral protocol at the final checkpoint'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / 'execution_neutral_final.png', dpi=180); fig.savefig(output / 'execution_neutral_final.pdf'); plt.close(fig)
    receipt = dict(status='generated', visual_review='pending', source=str(results.get('source', 'execution/results.json')), figures=[p.name for p in output.glob('*.png')])
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--results', required=True); parser.add_argument('--output', required=True); args = parser.parse_args()
    plot(json.loads(Path(args.results).read_text()), args.output)
