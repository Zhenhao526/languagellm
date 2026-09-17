"""Plots for the pre-registered conditional-Q interaction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import metrics


def load(path):
    data = json.loads(Path(path).read_text())
    if 'runs' not in data:
        raise ValueError('Results must contain runs')
    return data


def plot(results, output):
    output = Path(output); output.mkdir(parents=False, exist_ok=False); runs = results['runs']; seeds = sorted({r['seed'] for r in runs})
    by = {(r['seed'], r['schedule'], r['live']): r for r in runs}; updates = metrics.UPDATES
    interactions = []
    for seed in seeds:
        q = {(s, live): np.asarray([row['target_trajectory']['conditional_q_rate'] for row in sorted(by[seed, s, live]['trajectory'], key=lambda x: x['update'])])
             for s in ('static', 'rematched') for live in (False, True)}
        interactions.append((q['rematched', True] - q['rematched', False]) - (q['static', True] - q['static', False]))
    values = np.asarray(interactions); mean = values.mean(0) * 100; lo, hi = np.percentile(values, [2.5, 97.5], axis=0) * 100
    fig, ax = plt.subplots(figsize=(7.4, 4.5)); ax.axhline(0, color='0.35', lw=.8)
    ax.plot(updates, mean, marker='o', label='conditional Q interaction')
    ax.fill_between(updates, lo, hi, alpha=.18); ax.set(xlabel='updates', ylabel='interaction (pp)',
        title='Multi-legal coordination: rematching × communication'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / 'conditional_q_interaction_trajectories.png', dpi=180); fig.savefig(output / 'conditional_q_interaction_trajectories.pdf'); plt.close(fig)
    auc = values.mean(1) if False else np.asarray([metrics.centered_auc(v) for v in values]) * 100
    fig, ax = plt.subplots(figsize=(7.4, 4.5)); y = np.arange(len(auc)); ax.axvline(0, color='0.35', lw=.8)
    ax.scatter(auc, y, color='#386cb0'); ax.axvline(auc.mean(), color='#d95f02', lw=1.6, label=f'mean {auc.mean():.2f}')
    ax.set_yticks(y, [str(s) for s in seeds]); ax.set(xlabel='conditional Q interaction centered time AUC (pp)', title='Seed-level interaction'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / 'conditional_q_interaction_forest.png', dpi=180); fig.savefig(output / 'conditional_q_interaction_forest.pdf'); plt.close(fig)
    endpoint = [float(np.mean([by[s, schedule, live]['final']['new_layouts']['conditional_q_rate'] for s in seeds])) * 100 for schedule, live in (("static", False), ("static", True), ("rematched", False), ("rematched", True))]
    fig, ax = plt.subplots(figsize=(6.6, 4.2)); ax.bar(np.arange(4), endpoint, color=['#4c78a8', '#9ecae1', '#f58518', '#ffbf79'])
    ax.set_xticks(np.arange(4), ['static\nsilent', 'static\nlive', 'rematched\nsilent', 'rematched\nlive']); ax.set_ylabel('conditional Q (%)'); ax.set_title('New-layout endpoint')
    fig.tight_layout(); fig.savefig(output / 'conditional_q_endpoint.png', dpi=180); fig.savefig(output / 'conditional_q_endpoint.pdf'); plt.close(fig)
    receipt = dict(status='generated', visual_review='pending', source=str(results.get('source', 'execution/results.json')),
                   figures=[p.name for p in output.glob('*.png')])
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--results', required=True); parser.add_argument('--output', required=True); args = parser.parse_args(); plot(load(args.results), args.output)
