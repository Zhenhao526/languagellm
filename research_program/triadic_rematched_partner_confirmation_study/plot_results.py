"""Plot compact rematching results without reading checkpoints."""
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


def interaction_series(runs, key):
    by = {(r['seed'], r['schedule'], r['live']): r for r in runs}; rows = []
    for seed in sorted({r['seed'] for r in runs}):
        values = {(schedule, live): metrics._values(by, seed, schedule, live, key)
                  for schedule in ('static', 'rematched') for live in (False, True)}
        rows.append((values['rematched', True] - values['rematched', False]) -
                    (values['static', True] - values['static', False]))
    return np.asarray(rows)


def plot(results, output):
    output = Path(output); output.mkdir(parents=False, exist_ok=False); runs = results['runs']
    actor = interaction_series(runs, 'target_pair_actor_action_rate'); q = interaction_series(runs, 'q_rate')
    updates = metrics.UPDATES; actor_mean = actor.mean(0) * 100; q_mean = q.mean(0) * 100
    actor_lo, actor_hi = np.percentile(actor, 2.5, axis=0) * 100, np.percentile(actor, 97.5, axis=0) * 100
    q_lo, q_hi = np.percentile(q, 2.5, axis=0) * 100, np.percentile(q, 97.5, axis=0) * 100
    fig, ax = plt.subplots(figsize=(7.4, 4.5)); ax.axhline(0, color='0.35', lw=.8)
    ax.plot(updates, actor_mean, marker='o', label='target-pair actor exact-action rate')
    ax.fill_between(updates, actor_lo, actor_hi, alpha=.16)
    ax.plot(updates, q_mean, marker='s', label='team Q (both endpoints)'); ax.fill_between(updates, q_lo, q_hi, alpha=.12)
    ax.set(xlabel='updates', ylabel='rematched × communication interaction (pp)', title='Random partner rematching: trajectories')
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(output / 'rematching_trajectories.png', dpi=180); fig.savefig(output / 'rematching_trajectories.pdf'); plt.close(fig)

    rows = results['primary']['primary']['by_seed']; vals = np.asarray([r['primary_centered_AUC'] for r in rows]) * 100
    fig, ax = plt.subplots(figsize=(7.4, 4.5)); y = np.arange(len(vals)); ax.axvline(0, color='0.35', lw=.8)
    ax.scatter(vals, y, color='#386cb0'); ax.axvline(vals.mean(), color='#d95f02', lw=1.6, label=f'mean {vals.mean():.2f}')
    ax.set_yticks(y, [f"{r['seed']} ({r['heldout_pair_name']})" for r in rows]); ax.set(xlabel='primary centered time AUC (pp)', title='Rematching × communication: seed-level primary'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / 'rematching_seed_forest.png', dpi=180); fig.savefig(output / 'rematching_seed_forest.pdf'); plt.close(fig)

    cells = {}
    for schedule in ('static', 'rematched'):
        for live in (False, True):
            cells[schedule, live] = np.mean([r['trajectory'][-1]['target_trajectory']['target_pair_actor_action_rate'] for r in runs if r['schedule'] == schedule and r['live'] == live]) * 100
    labels = ['static\nsilent', 'static\nlive', 'rematched\nsilent', 'rematched\nlive']; values = [cells['static', False], cells['static', True], cells['rematched', False], cells['rematched', True]]
    fig, ax = plt.subplots(figsize=(7.4, 4.5)); ax.bar(np.arange(4), values, color=['#4c78a8', '#f58518', '#4c78a8', '#f58518']); ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel('held-out pair actor exact-action rate (%)'); ax.set_title('Held-out pair endpoint at 6000 updates'); fig.tight_layout(); fig.savefig(output / 'rematching_endpoint.png', dpi=180); fig.savefig(output / 'rematching_endpoint.pdf'); plt.close(fig)
    receipt = dict(status='generated', visual_review='pending', source=str(Path(results.get('source', 'execution/results.json'))), figures=[p.name for p in output.glob('*.png')])
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--results', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); plot(load(args.results), args.output)
