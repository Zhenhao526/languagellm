"""Plot compact partner-holdout results without reading model checkpoints."""
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
    by = {(r['seed'], r['payoff'], r['rule'], r['live']): r for r in runs}
    series = []
    for seed in sorted({r['seed'] for r in runs}):
        for rule in ('strict', 'reciprocal'):
            cells = {}
            for payoff in ('partial', 'all_or_nothing'):
                for live in (False, True):
                    rows = sorted(by[seed, payoff, rule, live]['trajectory'], key=lambda x: x['update'])
                    cells[payoff, live] = np.array([row['target_trajectory'][key] for row in rows])
            series.append(np.array([(cells['all_or_nothing', True][i] - cells['all_or_nothing', False][i]) -
                                    (cells['partial', True][i] - cells['partial', False][i]) for i in range(6)]))
    return np.array(series)


def plot(results, output):
    output = Path(output); output.mkdir(parents=False, exist_ok=False)
    runs = results['runs']; updates = metrics.UPDATES
    actor = interaction_series(runs, 'target_pair_actor_action_rate')
    q = interaction_series(runs, 'q_rate')
    actor_mean, actor_lo, actor_hi = actor.mean(0) * 100, np.percentile(actor, 2.5, axis=0) * 100, np.percentile(actor, 97.5, axis=0) * 100
    q_mean, q_lo, q_hi = q.mean(0) * 100, np.percentile(q, 2.5, axis=0) * 100, np.percentile(q, 97.5, axis=0) * 100
    fig, ax = plt.subplots(figsize=(7.4, 4.5)); ax.axhline(0, color='0.35', lw=.8)
    ax.plot(updates, actor_mean, marker='o', label='held-out pair actor exact-action rate')
    ax.fill_between(updates, actor_lo, actor_hi, alpha=.16)
    ax.plot(updates, q_mean, marker='s', label='team Q (both endpoints)')
    ax.fill_between(updates, q_lo, q_hi, alpha=.12)
    ax.set(xlabel='updates', ylabel='payoff × communication interaction (pp)', title='Partner-pair holdout: training trajectories')
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(output/'partner_holdout_trajectories.png', dpi=180); fig.savefig(output/'partner_holdout_trajectories.pdf'); plt.close(fig)

    primary = results['primary']['primary']; rows = primary['by_seed']; vals = np.array([r['primary_centered_AUC'] for r in rows]) * 100
    fig, ax = plt.subplots(figsize=(7.4, 4.5)); y = np.arange(len(vals)); ax.axvline(0, color='0.35', lw=.8)
    ax.scatter(vals, y, color='#386cb0'); ax.axvline(vals.mean(), color='#d95f02', lw=1.6, label=f'mean {vals.mean():.2f}')
    ax.set_yticks(y, [f"{r['seed']} ({r['heldout_pair_name']})" for r in rows]); ax.set(xlabel='primary centered time AUC (pp)', title='Partner-pair holdout: seed-level primary'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(output/'partner_holdout_seed_forest.png', dpi=180); fig.savefig(output/'partner_holdout_seed_forest.pdf'); plt.close(fig)

    # Endpoint cells from the compact target trajectories, averaged over seeds.
    cells = {}
    for payoff in ('partial', 'all_or_nothing'):
        for rule in ('strict', 'reciprocal'):
            for live in (False, True):
                vals = [r['trajectory'][-1]['target_trajectory']['target_pair_actor_action_rate'] for r in runs if r['payoff'] == payoff and r['rule'] == rule and r['live'] == live]
                cells[payoff, rule, live] = np.mean(vals) * 100
    labels = ['partial\nstrict', 'partial\nreciprocal', 'all-or-nothing\nstrict', 'all-or-nothing\nreciprocal']; keys = [('partial','strict'),('partial','reciprocal'),('all_or_nothing','strict'),('all_or_nothing','reciprocal')]
    x = np.arange(4); width=.36
    fig, ax = plt.subplots(figsize=(7.4, 4.5)); ax.bar(x-width/2, [cells[k+(False,)] for k in keys], width, label='silent'); ax.bar(x+width/2, [cells[k+(True,)] for k in keys], width, label='live')
    ax.set_xticks(x, labels); ax.set_ylabel('held-out pair actor exact-action rate (%)'); ax.set_title('Partner-pair holdout: endpoint at 6000 updates'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(output/'partner_holdout_endpoint.png', dpi=180); fig.savefig(output/'partner_holdout_endpoint.pdf'); plt.close(fig)
    receipt = dict(status='generated', visual_review='pending', source=str(Path(results.get('source', 'execution/results.json'))), figures=[p.name for p in output.glob('*.png')])
    (output/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--results', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); plot(load(args.results), args.output)
