"""Generate compact figures for the compound partner-direction probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _seed_mean(row, key):
    return float(np.mean([float(sender[key]) for sender in row['by_sender']]))


def main(results_path, output):
    results = json.loads(Path(results_path).read_text())
    rows = results.get('rows', [])
    if len(rows) != 64:
        raise ValueError('Expected 64 probe rows')
    output = Path(output); output.mkdir(parents=False, exist_ok=False)
    seeds = sorted({r['seed'] for r in rows})
    by = {(r['seed'], r['schedule'], r['live']): r for r in rows}

    partner_interactions = []
    plan_interactions = []
    for seed in seeds:
        partner = {(schedule, live): _seed_mean(by[seed, schedule, live], 'partner_transfer')
                   for schedule in ('static', 'rematched') for live in (False, True)}
        plan = {(schedule, live): _seed_mean(by[seed, schedule, live], 'plan_transfer')
                for schedule in ('static', 'rematched') for live in (False, True)}
        partner_interactions.append((partner['rematched', True] - partner['rematched', False]) -
                                    (partner['static', True] - partner['static', False]))
        plan_interactions.append((plan['rematched', True] - plan['rematched', False]) -
                                 (plan['static', True] - plan['static', False]))

    values = np.asarray(partner_interactions) * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.3)); ax.axvline(0, color='0.35', lw=.8)
    ax.scatter(values, np.arange(len(values)), color='#386cb0')
    ax.axvline(values.mean(), color='#d95f02', lw=1.6, label=f'mean {values.mean():.2f} pp')
    ax.set_yticks(np.arange(len(values)), [str(seed) for seed in seeds])
    ax.set_xlabel('partner-edge directional interaction (pp)')
    ax.set_title('Compound W1 probe: seed-level partner response')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / 'partner_direction_forest.png', dpi=180)
    fig.savefig(output / 'partner_direction_forest.pdf'); plt.close(fig)

    labels = ['static\nsilent', 'static\nlive', 'rematched\nsilent', 'rematched\nlive']
    condition_values = []
    for schedule, live in (('static', False), ('static', True), ('rematched', False), ('rematched', True)):
        condition_values.append(float(np.mean([
            _seed_mean(by[seed, schedule, live], 'partner_transfer') for seed in seeds
        ])) * 100)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(np.arange(4), condition_values,
           color=['#4c78a8', '#9ecae1', '#f58518', '#ffbf79'])
    ax.set_xticks(np.arange(4), labels); ax.set_ylabel('partner-edge transfer (pp)')
    ax.set_title('Compound partner response by condition')
    fig.tight_layout(); fig.savefig(output / 'partner_direction_conditions.png', dpi=180)
    fig.savefig(output / 'partner_direction_conditions.pdf'); plt.close(fig)

    (output / 'receipt.json').write_text(json.dumps(
        dict(status='generated', visual_review='pending',
             figures=[p.name for p in output.glob('*.png')]),
        ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status='generated', figures=[p.name for p in output.glob('*.png')]), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--results', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    main(args.results, args.output)
