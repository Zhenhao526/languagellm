"""Generate compact figures for the factorized directional probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import metrics


def main(results_path, output):
    results = json.loads(Path(results_path).read_text())
    rows = results.get('rows', [])
    if len(rows) != 64:
        raise ValueError('Expected 64 probe rows')
    output = Path(output); output.mkdir(parents=False, exist_ok=False)
    seeds = sorted({r['seed'] for r in rows})
    by = {(r['seed'], r['schedule'], r['live']): r for r in rows}
    transfer = []
    for seed in seeds:
        cell = {(s, live): float(np.mean([np.mean([z['by_axis'][a]['transfer'] for a in metrics.AXES]) for z in by[seed, s, live]['by_sender']]))
                for s in ('static', 'rematched') for live in (False, True)}
        transfer.append((cell['rematched', True] - cell['rematched', False]) -
                        (cell['static', True] - cell['static', False]))
    values = np.asarray(transfer) * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.3)); ax.axvline(0, color='0.35', lw=.8)
    ax.scatter(values, np.arange(len(values)), color='#386cb0')
    ax.axvline(values.mean(), color='#d95f02', lw=1.6, label=f'mean {values.mean():.2f} pp')
    ax.set_yticks(np.arange(len(values)), [str(s) for s in seeds]); ax.set_xlabel('directional transfer interaction (pp)'); ax.set_title('Factorized W1 probe: seed-level interaction'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / 'directional_transfer_forest.png', dpi=180); fig.savefig(output / 'directional_transfer_forest.pdf'); plt.close(fig)
    endpoint = []
    for schedule, live in (('static', False), ('static', True), ('rematched', False), ('rematched', True)):
        endpoint.append(float(np.mean([np.mean([z['by_axis']['kind']['transfer'], z['by_axis']['length']['transfer'], z['by_axis']['destination']['transfer']]) for z in by[seed, schedule, live]['by_sender']])) * 100)
    fig, ax = plt.subplots(figsize=(6.6, 4.2)); ax.bar(np.arange(4), endpoint, color=['#4c78a8', '#9ecae1', '#f58518', '#ffbf79'])
    ax.set_xticks(np.arange(4), ['static\nsilent', 'static\nlive', 'rematched\nsilent', 'rematched\nlive']); ax.set_ylabel('intervention transfer (pp)'); ax.set_title('Directional transfer by condition')
    fig.tight_layout(); fig.savefig(output / 'directional_transfer_conditions.png', dpi=180); fig.savefig(output / 'directional_transfer_conditions.pdf'); plt.close(fig)
    (output / 'receipt.json').write_text(json.dumps(dict(status='generated', visual_review='pending', figures=[p.name for p in output.glob('*.png')]), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status='generated', figures=[p.name for p in output.glob('*.png')]), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--results', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.results, args.output)
