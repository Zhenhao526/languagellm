"""Generate compact FI pilot trajectory and endpoint figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(results_path, output):
    data = json.loads(Path(results_path).read_text()); runs = data['runs']
    if len(runs) != 32: raise ValueError('Expected 32 FI runs')
    output = Path(output); output.mkdir(parents=False, exist_ok=False)
    updates = np.asarray([row['update'] for row in runs[0]['trajectory']], dtype=float)
    colors = {('static', False): '#4c78a8', ('static', True): '#9ecae1',
              ('rematched', False): '#f58518', ('rematched', True): '#ffbf79'}
    labels = {('static', False): 'static silent', ('static', True): 'static live',
              ('rematched', False): 'rematched silent', ('rematched', True): 'rematched live'}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
    for metric, ax, ylabel in (('target_pair_legal_rate', axes[0], 'target-pair legal rate | physical'),
                               ('physical_execution_rate', axes[1], 'physical execution rate')):
        for schedule in ('static', 'rematched'):
            for live in (False, True):
                vals = np.asarray([[row['target_trajectory'][metric] for row in run['trajectory']]
                                   for run in runs if run['schedule'] == schedule and run['live'] == live])
                ax.plot(updates, vals.mean(axis=0) * 100, marker='o', lw=1.6,
                        color=colors[schedule, live], label=labels[schedule, live])
        ax.set_xlabel('update'); ax.set_ylabel(ylabel + ' (%, mean)')
        ax.grid(alpha=.2)
    axes[0].set_title('FI target-pair choice'); axes[1].set_title('FI execution')
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(output / 'fi_capacity_trajectories.png', dpi=180)
    fig.savefig(output / 'fi_capacity_trajectories.pdf'); plt.close(fig)

    final_values = []
    for schedule, live in (('static', False), ('static', True), ('rematched', False), ('rematched', True)):
        final_values.append(float(np.mean([
            run['final']['new_layouts']['target_pair_legal_rate'] * 100
            for run in runs if run['schedule'] == schedule and run['live'] == live
        ])))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(np.arange(4), final_values, color=[colors[s, l] for s, l in
           (('static', False), ('static', True), ('rematched', False), ('rematched', True))])
    ax.set_xticks(np.arange(4), ['static\nsilent', 'static\nlive', 'rematched\nsilent', 'rematched\nlive'])
    ax.set_ylabel('target-pair legal rate | physical (%)'); ax.set_title('FI final new-layout capacity')
    fig.tight_layout(); fig.savefig(output / 'fi_capacity_endpoint.png', dpi=180)
    fig.savefig(output / 'fi_capacity_endpoint.pdf'); plt.close(fig)
    (output / 'receipt.json').write_text(json.dumps(dict(status='generated', visual_review='pending',
                                                         figures=[p.name for p in output.glob('*.png')]),
                                                    ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status='generated', figures=[p.name for p in output.glob('*.png')]), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--results', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.results, args.output)
