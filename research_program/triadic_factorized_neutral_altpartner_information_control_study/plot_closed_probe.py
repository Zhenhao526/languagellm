"""Plots for the same-policy closed-channel intervention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(results_path, output):
    output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    data = json.loads(Path(results_path).read_text()); rows = data['rows']
    groups = [('static', 'live'), ('rematched', 'live'), ('static', 'own'), ('rematched', 'own')]
    labels = ['static\nlive-trained', 'rematched\nlive-trained', 'static\nown-trained', 'rematched\nown-trained']
    def mean(mode, key, schedule, trained):
        selected = [r for r in rows if r['schedule'] == schedule and r['trained_channel'] == trained]
        return float(np.mean([r[mode][key] for r in selected]) * 100)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))
    for ax, key, ylabel in zip(axes, ('q_rate', 'physical_execution_rate'), ('team Q (%)', 'physical execution (%)')):
        x = np.arange(4); width = .36
        natural = [mean('natural', key, *g) for g in groups]; closed = [mean('closed', key, *g) for g in groups]
        ax.bar(x - width / 2, natural, width, label='natural', color='#386cb0')
        ax.bar(x + width / 2, closed, width, label='closed', color='#fdb462')
        ax.set_xticks(x, labels); ax.set_ylabel(ylabel); ax.set_ylim(0, 100); ax.grid(axis='y', alpha=.18)
    axes[0].legend(frameon=False, loc='upper left'); fig.suptitle('Same-policy cross-agent channel closure'); fig.tight_layout()
    fig.savefig(output / 'closed_channel_endpoint.png', dpi=180); fig.savefig(output / 'closed_channel_endpoint.pdf'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.5)); x = np.arange(4); width = .24
    keys = [('q_rate', 'Q'), ('physical_execution_rate', 'physical'), ('target_pair_legal_rate', 'target pair')]
    for j, (key, label) in enumerate(keys):
        values = [mean('natural', key, *g) - mean('closed', key, *g) for g in groups]
        ax.bar(x + (j - 1) * width, values, width, label=label)
    ax.axhline(0, color='0.35', lw=.8); ax.set_xticks(x, labels); ax.set_ylabel('natural − closed (pp)'); ax.set_title('Effect of closing cross-agent routing'); ax.legend(frameon=False, ncol=3); ax.grid(axis='y', alpha=.18)
    fig.tight_layout(); fig.savefig(output / 'closed_channel_differences.png', dpi=180); fig.savefig(output / 'closed_channel_differences.pdf'); plt.close(fig)
    receipt = dict(status='generated', visual_review='pending', source=str(Path(results_path).resolve()), figures=[p.name for p in output.glob('*.png')])
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--results', required=True); parser.add_argument('--output', required=True); args = parser.parse_args(); main(args.results, args.output)
