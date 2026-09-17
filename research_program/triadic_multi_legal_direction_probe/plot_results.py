"""Plot the posthoc directional probe summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(results, output):
    data = json.loads(Path(results).read_text()); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = data['primary']['seed_rows']
    live = np.asarray([r['plan_transfer_live'] for r in rows]) * 100
    silent = np.asarray([r['plan_transfer_silent'] for r in rows]) * 100
    fig, ax = plt.subplots(figsize=(8, 4.8)); x = np.arange(16)
    ax.axhline(0, color='0.4', lw=0.8); ax.plot(x, live, 'o-', label='live', color='#1f77b4'); ax.plot(x, silent, 'o-', label='silent', color='#d62728')
    ax.set(xlabel='paired seed index', ylabel='plan transfer (percentage points)', title='Directional W1 transfer on new layouts')
    ax.legend(frameon=False); ax.grid(axis='y', alpha=.25); fig.tight_layout(); fig.savefig(output / 'direction_seed_trajectories.png', dpi=180); fig.savefig(output / 'direction_seed_trajectories.pdf'); plt.close(fig)
    sender = data['primary']['sender_statistics']; means = [row['mean_percentage_points'] for row in sender]; lo = [100 * row['plan_transfer']['ci95_lower'] for row in sender]; hi = [100 * row['plan_transfer']['ci95_upper'] for row in sender]
    fig, ax = plt.subplots(figsize=(6, 4.4)); xx = np.arange(3); ax.axhline(0, color='0.4', lw=.8); ax.errorbar(xx, means, yerr=[np.asarray(means)-lo, np.asarray(hi)-means], fmt='o', capsize=4, color='#1f77b4'); ax.set_xticks(xx, ['sender A', 'sender B', 'sender C']); ax.set_ylabel('live − silent transfer (percentage points)'); ax.set_title('Sender-specific directional effect'); ax.grid(axis='y', alpha=.25); fig.tight_layout(); fig.savefig(output / 'direction_sender_forest.png', dpi=180); fig.savefig(output / 'direction_sender_forest.pdf'); plt.close(fig)
    receipt = dict(status='passed', results=str(Path(results).resolve()), figures=['direction_seed_trajectories.png', 'direction_sender_forest.png'], visual_review='pending')
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--results', required=True); parser.add_argument('--output', required=True); args = parser.parse_args(); print(json.dumps(plot(args.results, args.output), ensure_ascii=False))
