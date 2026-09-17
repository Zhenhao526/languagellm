"""Plot compact multi-legal-plan results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import metrics


def load(path):
    data = json.loads(Path(path).read_text());
    if 'runs' not in data: raise ValueError('Results must contain runs')
    return data


def plot(results, output):
    output = Path(output); output.mkdir(parents=False, exist_ok=False); runs = results['runs']; by = {(r['seed'], r['live']): r for r in runs}; updates = metrics.UPDATES
    q = np.asarray([[row['target_trajectory']['q_rate'] for row in sorted(by[seed, live]['trajectory'], key=lambda x: x['update'])] for seed in sorted({r['seed'] for r in runs}) for live in (False, True)])
    q_pairs = q[1::2] - q[0::2]; mean = q_pairs.mean(0) * 100; lo, hi = np.percentile(q_pairs, [2.5, 97.5], axis=0) * 100
    fig, ax = plt.subplots(figsize=(7.4, 4.5)); ax.axhline(0, color='0.35', lw=.8); ax.plot(updates, mean, marker='o', label='team Q (live − silent)'); ax.fill_between(updates, lo, hi, alpha=.18); ax.set(xlabel='updates', ylabel='communication effect (pp)', title='Two-legal-plan coordination: trajectories'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(output/'multi_legal_trajectories.png', dpi=180); fig.savefig(output/'multi_legal_trajectories.pdf'); plt.close(fig)
    rows = results['primary']['primary']['by_seed']; vals = np.asarray([r['q_centered_AUC'] for r in rows]) * 100; fig, ax = plt.subplots(figsize=(7.4, 4.5)); y=np.arange(len(vals)); ax.axvline(0,color='0.35',lw=.8); ax.scatter(vals,y,color='#386cb0'); ax.axvline(vals.mean(),color='#d95f02',lw=1.6,label=f'mean {vals.mean():.2f}'); ax.set_yticks(y,[str(r['seed']) for r in rows]); ax.set(xlabel='Q live − silent centered time AUC (pp)',title='Two-legal-plan coordination: seed-level Q'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(output/'multi_legal_seed_forest.png',dpi=180); fig.savefig(output/'multi_legal_seed_forest.pdf'); plt.close(fig)
    cells=[]
    for live in (False,True): cells.append(np.mean([r['trajectory'][-1]['target_trajectory']['q_rate'] for r in runs if r['live']==live])*100)
    fig, ax=plt.subplots(figsize=(6.4,4.2)); ax.bar([0,1],cells,color=['#4c78a8','#f58518']); ax.set_xticks([0,1],['silent','live']); ax.set_ylabel('team Q (%)'); ax.set_title('Two-legal-plan endpoint at 6000 updates'); fig.tight_layout(); fig.savefig(output/'multi_legal_endpoint.png',dpi=180); fig.savefig(output/'multi_legal_endpoint.pdf'); plt.close(fig)
    receipt=dict(status='generated',visual_review='pending',source=str(Path(results.get('source','execution/results.json'))),figures=[p.name for p in output.glob('*.png')]); (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(receipt,ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--results',required=True); parser.add_argument('--output',required=True); args=parser.parse_args(); plot(load(args.results),args.output)
