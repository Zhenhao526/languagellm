"""Produce a compact scientific plot and paired summaries of the actual runs."""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='pilot_001')
    args = parser.parse_args()
    folder = ROOT / 'results' / args.name
    results = json.loads((folder / 'results.json').read_text())
    config = json.loads((folder / 'config.json').read_text())
    paired = []
    for seed in config['seeds']:
        a = next(x for x in results if x['seed'] == seed and x['condition'] == 'communicate')
        b = next(x for x in results if x['seed'] == seed and x['condition'] == 'silent')
        paired.append({'seed': seed, 'communication': a['normal']['mean_reward_per_step'],
                       'silent': b['normal']['mean_reward_per_step'],
                       'shuffled': a['shuffle']['mean_reward_per_step'],
                       'communication_minus_silent': a['normal']['mean_reward_per_step'] - b['normal']['mean_reward_per_step'],
                       'communication_minus_shuffled': a['normal']['mean_reward_per_step'] - a['shuffle']['mean_reward_per_step'],
                       'stochastic_communication': a['stochastic']['mean_reward_per_step'],
                       'stochastic_silent': b['stochastic']['mean_reward_per_step'],
                       'training_image_communication': a['train_images']['mean_reward_per_step']})
    (folder / 'paired_summary.json').write_text(json.dumps(paired, indent=2))
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    colors = ['#2563eb', '#0d9488', '#a855f7']
    for idx, seed in enumerate(config['seeds']):
        for condition, style in [('communicate', '-'), ('silent', '--')]:
            curve = json.loads((folder / f'{condition}_s{seed}' / 'learning_curve.json').read_text())
            axes[0].plot([r['update'] for r in curve], [r['normal']['mean_reward_per_step'] for r in curve],
                         style, color=colors[idx % 3], label=f'{seed}: {condition}')
    axes[0].axhline(5/7, color='#64748b', linewidth=1, linestyle=':', label='No-message expected upper bound')
    axes[0].set(xlabel='Training updates', ylabel='Held-out resource success', ylim=(.35, 1.02))
    axes[0].legend(fontsize=7, loc='lower right')
    for idx, row in enumerate(paired):
        axes[1].plot([0, 1, 2], [row['communication'], row['shuffled'], row['silent']], 'o-',
                     color=colors[idx % 3], label=f'Seed {row["seed"]}')
    axes[1].axhline(5/7, color='#64748b', linewidth=1, linestyle=':')
    axes[1].set(xticks=[0, 1, 2], xticklabels=['Communication', 'Messages shuffled', 'Trained silent'],
                ylabel='Final held-out resource success', ylim=(.35, 1.02))
    axes[1].tick_params(axis='x', labelsize=9)
    axes[1].legend(fontsize=9)
    fig.suptitle('Frozen DINOv2-L + independent learning interfaces | 3 paired seeds')
    fig.savefig(folder / 'pilot_results.png', dpi=180)
    fig.savefig(folder / 'pilot_results.pdf')
    plt.close(fig)
    print(json.dumps(paired, indent=2))

if __name__ == '__main__':
    main()
