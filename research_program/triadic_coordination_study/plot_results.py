"""Plot all four paired seeds and fixed double-heldout monitoring checkpoints."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RUN = HERE / 'results/mean_vs_log_001'


def main():
    source = RUN / 'execution/results.json'
    data = json.loads(source.read_text())
    assert data['status'] == 'completed' and len(data['runs']) == 8
    output = RUN / 'figures_001'
    output.mkdir(exist_ok=False)
    seeds = [45101, 45102, 45103, 45104]
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), layout='constrained')
    fig.suptitle('Triadic collection: mean J vs mean log J', fontsize=15)
    rows = data['primary_comparison']['seed_pairs']
    for row, color in zip(rows, colors):
        assert row['seed'] == seeds[len(axes[0].lines)]
        y = [100 * row['mean_J_full_success_rate'], 100 * row['mean_log_J_full_success_rate']]
        axes[0].plot([0, 1], y, 'o-', color=color, label=str(row['seed']), alpha=.9)
    axes[0].set_xticks([0, 1], ['mean J', 'mean log J'])
    axes[0].set_xlim(-.2, 1.2)
    axes[0].set_title('Primary endpoint: all 8,244 heldout worlds')
    axes[0].set_ylabel('Full success (%)')
    axes[0].legend(title='Paired seed', loc='upper left')
    for result in data['runs']:
        color = colors[seeds.index(result['seed'])]
        style = '--' if result['objective'] == 'mean_J' else '-'
        x = [m['update'] for m in result['monitor']]
        y = [100*m['monitor']['new_needs_and_layouts']['greedy_full_success_rate'] for m in result['monitor']]
        axes[1].plot(x, y, marker='o', ms=3, linestyle=style, color=color, alpha=.9)
    axes[1].set_title('Fixed monitoring subset: 1,024 worlds')
    axes[1].set_xlabel('Training update')
    axes[1].set_xlim(-100, 6100)
    axes[1].legend(handles=[Line2D([0],[0],color='#444',linestyle='--',label='mean J'),
        Line2D([0],[0],color='#444',linestyle='-',label='mean log J')],loc='upper left')
    for ax in axes:
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=.2)
        ax.axhline(99, color='#777', linewidth=.8, linestyle=':')
    fig.supxlabel('Four paired initializations, one fixed split; dotted line = 99% candidate screen. No messages.', fontsize=9)
    for extension in ('png', 'svg'):
        fig.savefig(output / ('paired_objectives.' + extension), dpi=180)
    plt.close(fig)
    receipt = {'status': 'plotted_all_fixed_primary_pairs_and_checkpoints',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'figures': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()},
        'visual_review_pending': True, 'training_or_forward_calls': 0}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'output':str(output),'status':receipt['status']}))


if __name__ == '__main__':
    main()
