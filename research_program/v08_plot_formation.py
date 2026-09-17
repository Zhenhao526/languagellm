"""Plot the complete, fixed-photo post-hoc U/N trajectory; no neural imports."""
from pathlib import Path
from hashlib import sha256
import json
import math

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'v08_formation_trajectory_001'
SOURCE = BASE / 'execution/results.json'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads(SOURCE.read_text())
    assert data['status'] == 'complete' and data['counts']['directions'] == 960
    selected = {(r['reward_kind'], r['update']): r for r in data['summaries']
                if r['family'] == 'split' and r['partition'] == 'heldout'}
    updates = [0, 100, 300, 600, 1200, 1800, 2100, 2400]
    kinds = ['additive', 'mixed', 'joint']
    assert set(selected) == {(k, u) for k in kinds for u in updates}
    outputs = [BASE / f'formation_trajectory.{ext}' for ext in ['png', 'svg', 'json']]
    assert not any(p.exists() for p in outputs)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    values = {}
    for ax, metric, title in zip(axes, ['U_full49', 'N_both'],
                                ['Receiver coverage U', 'Natural joint success N']):
        maximum = 0
        for kind, color, label in zip(kinds, ['#1D5B75', '#AA7725', '#7B438C'],
                                     ['Additive (λ = 0)', 'Mixed (λ = 0.5)', 'Joint (λ = 1)']):
            rows = [selected[kind, u]['metrics'][metric] for u in updates]
            means = [100 * r['mean'] for r in rows]
            by_seed = [[100 * r['seed_values'][s] for r in rows] for s in range(4)]
            for sequence in by_seed:
                ax.plot(updates, sequence, color=color, lw=.8, alpha=.17)
                maximum = max(maximum, *sequence)
            ax.plot(updates, means, marker='o', ms=3.5, color=color, lw=2.2, label=label)
            values[metric + '/' + kind] = {'mean_percent': means, 'seed_percent': by_seed}
        ax.set_ylim(0, math.ceil((maximum + 1) / 10) * 10)
        ax.set_xlim(-55, 2455)
        ax.set_xticks([0, 600, 1200, 1800, 2400])
        ax.set_title(title, loc='left', fontsize=13, pad=13)
        ax.set_xlabel('Training update', labelpad=10)
        ax.set_ylabel('Percent')
        ax.grid(axis='y', color='#E2E6EA', lw=.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=7)
        ax.spines['left'].set_color('#B7BEC5')
        ax.spines['bottom'].set_color('#B7BEC5')
    axes[0].annotate('6.25%', (600, 6.25), xytext=(14, -13), textcoords='offset points',
                     color='#7B438C', fontsize=11, weight='bold')
    axes[1].annotate('0.26%', (600, 100 * selected['joint', 600]['metrics']['N_both']['mean']),
                     xytext=(14, 11), textcoords='offset points', color='#7B438C', fontsize=11, weight='bold')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.065, .089), ncol=3, frameon=False)
    fig.suptitle('How held-out coverage and natural use change during learning', x=.075,
                 ha='left', fontsize=16, weight='bold', y=.955)
    fig.text(.075, .874, 'Same 16 photo pairs at every checkpoint. Faint lines: individual seeds; bold lines: four-seed means.',
             fontsize=10, color='#4E5963')
    fig.text(.075, .039, 'All 8 stored checkpoints are shown. Curves join observations; the exact onset or minimum between them is unknown.',
             fontsize=9.3, color='#4E5963')
    fig.text(.075, .01, 'Post-hoc probe of 60 completed runs; no new training. Each seed averages 3 splits × 2 directions. Panel y-axis ranges differ.',
             fontsize=9.2, color='#4E5963')
    fig.subplots_adjust(left=.075, right=.965, top=.76, bottom=.27, wspace=.25)
    fig.savefig(outputs[0], dpi=180, facecolor='white')
    fig.savefig(outputs[1], facecolor='white')
    plt.close(fig)
    def digest(p): return sha256(p.read_bytes()).hexdigest()
    receipt = {'source_sha256': digest(SOURCE), 'script_sha256': digest(Path(__file__)),
               'matplotlib': matplotlib.__version__, 'updates': updates, 'values': values,
               'outputs_sha256': {p.name: digest(p) for p in outputs[:2]},
               'inference_calls': 0, 'selection': 'All preselected checkpoints, seeds and conditions; held-out panels.'}
    with outputs[2].open('x') as f: json.dump(receipt, f, indent=2)
    print(json.dumps({'status': 'rendered', 'outputs': [str(p) for p in outputs]}))


if __name__ == '__main__':
    main()
