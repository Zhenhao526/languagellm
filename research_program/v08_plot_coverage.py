"""Plot audited held-out U/J; no model dependencies or new evaluation."""
from pathlib import Path
from hashlib import sha256
import json
import platform

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'v08_receiver_coverage_001.json'
AUDIT = ROOT / 'v08_覆盖分析独立核验_真实.json'


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data, audit = json.loads(SOURCE.read_text()), json.loads(AUDIT.read_text())
    assert audit['status'] == 'passed_independent_real_recount'
    assert audit['reviewed_result_sha256'] == digest(SOURCE)
    selected = {r['reward_kind']: r for r in data['groups']
                if r['family'] == 'split' and r['subset'] == 'heldout'}
    kinds = ['additive', 'mixed', 'joint']
    seeds = [27101, 27102, 27103, 27104]
    assert set(selected) == set(kinds)
    outputs = [ROOT / f'v08_coverage_comparison_001.{ext}' for ext in ['png', 'svg', 'json']]
    assert not any(p.exists() for p in outputs), 'Preserve previous figure artifacts'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4), sharey=True)
    saved = {}
    for ax, field, title, subtitle, color in zip(
        axes, ['U_rate', 'endpoint_J_rate'],
        ['Receiver coverage U', 'Actual joint success J'],
        ['A jointly correct code exists', 'The natural message succeeds for both goals'],
        ['#1D5B75', '#AC4A26']):
        by_seed = []
        for seed in seeds:
            values = [100 * next(r[field] for r in selected[k]['per_seed'] if r['seed'] == seed) for k in kinds]
            by_seed.append(values)
            ax.plot(range(3), values, color='#A9AFB7', lw=1.05, marker='o', ms=4, alpha=.8, zorder=1)
        means = [sum(row[j] for row in by_seed) / 4 for j in range(3)]
        ax.plot(range(3), means, color=color, lw=2.8, marker='o', ms=7, zorder=3)
        for j, value in enumerate(means):
            ax.annotate(f'{value:.2f}%', (j, value), xytext=(0, 10), textcoords='offset points',
                        ha='center', color=color, weight='bold', fontsize=12,
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.2, 'alpha': .9})
        ax.set_title(title + '\n' + subtitle, fontsize=12, loc='left', pad=17)
        ax.set_xticks(range(3), ['Additive\nλ = 0', 'Mixed\nλ = 0.5', 'Joint\nλ = 1'])
        ax.set_xlim(-.18, 2.18)
        ax.set_ylim(0, 62)
        ax.set_yticks(range(0, 61, 10))
        ax.grid(axis='y', color='#E4E8EC', lw=.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=7)
        ax.spines['left'].set_color('#B6BDC4')
        ax.spines['bottom'].set_color('#B6BDC4')
        saved[field] = {'seed_order': seeds, 'seed_percent': by_seed, 'mean_percent': means}
    axes[0].set_ylabel('Percent', labelpad=10)
    fig.suptitle('Held-out combinations: coverage and actual use differ', x=.075, ha='left',
                 fontsize=17, weight='bold', y=.97)
    fig.legend(handles=[Line2D([0], [0], color='#A9AFB7', marker='o', lw=1, label='Individual training seed'),
                        Line2D([0], [0], color='#303942', marker='o', lw=2.5, label='Mean of four seeds')],
               loc='lower left', bbox_to_anchor=(.065, .102), frameon=False, ncol=2, fontsize=10)
    fig.text(.075, .047, 'Within each seed: 3 held-out splits × 2 directions. The same 49-code channel is used in all conditions.',
             fontsize=9.5, color='#525C65')
    fig.text(.075, .014, 'U: 144 map/direction/split/seed units per condition. J: 23,040 paired test worlds. These are not independent training samples.',
             fontsize=9.2, color='#525C65')
    fig.subplots_adjust(left=.075, right=.965, top=.73, bottom=.255, wspace=.22)
    fig.savefig(outputs[0], dpi=180, facecolor='white')
    fig.savefig(outputs[1], facecolor='white')
    plt.close(fig)
    receipt = {'source_sha256': digest(SOURCE), 'independent_audit_sha256': digest(AUDIT),
               'script_sha256': digest(Path(__file__)), 'python': platform.python_version(),
               'matplotlib': matplotlib.__version__, 'values': saved,
               'outputs_sha256': {p.name: digest(p) for p in outputs[:2]},
               'new_model_calls': 0, 'error_bars': 'none; all four seed values shown'}
    with outputs[2].open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps({'status': 'rendered', 'outputs': [str(p) for p in outputs]}))


if __name__ == '__main__':
    main()
