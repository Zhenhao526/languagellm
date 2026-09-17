"""Scientific figures from completed frozen numerical analysis; no inference."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/gain_001/analysis.json'
OUT = ROOT / 'results/gain_001/figures'
CELLS = [('additive', 1), ('additive', 3), ('joint', 1), ('joint', 3)]
SEEDS = [28101, 28102, 28103, 28104]
COLORS = {'additive': '#2871b5', 'joint': '#cf5a28'}


def main():
    result = json.loads(DATA.read_text())
    assert result['status'] == 'complete'
    OUT.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    trajectory = result['same_photo_trajectory']['seed_values']
    endpoints = result['endpoint_seed_values']
    plotted = {'curves': [], 'endpoints': []}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, metric, title in zip(axes, ['U_full49', 'N_both'],
                                 ['Complete-code coverage U', 'Natural two-goal success N']):
        for kind, gain in CELLS:
            rows = [r for r in trajectory if r['family'] == 'split' and r['partition'] == 'heldout'
                    and r['kind'] == kind and r['gain'] == gain]
            steps = sorted({r['update'] for r in rows})
            values = np.array([[next(r[metric] for r in rows if r['seed'] == s and r['update'] == u)
                                for u in steps] for s in SEEDS])
            assert values.shape == (4, 8)
            style = '-' if gain == 1 else '--'
            for vals in values:
                ax.plot(steps, 100*vals, color=COLORS[kind], linestyle=style, lw=.7, alpha=.2)
            ax.plot(steps, 100*values.mean(0), color=COLORS[kind], linestyle=style, lw=2,
                    marker='o', ms=3, label=f'{kind}, gain {gain}')
            plotted['curves'].append(dict(metric=metric, kind=kind, gain=gain, steps=steps,
                                          seeds=SEEDS, values=values.tolist()))
        ax.set(title=title, xlabel='Training updates', ylabel='Held-out maps (%)', xlim=(0, 2400))
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.15)
        ax.axvline(1200, color='#888888', linestyle=':', lw=.9)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Policy gain × reward rule: formation trajectories', fontsize=13)
    fig.text(.5, .015, 'Thin lines: 4 independent seeds; bold lines: mean. Splits/directions averaged within seed. Panel y-scales differ.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94))
    for ext in ('png', 'svg'):
        fig.savefig(OUT/f'formation.{ext}', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = []
    for kind, gain in CELLS:
        rows = [r for r in endpoints if r['family'] == 'split' and r['partition'] == 'heldout'
                and r['mode'] == 'normal' and r['kind'] == kind and r['gain'] == gain]
        assert len(rows) == 4
        ys = [next(r['J'] for r in rows if r['seed'] == seed) for seed in SEEDS]
        values.append(ys)
        plotted['endpoints'].append(dict(kind=kind, gain=gain, seeds=SEEDS, J=ys))
    values = 100*np.array(values)
    for i, seed in enumerate(SEEDS):
        ax.plot(range(4), values[:, i], '-o', color=plt.cm.tab10(i), alpha=.7, lw=1, ms=4,
                label=str(seed))
    ax.scatter(range(4), values.mean(1), marker='_', color='black', s=600, linewidths=2,
               label='Mean', zorder=5)
    ax.set(xticks=range(4), xticklabels=['Additive / 1', 'Additive / 3', 'Joint / 1', 'Joint / 3'],
           ylabel='Held-out J at update 2400 (%)', xlabel='Reward rule / policy gain',
           title='Actual two-goal success: all four independent seeds')
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=5, fontsize=8)
    ax.grid(axis='y', alpha=.15)
    fig.text(.5, .01, 'J uses endpoint evaluation worlds; it has a different photo-pair sample from N. Connecting lines identify seeds.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for ext in ('png', 'svg'):
        fig.savefig(OUT/f'endpoint.{ext}', dpi=180)
    plt.close(fig)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = dict(source=str(DATA), source_sha256=digest(DATA), plot_source_sha256=digest(Path(__file__)),
                   figures={p.name: digest(p) for p in sorted(OUT.glob('*'))}, plotted_values=plotted,
                   scope='Descriptive plots only; no confidence intervals or additional independent samples.')
    (OUT/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status='complete', out=str(OUT))))


if __name__ == '__main__':
    main()
