"""Descriptive figures from completed probability supplement; no model calls."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'results/probability_001/execution/results.json'
OUT = ROOT/'results/probability_001/figures'
SEEDS = [28101, 28102, 28103, 28104]
CELLS = [('additive', 1), ('additive', 3), ('joint', 1), ('joint', 3)]
METRICS = [('U', '#697380', '--', 'U: maps with a correct greedy code'),
           ('C', '#2573b7', '-', 'C: best-code success probability'),
           ('E_G', '#d65c23', '-', 'E_G: natural-code success probability')]


def main():
    data = json.loads(DATA.read_text())
    assert data['status'] == 'complete'
    OUT.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    plotted = []
    for ax, (kind, gain) in zip(axes.flat, CELLS):
        rows = [r for r in data['seed_values'] if r['family'] == 'split' and
                r['partition'] == 'heldout' and r['kind'] == kind and r['gain'] == gain]
        steps = sorted({r['update'] for r in rows})
        assert len(rows) == 32 and len(steps) == 8
        for metric, color, style, label in METRICS:
            values = np.array([[next(r[metric] for r in rows if r['seed'] == s and r['update'] == u)
                                for u in steps] for s in SEEDS])
            for ys in values:
                ax.plot(steps, 100*ys, color=color, linestyle=style, alpha=.20, lw=.7)
            ax.plot(steps, 100*values.mean(0), color=color, linestyle=style,
                    label=label, marker='o', ms=3, lw=2)
            plotted.append(dict(kind=kind, gain=gain, metric=metric, seeds=SEEDS,
                                updates=steps, values=values.tolist()))
        ax.set(title=f'{kind.capitalize()} reward / policy gain {gain}',
               xlim=(0, 2400), ylim=(0, 65), xticks=[0, 300, 600, 1200, 1800, 2400])
        ax.grid(axis='y', alpha=.15)
    for ax in axes[-1]: ax.set_xlabel('Training updates')
    for ax in axes[:, 0]: ax.set_ylabel('Held-out maps: rate / probability (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .938),
               frameon=False, ncol=3, fontsize=9)
    fig.suptitle('Greedy coverage and probabilistic success follow different trajectories', y=.98, fontsize=14)
    fig.text(.5, .043, 'All panels share one scale. Thin lines: 4 seeds; bold lines: mean. Splits and directions averaged within seed.',
             ha='center', fontsize=9)
    fig.text(.5, .019, 'U counts maps; C and E_G are temperature-1 probabilities. C assumes an oracle chooses the same complete code for both goals.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .89))
    for ext in ('png', 'svg'): fig.savefig(OUT/f'probability_trajectory.{ext}', dpi=180)
    plt.close(fig)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = dict(source=str(DATA), source_sha256=sha(DATA), source_code_sha256=sha(Path(__file__)),
                   figures={p.name: sha(p) for p in sorted(OUT.iterdir())}, plotted_values=plotted,
                   scope='Descriptive post-hoc illustration of all four fixed conditions, eight times and four seeds; no new metric or inference.')
    with (OUT/'receipt.json').open('x') as f: json.dump(receipt, f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status='complete', output=str(OUT))))


if __name__ == '__main__': main()
