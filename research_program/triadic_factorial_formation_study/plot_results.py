"""Predetermined figures from the audited JSON summary."""
from datetime import datetime, timezone
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import metrics

HERE = Path(__file__).resolve().parent


def read(path):
    with Path(path).open() as stream: return json.load(stream)


def write(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def render(summary, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 190, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = {'factorial_holdout': '#286090', 'saturated': '#be6231'}
    styles = {'live': '-', 'silent': '--'}
    trajectory = {(row['regime'], row['rule'], row['live'], row['update']): row for row in summary['target_trajectory_summaries']}
    x = np.asarray(metrics.UPDATES)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for rule, ax in zip(metrics.RULES, axes):
        for regime in metrics.REGIMES:
            for live in (True, False):
                label = f'{regime.replace("_", " ")} / {"live" if live else "silent"}'
                y = [100 * trajectory[regime, rule, live, int(t)]['factor_response']['heldout_changed_actor']['Q'] for t in x]
                ax.plot(x, y, label=label, color=colors[regime], linestyle=styles['live' if live else 'silent'], marker='o', markersize=3, linewidth=1.8)
        ax.set_title(f'{rule.capitalize()} rule'); ax.set_xlabel('Training updates'); ax.set_ylabel('Held-out changed-actor Q (%)'); ax.set_xlim(0, 6000); ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Communication response on unseen object×length resource predicates', fontsize=13, y=.99)
    fig.text(.04, .01, 'Solid = live channel; dashed = silent. Curves average16 paired initializations. Q is a behavioral response, not a language score.', fontsize=8, color='#444')
    fig.subplots_adjust(left=.07, right=.98, bottom=.13, top=.88, wspace=.22)
    for suffix in ('png', 'pdf'): fig.savefig(output / f'01_factorial_trajectory.{suffix}')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    rows = summary['primary']['by_seed']; vals = [100 * row['rule_mean_centered_AUC'] for row in rows]
    ax.scatter(np.arange(len(vals)), vals, color='#286090', s=24, label='One paired initialization')
    mean = 100 * summary['primary']['statistics']['mean']; lo = 100 * summary['primary']['statistics']['ci95_lower']; hi = 100 * summary['primary']['statistics']['ci95_upper']
    ax.errorbar(len(vals) + 1, mean, yerr=np.array([[mean - lo], [hi - mean]]), fmt='D', color='#111', capsize=5, linewidth=1.6, label='Mean ± approximate t15 interval')
    ax.axhline(0, color='#707070', linewidth=.8); ax.set_xticks(list(range(len(vals))) + [len(vals) + 1], labels=[str(row['seed']) for row in rows] + ['Mean']);
    for tick in ax.get_xticklabels()[:-1]: tick.set_rotation(60); tick.set_horizontalalignment('right'); tick.set_fontsize(7)
    ax.set_ylabel('Regime interaction centered AUC (percentage points)'); ax.set_title('Factorial holdout × communication interaction'); ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.text(.06, .02, 'Primary = [(factorial live−silent) − (saturated live−silent)], centered at update0, trapezoid AUC /6000; strict and reciprocal rules averaged.', fontsize=8, color='#444')
    fig.subplots_adjust(left=.09, right=.98, bottom=.2, top=.9)
    for suffix in ('png', 'pdf'): fig.savefig(output / f'02_factorial_primary.{suffix}')
    plt.close(fig)


def execute(summary_path, output='figures_001'):
    summary_path = Path(summary_path).resolve(); receipt_path = summary_path.parent / 'receipt.json'; receipt = read(receipt_path)
    require(receipt['status'] == 'passed' and receipt['outputs_sha256'][str(summary_path)] == sha(summary_path), 'Passed summary required')
    destination = Path(output); destination = destination if destination.is_absolute() else summary_path.parent.parent / destination; require(not destination.exists(), 'Never overwrite figures'); destination.mkdir(parents=True)
    summary = read(summary_path); data_path = destination / 'plot_data.json'; write(data_path, dict(updates=list(metrics.UPDATES), target=metrics.TARGET, primary=summary['primary'], interaction_curves=summary['interaction_curves']))
    render(summary, destination)
    files = [data_path] + [destination / f'01_factorial_trajectory.{s}' for s in ('png', 'pdf')] + [destination / f'02_factorial_primary.{s}' for s in ('png', 'pdf')]
    out = dict(status='passed', at=datetime.now(timezone.utc).isoformat(), inputs_sha256={str(p): sha(p) for p in (summary_path, receipt_path, HERE/'plot_results.py', HERE/'metrics.py')}, outputs_sha256={str(p): sha(p) for p in files}, model_forwards=0, npz_reads=0, visual_review='Pending actual PNG inspection.')
    write(destination / 'receipt.json', out); return dict(status='passed', output=str(destination), outputs_sha256=out['outputs_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--summary', required=True); parser.add_argument('--output', default='figures_001'); args = parser.parse_args(); print(json.dumps(execute(args.summary, args.output), ensure_ascii=False))

