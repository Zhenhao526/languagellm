"""Deterministic figures for the partner-switching payoff pilot."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import design

HERE = Path(__file__).resolve().parent


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def render(summary, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 190, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = {'partial': '#286090', 'all_or_nothing': '#be6231'}
    styles = {True: '-', False: '--'}
    trajectory = {(row['payoff'], row['rule'], row['live'], row['update']): row
                  for row in summary['trajectory_summaries']}
    x = np.asarray(design.UPDATES, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for rule, ax in zip(design.RULES, axes):
        for payoff in design.PAYOFFS:
            for live in (True, False):
                y = [100 * trajectory[payoff, rule, live, int(t)]['correct_executed_pair_rate'] for t in x]
                label = f'{payoff.replace("_", " ")} / {"live" if live else "silent"}'
                ax.plot(x, y, color=colors[payoff], linestyle=styles[live], marker='o',
                        markersize=3, linewidth=1.8, label=label)
        ax.set_title(f'{rule.capitalize()} execution rule')
        ax.set_xlabel('Training updates')
        ax.set_xlim(0, 6000); ax.set_ylim(0, 100); ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('Correct executed-pair rate (%)')
    fig.suptitle('Communication and payoff ecology on held-out layouts', fontsize=13, y=.99)
    fig.text(.04, .01, 'Solid = live channel; dashed = silent. Curves average four paired developmental seeds. All worlds remain in the denominator.', fontsize=8, color='#444')
    fig.subplots_adjust(left=.07, right=.98, bottom=.13, top=.88, wspace=.15)
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'01_partner_switch_trajectory.{suffix}')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    curves = {row['rule']: row for row in summary['interaction_curves']}
    for rule, ax in zip(design.RULES, axes):
        curve = curves[rule]
        ax.plot(x, 100 * np.asarray(curve['centered_interaction']), color='#111', marker='o', linewidth=2,
                label='Payoff × communication interaction')
        ax.axhline(0, color='#777', linewidth=.8)
        ax.set_title(f'{rule.capitalize()} rule')
        ax.set_xlabel('Training updates'); ax.set_ylabel('Centered interaction (pp)')
        ax.set_xlim(0, 6000); ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Does an all-or-nothing ecology increase communication dependence?', fontsize=13, y=.99)
    fig.text(.06, .01, 'Interaction = (all-or-nothing live−silent) − (partial live−silent), centered at update 0; curves average four seeds.', fontsize=8, color='#444')
    fig.subplots_adjust(left=.08, right=.98, bottom=.14, top=.88, wspace=.2)
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'02_partner_switch_interaction.{suffix}')
    plt.close(fig)

    rows = summary['primary']['by_seed']
    means = np.asarray([100 * row['rule_mean_centered_AUC'] for row in rows])
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.scatter(np.arange(len(means)), means, color='#286090', s=32)
    ax.axhline(100 * summary['primary']['mean_centered_AUC'], color='#111', linewidth=1.6,
                label=f'Mean = {100 * summary["primary"]["mean_centered_AUC"]:.2f} pp')
    ax.axhline(0, color='#777', linewidth=.8)
    ax.set_xticks(np.arange(len(means)), [str(row['seed']) for row in rows])
    ax.set_ylabel('Centered interaction AUC (pp)'); ax.set_title('Seed-level developmental pilot')
    ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.subplots_adjust(left=.12, right=.98, bottom=.16, top=.88)
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'03_partner_switch_primary.{suffix}')
    plt.close(fig)


def execute(summary_path, output):
    summary_path = Path(summary_path).resolve(); output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite figures')
    summary = read(summary_path); require(summary['status'] == 'completed_json_only_summary', 'Completed summary required')
    output.mkdir(parents=True)
    data_path = output / 'plot_data.json'
    write(data_path, dict(updates=list(design.UPDATES), trajectory=summary['trajectory_summaries'],
                          interaction_curves=summary['interaction_curves'], primary=summary['primary']))
    render(summary, output)
    files = [data_path] + [output / f'{prefix}.{suffix}' for prefix in
                           ('01_partner_switch_trajectory', '02_partner_switch_interaction', '03_partner_switch_primary')
                           for suffix in ('png', 'pdf')]
    receipt = dict(status='passed', at=datetime.now(timezone.utc).isoformat(),
                   inputs_sha256={str(summary_path): sha(summary_path), str(HERE / 'plot_results.py'): sha(HERE / 'plot_results.py')},
                   outputs_sha256={str(path): sha(path) for path in files}, model_forwards=0, npz_reads=0,
                   visual_review='Pending actual PNG inspection.')
    write(output / 'receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--summary', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(execute(args.summary, args.output), ensure_ascii=False, indent=2))
