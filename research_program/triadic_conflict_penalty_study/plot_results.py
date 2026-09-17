"""Generate figures from frozen JSON results only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from . import design

DOSE_RULES = ('c0', 'c25', 'c50', 'c75', 'c100')
DOSES = np.asarray([design.PENALTIES[r] for r in DOSE_RULES])
COLORS = {'c0': '#1b9e77', 'c25': '#66a61e', 'c50': '#e6ab02',
          'c75': '#d95f02', 'c100': '#7570b3', 'strict': '#000000'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def runs_by(data):
    return {(r['seed'], r['rule'], r['information'], r['schedule'], bool(r['live'])): r
            for r in data['runs']}


def traj(run, key):
    rows = sorted(run['trajectory'], key=lambda x: int(x['update']))
    return np.asarray([x['target_trajectory'][key] for x in rows])


def final(run, key):
    return float(run['final']['new_layouts'][key])


def mean_se(values):
    x = np.asarray(values, dtype=float)
    return x.mean(axis=0), x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])


def style(ax):
    ax.grid(axis='y', color='#dddddd', linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def figure_dose_trajectories(by, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    updates = np.asarray(design.UPDATES)
    for ax, info in zip(axes, ('PL', 'FI')):
        for rule in DOSE_RULES:
            streams = []
            for seed in design.SEEDS:
                sched = []
                for schedule in design.SCHEDULES:
                    sched.append(traj(by[seed, rule, info, schedule, True], 'target_pair_legal_rate')
                                 - traj(by[seed, rule, info, schedule, False], 'target_pair_legal_rate'))
                streams.append(np.mean(sched, axis=0))
            mean, se = mean_se(streams)
            ax.plot(updates, 100 * mean, marker='o', label=f'c={design.PENALTIES[rule]:.2g}',
                    color=COLORS[rule])
            ax.fill_between(updates, 100 * (mean - 1.96 * se), 100 * (mean + 1.96 * se),
                            color=COLORS[rule], alpha=.10)
        strict = []
        for seed in design.SEEDS:
            sched = [traj(by[seed, 'strict', info, s, True], 'target_pair_legal_rate')
                     - traj(by[seed, 'strict', info, s, False], 'target_pair_legal_rate')
                     for s in design.SCHEDULES]
            strict.append(np.mean(sched, axis=0))
        mean, se = mean_se(strict)
        ax.plot(updates, 100 * mean, '--', color=COLORS['strict'], label='strict')
        ax.fill_between(updates, 100 * (mean - 1.96 * se), 100 * (mean + 1.96 * se),
                        color=COLORS['strict'], alpha=.07)
        ax.axhline(0, color='#777777', linewidth=.8)
        ax.set_title(info)
        ax.set_xlabel('训练更新')
        style(ax)
    axes[0].set_ylabel('live − silent 目标搭档率（百分点）')
    axes[1].legend(frameon=False, fontsize=9, ncol=2)
    fig.suptitle('第三人冲突成本的通信—搭档选择剂量反应', y=1.02)
    fig.tight_layout()
    path = out / 'dose_response_trajectories.png'; fig.savefig(path, dpi=220, bbox_inches='tight')
    fig.savefig(out / 'dose_response_trajectories.pdf', bbox_inches='tight'); plt.close(fig)


def figure_dose_slopes(summary, out):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(2); width = .34
    for off, info in zip((-width / 2, width / 2), ('PL', 'FI')):
        item = summary['primary']['dose_response'][info]['target_pair_legal_rate']
        y = 100 * item['slope_centered_AUC']['mean']
        lo = 100 * (y / 100 - item['slope_centered_AUC']['mean'] + item['slope_centered_AUC']['ci95_lower'])
        hi = 100 * (item['slope_centered_AUC']['ci95_upper'] - item['slope_centered_AUC']['mean'] + y / 100)
        ax.bar(x + off, [y, 0], width, label=info, color=['#377eb8', '#e41a1c'][0 if info == 'PL' else 1])
        ax.errorbar(x[0] + off, y, yerr=[[y - lo], [hi - y]], color='black', capsize=4, fmt='none')
    # Use a clearer single category: replace the placeholder FI bar above.
    ax.clear()
    vals=[]; lows=[]; highs=[]
    for info in ('PL','FI'):
        item=summary['primary']['dose_response'][info]['target_pair_legal_rate']['slope_centered_AUC']
        vals.append(100*item['mean']); lows.append(100*(item['mean']-item['ci95_lower'])); highs.append(100*(item['ci95_upper']-item['mean']))
    ax.bar(np.arange(2), vals, color=['#377eb8','#e41a1c'], width=.58)
    ax.errorbar(np.arange(2), vals, yerr=[lows, highs], fmt='none', color='black', capsize=5)
    ax.axhline(0, color='#777777', linewidth=.8)
    ax.set_xticks(np.arange(2), ['PL', 'FI'])
    ax.set_ylabel('目标搭档率剂量斜率（百分点 / penalty）')
    ax.set_title('live − silent 增益随第三人冲突成本的斜率')
    style(ax)
    fig.tight_layout()
    path=out/'dose_slope.png'; fig.savefig(path,dpi=220,bbox_inches='tight'); fig.savefig(out/'dose_slope.pdf',bbox_inches='tight'); plt.close(fig)


def figure_mechanism(by, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = ['c0', 'c50', 'c100', 'strict']
    x = np.arange(len(labels)); width=.36
    for ax, metric, title in zip(axes, ('physical_execution_rate', 'target_pair_legal_rate'),
                                 ('物理执行率', '目标搭档率 | physical')):
        for off, live in zip((-width/2, width/2), (False, True)):
            vals=[]
            for rule in labels:
                vals.append(100*np.mean([final(by[seed, rule, 'PL', 'static', live], metric)
                                         for seed in design.SEEDS]))
            ax.bar(x+off, vals, width, label='live' if live else 'silent',
                   color='#4daf4a' if live else '#999999')
        ax.set_xticks(x, labels); ax.set_title(title); ax.set_ylabel('%')
        style(ax)
    axes[0].legend(frameon=False)
    fig.suptitle('PL static 新布局上的执行—选择分解', y=1.02)
    fig.tight_layout()
    path=out/'mechanism_decomposition.png'; fig.savefig(path,dpi=220,bbox_inches='tight'); fig.savefig(out/'mechanism_decomposition.pdf',bbox_inches='tight'); plt.close(fig)


def figure_reward(by, out):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = DOSE_RULES + ('strict',); x=np.arange(len(labels)); width=.36
    for off, live in zip((-width/2,width/2),(False,True)):
        vals=[]; conflicts=[]
        for rule in labels:
            vals.append(100*np.mean([final(by[seed, rule, 'PL', 'static', live], 'reward_rate') for seed in design.SEEDS]))
            conflicts.append(100*np.mean([final(by[seed, rule, 'PL', 'static', live], 'conflict_world_rate') for seed in design.SEEDS]))
        ax.bar(x+off, vals, width, label='live' if live else 'silent', color='#984ea3' if live else '#bdbdbd')
    ax.set_xticks(x, ['0','.25','.5','.75','1','strict']); ax.set_ylabel('惩罚后团队 reward（%）')
    ax.set_title('PL static 新布局上的实际惩罚后收益')
    style(ax); ax.legend(frameon=False)
    fig.tight_layout()
    path=out/'penalized_reward.png'; fig.savefig(path,dpi=220,bbox_inches='tight'); fig.savefig(out/'penalized_reward.pdf',bbox_inches='tight'); plt.close(fig)


def main(source, summary_path, output):
    source=Path(source).resolve(); summary_path=Path(summary_path).resolve(); output=Path(output).resolve()
    require(not output.exists(), 'Never overwrite figure output')
    output.mkdir(parents=False)
    data=json.loads((source/'execution'/'results.json').read_text())
    summary=json.loads((summary_path/'results.json').read_text())['summary']
    by=runs_by(data)
    figure_dose_trajectories(by,output)
    figure_dose_slopes(summary,output)
    figure_mechanism(by,output)
    figure_reward(by,output)
    files=sorted(str(p.name) for p in output.iterdir() if p.suffix in ('.png','.pdf'))
    receipt=dict(status='passed',source=str(source),summary=str(summary_path),files=files,
                 figure_sha256={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in files},
                 visual_review='pending')
    (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',required=True); parser.add_argument('--summary',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); main(args.source,args.summary,args.output)
