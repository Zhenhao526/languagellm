"""Print-sized figures from the completed formation data; no reanalysis/training.

8.3-inch figures are intended to fit a 16.6-cm Word text area at about 79%.
The existing large scientific plots and every saved measurement stay unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_formation_confirm import load, NAMES, COLORS

ROOT = Path(__file__).resolve().parent
METRIC = 'mean_reward_per_step'


def save(fig, folder, stem):
    fig.savefig(folder/f'{stem}.png', dpi=240)
    fig.savefig(folder/f'{stem}.pdf')
    plt.close(fig)


def draw(folder):
    config, _, runs = load(folder)
    summary = json.loads((folder/'formation_confirm_summary.json').read_text())
    plt.rcParams.update({'font.sans-serif': ['PingFang SC', 'Arial Unicode MS', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 12.8,
                         'axes.titlesize': 13.4, 'axes.labelsize': 12.8,
                         'xtick.labelsize': 12.4, 'ytick.labelsize': 12.4,
                         'legend.fontsize': 12.2, 'axes.spines.top': False,
                         'axes.spines.right': False})
    seeds = config['seeds']
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 4.8), constrained_layout=True,
                             gridspec_kw={'wspace': .12})
    conditions = config['conditions'][:3]
    offsets = np.linspace(-.035, .035, len(seeds))
    for s, offset in zip(seeds, offsets):
        y = [runs[c, s]['result']['native_task']['full']['greedy_success']*100 for c in conditions]
        axes[0].plot(np.arange(3)+offset, y, color='#9DA7AC', lw=.8, alpha=.58, zorder=1)
    for i, c in enumerate(conditions):
        y = [runs[c,s]['result']['native_task']['full']['greedy_success']*100 for s in seeds]
        axes[0].scatter(i+offsets, y, color=COLORS[c], s=30, zorder=3)
        axes[0].plot([i-.15, i+.15], [np.mean(y)]*2, lw=2.8, color='#222222', zorder=4)
    axes[0].set(title='相同起点与预算', xticks=np.arange(3),
                xticklabels=['课程\n通信', '直接\n完整', '课程\n恒0'],
                ylim=(66, 102), yticks=[70,80,90,100], xlim=(-.35,2.35),
                ylabel='互补任务原生成绩（%）')
    for i, name in enumerate(summary['primary_comparisons']):
        d = summary['primary_comparisons'][name]
        values = np.asarray(d['by_seed_order'])*100
        lo, hi = np.asarray(d['paired_bootstrap_percentile_95_interval'])*100
        mean = d['mean']*100
        axes[1].scatter(i-.13+offsets, values, s=25, color='#6D8F9A', alpha=.72, zorder=2)
        axes[1].errorbar(i+.1, mean, yerr=[[mean-lo],[hi-mean]], fmt='o', color='#165E59',
                         lw=2, capsize=4.5, markersize=5, zorder=4)
    axes[1].axhline(0, color='#92999D', lw=1)
    axes[1].set(title='配对差与95%区间', xticks=[0,1], xticklabels=['课程−\n直接完整', '课程−\n恒0原任务'],
                ylabel='成功率差（百分点）', ylim=(-.8,32), yticks=[0,10,20,30], xlim=(-.43,1.35))
    save(fig, folder, 'formation_report_paired')

    fig, axes = plt.subplots(2, 2, figsize=(8.3, 7.1), constrained_layout=True)
    x = np.asarray(config['checkpoints'])
    titles = {'course_communication': '课程＋通信', 'direct_communication': '直接完整＋通信',
              'course_blocked': '课程＋恒定接收0', 'course_substitutable': '课程＋资源可替代'}
    for ax, c in zip(axes.flat, config['conditions']):
        for mode, line in (('normal','-'), ('shuffle','--')):
            key = 'blank' if c == 'course_blocked' else mode
            ys = np.array([[r['full'][key][METRIC]*100 for r in runs[c,s]['curve']] for s in seeds])
            for y in ys:
                ax.plot(x,y,line,color=COLORS[c],lw=.8,alpha=.17,zorder=1)
            ax.plot(x,ys.mean(0),line,color=COLORS[c],lw=2.4,zorder=3)
            if c == 'course_blocked':
                break
        if c == 'course_substitutable':
            ax.axhline(100,color='#747C81',ls=':',lw=1.25,zorder=2)
            ax.text(610,96,'原任务恒100%',ha='center',va='top',fontsize=12.2,color='#50575B')
        ax.axhline(100*5/7,color='#ACB2B6',ls=':',lw=.85,zorder=0)
        for boundary in (600,900):
            ax.axvline(boundary,color='#AFB5B8',ls=':',lw=.9,zorder=0)
        ax.set(title=titles[c], xlim=(0,1200), ylim=(40,103),
               xticks=[0,600,1200], yticks=[50,75,100], xlabel='社会训练更新')
    axes[0,0].set_ylabel('完整场景成功率（%）')
    axes[1,0].set_ylabel('完整场景成功率（%）')
    save(fig, folder, 'formation_report_process')
    captions = {
        'figure_width_inches':8.3,
        'intended_word_width_cm':16.6,
        'main_font_points_at_source':12.8,
        'main_font_points_at_intended_word_width':12.8*16.6/(8.3*2.54),
        'formation_report_paired': '左：各条件保持原训练消息规则的完整互补任务成绩，恒0组评估仍恒0；细线连接相同种子，每点为一个群体，黑线为均值。右：十个种子的配对差及10,000次种子重抽样95%区间。点的少量水平位移仅用于避免重叠。可替代任务的恒满分不参加能力比较。',
        'formation_report_process': '细线保留十个群体，粗线为均值；实线为正常消息，虚线为打乱消息。恒0面板始终接收0，仅画其原任务成绩。可替代面板的实/虚线是共同互补资源诊断，顶部点线为其原生任务恒100%；其余水平点线为单对无当前消息期望上界5/7。竖线标出600与900次更新的阶段边界。过程每任务每模式2,048例。',
        'source_summary_sha256':hashlib.sha256((folder/'formation_confirm_summary.json').read_bytes()).hexdigest(),
        'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (folder/'formation_report_figure_captions.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2)+'\n')
    (folder/'analysis_source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    hashes=json.loads((folder/'analysis_source/source_hashes.json').read_text())
    hashes[Path(__file__).name]=captions['plot_source_sha256']
    (folder/'analysis_source/source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',nargs='?',type=Path,default=ROOT/'results/formation_confirm_001')
    draw(parser.parse_args().directory)
