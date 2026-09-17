"""Two scientific figures from the frozen support and independent analysis.

The diagram-only preview never creates the formal output directory. Result
figures require an existing complete analysis and passed raw validation. All
success axes are fixed to0..100%; source-level values and both donor assignments
are retained. No model, image, production metric, or training code is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'redesign_v0.9/.analysis_deps'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

ARMS = ('target_paths3', 'target_paths2')
COLORS = {'target_paths3': '#1876BD', 'target_paths2': '#D06528'}
LABELS = {'target_paths3': '3 条关联路径', 'target_paths2': '2 条关联路径'}
GROUP = 'common_target6'
SUPPORT = PROJECT / 'paper_program/mechanism_pressure_20260916/support_enumeration.json'
SUPPORT_SHA = '46bc6633b7cdf00f77f2690800f5bf180c187e8438370b5218635e8006dee688'
MARKERS = ('o', 's', '^', 'D')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def setup():
    font = font_manager.findfont('PingFang SC', fallback_to_default=False)
    plt.rcParams.update({'font.family': ['PingFang SC', 'DejaVu Sans'],
                         'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'axes.unicode_minus': False, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'savefig.facecolor': 'white',
                         'axes.spines.top': False, 'axes.spines.right': False})
    return font


def save(fig, directory, stem):
    paths = []
    for suffix in ('png', 'pdf'):
        path = directory / f'{stem}.{suffix}'
        fig.savefig(path, dpi=190, bbox_inches='tight', pad_inches=.13)
        paths.append(path)
    plt.close(fig)
    return {p.name: sha(p) for p in paths}


def diagram(directory):
    if sha(SUPPORT) != SUPPORT_SHA:
        raise ValueError('frozen support enumeration hash mismatch')
    data = read(SUPPORT)
    panel = data['panels'][0]
    if panel['panel'] != 1 or panel['coordinate_permutation'] != list(range(6)):
        raise ValueError('diagram requires prespecified p1, not a selected example')
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.6))
    fig.subplots_adjust(left=.085, right=.97, top=.79, bottom=.20, wspace=.30)
    fig.suptitle('两种社会训练共现支持（固定 p1）', fontsize=17, y=.985)
    fig.text(.5, .913, '每个格子是一种“食物地点 × 水地点”布局；两臂均训练 18 图、每行每列 3 图',
             ha='center', fontsize=11.5)
    cells = {}
    for ax, arm in zip(axes, ARMS):
        condition = panel['conditions'][arm]
        adjacency = np.asarray(condition['adjacency'], dtype=int)
        target = {tuple(p) for p in condition['target_pairs']}
        held = {tuple(p) for p in condition['other_held_pairs']}
        if (adjacency.shape != (6, 6) or adjacency.sum() != 18 or len(target) != 6 or len(held) != 6
                or set(map(tuple, np.argwhere(adjacency))) & (target | held)):
            raise ValueError('unexpected support diagram schema')
        cells[arm] = {'training': np.argwhere(adjacency).tolist(),
                      'common_P': [list(x) for x in sorted(target)],
                      'other_held': [list(x) for x in sorted(held)]}
        for f in range(6):
            for w in range(6):
                face, label, text_color = '#F7F7F7', '×', '#AAAEB3'
                if adjacency[f, w]:
                    face, label, text_color = COLORS[arm], '', 'white'
                elif (f, w) in target:
                    face, label, text_color = '#FFFFFF', 'P', '#1C2634'
                elif (f, w) in held:
                    face, label, text_color = '#D9DDE2', 'Q', '#515963'
                elif f != w:
                    raise ValueError('unclassified legal layout')
                ax.add_patch(Rectangle((w - .5, f - .5), 1, 1, facecolor=face,
                                       edgecolor='white', linewidth=1.5))
                if label:
                    ax.text(w, f, label, ha='center', va='center', fontsize=15,
                            color=text_color, weight='bold' if label == 'P' else 'normal')
        ax.set(xlim=(-.5, 5.5), ylim=(5.5, -.5), xticks=range(6), yticks=range(6),
               xlabel='水的位置', ylabel='食物的位置')
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        paths = set(condition['target_three_paths'])
        if len(paths) != 1:
            raise ValueError('each prespecified P target must have equal path count')
        ax.set_title(f"{LABELS[arm]} / P 目标\n二部四环：{condition['four_cycles']} 个",
                     color=COLORS[arm], fontsize=13, pad=9)
    handles = [Patch(facecolor=COLORS[ARMS[0]], label='3 路径臂训练图'),
               Patch(facecolor=COLORS[ARMS[1]], label='2 路径臂训练图'),
               Patch(facecolor='white', edgecolor='#68717D', label='P：共同留出 6 图'),
               Patch(facecolor='#D9DDE2', label='Q：其他留出 6 图')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .065),
               ncol=2, frameon=False, fontsize=11, columnspacing=2.2, handlelength=1.6)
    fig.text(.5, .015, '× 为排除的同址布局。关联路径不是行动路线；支持图没有直接提供给主体。',
             ha='center', fontsize=11, color='#3E4650')
    return save(fig, directory, '01_support_design'), cells


def style_success(ax, title, ylabel='双资源成功率 J（%）'):
    ax.set_title(title, loc='left', pad=10)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', color='#C8CDD3', alpha=.55, linewidth=.7)
    ax.set_axisbelow(True)


def result_figure(directory, analysis):
    if analysis['status'] != 'complete':
        raise ValueError('only a complete analysis can be plotted')
    seeds = analysis['seeds']
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError('final figure requires all four sources; no source selection')
    rows = {(r['seed'], r['condition']): r for r in analysis['seed_rows']}
    if set(rows) != {(s, arm) for s in seeds for arm in ARMS}:
        raise ValueError('source by condition matrix is incomplete')
    agg = analysis['aggregate']
    times = [p['update'] for p in agg[ARMS[0]]['curve']]
    if times != [0, 100, 600, 1200, 2100, 2400]:
        raise ValueError('all six formal checkpoints are required')
    evidence = {'seeds': seeds, 'times': times, 'curve': {}, 'endpoint': {},
                'support_endpoints': {}, 'recombination': {},
                'y_limits_percent': [0, 100], 'no_CI_or_source_filter': True}
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.3))
    fig.subplots_adjust(left=.078, right=.985, top=.835, bottom=.20, hspace=.43, wspace=.26)
    fig.suptitle('共同经验支持与未训练配对表达', fontsize=17, y=.99)
    fig.text(.5, .938, '共同 P6 自然表现与固定供体重组；四个来源全部保留，所有面板纵轴均为 0–100%',
             ha='center', fontsize=12)
    handles = [Line2D([0], [0], color=COLORS[a], marker='o', label=LABELS[a], lw=2) for a in ARMS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .916), ncol=2,
               frameon=False, fontsize=12)
    a, b, c, d = axes.flat
    # A: Means are computed from the already source-aggregated analysis. Thin
    # lines show every source, not confidence bands or selected examples.
    for arm in ARMS:
        vals = np.asarray([[p['scores'][GROUP]['pooled']['J'] * 100
                            for p in rows[seed, arm]['curve']] for seed in seeds])
        for trajectory in vals:
            a.plot(times, trajectory, color=COLORS[arm], alpha=.22, lw=.8)
        means = np.asarray([p['scores'][GROUP]['pooled']['J'] * 100 for p in agg[arm]['curve']])
        if not np.allclose(means, vals.mean(0), atol=1e-9):
            raise ValueError('aggregate/source curve mismatch')
        a.plot(times, means, color=COLORS[arm], marker='o', ms=4.5, lw=2.2)
        evidence['curve'][arm] = {'source_percent': vals.tolist(), 'mean_percent': means.tolist()}
    style_success(a, 'A  共同 P6：形成过程')
    a.set_xlim(0, 2400)
    a.set_xticks([0, 600, 1200, 1800, 2400])
    a.set_xlabel('共同学习更新次数')
    # B: Fixed, source-specific horizontal offsets avoid obscuring exact ties.
    offsets = np.linspace(-.10, .10, 4)
    values = np.asarray([[rows[seed, arm]['scores'][GROUP]['pooled']['J'] * 100
                          for arm in ARMS] for seed in seeds])
    for i, seed in enumerate(seeds):
        xs = np.arange(2) + offsets[i]
        b.plot(xs, values[i], color='#7E858E', lw=.9, alpha=.65, zorder=1)
        for j, arm in enumerate(ARMS):
            b.scatter(xs[j], values[i, j], marker=MARKERS[i], color=COLORS[arm],
                      edgecolor='white', linewidth=.4, s=44, zorder=3, clip_on=False)
    b.scatter([0, 1], values.mean(0), marker='_', color='black', s=390, linewidth=2.8,
              zorder=4, clip_on=False)
    style_success(b, 'B  共同 P6：终点逐来源配对')
    b.set_xlim(-.32, 1.32)
    b.set_xticks([0, 1], [LABELS[a] for a in ARMS])
    seed_handles = [Line2D([0], [0], marker=MARKERS[i], color='#777777', lw=0,
                          markersize=5, label=str(seed)) for i, seed in enumerate(seeds)]
    b.legend(handles=seed_handles, ncol=2, fontsize=9.5, frameon=False, loc='upper right')
    b.set_xlabel('灰线连接相同来源；黑横线为均值')
    evidence['endpoint'] = {'source_percent': values.tolist(), 'mean_percent': values.mean(0).tolist()}
    # C: Each arm's train18 is its own support; target P6 is shared.
    for j, arm in enumerate(ARMS):
        vals = np.asarray([[rows[seed, arm]['scores'][group]['pooled']['J'] * 100
                            for group in ('train18', GROUP)] for seed in seeds])
        xs = np.arange(2) + (j - .5) * .34
        c.bar(xs, vals.mean(0), width=.30, color=COLORS[arm], alpha=.75, zorder=2)
        for i in range(4):
            c.scatter(xs + offsets[i] * .60, vals[i], marker=MARKERS[i], s=25,
                      color=COLORS[arm], edgecolor='white', linewidth=.5, zorder=3,
                      clip_on=False)
        evidence['support_endpoints'][arm] = vals.tolist()
    style_success(c, 'C  终点：各自训练图与共同留出图')
    c.set_xticks([0, 1], ['各自训练 18 图', '共同留出 P6'])
    c.set_xlabel('训练集合不同；P6 测试集合相同')
    # D: Both fixed donor assignments, observed and recoded reference, retained.
    # Small circles show four source observed values; the larger circle and X
    # are aggregate observed and mean over the fixed199 recoding references.
    for j, arm in enumerate(ARMS):
        evidence['recombination'][arm] = {}
        for k, assignment in enumerate(('FW', 'WF')):
            metric = f'recombine_{assignment}_J'
            summary = agg[arm]['recombination_null']['pooled'][metric]
            source = [rows[seed, arm]['recombination_null']['pooled'][metric] for seed in seeds]
            if any(len(v['null']) != 199 for v in [summary] + source):
                raise ValueError('all199 prespecified complete-code bijections are required')
            observed = np.asarray([v['observed'] * 100 for v in source])
            null = np.asarray([v['null_mean'] * 100 for v in source])
            if not np.allclose([summary['observed'] * 100, summary['null_mean'] * 100],
                               [observed.mean(), null.mean()], atol=1e-9):
                raise ValueError('null/source aggregate mismatch')
            x = k + (j - .5) * .36
            for i in range(4):
                d.scatter(x + offsets[i] * .55, observed[i], marker=MARKERS[i], s=23,
                          color=COLORS[arm], alpha=.62, linewidth=.45, edgecolor='white',
                          zorder=2, clip_on=False)
            d.plot([x, x], [null.mean(), observed.mean()], color=COLORS[arm], ls=':', lw=1.3)
            d.scatter(x, observed.mean(), color=COLORS[arm], s=72, marker='o',
                      edgecolor='white', linewidth=.8, zorder=4, clip_on=False)
            d.scatter(x, null.mean(), color=COLORS[arm], s=75, marker='x', linewidth=2,
                      zorder=5, clip_on=False)
            evidence['recombination'][arm][assignment] = {
                'source_observed_percent': observed.tolist(), 'source_null_mean_percent': null.tolist(),
                'observed_mean_percent': float(observed.mean()), 'null_mean_percent': float(null.mean())}
    style_success(d, 'D  P6 人工重组：两种固定供体分配')
    d.set_xlim(-.45, 1.45)
    d.set_xticks([0, 1], ['FW：食物 token0 + 水 token1', 'WF：水 token0 + 食物 token1'])
    d.tick_params(axis='x', labelsize=10)
    d.set_xlabel('两种分配均报告，不择优')
    d.legend(handles=[Line2D([0], [0], color='#555555', marker='o', lw=0,
                            markersize=6, label='原始重组均值'),
                      Line2D([0], [0], color='#555555', marker='x', lw=0,
                            markersize=7, label='199 双射参照均值')],
             loc='upper right', fontsize=10, frameon=False)
    fig.text(.5, .108, '来源内先平均 3 个坐标重复和 2 个通信方向，再平均 4 个来源；细线、小点不代表置信区间。',
             ha='center', fontsize=11)
    fig.text(.5, .066, 'P6 是一一配对，单资源位置即可确定另一资源；自然高分或人工重组高分均不能独立证明语法。',
             ha='center', fontsize=11)
    fig.text(.5, .025, '人工重组由实验者选择训练布局供体；199 次完整码重命名是结构参照，不是独立训练重复。',
             ha='center', fontsize=11, color='#414954')
    return save(fig, directory, '02_support_outcomes'), evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--diagram-only', action='store_true')
    parser.add_argument('--preview-dir', type=Path, default=ROOT / 'figures_preview')
    args = parser.parse_args()
    font = setup()
    if args.diagram_only:
        directory = args.preview_dir.resolve()
        formal = ROOT / 'results/support_001'
        if directory == formal or formal in directory.parents:
            parser.error('diagram-only must not create the formal runner output directory')
        directory.mkdir(parents=True, exist_ok=True)
        outputs, cells = diagram(directory)
        record = {'status': 'diagram_preview', 'support_sha256': SUPPORT_SHA,
                  'plot_source_sha256': sha(__file__), 'font_path': font,
                  'figure_sha256': outputs, 'matrix_cells': cells, 'model_results_read': False}
    else:
        if args.out is None:
            parser.error('--out is required for the final two figures')
        out = args.out.resolve()
        if not out.is_dir() or not (out / 'analysis.json').is_file():
            parser.error('existing completed output required; will not create the runner parent')
        analysis = read(out / 'analysis.json')
        validation = read(out / 'raw_validation.json')
        if not validation['passed'] or not analysis['formal']:
            raise ValueError('passed independent formal raw analysis required')
        if validation.get('analysis_sha256') != sha(out / 'analysis.json'):
            raise ValueError('raw validation does not bind current analysis')
        directory = out / 'figures'
        directory.mkdir(exist_ok=True)
        output1, cells = diagram(directory)
        output2, values = result_figure(directory, analysis)
        record = {'status': 'rendered_pending_visual_QA', 'support_sha256': SUPPORT_SHA,
                  'plot_source_sha256': sha(__file__), 'analysis_sha256': sha(out / 'analysis.json'),
                  'raw_validation_sha256': sha(out / 'raw_validation.json'), 'font_path': font,
                  'figure_sha256': {**output1, **output2}, 'matrix_cells': cells,
                  'plotted_values': values, 'axes_fixed_before_results': True,
                  'source_and_assignment_selection': False}
    write(directory / 'figure_source.json', record)
    print(json.dumps({'status': record['status'], 'directory': str(directory),
                      'files': list(record['figure_sha256'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
