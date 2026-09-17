"""Render v0.32 joint-payoff design and outcomes from independent analysis."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
import support
sys.path.insert(0, str(PROJECT / 'redesign_v0.9/.analysis_deps'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle

CONDITIONS = ('fixed_partners', 'rotating_partners')
COLORS = {'fixed_partners': '#1B6685', 'rotating_partners': '#C4513A'}
LABELS = {'fixed_partners': '固定伙伴', 'rotating_partners': '轮换伙伴'}
PRIVATE = {0: '#3C78A8', 1: '#D77A34'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def pct(value):
    return 100.0 * float(value)


def setup():
    try:
        font = font_manager.findfont('PingFang SC', fallback_to_default=False)
    except Exception:
        font = font_manager.findfont('DejaVu Sans')
    plt.rcParams.update({
        'font.family': ['PingFang SC', 'DejaVu Sans'], 'font.size': 10.5,
        'axes.titlesize': 12.5, 'axes.labelsize': 10.5,
        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'axes.unicode_minus': False, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'savefig.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    return font


def save(fig, directory, stem):
    result = {}
    for ext in ('png', 'pdf'):
        path = directory / f'{stem}.{ext}'
        fig.savefig(path, dpi=220, bbox_inches='tight', pad_inches=.14)
        result[path.name] = sha(path)
    plt.close(fig)
    return result


def draw_agent(ax, xy, label, private_type):
    x, y = xy
    ax.add_patch(Circle((x, y), .23, facecolor=PRIVATE[private_type], edgecolor='white', linewidth=1.8, zorder=3))
    ax.text(x, y, label, color='white', ha='center', va='center', weight='bold', fontsize=12, zorder=4)
    ax.text(x, y - .38, f'私有类型 {private_type}', color=PRIVATE[private_type], ha='center', va='top', fontsize=9)


def diagram(directory):
    design = read(ROOT / 'support_design.json')
    fig = plt.figure(figsize=(12.2, 6.25))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.1, 1.0), wspace=.28, left=.055, right=.975, top=.80, bottom=.23)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('A  伙伴制度：训练中的唯一操纵', loc='left', pad=10)
    positions = {0: (0, 1.15), 1: (1.8, 1.15), 2: (0, -.55), 3: (1.8, -.55)}
    for i in range(4):
        draw_agent(ax, positions[i], str(i), int(design['population']['private_types'][i]))
    # Fixed edges are solid; the alternating matching is shown as dashed arrows.
    for i, j in design['population']['fixed_matchings'][0]:
        x1, y1 = positions[i]; x2, y2 = positions[j]
        ax.add_patch(FancyArrowPatch((x1 + .24, y1), (x2 - .24, y2), arrowstyle='<->',
                                     mutation_scale=12, linewidth=2.3, color=COLORS['fixed_partners']))
    for i, j in design['population']['rotating_matchings'][1]:
        x1, y1 = positions[i]; x2, y2 = positions[j]
        ax.add_patch(FancyArrowPatch((x1 + .18, y1 + .08), (x2 - .18, y2 + .08), arrowstyle='<->',
                                     mutation_scale=11, linewidth=1.9, linestyle=(0, (4, 3)),
                                     color=COLORS['rotating_partners']))
    ax.text(.9, 1.72, '固定伙伴： (0,1) + (2,3) 每一步重复', ha='center', color=COLORS['fixed_partners'], fontsize=10)
    ax.text(.9, -1.30, '轮换伙伴：偶数步固定；奇数步 (0,3) + (2,1)', ha='center', color=COLORS['rotating_partners'], fontsize=10)
    ax.text(.9, -1.70, '四个主体、两种冻结私有视觉类型；每个更新四个有向通信批次', ha='center', color='#4d5964', fontsize=9.5)
    ax.set(xlim=(-.55, 2.35), ylim=(-2.00, 2.05), aspect='equal'); ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title('B  共同目标与训练支持（p1）', loc='left', pad=10)
    target = {tuple(x) for x in design['target12_pairs']}
    train = {tuple(x) for x in design['training12_pairs']}
    for f in range(6):
        for w in range(6):
            if f == w:
                face, label, color = '#454d56', '×', 'white'
            elif (f, w) in target:
                face, label, color = '#F3C66A', 'T', '#5E3E00'
            elif (f, w) in train:
                face, label, color = '#76A9C9', 'S', '#12364A'
            else:
                face, label, color = '#E7EAED', '', '#555'
            ax.add_patch(Rectangle((w - .5, f - .5), 1, 1, facecolor=face, edgecolor='white', linewidth=1.3))
            if label:
                ax.text(w, f, label, ha='center', va='center', fontsize=11, color=color, weight='bold')
    ax.set(xlim=(-.5, 5.5), ylim=(5.5, -.5), xticks=range(6), yticks=range(6), xlabel='水位置', ylabel='食物位置')
    ax.set_aspect('equal'); ax.tick_params(length=0)
    ax.legend(handles=[Patch(facecolor='#F3C66A', label='T：12条共同目标边'), Patch(facecolor='#76A9C9', label='S：12条训练边'), Patch(facecolor='#E7EAED', label='留出布局'), Patch(facecolor='#454d56', label='×：同址排除')],
              loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2, frameon=False, fontsize=9)
    fig.suptitle('v0.32：联合收益压力下的固定伙伴与轮换伙伴', fontsize=17, y=.98)
    fig.text(.5, .025, '两种条件共享目标图、训练图、视觉输入和采样流；只改变伙伴配对时序，并让一对主体共享四行动联合收益。', ha='center', fontsize=10.5, color='#3f4b57')
    cells = {'target_pairs': sorted(map(list, target)), 'training_pairs': sorted(map(list, train)), 'fixed_matching': design['population']['fixed_matchings'][0], 'rotating_matching_odd': design['population']['rotating_matchings'][1]}
    return save(fig, directory, '01_partner_design'), cells


def axis_percent(ax, title, ylabel='百分比'):
    ax.set_title(title, loc='left', pad=8)
    ax.set_ylim(0, 100); ax.set_yticks(np.arange(0, 101, 20)); ax.set_ylabel(ylabel)
    ax.grid(axis='y', color='#C9CED3', alpha=.58, linewidth=.7); ax.set_axisbelow(True)


def result_figure(directory, analysis):
    if analysis['status'] != 'complete' or not analysis['formal']:
        raise ValueError('formal independent analysis required')
    seeds = analysis['seeds']; aggregate = analysis['aggregate']
    rows = {(r['seed'], r['condition']): r for r in analysis['seed_rows']}
    times = [x['update'] for x in aggregate[CONDITIONS[0]]['curve']]
    if set(rows) != {(s, c) for s in seeds for c in CONDITIONS}:
        raise ValueError('incomplete formal matrix')
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.0))
    fig.subplots_adjust(left=.06, right=.985, top=.83, bottom=.16, hspace=.47, wspace=.27)
    fig.suptitle('v0.32：联合收益提高了消息复用，但语义迁移仍弱', fontsize=17, y=.985)
    fig.text(.5, .942, '细线为4个初始化来源；粗线为来源均值。所有纵轴预先固定为0–100%。', ha='center', fontsize=10.5)
    handles = [Line2D([0], [0], color=COLORS[c], marker='o', lw=2.2, label=LABELS[c]) for c in CONDITIONS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .912), ncol=2, frameon=False, fontsize=10.5)
    evidence = {'times': times, 'target_J': {}, 'agreement': {}, 'endpoint': {}, 'pair_endpoint': {}, 'recombination': {}}

    # A: target and train curves.
    ax = axes[0, 0]
    for c in CONDITIONS:
        vals = np.asarray([[pct(row['scores']['target12']['pooled']['J']) for row in rows[s, c]['curve']] for s in seeds])
        mean = np.asarray([pct(item['scores']['target12']['pooled']['J']) for item in aggregate[c]['curve']])
        if not np.allclose(mean, vals.mean(0), atol=1e-9): raise ValueError('target curve mismatch')
        for v in vals: ax.plot(times, v, color=COLORS[c], alpha=.23, lw=.8)
        ax.plot(times, mean, color=COLORS[c], marker='o', ms=4, lw=2.2)
        evidence['target_J'][c] = {'source_percent': vals.tolist(), 'mean_percent': mean.tolist()}
    axis_percent(ax, 'A  目标12：双资源自然成功 J', 'J（%）'); ax.set_xlabel('群体更新次数'); ax.set_xlim(0, 2400); ax.set_xticks(times)

    # B: same-type agreement, including both message tokens.
    ax = axes[0, 1]
    for c in CONDITIONS:
        vals = {k: np.asarray([pct(item['agreement'][k]) for item in aggregate[c]['agreement']]) for k in ('full_message_agreement', 'token0_agreement', 'token1_agreement')}
        for key, ls, alpha in (('full_message_agreement', '-', 1), ('token0_agreement', '--', .75), ('token1_agreement', ':', .75)):
            ax.plot(times, vals[key], color=COLORS[c], linestyle=ls, lw=2 if alpha == 1 else 1.25, alpha=alpha)
        evidence['agreement'][c] = {k: v.tolist() for k, v in vals.items()}
    axis_percent(ax, 'B  同私有类型主体的消息一致率', '一致率（%）'); ax.set_xlabel('群体更新次数'); ax.set_xlim(0, 2400); ax.set_xticks(times)
    ax.legend(handles=[Line2D([0], [0], color='#555', lw=2, label='完整双token'), Line2D([0], [0], color='#555', lw=1.3, ls='--', label='token0'), Line2D([0], [0], color='#555', lw=1.3, ls=':', label='token1')], frameon=False, fontsize=8.5, loc='lower right')

    # C: endpoint components.
    ax = axes[0, 2]
    labels = ['训练12 J', '目标12 J', '目标食物', '目标水', '伙伴对均值']
    x = np.arange(len(labels)); width = .35
    for j, c in enumerate(CONDITIONS):
        m = aggregate[c]['scores']; target = m['target12']['pooled']; train = m['train12']['pooled']
        values = [pct(train['J']), pct(target['J']), pct(target['food']), pct(target['water']), pct((target['food_partner_pair_J'] + target['water_partner_pair_J']) / 2)]
        ax.bar(x + (j - .5) * width, values, width=width, color=COLORS[c], alpha=.82, label=LABELS[c]); evidence['endpoint'][c] = dict(labels=labels, percent=values)
    axis_percent(ax, 'C  终点表现分解', '成功率（%）'); ax.set_xticks(x, labels, rotation=20, ha='right'); ax.legend(frameon=False, fontsize=8.5, loc='upper left')

    # D: each ordered cross-type protocol.
    ax = axes[1, 0]
    pair_keys = sorted(aggregate[CONDITIONS[0]]['pair_endpoint_J'])
    x = np.arange(len(pair_keys))
    for j, c in enumerate(CONDITIONS):
        values = [pct(aggregate[c]['pair_endpoint_J'][k]) for k in pair_keys]
        ax.bar(x + (j - .5) * width, values, width=width, color=COLORS[c], alpha=.82, label=LABELS[c]); evidence['pair_endpoint'][c] = dict(keys=pair_keys, percent=values)
    axis_percent(ax, 'D  8个有序跨类型伙伴协议', '目标12 J（%）'); ax.set_xticks(x, [k.replace('i', '').replace('_j', '→') for k in pair_keys], rotation=45, ha='right'); ax.legend(frameon=False, fontsize=8.5, loc='upper left')

    # E: held-out transfer.
    ax = axes[1, 1]
    labels = ['训练12', '目标12', '留出18']
    x = np.arange(len(labels))
    for j, c in enumerate(CONDITIONS):
        s = aggregate[c]['scores']; values = [pct(s['train12']['pooled']['J']), pct(s['target12']['pooled']['J']), pct(s['held18']['pooled']['J'])]
        ax.bar(x + (j - .5) * width, values, width=width, color=COLORS[c], alpha=.82, label=LABELS[c])
    axis_percent(ax, 'E  训练—目标—留出迁移', 'J（%）'); ax.set_xticks(x, labels); ax.legend(frameon=False, fontsize=8.5, loc='upper left')

    # F: recombination against full-code relabeling reference.
    ax = axes[1, 2]
    labels = ['FW 原始', 'FW 参照', 'WF 原始', 'WF 参照']; x = np.arange(len(labels)); evidence['recombination'] = {}
    for j, c in enumerate(CONDITIONS):
        n = aggregate[c]['recombination_null']['pooled']; vals = []
        for orient in ('FW', 'WF'):
            z = n[f'recombine_{orient}_J']; vals.extend([pct(z['observed']), pct(z['null_mean'])])
        ax.bar(x + (j - .5) * width, vals, width=width, color=COLORS[c], alpha=.82, label=LABELS[c]); evidence['recombination'][c] = dict(labels=labels, percent=vals)
    axis_percent(ax, 'F  片段重组与整码双射参照', '目标12 J（%）'); ax.set_xticks(x, labels, rotation=20, ha='right'); ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.text(.5, .087, '伙伴对要求同一资源的两个目标边同时正确；它远低于单边准确率时，单资源策略仍是合理解释。', ha='center', fontsize=10.2)
    fig.text(.5, .052, '完整消息一致率只比较同一私有类型的独立主体副本；重组与199项整码双射均为离线结构参照。', ha='center', fontsize=10.2, color='#43505c')
    return save(fig, directory, '02_partner_outcomes'), evidence


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path); ap.add_argument('--diagram-only', action='store_true'); ap.add_argument('--preview-dir', type=Path, default=ROOT / 'figures_preview'); args = ap.parse_args(); font = setup()
    if args.diagram_only:
        directory = args.preview_dir.resolve(); directory.mkdir(parents=True, exist_ok=True)
        figures, cells = diagram(directory)
        record = {'status': 'diagram_preview', 'support_design_sha256': sha(ROOT / 'support_design.json'), 'plot_source_sha256': sha(__file__), 'font_path': font, 'figure_sha256': figures, 'matrix_cells': cells, 'model_results_read': False}
    else:
        if args.out is None: ap.error('--out is required')
        out = args.out.resolve()
        if not out.is_dir() or not (out / 'analysis.json').is_file(): ap.error('existing analyzed output required')
        analysis = read(out / 'analysis.json'); validation = read(out / 'raw_validation.json')
        if not validation['passed'] or not analysis['formal'] or validation.get('analysis_sha256') != sha(out / 'analysis.json'): raise ValueError('independent formal analysis binding failed')
        directory = out / 'figures'; directory.mkdir(exist_ok=True)
        first, cells = diagram(directory); second, evidence = result_figure(directory, analysis)
        record = {'status': 'rendered_pending_visual_QA', 'support_design_sha256': sha(ROOT / 'support_design.json'), 'plot_source_sha256': sha(__file__), 'analysis_sha256': sha(out / 'analysis.json'), 'raw_validation_sha256': sha(out / 'raw_validation.json'), 'font_path': font, 'figure_sha256': {**first, **second}, 'matrix_cells': cells, 'plotted_values': evidence, 'axes_fixed_before_results': True, 'source_and_condition_selection': False}
    write(directory / 'figure_source.json', record)
    print(json.dumps({'status': record['status'], 'directory': str(directory), 'files': list(record['figure_sha256'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
