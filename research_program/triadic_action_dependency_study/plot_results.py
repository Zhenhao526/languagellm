"""Three static scientific figures from a completed read-only summary only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

SEEDS = (51101, 51102, 51103, 51104)
CONDITIONS = ('FI_silent', 'FI_live', 'PL_silent', 'PL_live', 'LL_silent', 'LL_live')
PARTS = ('train', 'new_needs', 'new_layouts', 'new_needs_and_layouts')
STEPS = (0, 100, 500, 1500, 3000, 6000)
AXES = ('kind', 'length', 'destination', 'macro')
FONT = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
PART_NAMES = dict(train='训练关系／布局', new_needs='新需求关系', new_layouts='新布局', new_needs_and_layouts='需求关系与布局双留出')
INFO_NAMES = dict(FI='需求与布局完整可见', PL='私有需求／公开布局', LL='私有需求／局部布局')
AXIS_NAMES = dict(kind='对象种类', length='长短属性', destination='目的地', macro='三轴等权：主量')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content(row, axis='macro'):
    if axis == 'macro': return row['content']['macro']['both_endpoints_apt']
    return row['content']['by_axis'][axis]['metrics']['both_endpoints_apt']


def figure_data(summary):
    assert summary['status'] == 'completed_read_only_summary'
    rows = summary['compact_records']
    finals = {(r['seed'], r['condition'], r['partition'], r['mode']): r for r in rows if r['kind'] == 'full_endpoint'}
    monitors = {(r['seed'], r['condition'], r['partition'], r['mode'], r['checkpoint']): r for r in rows if r['kind'] == 'two_background_monitor'}
    assert len(rows) == 1344 and len(finals) == 192 and len(monitors) == 1152
    for row in finals.values(): assert row['semantic_scope']['coverage'] == 'full_partition'
    for row in monitors.values():
        assert row['semantic_scope']['coverage'] == 'all_needs_fixed_background_subset'
        assert row['semantic_scope']['saved_background_count'] == 2
    endpoint = {axis: {cond: [content(finals[s, cond, 'new_needs_and_layouts', 'natural'], axis) for s in SEEDS]
        for cond in ('PL_live', 'LL_live')} for axis in AXES}
    world = {part: {cond: {mode: {metric: [finals[s, cond, part, mode]['world'][metric] for s in SEEDS]
        for metric in ('reward_mean', 'full_success_rate')} for mode in ('natural', 'closed')}
        for cond in CONDITIONS} for part in PARTS}
    trajectory = {part: {cond: {mode: [[content(monitors[s, cond, part, mode, t]) for t in STEPS] for s in SEEDS]
        for mode in ('natural', 'closed')} for cond in CONDITIONS} for part in PARTS}
    primary = summary['primary_comparison']['primary']
    assert primary['partition'] == 'new_needs_and_layouts' and primary['contrast'] == 'PL_live_minus_LL_live'
    for row, pl, ll in zip(primary['seed_values'], endpoint['macro']['PL_live'], endpoint['macro']['LL_live']):
        assert abs(row['difference'] - (pl - ll)) <= 1e-14
    return dict(endpoint=endpoint, world=world, trajectory=trajectory, primary=primary,
        scope='Complete endpoint and fixed-two-background update6000 are distinct records; no full endpoint inserted into monitor curves.')


def plot(summary_path, out):
    summary_path, out = Path(summary_path).resolve(), Path(out).resolve()
    assert not out.exists(), 'Refuse to overwrite figures'
    summary = json.loads(summary_path.read_text()); data = figure_data(summary)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    assert FONT.is_file(), 'Required Chinese font missing'
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(FONT)).get_name(),
        'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.dpi': 120, 'savefig.dpi': 180, 'axes.titleweight': 'normal'})
    out.mkdir(parents=True); files = []; x = np.arange(4); labels = [str(s) for s in SEEDS]
    colors = dict(FI='#69717a', PL='#14759b', LL='#c47832')
    def finish(fig, name):
        for ext in ('.png', '.pdf'):
            path = out / (name + ext); fig.savefig(path, bbox_inches='tight')
            files.append(dict(path=path.name, sha256=sha(path)))
        plt.close(fig)
    def signed_limit(values, floor=1.):
        return max(floor, max(abs(float(v)) for v in values) * 1.18)

    fig, axes = plt.subplots(2, 4, figsize=(14.7, 7.3), sharey='row'); deltas = []
    for j, axis in enumerate(AXES):
        pl, ll = [np.asarray(data['endpoint'][axis][cond]) for cond in ('PL_live', 'LL_live')]
        delta = (pl - ll) * 100; deltas.extend(delta)
        for i in range(4): axes[0, j].plot([i - .08, i + .08], [ll[i], pl[i]], color='#b6c0c7', lw=1)
        axes[0, j].scatter(x - .08, ll, c=colors['LL'], marker='s', label='LL：局部布局', s=35)
        axes[0, j].scatter(x + .08, pl, c=colors['PL'], marker='o', label='PL：公开布局', s=35)
        axes[0, j].set_title(AXIS_NAMES[axis]); axes[0, j].set_ylim(-.025, 1.025)
        axes[1, j].axhline(0, color='#777', lw=.8)
        axes[1, j].vlines(x, 0, delta, color=colors['PL'], lw=1.2)
        axes[1, j].scatter(x, delta, c=colors['PL'], s=35)
        for row in (0, 1):
            axes[row, j].set_xticks(x, labels, rotation=35); axes[row, j].grid(axis='y', alpha=.15)
    limit = signed_limit(deltas)
    for ax in axes[1]: ax.set_ylim(-limit, limit)
    axes[0, 0].set_ylabel('内容两端听者均适切比例')
    axes[1, 0].set_ylabel('PL_live − LL_live（百分点）')
    axes[0, 0].legend(fontsize=8, loc='upper right')
    fig.suptitle('6000 步完整双留出终点：三个内容轴与四个配对种子', fontsize=15, y=.99)
    fig.text(.5, .015, '主要量为右侧三轴等权差；未按成功或固定搭档筛选。三个轴正确行动数均为 1，仍不能认为各轴全部难度相等。', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .96)); finish(fig, 'full_endpoint_content_all_seeds')

    fig, axes = plt.subplots(4, 4, figsize=(15.8, 12.4), sharey='row')
    close_deltas = {metric: [] for metric in ('reward_mean', 'full_success_rate')}
    for j, part in enumerate(PARTS):
        for ci, condition in enumerate(CONDITIONS):
            info, visibility = condition.split('_'); offset = (ci - 2.5) * .055
            marker = 'o' if visibility == 'live' else 'x'
            label = info + ('开放' if visibility == 'live' else '静默')
            for ri, metric in enumerate(('reward_mean', 'full_success_rate')):
                v = np.asarray(data['world'][part][condition]['natural'][metric])
                axes[ri, j].plot(x + offset, v, color=colors[info], marker=marker, ls='-' if visibility == 'live' else ':', lw=.8, ms=4, alpha=.9, label=label)
                axes[ri, j].set_ylim(-.025, 1.025)
                if visibility == 'live':
                    closed = np.asarray(data['world'][part][condition]['closed'][metric]); delta = 100 * (v - closed)
                    close_deltas[metric].extend(delta)
                    axes[ri + 2, j].plot(x + (ci // 2 - 1) * .08, delta, color=colors[info], marker='o', lw=.9, ms=4, label=info)
        axes[0, j].set_title(PART_NAMES[part], fontsize=10)
        for ri in range(4):
            axes[ri, j].set_xticks(x, labels, rotation=35, fontsize=8)
            axes[ri, j].grid(axis='y', alpha=.15)
            if ri >= 2: axes[ri, j].axhline(0, color='#777', lw=.8)
    for ri, metric in enumerate(('reward_mean', 'full_success_rate')):
        limit = signed_limit(close_deltas[metric])
        for ax in axes[ri + 2]: ax.set_ylim(-limit, limit)
    for row, text in enumerate(('原生平均 R', '团队满分率', 'R：自然 − 关闭（×100）', '满分率：自然 − 关闭（百分点）')):
        axes[row, 0].set_ylabel(text)
    axes[0, 0].legend(fontsize=7, ncol=2, loc='upper right')
    axes[2, 0].legend(fontsize=8, ncol=3, loc='upper right')
    fig.suptitle('相同训练预算的完整终点：任务表现与关闭他人通道后的变化', fontsize=15, y=.992)
    fig.text(.5, .012, '上两行保留六个条件；下两行对三种 live 政策从第一窗关闭并重生，未重新训练。四种子均保留；关闭效应本身不能赋予消息词义。', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .97)); finish(fig, 'full_endpoint_task_and_closure')

    fig, axes = plt.subplots(3, 4, figsize=(16, 10.4), sharex=True, sharey=True)
    tx = np.arange(len(STEPS))
    for i, info in enumerate(('FI', 'PL', 'LL')):
        for j, part in enumerate(PARTS):
            ax = axes[i, j]
            for condition, mode, color, style, label in (
                    (info + '_silent', 'natural', '#777777', ':', '从头静默'),
                    (info + '_live', 'natural', colors[info], '-', '自然消息'),
                    (info + '_live', 'closed', '#a63849', '--', '当前政策关闭通道')):
                y = np.asarray(data['trajectory'][part][condition][mode])
                for row in y: ax.plot(tx, row, color=color, ls=style, lw=.65, alpha=.18)
                ax.plot(tx, y.mean(0), color=color, ls=style, lw=1.65, marker='o', ms=3, label=label)
            ax.set_ylim(-.025, 1.025); ax.grid(axis='y', alpha=.15)
            ax.set_xticks(tx, [str(t) for t in STEPS], rotation=45, fontsize=8)
            if i == 0: ax.set_title(PART_NAMES[part], fontsize=10)
            if j == 0: ax.set_ylabel(info + '：内容两端均适切\n' + INFO_NAMES[info], fontsize=9)
            if i == 2: ax.set_xlabel('保存更新数（检查点等距展示）', fontsize=9)
    axes[0, 0].legend(fontsize=7, loc='upper right')
    fig.suptitle('形成监测：各分区全部需求 × 两个固定背景，六个保存时点', fontsize=15, y=.99)
    fig.text(.5, .013, '细线为全部四种子，粗线为均值；横轴等距不代表更新间隔相等。6000 点仍是两个背景，未插入完整终点。PL/LL 静默与关闭的零为结构参照。', ha='center', fontsize=8.8)
    fig.tight_layout(rect=(0, .045, 1, .966)); finish(fig, 'two_background_content_trajectory')
    (out / 'plot_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    receipt = dict(status='plotted_requires_actual_visual_review', completed_at=datetime.now(timezone.utc).isoformat(),
        summary_path=str(summary_path), summary_sha256=sha(summary_path), plot_source_sha256=sha(__file__),
        python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
        font_path=str(FONT), font_sha256=sha(FONT), files=files, plot_data_sha256=sha(out / 'plot_data.json'),
        neural_forward_calls=0, training_calls=0, complete_endpoint_separate_from_monitor=True)
    (out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args(); print(json.dumps(plot(args.summary, args.out), ensure_ascii=False))
