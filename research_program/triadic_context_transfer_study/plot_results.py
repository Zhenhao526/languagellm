"""Three standalone figures from a completed read-only transfer summary."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import numpy as np

SEEDS = (51101, 51102, 51103, 51104)
PARTS = ('train', 'new_needs', 'new_layouts', 'new_needs_and_layouts')
PART_LABELS = ('训练支持', '新需求关系', '新布局', '新需求关系与布局')
COLORS = ('#176B91', '#C56B29')


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def extract(summary):
    if summary['status'] != 'completed_read_only_summary':
        raise ValueError('A complete read-only summary is required')
    records = summary['policies']
    lookup = {(row['seed'], row['condition']): row for row in records}
    expected = {(s, c) for s in SEEDS for c in ('PL_silent', 'PL_live', 'LL_silent', 'LL_live')}
    if len(records) != 16 or set(lookup) != expected:
        raise ValueError('All 16 policies required')
    rows = []
    for seed in SEEDS:
        for condition in ('PL_live', 'LL_live'):
            for part in PARTS:
                cell = lookup[seed, condition]['partitions'][part]
                for layer in ('eligible', 'all_other'):
                    data = cell[layer]
                    rows.append(dict(seed=seed, condition=condition, partition=part, support=layer,
                        T=data['contrast']['macro']['target_apt_gain'],
                        C_difference=data['contrast']['macro']['conservative_target_apt_gain'],
                        by_axis={a: data['contrast']['by_axis'][a]['values']['target_apt_gain']
                                 for a in ('kind', 'length', 'destination')},
                        same=data['remote_same_both']['macro'], opposite=data['remote_opposite_both']['macro']))
    return rows


def plot(summary_path, out):
    summary_path, out = Path(summary_path).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('Refuse to overwrite figures')
    summary = read(summary_path)
    data = extract(summary)
    out.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if not font.is_file():
        raise ValueError('Required Chinese font is unavailable')
    plt.rcParams.update({'font.family': FontProperties(fname=font).get_name(), 'axes.unicode_minus': False,
        'font.size': 10, 'axes.titlesize': 12, 'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': 170})
    outputs = {}
    lookup = {(r['seed'], r['condition'], r['partition'], r['support']): r for r in data}

    def save(fig, name):
        for suffix in ('.png', '.pdf'):
            target = out / (name + suffix)
            fig.savefig(target, bbox_inches='tight', facecolor='white')
            outputs[target.name] = dict(sha256=sha(target), bytes=target.stat().st_size)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.1))
    for ax, axis, label in zip(axes.flat, ('kind', 'length', 'destination', 'macro'), ('对象种类', '长短属性', '目的地', '三轴等权平均')):
        for offset, condition, color, caption in zip((-.10, .10), ('PL_live', 'LL_live'), COLORS, ('PL：布局公开', 'LL：局部观察')):
            values = [lookup[s, condition, PARTS[-1], 'eligible'] for s in SEEDS]
            y = np.asarray([r['T'] if axis == 'macro' else r['by_axis'][axis] for r in values]) * 100
            ax.plot(np.arange(4) + offset, y, 'o-', color=color, linewidth=1, label=caption)
        ax.axhline(0, color='#777777', linewidth=.7)
        ax.set_xticks(range(4), [str(s) for s in SEEDS]); ax.set_title(label)
        ax.set_ylabel('相反需求包 − 同需求包（百分点）')
        ax.grid(axis='y', alpha=.18)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('完整双留出：目标材料两端均换位时的整包迁移作用', fontsize=15, y=.985)
    fig.text(.5, .01, '每点为一个既有政策种子；两个方向等权。听者单方向正确率差，不是上一轮两端同时正确率。', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95)); save(fig, 'eligible_transfer_all_seeds')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.7))
    markers = ('o', 's', '^', 'D')
    for column, condition in enumerate(('PL_live', 'LL_live')):
        for index, seed in enumerate(SEEDS):
            r = lookup[seed, condition, PARTS[-1], 'eligible']
            top = [r['same']['target_apt'], r['opposite']['target_apt'],
                   r['same']['copy_donor_same_target_apt'], r['same']['copy_donor_opposite_target_apt']]
            axes[0, column].plot(np.arange(4) + (index - 1.5) * .035, np.asarray(top) * 100,
                marker=markers[index], linewidth=.8, alpha=.85, label=str(seed))
            bottom = [r['same']['conservative_gate'], r['same']['conservative_target_apt'], r['opposite']['conservative_target_apt']]
            axes[1, column].plot(np.arange(3) + (index - 1.5) * .035, np.asarray(bottom) * 100,
                marker=markers[index], linewidth=.8, alpha=.85, label=str(seed))
        axes[0, column].set_xticks(range(4), ('同需求包', '相反需求包', '复制同需求\n供体动作', '复制相反需求\n供体动作'))
        axes[1, column].set_xticks(range(3), ('共同门 G 通过', '同需求包 C', '相反需求包 C'))
        for row in range(2):
            axes[row, column].set_ylim(-1, 101); axes[row, column].set_ylabel('原权重下比例（%）')
            axes[row, column].grid(axis='y', alpha=.18)
        axes[0, column].set_title(condition.replace('_live', '') + '：反事实目标适切与复制参照')
        axes[1, column].set_title('C = G × 反事实目标适切；不缩减分母')
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle('完整双留出：绝对率、供体实际动作与共同门', fontsize=15, y=.985)
    fig.text(.5, .01, '主量支持、三轴等权；G 随政策及方向可变，C 不是控制所有机制后的直接效应。', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95)); save(fig, 'target_rates_copy_and_gate')

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    for ax, part, label in zip(axes.flat, PARTS, PART_LABELS):
        for offset, support, style, caption in zip((-.10, .10), ('eligible', 'all_other'), ('o-', 's--'), ('两目标均换位', '全部其他布局')):
            y = [lookup[s, 'PL_live', part, support]['T'] - lookup[s, 'LL_live', part, support]['T'] for s in SEEDS]
            ax.plot(np.arange(4) + offset, np.asarray(y) * 100, style, linewidth=1, label=caption)
        ax.axhline(0, color='#777777', linewidth=.7); ax.grid(axis='y', alpha=.18)
        ax.set_xticks(range(4), [str(s) for s in SEEDS]); ax.set_title(label)
        ax.set_ylabel('T_PL − T_LL（百分点）')
    axes[0, 0].legend(frameon=False)
    fig.suptitle('全部四分区：主量支持与完整供体支持', fontsize=15, y=.985)
    fig.text(.5, .01, '保留全部四个配对种子；背景与供体是重复测量，不是独立群体。全包结果不证明字符或片段的组合复用。', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95)); save(fig, 'support_comparison_all_partitions')

    (out / 'plot_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    report = dict(status='figures_created_awaiting_visual_review', source_summary_sha256=sha(summary_path),
        plot_source_sha256=sha(__file__), outputs=outputs, plot_data_sha256=sha(out / 'plot_data.json'),
        font_path=str(font), font_sha256=sha(font), matplotlib=matplotlib.__version__,
        scope='All four paired seeds, complete endpoints only; plots do not introduce an additional outcome or model run.')
    (out / 'figures.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    plot(args.summary, args.out)
