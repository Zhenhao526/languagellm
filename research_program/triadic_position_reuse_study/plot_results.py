"""Two scientific figures from completed JSON summaries only; no model imports.

Run after the main batch completes:
  research_program/.plotting_venv/bin/python -m \
    research_program.triadic_position_reuse_study.plot_results \
    --run research_program/triadic_position_reuse_study/results/position_001

Outputs are exclusive PNG/PDF files under RUN/figures, plus plotted values/hashes.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import platform

import numpy as np

SEEDS = (51101, 51102, 51103, 51104)
CONDITIONS = ('PL_silent', 'PL_live', 'LL_silent', 'LL_live')
LIVE = ('PL_live', 'LL_live')
AXIS_LABELS = ('对象', '长短', '目的地')
CONDITION_LABELS = {'PL_live': 'PL：公开布局', 'LL_live': 'LL：局部观察'}
CONDITION_COLORS = {'PL_live': '#176B91', 'LL_live': '#C56B29'}
FONT = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def close(left, right):
    return bool(np.allclose(left, right, rtol=0, atol=2e-12))


def extract_data(result, summaries):
    """Validate saved arithmetic without calling the production metric functions."""
    require(result.get('status') == 'completed', 'Only a complete batch can be plotted')
    expected = {(seed, condition) for seed in SEEDS for condition in CONDITIONS}
    lookup = {(row['seed'], row['condition']): row for row in summaries}
    require(len(summaries) == 16 and set(lookup) == expected, 'All16 policy summaries required')
    rows = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            summary = lookup[seed, condition]
            matrix = np.asarray(summary['matrix_rows_discovered_axis_columns_test_axis'], dtype=float)
            require(matrix.shape == (3, 3) and np.isfinite(matrix).all()
                    and np.all(np.abs(matrix) <= 1+2e-12), 'Invalid3×3 rate-contrast matrix')
            diagonal = float(summary['diagonal_effect'])
            offdiagonal = float(summary['offdiagonal_effect'])
            baseline = float(summary['uniform_position_effect'])
            selectivity = float(summary['selectivity'])
            whole = float(summary['matched_references']['contrast']['macro']['target_apt'])
            numbers = (diagonal, offdiagonal, baseline, selectivity, whole)
            require(all(math.isfinite(x) for x in numbers), 'Non-finite summary')
            require(close(diagonal, np.trace(matrix)/3), 'D does not equal saved matrix diagonal mean')
            require(close(offdiagonal, (matrix.sum()-np.trace(matrix))/6), 'Off-diagonal mean mismatch')
            require(close(selectivity, diagonal-offdiagonal), 'S does not equal D minus off-diagonal mean')
            require(close(baseline, summary['uniform_position']['contrast']['macro']['target_apt']), 'UniformB mismatch')
            require(close(summary['diagonal_minus_uniform_position'], diagonal-baseline), 'D−B mismatch')
            require(all(abs(x) <= 1+2e-12 for x in (diagonal, offdiagonal, baseline, whole)), 'Rate contrast outside[-1,1]')
            if condition.endswith('_silent'):
                require(close(matrix, 0) and close([diagonal, offdiagonal, baseline, selectivity, whole], 0),
                        'Silent alias has a nonzero effect')
            else:
                rows.append(dict(seed=seed, condition=condition, matrix=matrix.tolist(),
                    D=diagonal, B=baseline, whole_T=whole, S=selectivity,
                    offdiagonal=offdiagonal, D_minus_B=diagonal-baseline))
    pairs = result['primary']['paired_seeds']
    pair_lookup = {p['seed']: p for p in pairs}
    require(len(pairs) == 4 and set(pair_lookup) == set(SEEDS), 'All four paired seeds required')
    paired = []
    for seed in SEEDS:
        pl = lookup[seed, 'PL_live']['selectivity']
        ll = lookup[seed, 'LL_live']['selectivity']
        saved = pair_lookup[seed]
        require(close([saved['PL'], saved['LL'], saved['difference']], [pl, ll, pl-ll]), 'Primary paired values mismatch')
        paired.append(dict(seed=seed, PL_S=float(pl), LL_S=float(ll), difference=float(saved['difference'])))
    mean_difference = float(result['primary']['mean_difference'])
    require(close(mean_difference, np.mean([p['difference'] for p in paired])), 'Primary mean mismatch')
    return dict(policy_values=rows, paired_selectivity=paired, mean_selectivity_difference=mean_difference,
        quantities=dict(matrix='opposite−same target-action rate for discovered axis q and tested axis r',
            D='mean diagonal of the3×3 rate-contrast matrix', B='uniform mean of all8 position interventions on the same rows',
            whole_T='matched whole-two-window-packet opposite−same target-action rate, a larger-intervention reference',
            S='D minus mean of the6 off-diagonal entries; selectivity contrast, not a success rate',
            paired='S_PL−S_LL within the same existing training seed'),
        display_units='All plotted numbers multiply the saved rate-scale values by100. Percentage points for effects/contrasts, not percent success.',
        sample_scope='Fixed144 undirected double-held-out case/background rows, both directions; not the complete domain. All4 existing paired seeds retained.')


def load_data(run):
    run = Path(run).resolve()
    result_path = run/'execution/results.json'
    status_path = run/'execution/status.json'
    require(read(status_path).get('status') == 'completed', 'Main execution has not completed')
    result = read(result_path)
    sources = {str(result_path): sha(result_path), str(status_path): sha(status_path)}
    summaries = []
    references = result.get('policy_summaries', [])
    require(len(references) == 16, 'All16 summary references required')
    for reference in references:
        path = Path(reference['path'])
        require(sha(path) == reference['sha256'], 'Policy summary SHA changed: '+str(path))
        summary = read(path)
        require((summary['seed'], summary['condition']) == (reference['seed'], reference['condition']), 'Policy summary identity mismatch')
        summaries.append(summary)
        sources[str(path)] = reference['sha256']
    return extract_data(result, summaries), sources


def label_number(value, digits=1):
    """Normalize only displayed rounded−0; never change stored data or signs."""
    text = f'{value:+.{digits}f}'
    return f'{0:.{digits}f}' if float(text) == 0 else text


def symmetric_limit(values, *, minimum=1., step=1., margin=1.15):
    maximum = max(abs(float(v)) for v in values)
    return max(minimum, math.ceil(maximum*margin/step)*step)


def draw(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    require(FONT.is_file(), 'Required Chinese font is unavailable')
    plt.rcParams.update({'font.family': FontProperties(fname=FONT).get_name(),
        'axes.unicode_minus': False, 'font.size': 10.5, 'axes.titlesize': 12,
        'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 170,
        'pdf.fonttype': 42})
    lookup = {(r['seed'], r['condition']): r for r in data['policy_values']}
    outputs = {}

    def save(fig, stem):
        for suffix in ('.png', '.pdf'):
            path = out/(stem+suffix)
            fig.savefig(path, bbox_inches='tight', facecolor='white')
            outputs[path.name] = dict(sha256=sha(path), bytes=path.stat().st_size)
        plt.close(fig)

    matrix_values = np.asarray([r['matrix'] for r in data['policy_values']])*100
    limit = min(100., symmetric_limit(matrix_values.ravel(), minimum=5., step=5., margin=1.))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig = plt.figure(figsize=(14.8, 8.4))
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, .065],
        left=.10, right=.955, top=.865, bottom=.21, wspace=.36, hspace=.42)
    image = None
    for row, condition in enumerate(LIVE):
        for col, seed in enumerate(SEEDS):
            ax = fig.add_subplot(grid[row, col])
            matrix = np.asarray(lookup[seed, condition]['matrix'])*100
            image = ax.imshow(matrix, cmap='RdBu_r', norm=norm)
            ax.set_xticks(range(3), AXIS_LABELS)
            ax.set_yticks(range(3), AXIS_LABELS)
            if row == 0:
                ax.set_title(str(seed), pad=10)
            if row == 1:
                ax.set_xlabel('评价轴 r', labelpad=7)
            if col == 0:
                ax.set_ylabel(CONDITION_LABELS[condition]+'\n发现轴 q', labelpad=9,
                              color=CONDITION_COLORS[condition])
            for q in range(3):
                for r in range(3):
                    color = 'white' if abs(matrix[q, r]) > .60*limit else '#202020'
                    ax.text(r, q, label_number(matrix[q, r]), ha='center', va='center', fontsize=11.5, color=color)
                ax.add_patch(Rectangle((q-.49, q-.49), .98, .98, fill=False, linewidth=1.2, edgecolor='#303030'))
            ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=1.)
            ax.tick_params(which='minor', bottom=False, left=False)
    bar = fig.colorbar(image, cax=fig.add_subplot(grid[:, 4]))
    bar.set_label('相反需求 − 同需求：目标正确率差（百分点）', labelpad=12)
    fig.suptitle('训练选定位置在新需求／新布局中的跨轴作用', y=.96, fontsize=16)
    fig.text(.10, .125, '行按训练为各发送者选择的位置聚合；列为验证需求轴。边框只标对角，并不表示它应当占优。', fontsize=10)
    fig.text(.10, .085, '8个开放政策共用对称色阶。固定144个案例×背景行、双向；全部失败与合法未见混合包保留。', fontsize=10)
    fig.text(.10, .045, '每格是方向作用差，不是正确率；同一位置允许被多个发现轴选中。', fontsize=10)
    save(fig, 'selected_positions_matrix')

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.))
    fig.subplots_adjust(left=.09, right=.975, top=.855, bottom=.185, hspace=.68, wspace=.34)
    metric_styles = [('D', 'D：发现位置对角均值', '#176B91', 'o', -.16),
                     ('B', 'B：8位置均匀参照', '#707070', 's', 0.),
                     ('whole_T', '整包T：同样本方向作用', '#24866D', '^', .16)]
    top_values = [r[key]*100 for r in data['policy_values'] for key in ('D', 'B', 'whole_T')]
    top_limit = symmetric_limit(top_values)
    for column, condition in enumerate(LIVE):
        ax = axes[0, column]
        for key, label, color, marker, offset in metric_styles:
            values = [lookup[seed, condition][key]*100 for seed in SEEDS]
            ax.scatter(np.arange(4)+offset, values, color=color, marker=marker, s=53, zorder=3)
        ax.axhline(0, color='#777777', linewidth=.8)
        ax.set_ylim(-top_limit, top_limit)
        ax.set_xticks(range(4), [str(s) for s in SEEDS])
        ax.set_xlabel('既有训练初始化')
        ax.set_ylabel('目标正确率差（百分点）')
        ax.set_title(CONDITION_LABELS[condition]+'：不同干预参照', color=CONDITION_COLORS[condition])
        ax.grid(axis='y', alpha=.2)
    handles = [Line2D([], [], marker=marker, color=color, linestyle='None', markersize=7, label=label)
               for _, label, color, marker, _ in metric_styles]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .929), ncol=3, frameon=False, fontsize=10.5)

    ax = axes[1, 0]
    values_s = []
    for i, seed in enumerate(SEEDS):
        pl, ll = [lookup[seed, c]['S']*100 for c in LIVE]
        values_s.extend((pl, ll))
        ax.plot([i-.07, i+.07], [pl, ll], color='#999999', linewidth=1., zorder=1)
    for condition, offset in zip(LIVE, (-.07, .07)):
        ax.scatter(np.arange(4)+offset, [lookup[s, condition]['S']*100 for s in SEEDS],
            color=CONDITION_COLORS[condition], s=54, label=condition.split('_')[0], zorder=3)
    slimit = symmetric_limit(values_s)
    ax.set_ylim(-slimit, slimit)
    ax.set_xticks(range(4), [str(s) for s in SEEDS])
    ax.set_xlabel('同种子连接PL与LL')
    ax.set_ylabel('选择性差S（百分点）')
    ax.set_title('S = D − 非对角均值')
    ax.legend(frameon=False, ncol=2, loc='upper right')
    ax.axhline(0, color='#777777', linewidth=.8)
    ax.grid(axis='y', alpha=.2)

    ax = axes[1, 1]
    differences = [p['difference']*100 for p in data['paired_selectivity']]
    mean = data['mean_selectivity_difference']*100
    positions = list(range(4))+[4.6]
    all_differences = differences+[mean]
    dlimit = symmetric_limit(all_differences, margin=1.35)
    for i, value in enumerate(all_differences):
        ax.scatter(positions[i], value, marker='D' if i == 4 else 'o',
                   color='#222222' if i == 4 else '#6D528A', s=60, zorder=3)
        ax.annotate(label_number(value, 2), (positions[i], value),
                    xytext=(0, 9 if value >= 0 else -14), textcoords='offset points',
                    ha='center', va='bottom' if value >= 0 else 'top', fontsize=10)
    ax.axvline(4., color='#AAAAAA', linestyle=':', linewidth=.8)
    ax.axhline(0, color='#777777', linewidth=.8)
    ax.set_ylim(-dlimit, dlimit)
    ax.set_xlim(-.5, 5.1)
    ax.set_xticks(positions, [str(s) for s in SEEDS]+['4种子\n均值'])
    ax.set_xlabel('固定主比较；均值不是第5个社会')
    ax.set_ylabel('S_PL − S_LL（百分点）')
    ax.set_title('配对选择性差：保留所有方向')
    ax.grid(axis='y', alpha=.2)
    fig.suptitle('方向作用、位置参照与选择性主比较', y=.982, fontsize=16)
    fig.text(.09, .105, '上排D、B与整包T是方向作用；下排S是矩阵对角与非对角之差。它们都不是任务成功率。', fontsize=10)
    fig.text(.09, .065, '整包T使用相同验证行，但替换量大于单位置，只作剂量不同的参照；B为全部8位置的精确均值。', fontsize=10)
    fig.text(.09, .025, '固定双留出平衡样本、4个既有训练初始化；无显著性检验，不据正的S单独认定成分复用。', fontsize=10)
    save(fig, 'position_effects_and_selectivity')
    return outputs, dict(matplotlib=matplotlib.__version__, font_path=str(FONT), font_sha256=sha(FONT),
                         matrix_symmetric_color_limit_percentage_points=limit)


def plot(run, out=None):
    run = Path(run).resolve()
    out = Path(out).resolve() if out is not None else run/'figures'
    require(not out.exists(), 'Refuse to overwrite a figure directory')
    data, sources = load_data(run)
    out.mkdir(parents=True, exist_ok=False)
    try:
        outputs, style = draw(data, out)
        values = out/'plot_data.json'
        values.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        receipt = dict(status='figures_created_awaiting_visual_review', at=datetime.now(timezone.utc).isoformat(),
            source_result_and_summary_sha256=sources, plot_source_sha256=sha(__file__),
            plot_data_sha256=sha(values), outputs=outputs, style=style,
            runtime=dict(python=platform.python_version(), numpy=np.__version__),
            policy_count_read=16, live_policies_plotted=8, parameter_loads=0, neural_forwards=0, training_updates=0,
            visual_review='Pending actual PNG inspection; PDF generation alone is not visual verification.')
        (out/'figures.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps(dict(status='failed', error=repr(error),
            at=datetime.now(timezone.utc).isoformat(), plot_source_sha256=sha(__file__)), ensure_ascii=False, indent=2)+'\n')
        raise
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', help='Default: RUN/figures; must not already exist')
    arguments = parser.parse_args()
    print(json.dumps(plot(arguments.run, arguments.out), ensure_ascii=False))
