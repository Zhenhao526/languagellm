"""Three figures from the complete frozen JSON summaries; no network imports.

After main completion:
  research_program/.plotting_venv/bin/python -m \
    research_program.triadic_formation_trajectory_study.plot_results \
    --run research_program/triadic_formation_trajectory_study/results/formation_001

The output directory must be new. Actual PNG inspection remains a separate step.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import platform

import numpy as np

STEPS = (0, 100, 500, 1500, 3000, 6000)
SEEDS = (51101, 51102, 51103, 51104)
CONDITIONS = ('PL_silent', 'PL_live', 'LL_silent', 'LL_live')
LIVE = ('PL_live', 'LL_live')
COLORS = {'PL_live': '#176B91', 'LL_live': '#C56B29'}
LABELS = {'PL_live': 'PL：公开布局', 'LL_live': 'LL：局部观察'}
FONT = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
CURVE_KEYS = ('diagonal_effect', 'offdiagonal_effect', 'selectivity',
              'uniform_position_effect', 'diagonal_minus_uniform_position')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    return bool(np.allclose(a, b, rtol=0, atol=2e-12))


def area(values):
    a = np.asarray(values, dtype=float)
    return float(np.dot((a[:-1]+a[1:])/2, np.diff(STEPS))/6000)


def finite_array(values, shape, description):
    a = np.asarray(values, dtype=float)
    require(a.shape == shape and np.isfinite(a).all(), 'Invalid '+description)
    return a


def extract_data(result, summaries):
    """Validate stored arithmetic, retaining all policies and all signed values."""
    require(result.get('status') == 'completed', 'Main result must be complete')
    expected = {(s, c) for s in SEEDS for c in CONDITIONS}
    lookup = {(s['seed'], s['condition']): s for s in summaries}
    require(len(summaries) == 16 and set(lookup) == expected, 'All16 policies required')
    rows = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            summary = lookup[seed, condition]
            require(tuple(summary['steps']) == STEPS, 'Do not substitute checkpoint times')
            curves = {}
            for source, destination, area_key in (
                    ('current_discovery_curves', 'current', 'normalized_areas'),
                    ('retrospective_final_position_curves', 'retrospective', 'retrospective_normalized_areas')):
                curves[destination] = {}
                for key in CURVE_KEYS:
                    values = finite_array(summary[source][key], (6,), source+'/'+key)
                    require(close(summary[area_key][key], area(values)), 'Stored normalized area mismatch')
                    curves[destination][key] = values.tolist()
                c = curves[destination]
                require(close(c['selectivity'], np.subtract(c['diagonal_effect'], c['offdiagonal_effect'])), 'S mismatch')
                require(close(c['diagonal_minus_uniform_position'], np.subtract(c['diagonal_effect'], c['uniform_position_effect'])), 'D−B mismatch')
            for key in CURVE_KEYS:
                require(close(curves['current'][key][-1], curves['retrospective'][key][-1]), 'Current/retrospective6000 anchor mismatch')
            matrix = finite_array(summary['cross_time_target_effect'], (6, 6), 'cross-time matrix')
            adjusted = finite_array(summary['cross_time_minus_receiver_same_time'], (6, 6), 'row-adjusted matrix')
            require(np.max(np.abs(matrix)) <= 1+2e-12, 'Target-rate contrast outside bounds')
            require(close(adjusted, matrix-np.diag(matrix)[:, None]), 'Row reference mismatch')
            require(close(summary['cross_time_same_time_effect'], np.diag(matrix)), 'Same-time diagonal mismatch')
            response, score = [], []
            for t in STEPS:
                discovery = summary['discovery_at_checkpoints'][str(t)]
                rr = finite_array(discovery['selected_response'], (3, 3), 'selected sender response')
                ss = finite_array(discovery['selected_scores'], (3, 3), 'selected sender score')
                require(np.all((rr >= 0) & (rr <= 1)), 'Sender response outside[0,1]')
                require(close(rr.mean(), discovery['mean_selected_response']), 'Sender response mean mismatch')
                require(close(ss.mean(), discovery['mean_selected_score']), 'Sender score mean mismatch')
                response.append(float(discovery['mean_selected_response']))
                score.append(float(discovery['mean_selected_score']))
            changes = summary['temporal_changes']
            require(len(changes) == 5 and [(v['earlier'], v['later']) for v in changes] == list(zip(STEPS[:-1], STEPS[1:])), 'All five adjacent intervals required')
            token_change = []
            for change in changes:
                rate = float(change['train_token_change_rate'])
                detail = finite_array(change['by_window_sender_position'], (2, 3, 4), 'token change detail')
                require(0 <= rate <= 1 and np.all((detail >= 0) & (detail <= 1)), 'Token-change rate outside[0,1]')
                require(close(detail.mean(), rate), 'Token-change mean mismatch')
                token_change.append(rate)
            if condition.endswith('_silent'):
                require(close(matrix, 0) and all(close(curves[k][q], 0) for k in curves for q in CURVE_KEYS),
                        'Silent invisible-intervention contrast must be zero')
            rows.append(dict(seed=seed, condition=condition, **curves,
                normalized_S_area=float(summary['normalized_areas']['selectivity']),
                retrospective_S_area=float(summary['retrospective_normalized_areas']['selectivity']),
                cross_time_target_effect=matrix.tolist(), cross_time_row_adjusted=adjusted.tolist(),
                same_time_whole_T=np.diag(matrix).tolist(), sender_response=response,
                sender_specificity_score=score, adjacent_token_change=token_change,
                temporal_changes=changes))
    row_lookup = {(r['seed'], r['condition']): r for r in rows}
    paired = result['primary']['paired_seeds']
    require(len(paired) == 4 and {p['seed'] for p in paired} == set(SEEDS), 'All four primary paired values required')
    p_lookup = {p['seed']: p for p in paired}
    primary = []
    for seed in SEEDS:
        pl, ll = (row_lookup[seed, c]['normalized_S_area'] for c in LIVE)
        p = p_lookup[seed]
        require(close([p['PL'], p['LL'], p['difference']], [pl, ll, pl-ll]), 'Primary paired area mismatch')
        primary.append(dict(seed=seed, PL=float(pl), LL=float(ll), difference=float(p['difference'])))
    require(close(result['primary']['mean_difference'], np.mean([p['difference'] for p in primary])), 'Primary mean mismatch')
    return dict(steps=list(STEPS), all_policy_values=rows, primary=primary,
        mean_difference=float(result['primary']['mean_difference']),
        units=dict(sender_response='percent of paired messages changing at the selected position; association',
            token_change='percent of all complete-train greedy token entries differing between adjacent saved times; not per update',
            effects='rate difference times100, in percentage points; not percent task success',
            normalized_area='trapezoid integral over optimization steps0–6000 divided by6000; displayed times100'),
        scope=dict(train_worlds=419904, validation_case_background_rows=144, directions=2,
            epochs_or_generations=False, actual_training_updates_added=0,
            endpoint6000='Reuse of previously frozen result, not a fresh confirmatory replicate',
            retrospective='Uses6000-selected positions at earlier times; future-informed descriptive reference',
            cross_time='Receiver rows; donor packet columns. Donor W2 includes donor-time interaction history.'))


def load_data(run):
    run = Path(run).resolve()
    result_path, status_path = run/'execution/results.json', run/'execution/status.json'
    require(read(status_path).get('status') == 'completed', 'Refuse plotting incomplete execution')
    result = read(result_path)
    require(tuple(result['config']['steps']) == STEPS, 'Unexpected saved configuration')
    sources = {str(p): sha(p) for p in (result_path, status_path)}
    plan = run/'plan.json'
    require(sha(plan) == result['plan_sha256'], 'Frozen plan identity changed')
    sources[str(plan)] = sha(plan)
    summaries = []
    for row in result['policies']:
        path = Path(row['path'])
        require(sha(path) == row['sha256'], 'Policy summary changed: '+str(path))
        summary = read(path)
        require((summary['seed'], summary['condition']) == (row['seed'], row['condition']), 'Policy identity mismatch')
        summaries.append(summary); sources[str(path)] = row['sha256']
    return extract_data(result, summaries), sources


def label(value, decimals=1):
    text = f'{value:+.{decimals}f}'
    return f'{0:.{decimals}f}' if float(text) == 0 else text


def limit(values, minimum=1., step=1., margin=1.15):
    return max(minimum, math.ceil(float(np.max(np.abs(values)))*margin/step)*step)


def draw(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle
    require(FONT.is_file(), 'Chinese font unavailable')
    plt.rcParams.update({'font.family': FontProperties(fname=FONT).get_name(),
        'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.dpi': 170})
    rows = {(r['seed'], r['condition']): r for r in data['all_policy_values']}
    live = [rows[s, c] for s in SEEDS for c in LIVE]
    outputs = {}

    def save(fig, stem):
        for suffix in ('.png', '.pdf'):
            path = out/(stem+suffix)
            fig.savefig(path, bbox_inches='tight', facecolor='white')
            outputs[path.name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        plt.close(fig)

    def time_axis(ax, effect=True):
        ax.set_xlim(-110, 6170)
        ax.set_xticks([0, 1500, 3000, 6000], ['0', '1500', '3000', '6000'])
        ax.set_xlabel('优化步骤')
        ax.axvline(6000, color='#888888', linestyle=':', linewidth=.8)
        if effect:
            ax.axhline(0, color='#888888', linewidth=.8)
        ax.grid(axis='y', alpha=.2)

    # Figure1: two requested local-effect curves; main four-pair area retained.
    fig = plt.figure(figsize=(15.8, 11.2))
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, .88], left=.075,
        right=.98, top=.86, bottom=.185, hspace=.7, wspace=.37)
    for ri, key in enumerate(('selectivity', 'diagonal_minus_uniform_position')):
        values = [np.asarray(r['current'][key])*100 for r in live]
        if key == 'selectivity':
            values += [np.asarray(r['retrospective'][key])*100 for r in live]
        bound = limit(values)
        for ci, seed in enumerate(SEEDS):
            ax = fig.add_subplot(grid[ri, ci]); ax.set_title(str(seed))
            for condition in LIVE:
                r = rows[seed, condition]; color = COLORS[condition]
                y = np.asarray(r['current'][key])*100
                ax.plot(STEPS, y, color=color, marker='o', markersize=3.7, linewidth=1.5)
                if key == 'selectivity':
                    ax.plot(STEPS, np.asarray(r['retrospective'][key])*100,
                            color=color, linestyle='--', linewidth=1.1, alpha=.65)
                ax.scatter([6000], [y[-1]], marker='D', color=color, s=25, zorder=4)
            ax.set_ylim(-bound, bound); time_axis(ax)
            if ci == 0:
                ax.set_ylabel(('选择性S' if ri == 0 else 'D − 均匀位置B')+'（百分点）')
    ax = fig.add_subplot(grid[2, :])
    paired = [p['difference']*100 for p in data['primary']]
    yy = paired+[data['mean_difference']*100]; xx = [0, 1, 2, 3, 4.5]
    bound = limit(yy, margin=1.4)
    for i, (x, y) in enumerate(zip(xx, yy)):
        ax.scatter(x, y, s=53, marker='D' if i == 4 else 'o',
                   color='#222222' if i == 4 else '#6D528A', zorder=3)
        ax.annotate(label(y, 3), (x, y), xytext=(0, 7 if y >= 0 else -10),
                    textcoords='offset points', ha='center', va='bottom' if y >= 0 else 'top')
    ax.set_xticks(xx, [str(s) for s in SEEDS]+['4种子等权均值'])
    ax.set_xlim(-.6, 5.1); ax.set_ylim(-bound, bound)
    ax.axhline(0, color='#888888', linewidth=.8); ax.axvline(3.8, color='#AAAAAA', linestyle=':', linewidth=.8)
    ax.set_ylabel('S归一化面积配对差\nPL − LL（百分点）')
    ax.set_title('固定主比较：六点梯形面积／6000；均值不增加独立样本')
    ax.grid(axis='y', alpha=.2)
    handles = [Line2D([], [], color=COLORS[c], label=LABELS[c], linewidth=2) for c in LIVE]
    handles += [Line2D([], [], color='#555555', marker='o', label='实线：当时训练支持选择'),
                Line2D([], [], color='#555555', linestyle='--', label='虚线：6000位置回看（仅S）'),
                Line2D([], [], color='#555555', linestyle='None', marker='D', label='6000：复用锚')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .938), ncol=3, frameon=False)
    fig.suptitle('优化过程中的单位置作用与选择性', y=.985, fontsize=16)
    fig.text(.075, .115, '全部4个既有种子；每点固定同一144个案例×背景行及双向，不筛成功，不插补未保存的步骤。', fontsize=10)
    fig.text(.075, .073, '虚线使用未来6000步的位置选择，仅为回看参照；6000步与旧批结果复用，不是独立确认。', fontsize=10)
    fig.text(.075, .031, '稀疏梯形面积不是准确形成时刻或独立学习速度；S、D−B均为功能对比，不是任务正确率。', fontsize=10)
    save(fig, 'formation_curves_and_primary_area')

    # Figure2: complete8raw+8row-adjusted grids, one color scale per estimand.
    raw_bound = limit([np.asarray(r['cross_time_target_effect'])*100 for r in live], minimum=5, step=5, margin=1.)
    adj_bound = limit([np.asarray(r['cross_time_row_adjusted'])*100 for r in live], minimum=5, step=5, margin=1.)
    fig = plt.figure(figsize=(16.8, 15.2))
    grid = fig.add_gridspec(4, 6, width_ratios=[1, 1, .07, 1, 1, .07],
        left=.065, right=.96, top=.885, bottom=.13, hspace=.58, wspace=.55)
    images = {}
    for ri, seed in enumerate(SEEDS):
        for group, (key, bound) in enumerate((('cross_time_target_effect', raw_bound), ('cross_time_row_adjusted', adj_bound))):
            for ci, condition in enumerate(LIVE):
                ax = fig.add_subplot(grid[ri, group*3+ci])
                mat = np.asarray(rows[seed, condition][key])*100
                images[group] = ax.imshow(mat, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
                ax.set_title(f'{condition.split("_")[0]} · {seed}', color=COLORS[condition], fontsize=11)
                ax.set_xticks(range(6), [str(t) for t in STEPS], rotation=50, ha='right', fontsize=8)
                ax.set_yticks(range(6), [str(t) for t in STEPS], fontsize=8)
                if ri == 3:
                    ax.set_xlabel('供体包优化步骤', fontsize=9)
                if ci == 0:
                    ax.set_ylabel('接收者优化步骤', fontsize=9)
                for r in range(6):
                    for s in range(6):
                        ax.text(s, r, label(mat[r, s]), ha='center', va='center', fontsize=7.7,
                                color='white' if abs(mat[r, s]) > .6*bound else '#202020')
                    ax.add_patch(Rectangle((r-.49, r-.49), .98, .98, fill=False, linewidth=.7, edgecolor='#555555'))
                ax.set_xticks(np.arange(-.5, 6, 1), minor=True); ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
                ax.grid(which='minor', color='white', linewidth=.45)
                ax.tick_params(which='minor', bottom=False, left=False)
    for group, column in enumerate((2, 5)):
        bar = fig.colorbar(images[group], cax=fig.add_subplot(grid[:, column]))
        bar.set_label('T（百分点）' if group == 0 else 'T − 同行同时间T（百分点）', fontsize=10)
    fig.suptitle('跨优化检查点的完整双窗包兼容性', y=.98, fontsize=16)
    fig.text(.24, .935, '整包T：相反需求 − 同需求', ha='center', fontsize=13)
    fig.text(.72, .935, '同行基准差：T(r,s) − T(r,r)', ha='center', fontsize=13)
    fig.text(.065, .077, '各半图8个政策共用对称色阶；行接收者、列供体包。边框是同时间，不把矩阵不对称独占解释为词典改变。', fontsize=10)
    fig.text(.065, .047, '供体第二窗含供体时点的伙伴互动，接收方第二窗与行动重新计算。右图对角为定义上的零，非独立成功证据。', fontsize=10)
    fig.text(.065, .017, '6000×6000是既有整包复用锚；所有横纵坐标均为优化步骤，不是世代。未新增新人学习或训练。', fontsize=10)
    save(fig, 'cross_time_whole_packet_matrices')

    # Figure3: distinguish training association, validation effect, train-form drift.
    fig = plt.figure(figsize=(15.8, 11.5))
    grid = fig.add_gridspec(3, 4, left=.075, right=.98, top=.87, bottom=.16, hspace=.59, wspace=.36)
    effect_bound = limit([np.asarray(r[key] if key == 'same_time_whole_T' else r['current'][key])*100
                          for r in live for key in ('diagonal_effect', 'same_time_whole_T')])
    mid = (np.asarray(STEPS[:-1])+np.asarray(STEPS[1:]))/2
    for ci, seed in enumerate(SEEDS):
        for ri in range(3):
            ax = fig.add_subplot(grid[ri, ci]); ax.set_title(str(seed))
            for condition in LIVE:
                row = rows[seed, condition]; color = COLORS[condition]
                if ri == 0:
                    y = np.asarray(row['sender_response'])*100
                    ax.plot(STEPS, y, color=color, marker='o', markersize=3.5)
                    ax.scatter([6000], [y[-1]], marker='D', color=color, s=24)
                elif ri == 1:
                    y = np.asarray(row['current']['diagonal_effect'])*100
                    ax.plot(STEPS, y, color=color, marker='o', markersize=3.5)
                    ax.plot(STEPS, np.asarray(row['same_time_whole_T'])*100, color=color, linestyle='--', linewidth=1.1)
                    ax.scatter([6000], [y[-1]], marker='D', color=color, s=24)
                else:
                    y = np.asarray(row['adjacent_token_change'])*100
                    ax.plot(mid, y, color=color, marker='o', markersize=4, linewidth=1.1)
                    for i in range(5):
                        ax.plot([STEPS[i], STEPS[i+1]], [y[i], y[i]], color=color, linewidth=1., alpha=.5)
            time_axis(ax, effect=ri == 1)
            ax.set_ylim((-effect_bound, effect_bound) if ri == 1 else (0, 100))
            if ci == 0:
                ax.set_ylabel(('选定位置消息响应（%）', '验证方向作用（百分点）', '区间内贪心符号变化（%）')[ri])
    handles = [Line2D([], [], color=COLORS[c], label=LABELS[c], linewidth=2) for c in LIVE]
    handles += [Line2D([], [], color='#555555', label='中排实线D：单位置对角'),
                Line2D([], [], color='#555555', linestyle='--', label='中排虚线T：同时间整包')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .942), ncol=2, frameon=False)
    fig.suptitle('发送关联、接收功能与消息形式变化分别观察', y=.985, fontsize=16)
    fig.text(.075, .10, '上排是训练支持中的消息变化关联；中排是验证干预作用，两者不是同一分母，也不以相关性推断语义。', fontsize=10)
    fig.text(.075, .060, '下排覆盖完整419904个固定训练世界的所有贪心符号；点位为相邻检查区间中点，横线标跨度，未除以步骤数。', fontsize=10)
    fig.text(.075, .020, '消息更稳定可伴随忽略通信；消息变化也可能保留功能。6000为旧结果复用，不把区间读数解释为起源时刻。', fontsize=10)
    save(fig, 'sender_response_effects_and_token_change')
    return outputs, dict(matplotlib=matplotlib.__version__, font_path=str(FONT), font_sha256=sha(FONT),
        cross_time_raw_symmetric_limit_pp=raw_bound, cross_time_adjusted_symmetric_limit_pp=adj_bound)


def plot(run, out=None):
    run = Path(run).resolve(); out = Path(out).resolve() if out is not None else run/'figures'
    require(not out.exists(), 'Refuse to overwrite figure directory')
    data, sources = load_data(run)
    out.mkdir(parents=True, exist_ok=False)
    try:
        outputs, style = draw(data, out)
        path = out/'plot_data.json'
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        receipt = dict(status='created_awaiting_actual_PNG_review', recorded_at=datetime.now(timezone.utc).isoformat(),
            source_sha256=sha(__file__), result_and_summary_sha256=sources,
            plot_data_sha256=sha(path), outputs=outputs, style=style,
            runtime=dict(python=platform.python_version(), numpy=np.__version__),
            policy_summaries_read=16, live_policies_plotted=8,
            parameter_loads=0, neural_forward_samples=0, training_updates=0,
            interpretation='Optimization steps only, not generations. Retrospective S is future-informed.6000 is a reused anchor.')
        (out/'figures.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        return receipt
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps(dict(status='failed', error=repr(error),
            recorded_at=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__)), ensure_ascii=False, indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', help='New directory; default RUN/figures')
    args = parser.parse_args()
    print(json.dumps(plot(args.run, args.out), ensure_ascii=False))
