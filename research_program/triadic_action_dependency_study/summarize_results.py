"""Pure aggregation of all completed runs; never loads policy parameters.

Main results are immutable inputs. Silent closed cells alias natural analysis.
Full endpoints and two-background monitors have distinct record identities.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import time
import traceback
import numpy as np
from . import dataset, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = metrics.SEEDS
CONDITIONS = metrics.CONDITIONS
PARTITIONS = metrics.PARTITIONS
STEPS = (0, 100, 500, 1500, 3000, 6000)
MODES = ('natural', 'closed')
require = dataset.require


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256((str(a.dtype) + '|' + str(a.shape)).encode() + a.tobytes()).hexdigest()


def validate_completed_grid(result):
    require(result.get('status') == 'completed' and result.get('completed_run_count') == 24, 'Wait for all24completed')
    runs = result['runs']
    require(len(runs) == 24 and [(r['seed'], r['condition']) for r in runs] == list(product(SEEDS, CONDITIONS)), 'Complete ordered grid required')
    for run in runs:
        require(run['updates'] == 6000 and set(run['final']) == set(PARTITIONS), 'Incomplete full endpoints')
        require([r['update'] for r in run['monitor']] == list(STEPS), 'Incomplete monitor checkpoints')
        for context in [run['final']] + [m['monitor'] for m in run['monitor']]:
            require(set(context) == set(PARTITIONS), 'Missing monitor partition')
            for part in PARTITIONS:
                require(set(context[part]) == set(MODES), 'Missing natural/closed cell')
    return runs


def verify_alias(natural, closed, condition):
    if condition.endswith('_silent'):
        expected = deepcopy(natural); expected['reused_natural'] = True
        require(closed == expected, 'Silent closed is not an exact natural-record alias')
        return True
    require(not natural['reused_natural'] and not closed['reused_natural'], 'Live result unexpectedly reuses natural')
    require(natural['path'] != closed['path'], 'Live closed must have a distinct recorded rollout')
    return False


def resolve_record(record, execution, seed, condition, kind, checkpoint, part, mode):
    basename = (f'final_{part}' if kind == 'full_endpoint' else f'monitor_{checkpoint:04d}_{part}') + '_' + mode + '.npz'
    expected = (execution / f'seed_{seed}_{condition}' / basename).resolve()
    recorded = Path(record['path'])
    if not recorded.is_absolute():
        candidates = [(ROOT / recorded).resolve(), (execution / recorded).resolve()]
        require(expected in candidates, 'Unexpected relative record path')
    else:
        require(recorded.resolve() == expected, 'Record belongs to wrong run/cell')
    require(expected.is_file() and sha(expected) == record['data_sha256'], 'Missing/changed raw NPZ ' + str(expected))
    return expected


def analyze_cell(values, record, spec, kind, condition, mode, pairs):
    required = {'states', 'state_indices', 'messages', 'action_indices', 'action_probabilities',
                'greedy_reward', 'executed', 'satisfied', 'conditional_exact_expected_reward',
                'conditional_exact_full_success_probability', 'conditional_exact_execution_probability'}
    require(set(values) == required, 'Unexpected saved record schema')
    expected_ids = np.arange(spec['world_count']) if kind == 'full_endpoint' else np.asarray(spec['monitor_indices'])
    require(np.array_equal(values['state_indices'], expected_ids), 'Full/monitor saved-world set differs')
    n = len(expected_ids)
    require(record['worlds'] == n and record['information'] == condition.split('_')[0], 'Recorded world count/information differs')
    received = condition.endswith('_live') and mode == 'natural'
    require(record['live'] == received, 'Recorded channel delivery differs')
    messages = values['messages']
    require(messages.shape == (n, 2, 3, 4) and messages.dtype.kind in 'iu'
            and np.all((messages >= 0) & (messages < 8)), 'Invalid saved messages')
    settled = metrics.settle_arrays(values['states'], values['action_indices'])
    require(np.array_equal(values['executed'], settled['executed']) and np.array_equal(values['satisfied'], settled['satisfied']), 'Saved individual settlement differs')
    world = metrics.world_metrics(states=values['states'], action_indices=values['action_indices'],
        action_probabilities=values['action_probabilities'], greedy_reward=values['greedy_reward'])
    for key in ('reward_mean', 'full_success_rate', 'role_success_rate', 'physical_execution_rate'):
        require(world[key] == record[key], 'Recorded ' + key + ' differs')
    for measured, saved in (('expected_reward_given_saved_messages', 'conditional_exact_expected_reward'),
                           ('full_probability_given_saved_messages', 'conditional_exact_full_success_probability'),
                           ('physical_probability_given_saved_messages', 'conditional_exact_execution_probability')):
        require(np.isfinite(values[saved]).all() and values[saved].shape == (n,), 'Invalid conditional statistic')
        require(abs(world[measured] - float(values[saved].mean())) <= 2e-12, 'Conditional expectation mean differs')
    semantic = metrics.semantic_metrics(spec=spec, states=values['states'], state_indices=values['state_indices'],
        action_indices=values['action_indices'], action_probabilities=values['action_probabilities'], pairs=pairs, include_rows=True)
    if condition.startswith(('PL_', 'LL_')) and not received:
        for group in ('content', 'role'):
            require(semantic[group]['macro']['both_endpoints_apt'] == 0, 'Same-observation closed/silent structural zero failed')
    require((semantic['coverage'] == 'full_partition') == (kind == 'full_endpoint'), 'Monitoring substituted for complete endpoint')
    return dict(world_metrics=world, semantic_metrics=semantic)


def compact_record(metadata, report):
    semantic = report['semantic_metrics']; world = report['world_metrics']
    return dict(**metadata,
        world={k: world[k] for k in (*metrics.COMPARISON_METRICS[1:], 'worlds', 'joint_action_count')},
        semantic_scope={k: semantic[k] for k in ('coverage', 'saved_worlds', 'saved_background_count', 'full_partition_worlds')},
        **{group: dict(macro=semantic[group]['macro'], availability=semantic[group]['availability'],
            by_axis={a: dict(metrics=cell['metrics'], complete_six_strata=cell['complete_six_strata'])
                     for a, cell in semantic[group]['by_axis'].items()}) for group in ('content', 'role')})


def numeric_leaves(value, prefix=''):
    if isinstance(value, dict):
        for key, child in value.items():
            if key != 'joint_action_distribution':
                yield from numeric_leaves(child, prefix + '/' + str(key))
    elif isinstance(value, list):
        for i, child in enumerate(value): yield from numeric_leaves(child, prefix + '/' + str(i))
    elif value is None or isinstance(value, (int, float, bool)):
        yield prefix.lstrip('/'), value


def save_cell(out, rid, metadata, report, case_catalog, csv_writer):
    semantic = report['semantic_metrics']
    labels = semantic.pop('case_rows'); means = semantic.pop('case_mean_values')
    require(labels == case_catalog['labels'], 'Per-case output order changed')
    row_payload = dict(case_index=np.arange(len(labels), dtype=np.int32), **means)
    npz = out / 'case_metrics' / (rid + '.npz')
    np.savez_compressed(npz, **row_payload)
    row_receipt = dict(path=str(npz.relative_to(out)), sha256=sha(npz),
        values={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=array_sha(v)) for k, v in row_payload.items()},
        reduction='Per-case means over every saved background, with both endpoints retained within the case.',
        reconstruction='All individual background values recoverable from source NPZ states/actions/probabilities and complete frozen case metadata.')
    target = out / 'reports' / (rid + '.json')
    write(target, dict(metadata=metadata, metrics=report, case_metrics=row_receipt,
                       case_metadata={k: case_catalog[k] for k in ('path', 'sha256')}))
    for key, value in numeric_leaves(report):
        csv_writer.writerow([rid, metadata['kind'], metadata['seed'], metadata['condition'],
            metadata['checkpoint'], metadata['partition'], metadata['mode'], key, value])
    return dict(report_id=rid, **metadata, report_path=str(target.relative_to(out)),
                report_sha256=sha(target), case_metrics=row_receipt), report


def summarize(results_path, out):
    started = time.perf_counter(); results_path = Path(results_path).resolve(); out = Path(out).resolve()
    require(not out.exists(), 'Never overwrite or resume a summary')
    result = read(results_path); runs = validate_completed_grid(result)
    execution, run_dir = results_path.parent, results_path.parent.parent
    plan, freeze, prepared = [read(run_dir / name) for name in ('plan.json', 'freeze.json', 'prepared.json')]
    require(result['plan_sha256'] == freeze['plan_sha256'] == sha(run_dir / 'plan.json'), 'Plan anchor differs')
    require(freeze['prepared_sha256'] == plan['prepared_sha256'] == sha(run_dir / 'prepared.json'), 'Prepared source differs')
    require(read(execution / 'status.json')['status'] == 'completed', 'Execution not allcompleted')
    source_hashes = {str(path): sha(path) for path in (results_path, execution / 'status.json', run_dir / 'plan.json', run_dir / 'freeze.json', run_dir / 'prepared.json', Path(__file__))}
    for relative, digest in plan['sources'].items():
        path = (ROOT / relative).resolve(); snap = run_dir / 'source_snapshot' / relative
        require(sha(path) == digest and sha(snap) == digest, 'Frozen source changed ' + relative)
        source_hashes[str(path)] = digest; source_hashes[str(snap)] = digest
    for module in (dataset, metrics):
        relative = str(Path(module.__file__).resolve().relative_to(ROOT))
        require(relative in plan['sources'], 'Measurement module not frozen in the main plan')
    out.mkdir(parents=True); (out / 'reports').mkdir(); (out / 'case_metrics').mkdir(); (out / 'case_metadata').mkdir()
    write(out / 'started.json', dict(status='read_only_summary_started', results_sha256=source_hashes[str(results_path)],
        summary_source_sha256=sha(__file__), no_model_or_training=True))
    case_catalogs, pairs = {}, {}
    for part in PARTITIONS:
        pair = dataset.content_pairs(prepared['partitions'][part]); pairs[part] = pair
        path = out / 'case_metadata' / (part + '.json'); write(path, pair)
        labels = [{k: r[k] for k in ('case_id', 'classification', 'axis_index', 'sender', 'listener', 'endpoint_need_indices')} for r in pair['rows']]
        case_catalogs[part] = dict(path=str(path.relative_to(out)), sha256=sha(path), labels=labels,
                                 case_count=len(labels), summary=pair['summary'])
    catalog, compact, enriched, closed_changes = [], [], [], []
    counts = dict(unique_npz_analyses=0, natural_analyses=0, live_closed_analyses=0,
                  silent_closed_aliases=0, logical_cells=0, full_endpoint_analyses=0, monitor_analyses=0,
                  saved_worlds_analyzed=0, neural_forwards=0, training=0)
    with (out / '完整分层指标.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['record_id', 'kind', 'seed', 'condition', 'checkpoint', 'partition', 'mode', 'metric_path', 'value'])
        for run in runs:
            seed, condition = run['seed'], run['condition']
            saved_run = execution / f'seed_{seed}_{condition}' / 'result.json'
            require(read(saved_run) == run, 'Per-run result differs from allcompleted aggregate')
            source_hashes[str(saved_run)] = sha(saved_run)
            final_enriched = dict(seed=seed, condition=condition, updates=6000, final={})
            contexts = [('two_background_monitor', m['update'], m['monitor']) for m in run['monitor']] + [('full_endpoint', 6000, run['final'])]
            for kind, checkpoint, cells in contexts:
                for part in PARTITIONS:
                    natural, closed = cells[part]['natural'], cells[part]['closed']
                    alias = verify_alias(natural, closed, condition)
                    analyzed, nat_values = {}, None
                    for mode in MODES:
                        metadata = dict(kind=kind, seed=seed, condition=condition, checkpoint=checkpoint, partition=part, mode=mode)
                        counts['logical_cells'] += 1
                        if mode == 'closed' and alias:
                            natural_entry = analyzed['natural']['entry']; report = analyzed['natural']['report']
                            entry = dict(**metadata, reused_natural=True, aliased_report_id=natural_entry['report_id'],
                                report_path=natural_entry['report_path'], report_sha256=natural_entry['report_sha256'],
                                case_metrics=natural_entry['case_metrics'])
                            counts['silent_closed_aliases'] += 1
                        else:
                            record = cells[part][mode]
                            path = resolve_record(record, execution, seed, condition, kind, checkpoint, part, mode)
                            source_hashes[str(path)] = record['data_sha256']
                            with np.load(path, allow_pickle=False) as z: values = {k: z[k] for k in z.files}
                            report = analyze_cell(values, record, prepared['partitions'][part], kind, condition, mode, pairs[part])
                            rid = f'{kind}_{seed}_{condition}_{checkpoint:04d}_{part}_{mode}'
                            metadata.update(source_path=str(path), source_sha256=record['data_sha256'], reused_natural=False)
                            entry, report = save_cell(out, rid, metadata, report, case_catalogs[part], writer)
                            counts['unique_npz_analyses'] += 1
                            counts['natural_analyses' if mode == 'natural' else 'live_closed_analyses'] += 1
                            counts['full_endpoint_analyses' if kind == 'full_endpoint' else 'monitor_analyses'] += 1
                            counts['saved_worlds_analyzed'] += len(values['states'])
                            if mode == 'natural':
                                nat_values = {k: values[k] for k in ('state_indices', 'action_indices', 'messages')}
                            else:
                                require(np.array_equal(nat_values['state_indices'], values['state_indices']), 'Closed pairing differs')
                                closed_changes.append(dict(**metadata,
                                    any_actor_action_change_rate=float(np.any(values['action_indices'] != nat_values['action_indices'], axis=1).mean()),
                                    per_actor_action_change_rate=(values['action_indices'] != nat_values['action_indices']).mean(axis=0).tolist(),
                                    any_generated_message_change_rate=float(np.any(values['messages'] != nat_values['messages'], axis=(1, 2, 3)).mean())))
                            del values
                        catalog.append(entry); compact.append(compact_record(metadata, report))
                        analyzed[mode] = dict(entry=entry, report=report)
                    if kind == 'full_endpoint':
                        final_enriched['final'][part] = {m: dict(source_record=cells[part][m], **analyzed[m]['report']) for m in MODES}
                    del nat_values, analyzed
            enriched.append(final_enriched)
            print(json.dumps(dict(summarized_seed=seed, condition=condition, unique_analyses=counts['unique_npz_analyses']), ensure_ascii=False), flush=True)
    require(counts['unique_npz_analyses'] == 1008 and counts['natural_analyses'] == 672
        and counts['live_closed_analyses'] == 336 and counts['silent_closed_aliases'] == 336
        and counts['logical_cells'] == 1344 and counts['full_endpoint_analyses'] == 144
        and counts['monitor_analyses'] == 864, 'Complete analysis budget differs')
    comparison = metrics.primary_comparison(enriched)
    require(all(sha(path) == digest for path, digest in source_hashes.items()), 'Source changed during summary')
    write(out / 'enriched_final_records.json', enriched)
    write(out / 'closed_action_changes.json', closed_changes)
    compact_cases = {part: {k: v for k, v in cell.items() if k != 'labels'} for part, cell in case_catalogs.items()}
    summary = dict(status='completed_read_only_summary', completed_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter() - started, source_sha256=source_hashes, counts=counts,
        primary_comparison=comparison, compact_records=compact, catalog=catalog, case_catalogs=compact_cases,
        enriched_final_records_sha256=sha(out / 'enriched_final_records.json'),
        closed_action_changes_sha256=sha(out / 'closed_action_changes.json'),
        limitations=['Metrics API reused, not an independent mathematical reimplementation; independent execution audit is separate.',
            'Four paired initializations; worlds, cases, endpoints and six conditions are not independent team replicates.',
            'Complete endpoints are separate from all-needs × two-background monitors, including update6000.',
            'PL/LL silent/closed both-endpoint zeros are structural same-observation checks.',
            'All content/role/descriptive cases retained; per-case background means saved; individual background values reconstructible from unchanged raw NPZ and complete case metadata.',
            'No natural-content or closure result establishes reusable message components or compositional grammar.',
            'Summary/plot sources are independently prepared; only the listed main-plan sources belong to its frozen source snapshot.'])
    write(out / 'summary.json', summary)
    primary = comparison['primary']; rows = comparison['partitions']['new_needs_and_layouts']['seed_rows']
    lines = ['# 新任务完整汇总', '', '全部24组与四分区已纳入；本次只读计算，没有模型前向。主要量是双留出完整终点内容三轴宏平均的 PL_live−LL_live。', '',
        '| 配对种子 | PL_live | LL_live | 差（百分点） |', '|---|---:|---:|---:|']
    for row in rows:
        pl = row['cells']['PL_live']['natural']['content_both_endpoints_apt']; ll = row['cells']['LL_live']['natural']['content_both_endpoints_apt']
        lines.append(f"| {row['seed']} | {100*pl:.4f}% | {100*ll:.4f}% | {100*(pl-ll):+.4f} |")
    lines += ['', f"四种子等权差：{100*primary['equal_seed_mean']:+.4f} 个百分点。", '',
        '1008份实际记录各分析一次，336个静默关闭单元仅引用自然分析；完整终点与两个固定背景的六时点轨迹分别标记。',
        '全部分轴、角色和失败保存在reports、case_metrics和完整分层指标.csv；原始逐世界消息、动作、概率及固定案例元数据可重构未压缩的逐背景指标。',
        '公开布局也减少需传播的观察信息，因此不称纯行动依赖效应；自然适切或关闭效应不直接证明组合语法。']
    (out / '汇总.md').write_text('\n'.join(lines) + '\n')
    return dict(status=summary['status'], output=str(out), counts=counts, primary=primary,
                elapsed_seconds=summary['elapsed_seconds'], summary_sha256=sha(out / 'summary.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(summarize(args.results, args.out), ensure_ascii=False))
    except BaseException as error:
        out = Path(args.out)
        if out.is_dir() and not (out / 'failure.json').exists():
            write(out / 'failure.json', dict(status='failed', error=str(error), traceback=traceback.format_exc(),
                  source_sha256=sha(__file__), automatic_retry=False))
        raise
