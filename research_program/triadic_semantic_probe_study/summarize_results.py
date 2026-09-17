"""Read-only aggregation of a COMPLETED semantic probe; no neural forwards.

Uses the frozen metrics API. Saves every row metric and all strata, including
zeros and negative contrasts. Two intervention directions stay within cases.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import time
import traceback
import numpy as np
from research_program.triadic_semantic_probe_study import dataset, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (49101, 49102, 49103, 49104)
CONDITIONS = ('FI_silent', 'FI_live', 'PI_silent', 'PI_live')
PARTITIONS = ('train', 'heldout_layouts')
CHECKPOINTS = (0, 100, 500, 1500, 3000, 6000)
WINDOWS = ('w1', 'w2', 'both')
MODES = ('sham',)+tuple(prefix+'_'+w for prefix in ('local_opposite', 'remote_same', 'remote_opposite') for w in WINDOWS)
LABELS = ('success_action_masks', 'listener', 'axis', 'sender', 'listener_partners', 'classification',
          'content_within_axis_weight', 'role_within_axis_weight')
sha, read, require, json_bytes, array_sha = dataset.sha, dataset.read, dataset.require, dataset.json_bytes, dataset.array_sha


def _resolve(path, results_path):
    p = Path(path)
    candidates = [p] if p.is_absolute() else [results_path.parent/p, ROOT/p]
    existing = {q.resolve() for q in candidates if q.is_file()}
    require(len(existing) == 1, 'Missing or ambiguous recorded path '+str(path))
    return existing.pop()


def load_record(record, results_path, source_hashes):
    path = _resolve(record['path'], results_path)
    require(sha(path) == record['data_sha256'], 'Recorded data SHA differs '+str(path))
    source_hashes[str(path)] = record['data_sha256']
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def dataset_labels(arrays, rows=None):
    return {k: arrays[k] if rows is None else arrays[k][rows] for k in LABELS}


def natural_report(values, arrays):
    indices = arrays['endpoint_indices']
    require(np.array_equal(values['state_indices'], np.arange(len(values['states']))), 'Natural records must be complete original-order arrays')
    report = metrics.natural_metrics(action_indices=values['action_indices'][indices],
        action_probs=values['action_probabilities'][indices], native_rewards=values['greedy_reward'][indices],
        **dataset_labels(arrays))
    return report


def combine_directions(pair, arrays, expected_rows):
    """Combine declared directions0/1, never concatenate as independent cases."""
    require(set(pair) == {0, 1}, 'Need both intervention directions exactly once')
    for direction, values in pair.items():
        require(np.array_equal(values['dataset_rows'], expected_rows), 'Intervention case order differs')
        require(np.array_equal(values['recipient_indices'], arrays['endpoint_indices'][expected_rows, direction]), 'Recipient endpoint differs')
    return {key: np.stack([pair[b][key] for b in (0, 1)], axis=1)
        for key in ('action_indices', 'action_probabilities', 'greedy_reward')}


def _compact(report, metadata):
    result = {k: metadata[k] for k in ('kind', 'seed', 'condition', 'checkpoint', 'partition', 'mode')}
    for group in ('content', 'role'):
        if group in report:
            result[group] = dict(macro=report[group]['macro'],
                by_axis={a: dict(n_case_worlds=v['n_case_worlds'], weight_sum=v['weight_sum'],
                    means={k: x['mean'] for k, x in v['metrics'].items()}) for a, v in report[group]['by_axis'].items()})
    return result


def flatten_numbers(value, prefix=''):
    if isinstance(value, dict):
        for key, child in value.items(): yield from flatten_numbers(child, prefix+'/'+str(key))
    elif isinstance(value, list):
        for i, child in enumerate(value): yield from flatten_numbers(child, prefix+'/'+str(i))
    elif value is None or isinstance(value, (int, float, bool)):
        yield prefix.lstrip('/'), value


def save_report(output, report_id, report, metadata, arrays, rows, writer):
    clean = metrics.summary_only(report)
    report_path = output/'reports'/(report_id+'.json')
    row_path = output/'row_metrics'/(report_id+'.npz')
    payload = dict(dataset_rows=rows, case_index=arrays['case_index'][rows],
        endpoint_indices=arrays['endpoint_indices'][rows], **report['row_values'])
    np.savez_compressed(row_path, **payload)
    row_metadata = dict(path=str(row_path.relative_to(output)), sha256=sha(row_path),
        arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=array_sha(v)) for k, v in payload.items()})
    report_path.write_bytes(json_bytes(dict(metadata=metadata, metrics=clean, row_metrics=row_metadata)))
    for metric_path, value in flatten_numbers(clean):
        writer.writerow([report_id, metadata['kind'], metadata['seed'], metadata['condition'],
            metadata['checkpoint'], metadata['partition'], metadata.get('mode', ''), metric_path, value])
    return dict(report_id=report_id, **metadata, report_file=str(report_path.relative_to(output)),
        report_sha256=sha(report_path), row_metrics=row_metadata,
        compact=_compact(report, metadata))


def primary_comparison(compact):
    index = {(r['seed'], r['condition'], r['checkpoint'], r['partition']): r for r in compact if r['kind'] == 'natural'}
    rows = []
    for seed in SEEDS:
        s, l = [index[seed, c, 6000, 'heldout_layouts'] for c in ('PI_silent', 'PI_live')]
        rows.append(dict(seed=seed,
            PI_silent_macro=s['content']['macro']['both_endpoints_apt'],
            PI_live_macro=l['content']['macro']['both_endpoints_apt'],
            PI_live_minus_silent=l['content']['macro']['both_endpoints_apt']-s['content']['macro']['both_endpoints_apt'],
            axis_rates={a: dict(PI_silent=s['content']['by_axis'][a]['means']['both_endpoints_apt'],
                PI_live=l['content']['by_axis'][a]['means']['both_endpoints_apt'],
                PI_live_minus_silent=l['content']['by_axis'][a]['means']['both_endpoints_apt']-s['content']['by_axis'][a]['means']['both_endpoints_apt']) for a in metrics.AXES}))
    return dict(endpoint=6000, recipient_partition='heldout_layouts', metric='content.both_endpoints_apt',
        seeds=rows, independent_paired_seeds=4,
        equal_seed_mean={k: sum(row[k] for row in rows)/4 for k in ('PI_silent_macro', 'PI_live_macro', 'PI_live_minus_silent')},
        reference=dict(PI_silent_zero='Structural under identical listener observation and deterministic greedy actions; not an empirical discovery.',
            global_fixed_pair_content_upper_bound=1/3), significance_test=None)


def summarize(results_path, dataset_dir, output):
    started = time.perf_counter(); results_path = Path(results_path).resolve()
    dataset_dir, output = Path(dataset_dir).resolve(), Path(output).resolve()
    require(not output.exists(), 'Refuse to overwrite summary')
    result = read(results_path); require(result['status'] == 'completed', 'Wait for allcompleted')
    natural, interventions = result['natural_records'], result['intervention_records']
    require(len(natural) == 192 and len(interventions) == 320, 'Incomplete logical record budget')
    natural_index = {(r['seed'], r['condition'], r['checkpoint'], r['partition']): r for r in natural}
    require(set(natural_index) == set(product(SEEDS, CONDITIONS, CHECKPOINTS, PARTITIONS)) and len(natural_index) == 192, 'Natural grid incomplete or duplicate')
    intervention_index = {(r['seed'], r['condition'], r['partition'], r['mode'], r['direction']): r for r in interventions}
    require(set(intervention_index) == set(product(SEEDS, ('PI_silent', 'PI_live'), PARTITIONS, MODES, (0, 1)))
        and len(intervention_index) == 320, 'Intervention grid incomplete or duplicate')
    require(all(r['checkpoint'] == 6000 for r in interventions), 'Only frozen endpoint interventions allowed')
    require(sum(bool(r['reused_saved_endpoint']) for r in natural) == 32
        and all(bool(r['reused_saved_endpoint']) == (r['checkpoint'] == 6000) for r in natural), 'Natural endpoint reuse differs')
    manifest = read(dataset_dir/'manifest.json')
    source_hashes = {str(results_path): sha(results_path), str(dataset_dir/'manifest.json'): sha(dataset_dir/'manifest.json'),
        str(Path(__file__)): sha(__file__), str(Path(metrics.__file__)): sha(metrics.__file__)}
    arrays = {}
    for part in PARTITIONS:
        path = dataset_dir/(part+'.npz'); require(sha(path) == manifest['outputs'][part+'.npz']['sha256'], 'Dataset NPZ changed')
        source_hashes[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as z: arrays[part] = {k: z[k] for k in z.files}
    output.mkdir(parents=True); (output/'reports').mkdir(); (output/'row_metrics').mkdir()
    catalog = []; sanity = []; counts = dict(natural_reports=0, intervention_reports=0, remote_contrasts=0, neural_forwards=0)
    with (output/'完整分层指标.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream); writer.writerow(['report_id','kind','seed','condition','checkpoint','partition','mode','metric_path','value'])
        for seed, condition, checkpoint, part in product(SEEDS, CONDITIONS, CHECKPOINTS, PARTITIONS):
            record = natural_index[seed, condition, checkpoint, part]
            values = load_record(record, results_path, source_hashes)
            report = natural_report(values, arrays[part])
            if condition == 'PI_silent':
                require(report['content']['macro']['both_endpoints_apt'] == 0 and report['role']['macro']['both_endpoints_apt'] == 0,
                    'Silent structural both-apt zero failed')
            metadata = dict(kind='natural', seed=seed, condition=condition, checkpoint=checkpoint, partition=part,
                mode='natural', source_records=[record], reused_saved_endpoint=record['reused_saved_endpoint'])
            rid = f'natural_{seed}_{condition}_{checkpoint:04d}_{part}'
            rows = np.arange(len(arrays[part]['case_index']), dtype=np.int64)
            catalog.append(save_report(output, rid, report, metadata, arrays[part], rows, writer)); counts['natural_reports'] += 1
            del values, report
        for seed, condition, part in product(SEEDS, ('PI_silent', 'PI_live'), PARTITIONS):
            a = arrays[part]; rows = np.flatnonzero(a['classification'] == 1).astype(np.int64)
            natural_values = load_record(natural_index[seed, condition, 6000, part], results_path, source_hashes)
            ni = a['endpoint_indices'][rows]
            na, np_ = natural_values['action_indices'][ni], natural_values['action_probabilities'][ni]
            reports, donor_info = {}, {}
            for mode in MODES:
                records = [intervention_index[seed, condition, part, mode, b] for b in (0, 1)]
                pair = {b: load_record(record, results_path, source_hashes) for b, record in enumerate(records)}
                combined = combine_directions(pair, a, rows)
                window = 'both' if mode == 'sham' else mode.rsplit('_', 1)[1]
                report = metrics.intervention_metrics(action_indices=combined['action_indices'], action_probs=combined['action_probabilities'],
                    natural_action_indices=na, natural_action_probs=np_, native_rewards=combined['greedy_reward'], window=window,
                    **dataset_labels(a, rows))
                if mode == 'sham' or condition == 'PI_silent':
                    require(np.array_equal(combined['action_indices'], na), 'Sham/silent action changed')
                    require(np.allclose(combined['action_probabilities'], np_, atol=2e-12, rtol=0), 'Sham/silent probabilities changed')
                    sanity.append(dict(seed=seed, condition=condition, partition=part, mode=mode,
                        identity='sham_or_silent', action_equal=True, max_probability_error=float(np.max(np.abs(combined['action_probabilities']-np_)))))
                if mode == 'local_opposite_both':
                    listener = a['listener'][rows]; ix=np.arange(len(rows))[:,None]; direction=np.arange(2)[None,:]
                    actual_actions=combined['action_indices'][ix,direction,listener[:,None]]
                    donor_actions=na[:,::-1][ix,direction,listener[:,None]]
                    actual_p=combined['action_probabilities'][ix,direction,listener[:,None]]
                    donor_p=np_[:,::-1][ix,direction,listener[:,None]]
                    require(np.array_equal(actual_actions,donor_actions) and np.allclose(actual_p,donor_p,atol=2e-12,rtol=0), 'Local both structural listener identity failed')
                    sanity.append(dict(seed=seed,condition=condition,partition=part,mode=mode,identity='local_both_listener_donor',
                        action_equal=True,max_probability_error=float(np.max(np.abs(actual_p-donor_p)))))
                metadata = dict(kind='intervention',seed=seed,condition=condition,checkpoint=6000,partition=part,mode=mode,source_records=records)
                rid=f'intervention_{seed}_{condition}_{part}_{mode}'
                catalog.append(save_report(output,rid,report,metadata,a,rows,writer)); counts['intervention_reports']+=1
                if mode.startswith('remote_'):
                    reports[mode]=report
                    donor_info[mode]={b:(pair[b]['donor_partition'].copy(),pair[b]['donor_indices'].copy()) for b in (0,1)}
                del pair,combined
            for window in WINDOWS:
                same,opposite='remote_same_'+window,'remote_opposite_'+window
                for b in (0,1):
                    ds,do=donor_info[same][b],donor_info[opposite][b]
                    require(np.array_equal(ds[0],do[0]) and np.array_equal(ds[0],a['remote_donor_partition'][rows]), 'Remote donor partitions not paired')
                    require(np.array_equal(ds[1],a['remote_donor_endpoint_indices'][rows,b])
                        and np.array_equal(do[1],a['remote_donor_endpoint_indices'][rows,1-b]), 'Remote same/opposite donor indices differ')
                report=metrics.remote_contrast(reports[same],reports[opposite])
                metadata=dict(kind='remote_contrast',seed=seed,condition=condition,checkpoint=6000,partition=part,mode=window,
                    matched_modes=[same,opposite])
                catalog.append(save_report(output,f'remote_{seed}_{condition}_{part}_{window}',report,metadata,a,rows,writer));counts['remote_contrasts']+=1
            del reports,donor_info,natural_values
    compact=[r['compact'] for r in catalog];primary=primary_comparison(compact)
    require(counts==dict(natural_reports=192,intervention_reports=160,remote_contrasts=48,neural_forwards=0), 'Summary report counts differ')
    require(all(sha(p)==digest for p,digest in source_hashes.items()), 'Sources changed during summary')
    summary=dict(status='completed_read_only_summary',created_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
        source_sha256=source_hashes,counts=counts,primary=primary,catalog=catalog,compact_records=compact,sanity=sanity,
        dataset_summary=manifest['summary'],remote_background_mapping=manifest['remote_background_mapping'],
        limits=['Four paired inherited team seeds; no new training or independent confirmation.',
            'PI_silent zero and local_both donor identity are structural checks, not evidence of semantics.',
            'Natural row unit is an unordered two-endpoint case; intervention directions are paired within it.',
            'Remote heldout refers only to recipient; five of six heldout layouts use a train-layout donor.',
            'Complete row metrics and all axis/partner/sender/listener strata are retained; no success filtering.',
            'Metrics API reused rather than independently reimplemented; execution/measurement audit is separate.'])
    (output/'summary.json').write_bytes(json_bytes(summary))
    lines=['# 单项需求诊断汇总','', '只读统计完成；本汇总没有神经前向。主量为留出recipient、6000步的content三轴等权，两端听者均适切。','',
        '| 种子 | PI静默 | PI开放 | 开放−静默（百分点） |','|---|---:|---:|---:|']
    for row in primary['seeds']:lines.append(f"| {row['seed']} | {100*row['PI_silent_macro']:.4f}% | {100*row['PI_live_macro']:.4f}% | {100*row['PI_live_minus_silent']:+.4f} |")
    means=primary['equal_seed_mean'];lines.append(f"| 四种子等权 | {100*means['PI_silent_macro']:.4f}% | {100*means['PI_live_macro']:.4f}% | {100*means['PI_live_minus_silent']:+.4f} |")
    lines += ['', '全部192自然、160双向合并干预及48远背景配对对比，见summary.json、reports与row_metrics；完整数值分层见完整分层指标.csv。',
        '', 'PI静默零与本地both供体重现是结构检查。固定一对搭档的内容参照为1/3；听者两端适切不等于团队两端满分。远背景留出只限定recipient，5/6供体来自训练布局。']
    (output/'汇总.md').write_text('\n'.join(lines)+'\n')
    return dict(status=summary['status'],output=str(output),counts=counts,summary_sha256=sha(output/'summary.json'),primary=primary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--dataset',type=Path,default=HERE/'dataset_001')
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    try: print(json.dumps(summarize(args.results,args.dataset,args.out),ensure_ascii=False))
    except BaseException as error:
        if args.out.is_dir() and not (args.out/'failure.json').exists():
            (args.out/'failure.json').write_bytes(json_bytes(dict(status='failed',error=str(error),traceback=traceback.format_exc(),
                summary_source_sha256=sha(__file__),automatic_retry=False)))
        raise
