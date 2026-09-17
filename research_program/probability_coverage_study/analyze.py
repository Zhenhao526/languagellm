"""Post-hoc probability coverage from frozen policy-gain JSON records only.

No model imports, weight loading, feature extraction or forward calls. Prepare
binds source bytes without computing new probability metrics. Execute processes
the complete 64 x 8 x 2 grid once; any failure is retained and stops the batch.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import numpy as np

STUDY = Path(__file__).resolve().parent
WORK = STUDY.parents[1]
OLD = WORK / 'research_program/policy_gain_study'
DEFAULT_BATCH = OLD / 'results/gain_001'
SEEDS = (28101, 28102, 28103, 28104)
KINDS, GAINS = ('additive', 'joint'), (1, 3)
UPDATES = (0, 100, 300, 600, 1200, 1800, 2100, 2400)
EARLY = (0, 100, 300, 600, 1200)
TIME_PAIRS = ((0, 300), (300, 600), (600, 1200), (0, 2400))
MAPS = tuple(permutations(range(6), 2))
CODES = tuple(product(range(7), repeat=2))
MATCHINGS = {1: ((0, 1), (2, 3), (4, 5)), 2: ((0, 2), (1, 4), (3, 5)),
             3: ((0, 3), (1, 5), (2, 4))}
CONDITIONS = tuple(f"{('split'+str(s)) if s else 'full'}_{k}_gain{g}"
                   for s, k, g in product((1, 2, 3, 0), KINDS, GAINS))
METRICS = ('C', 'E_G', 'C_minus_E_G', 'U', 'N')


def require(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        stream.write('\n')


def runtime():
    return dict(python=platform.python_version(), numpy=str(np.__version__), platform=platform.platform())


def partition(condition):
    prefix = condition.split('_')[0]
    split = int(prefix[-1]) if prefix.startswith('split') else 0
    held = [i for i, pair in enumerate(MAPS) if split and tuple(sorted(pair)) in MATCHINGS[split]]
    return [i for i in range(30) if i not in held], held


def prepare(output, batch=DEFAULT_BATCH):
    """Only metadata and file hashes; do not read per-record logits here."""
    require('torch' not in sys.modules and 'camp' not in sys.modules, 'Clean pure-numeric process required')
    output, batch = Path(output).resolve(), Path(batch).resolve()
    require(not output.exists(), 'Never overwrite preparation')
    require(batch == DEFAULT_BATCH.resolve(), 'This supplement is fixed to gain_001')
    status, manifest = read(batch/'status.json'), read(batch/'manifest.json')
    probe = read(batch/'probe/execution/results.json')
    old = read(batch/'analysis.json')
    diagnosis = read(batch/'analysis_dtype_diagnosis.json')
    recovery = read(batch/'analysis_recovery_001/result.json')
    audit = read(batch/'audit_results_002.json')
    require(status['status'] == 'complete' and status['completed_runs'] == status['expected_runs'] == 64,
            'All 64 source runs must be complete')
    require(manifest['seeds'] == list(SEEDS) and set(manifest['conditions']) == set(CONDITIONS), 'Source factorial grid changed')
    require(probe['status'] == 'complete' and probe['counts']['directions'] == 1024 and
            probe['counts']['receiver_maximum_ties'] == 0 and old['status'] == 'complete', 'Complete prior probe/analysis required')
    require(diagnosis['status'] == 'confirmed_margin_dtype_mismatch_only' and diagnosis['completed_records'] == 1024 and
            recovery['status'] == 'complete' and recovery['analysis_sha256'] == sha(batch/'analysis.json') and
            audit['status'] == 'passed' and audit['analysis_sha256'] == sha(batch/'analysis.json'), 'Old recovery/audit not bound')
    require(recovery['recovery_plan_sha256'] == sha(batch/'analysis_recovery_001/plan.json') and
            probe['plan_sha256'] == sha(batch/'probe/plan.json') == read(batch/'probe/freeze.json')['plan_sha256'],
            'Old plans changed')
    old_recovery_plan = read(batch/'analysis_recovery_001/plan.json')
    require(old_recovery_plan['independent_diagnosis_sha256'] == sha(batch/'analysis_dtype_diagnosis.json') and
            old_recovery_plan['original_failure_sha256'] == sha(batch/'analysis_failed_001.json') and
            old_recovery_plan['source_sha256'] == sha(OLD/'analysis_float32_replay.py') and
            old_recovery_plan['original_analysis_sha256'] == sha(OLD/'analysis.py') and
            diagnosis['script_sha256'] == sha(OLD/'diagnose_analysis_dtype.py') and
            all(sha(OLD/name) == digest for name, digest in diagnosis['frozen_source_sha256'].items()),
            'Old diagnosis/recovery source binding differs')
    inputs = {}
    for path, digest in manifest['source_hashes'].items():
        require(sha(path) == digest, f'Old frozen source changed: {path}')
        inputs[str(Path(path).resolve())] = digest
    for relative in ('status.json', 'manifest.json', 'analysis.json', 'analysis_failed_001.json',
        'analysis_dtype_diagnosis.json', 'analysis_recovery_001/plan.json', 'analysis_recovery_001/result.json',
        'audit_results_002.json', 'audit_execution_001/verification.json',
        'probe/plan.json', 'probe/freeze.json', 'probe/execution/results.json'):
        path = batch/relative; inputs[str(path)] = sha(path)
    for path in (OLD/'analysis_float32_replay.py', OLD/'diagnose_analysis_dtype.py'):
        inputs[str(path)] = sha(path)
    entries, seen = [], set()
    for row in probe['records']:
        key = tuple(row[k] for k in ('seed', 'condition', 'update', 'scout'))
        require(key not in seen, 'Duplicate source record'); seen.add(key)
        path = (batch/'probe/execution'/row['file']).resolve()
        require(path.is_relative_to((batch/'probe/execution/records').resolve()), 'Invalid source record path')
        require(sha(path) == row['sha256'], 'Source record changed')
        inputs[str(path)] = row['sha256']
        entries.append(dict(seed=key[0], condition=key[1], update=key[2], scout=key[3],
                            file=str(path), sha256=row['sha256']))
    require(len(entries) == 1024 and seen == set(product(SEEDS, CONDITIONS, UPDATES, range(2))), 'Incomplete source 1024 grid')
    sources = {str(STUDY/name): sha(STUDY/name) for name in ('analyze.py', 'tests.py', 'plan.md')}
    plan = dict(schema_version=1, created_at=now(), design='post_hoc_probability_coverage_gain001',
        timing='Designed after the old J/U/N results were visible; not preregistered before training.',
        batch=str(batch), source_analysis=str(batch/'analysis.json'), source_records=entries,
        seeds=list(SEEDS), conditions=list(CONDITIONS), updates=list(UPDATES), early_updates=list(EARLY),
        time_pairs=[list(pair) for pair in TIME_PAIRS], metrics=list(METRICS),
        temperature=1., dtype='float64', log_softmax='subtract maximum, then log(sum(exp(centered)))',
        probability_product='p_F(correct|same complete code) * p_W(correct|same complete code)',
        maximum_codes='All strictly equal float64 maximum product values; first in lexicographic 00..66 order.',
        numeric_failure='Nonfinite values or any probability/product underflow to zero stop the entire batch; no fallback.',
        original_counts='Restore stored logits to float32 solely for exact original actions/ties/margins validation.',
        unit='Mean splits and directions within each of four training seeds; keep full separate.',
        execution=dict(new_training=False, model_imports=0, weight_loads=0, feature_extraction=False,
            neural_forward_calls=0, automatic_retry=False, resume=False),
        expected=dict(runs=64, records=1024, map_records=30720, photo_messages=491520),
        source_files_sha256=inputs, analysis_sources_sha256=sources, prepare_runtime=runtime())
    output.mkdir(parents=True, exist_ok=False)
    (output/'code_snapshot').mkdir()
    for path in sources: shutil.copyfile(path, output/'code_snapshot'/Path(path).name)
    write_new(output/'plan.json', plan)
    write_new(output/'freeze.json', dict(status='prepared_not_executed', plan_sha256=sha(output/'plan.json')))
    return dict(status='prepared_not_executed', output=str(output), **plan['expected'])


def verify(output):
    output = Path(output).resolve()
    require(sha(output/'plan.json') == read(output/'freeze.json')['plan_sha256'], 'New plan changed')
    plan = read(output/'plan.json')
    for path, digest in plan['source_files_sha256'].items():
        require(sha(path) == digest, f'Frozen input changed: {path}')
    for path, digest in plan['analysis_sources_sha256'].items():
        require(sha(path) == digest and sha(output/'code_snapshot'/Path(path).name) == digest, 'New analysis source changed')
    return plan


def stable_log_softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    require(x.shape == (49, 2, 6) and np.isfinite(x).all(), 'Expected finite 49x2x6 logits')
    centered = x-x.max(axis=-1, keepdims=True)
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        result = centered-np.log(np.exp(centered).sum(axis=-1, keepdims=True))
        probability = np.exp(result)
    require(np.isfinite(result).all() and np.isfinite(probability).all() and (probability > 0).all(),
            'Nonfinite log probabilities or probability underflow: stop whole batch')
    require(np.allclose(probability.sum(-1), 1., rtol=0., atol=1e-14), 'Probability normalization failure')
    return result, probability


def measure(logits, messages, train_ids, heldout_ids):
    """Pure synthetic-testable kernel; all codes shared by the two goals."""
    x = np.asarray(logits, dtype=np.float64)
    logp, probabilities = stable_log_softmax(x)
    messages = np.asarray(messages)
    require(messages.shape == (30, 16, 2) and np.issubdtype(messages.dtype, np.integer) and
            ((messages >= 0) & (messages < 7)).all(), 'Expected actual integer 30x16 two-symbol messages')
    require(len(set(train_ids)) == len(train_ids) and len(set(heldout_ids)) == len(heldout_ids) and
            not set(train_ids)&set(heldout_ids) and set(train_ids)|set(heldout_ids) == set(range(30)), 'Map partition differs')
    target = np.asarray(MAPS)
    with np.errstate(under='ignore', invalid='raise'):
        success = (probabilities[:, 0, target[:, 0]]*probabilities[:, 1, target[:, 1]]).T
    require(success.shape == (30, 49) and np.isfinite(success).all() and (success > 0).all(),
            'Nonfinite joint probability or product underflow: stop whole batch')
    c = success.max(1)
    code_ids = messages[..., 0]*7+messages[..., 1]
    eg = np.take_along_axis(success, code_ids, axis=1)
    actions = x.argmax(-1)
    joint_correct = (actions[:, None, :] == target[None, :, :]).all(-1).T
    u = joint_correct.any(1)
    n = np.take_along_axis(joint_correct, code_ids, 1)
    require((eg <= c[:, None]).all() and not (n & ~u[:, None]).any(), 'Existential/natural consistency failure')
    map_rows = []
    for mid, sites in enumerate(MAPS):
        maximum_ids = np.flatnonzero(success[mid] == c[mid]).tolist()
        map_rows.append(dict(map_id=mid, locations=list(sites), partition='train' if mid in train_ids else 'heldout',
            C=float(c[mid]), max_code_ids=maximum_ids, first_max_code_id=maximum_ids[0],
            max_messages=[list(CODES[i]) for i in maximum_ids], first_max_message=list(CODES[maximum_ids[0]]),
            E_G=eg[mid].tolist(), C_minus_E_G=(c[mid]-eg[mid]).tolist(),
            U=bool(u[mid]), N=n[mid].tolist(), N_count=int(n[mid].sum())))
    summaries = {}
    for name, ids in [('train', train_ids), ('heldout', heldout_ids)]:
        if not ids: summaries[name] = None; continue
        summaries[name] = dict(C_sum=float(c[ids].sum()), C_n=len(ids), C=float(c[ids].mean()),
            E_G_sum=float(eg[ids].sum()), E_G_n=len(ids)*16, E_G=float(eg[ids].mean()),
            C_minus_E_G=float((c[ids, None]-eg[ids]).mean()),
            U_numerator=int(u[ids].sum()), U_denominator=len(ids), U=float(u[ids].mean()),
            N_numerator=int(n[ids].sum()), N_denominator=len(ids)*16, N=float(n[ids].mean()))
    return dict(receiver_log_probabilities=logp.tolist(), receiver_probabilities=probabilities.tolist(),
                maps=map_rows, summaries=summaries)


def check_old_record(record, measured):
    """Exact U/N counts and original float32 margin replay, not relaxed comparison."""
    receiver = record['receiver']; x64 = np.asarray(receiver['logits'], dtype=np.float64); x32 = x64.astype(np.float32)
    require(np.array_equal(x64, x32.astype(np.float64)), 'Stored logits not exactly float32')
    order = np.sort(x32, -1)
    require(receiver['actions'] == x32.argmax(-1).tolist() == x64.argmax(-1).tolist(), 'Original actions differ')
    counts = (x32 == x32.max(-1, keepdims=True)).sum(-1)
    require((counts == 1).all() and receiver['unique_max_count'] == counts.tolist(), 'Original unique maximum differs')
    require(receiver['top_two_margin'] == (order[..., -1]-order[..., -2]).tolist(), 'Original float32 margins differ')
    require(record['emitted'] == record['delivered'], 'Source channel not direct')
    require(len(record['analysis']['maps']) == 30, 'Original map count differs')
    for new, old in zip(measured['maps'], record['analysis']['maps']):
        require(new['map_id'] == old['map_id'] and new['locations'] == old['locations'] and
                new['partition'] == old['partition'] and new['U'] == old['U_full49'] == old['U_channel'] and
                new['N'] == old['natural_both'], 'Original per-map U/N exact replay failed')
    for name, new in measured['summaries'].items():
        old = record['analysis']['summaries'][name]
        if new is None: require(old is None, 'Original empty partition differs'); continue
        for field, old_key in [('U', 'U_full49'), ('N', 'N_both')]:
            require(old[old_key] == dict(numerator=new[field+'_numerator'], denominator=new[field+'_denominator'], rate=new[field]),
                    'Original aggregate U/N counts differ')
    return dict(map_U_checked=30, photo_N_checked=480, receiver_actions_checked=98,
                original_float32_margins_exact=True)


def contrasts(rows, keys, metrics):
    buckets = defaultdict(dict)
    for row in rows:
        key, cell = tuple(row[k] for k in keys), (row['seed'], row['kind'], row['gain'])
        require(cell not in buckets[key], 'Duplicate factorial cell'); buckets[key][cell] = row
    output = []
    for key, cells in sorted(buckets.items()):
        require(set(cells) == set(product(SEEDS, KINDS, GAINS)), 'Incomplete four-seed factorial grid')
        for metric in metrics:
            effects = defaultdict(list)
            for seed in SEEDS:
                a1, a3, j1, j3 = [cells[seed, k, g][metric] for k, g in
                                  [('additive', 1), ('additive', 3), ('joint', 1), ('joint', 3)]]
                values = dict(lambda_at_gain1=j1-a1, lambda_at_gain3=j3-a3, gain_at_lambda0=a3-a1,
                    gain_at_lambda1=j3-j1, interaction=(j3-j1)-(a3-a1), diagonal_joint3_minus_additive1=j3-a1)
                for name, value in values.items(): effects[name].append(value)
            output.append(dict(zip(keys, key), metric=metric, seeds=list(SEEDS), effects={name:
                dict(seed_values=v, mean=sum(v)/4, min=min(v), max=max(v)) for name, v in effects.items()}))
    return output


def summarize(entries):
    require(len(entries) == 1024 and {(r['seed'], r['condition'], r['update'], r['scout']) for r in entries} ==
            set(product(SEEDS, CONDITIONS, UPDATES, range(2))), 'Cannot summarize partial source grid')
    buckets = defaultdict(list)
    for row in entries:
        prefix, kind, gain = row['condition'].split('_')
        family = 'split' if prefix.startswith('split') else 'full'
        for part, values in row['summaries'].items():
            if values is not None:
                key = family, part, row['update'], row['seed'], kind, int(gain[4:])
                buckets[key].append(values)
    seed_rows = []
    for (family, part, update, seed, kind, gain), vals in sorted(buckets.items()):
        require(len(vals) == (6 if family == 'split' else 2), 'Incomplete split/direction group')
        require(len({v['C_n'] for v in vals}) == len({v['E_G_n'] for v in vals}) == 1, 'Unbalanced within-seed denominator')
        seed_rows.append(dict(family=family, partition=part, update=update, seed=seed, kind=kind, gain=gain,
            **{m: sum(v[m] for v in vals)/len(vals) for m in METRICS},
            U_numerator=sum(v['U_numerator'] for v in vals), U_denominator=sum(v['U_denominator'] for v in vals),
            N_numerator=sum(v['N_numerator'] for v in vals), N_denominator=sum(v['N_denominator'] for v in vals)))
    cells = defaultdict(dict)
    for row in seed_rows:
        key = row['family'], row['partition'], row['seed'], row['kind'], row['gain']
        cells[key][row['update']] = row
    auc, changes = [], []
    for (family, part, seed, kind, gain), points in sorted(cells.items()):
        require(set(points) == set(UPDATES), 'Incomplete eight-point curve')
        identity = dict(family=family, partition=part, seed=seed, kind=kind, gain=gain)
        auc.append(dict(identity, **{m+'_auc_0_1200': sum((points[l][m]+points[r][m])*.5*(r-l)
            for l, r in zip(EARLY[:-1], EARLY[1:]))/1200 for m in METRICS}))
        for left, right in TIME_PAIRS:
            changes.append(dict(identity, start_update=left, end_update=right,
                                **{m: points[right][m]-points[left][m] for m in METRICS}))
    means = group_means(seed_rows, ('family', 'partition', 'update', 'kind', 'gain'), METRICS)
    time_means = group_means(changes, ('family', 'partition', 'start_update', 'end_update', 'kind', 'gain'), METRICS)
    auc_metrics = tuple(m+'_auc_0_1200' for m in METRICS)
    return dict(seed_values=seed_rows, means=means, early_auc_seed_values=auc,
        early_auc_means=group_means(auc, ('family', 'partition', 'kind', 'gain'), auc_metrics),
        time_change_seed_values=changes, time_change_means=time_means,
        effects=dict(by_checkpoint=contrasts(seed_rows, ('family', 'partition', 'update'), METRICS),
                     early_auc=contrasts(auc, ('family', 'partition'), auc_metrics)))


def group_means(rows, keys, metrics):
    buckets = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in keys)
        require(row['seed'] not in buckets[key], 'Duplicate mean seed')
        buckets[key][row['seed']] = row
    output = []
    for key, cells in sorted(buckets.items()):
        require(set(cells) == set(SEEDS), 'Incomplete mean seed set')
        output.append(dict(zip(keys, key), seeds=list(SEEDS), metrics={metric:
            dict(seed_values=[cells[s][metric] for s in SEEDS], mean=sum(cells[s][metric] for s in SEEDS)/4)
            for metric in metrics}))
    return output


def anchor_old_summary(summary, old):
    identity = lambda r: tuple(r[k] for k in ('family', 'partition', 'update', 'seed', 'kind', 'gain'))
    old_rows = {identity(r): r for r in old['same_photo_trajectory']['seed_values']}
    require({identity(r) for r in summary['seed_values']} == set(old_rows), 'Old summary grid differs')
    for row in summary['seed_values']:
        prev = old_rows[identity(row)]
        require(row['U'] == prev['U_full49'] and row['N'] == prev['N_both'], 'Old seed U/N differs')
    short = lambda r: tuple(r[k] for k in ('family', 'partition', 'seed', 'kind', 'gain'))
    old_auc = {short(r): r for r in old['same_photo_trajectory']['early_auc_seed_values']}
    require({short(r) for r in summary['early_auc_seed_values']} == set(old_auc), 'Old AUC grid differs')
    for row in summary['early_auc_seed_values']:
        prev = old_auc[short(row)]
        require(row['U_auc_0_1200'] == prev['U_auc_0_1200'] and row['N_auc_0_1200'] == prev['N_auc_0_1200'], 'Old exact U/N AUC differs')
    return dict(seed_checkpoint_U_N_pairs=len(old_rows), seed_AUC_U_N_pairs=len(old_auc), exact=True)


def execute(output):
    output = Path(output).resolve(); plan = verify(output)
    require(runtime()['python'] == plan['prepare_runtime']['python'] and runtime()['numpy'] == plan['prepare_runtime']['numpy'],
            'Use the prepared pure-numeric runtime')
    directory = output/'execution'; directory.mkdir(exist_ok=False)
    (directory/'records').mkdir()
    write_new(directory/'started.json', dict(started_at=now(), plan_sha256=sha(output/'plan.json'), runtime=runtime()))
    started = time.monotonic(); context = {}
    try:
        entries = []
        for source in plan['source_records']:
            context = {k: source[k] for k in ('seed', 'condition', 'update', 'scout', 'file')}
            record = read(source['file'])
            require(all(record[k] == source[k] for k in ('seed', 'condition', 'update', 'scout')) and
                    record['collector'] == 1-record['scout'], 'Source record identity changed')
            train, held = partition(record['condition'])
            measured = measure(record['receiver']['logits'], record['emitted'], train, held)
            old_check = check_old_record(record, measured)
            result = dict(seed=record['seed'], condition=record['condition'], update=record['update'],
                scout=record['scout'], collector=record['collector'], source_record_file=source['file'],
                source_record_sha256=source['sha256'], photo_pairs=record['photo_pairs'], emitted=record['emitted'],
                old_exact_check=old_check, **measured)
            relative = f"records/s{record['seed']}_{record['condition']}_u{record['update']:04d}_s{record['scout']}.json"
            write_new(directory/relative, result)
            entries.append(dict(seed=record['seed'], condition=record['condition'], update=record['update'],
                scout=record['scout'], file=relative, sha256=sha(directory/relative), summaries=measured['summaries']))
        summary = summarize(entries)
        old_anchor = anchor_old_summary(summary, read(plan['source_analysis']))
        verify(output)
        require('torch' not in sys.modules and 'camp' not in sys.modules, 'Unexpected neural runtime import')
        result = dict(status='complete', completed_at=now(), elapsed_seconds=time.monotonic()-started,
            plan_sha256=sha(output/'plan.json'), runtime=runtime(), records=entries, **summary, old_summary_anchor=old_anchor,
            counts=dict(records=1024, maps=30720, photo_messages=491520, receiver_distributions=100352),
            scope='Post-hoc after known J/U/N; C/E_G use fixed temperature1 float64 probabilities, not original stochastic-mode expectation.',
            model_imports=0, weight_loads=0, neural_forward_calls=0)
        write_new(directory/'results.json', result)
        return dict(status='complete', output=str(directory/'results.json'), **result['counts'])
    except Exception as error:
        write_new(directory/'failure.json', dict(status='failed', failed_at=now(), context=context,
            elapsed_seconds=time.monotonic()-started, error=repr(error), traceback=traceback.format_exc(),
            automatic_retry=False, source_inputs_modified=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'execute'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--batch', type=Path, default=DEFAULT_BATCH)
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.out, args.batch)
    elif args.command == 'verify':
        plan = verify(args.out); result = dict(status='verified', **plan['expected'])
    else: result = execute(args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
