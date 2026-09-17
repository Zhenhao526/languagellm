"""Independent probability audit from saved logits, never model inference.

No study implementation is imported. The temperature is one. The same complete
message is used in both goal probabilities before maximizing over messages.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import permutations, product
import json
import math
from pathlib import Path
import sys
import time
import traceback
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
SEEDS = (28101, 28102, 28103, 28104)
UPDATES = (0, 100, 300, 600, 1200, 1800, 2100, 2400)
EARLY = (0, 100, 300, 600, 1200)
MAPS = tuple(permutations(range(6), 2))
CONDITIONS = tuple(f'{"split"+str(s) if s else "full"}_{k}_gain{g}'
                   for s, k, g in product((1, 2, 3, 0), ('additive', 'joint'), (1, 3)))
METRICS = ('C', 'E_G', 'C_minus_E_G', 'U', 'N')


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def partition(split):
    matchings = {0: (), 1: ((0, 1), (2, 3), (4, 5)),
                 2: ((0, 2), (1, 4), (3, 5)), 3: ((0, 3), (1, 5), (2, 4))}
    held = {p for a, b in matchings[split] for p in ((a, b), (b, a))}
    return {name: [i for i, p in enumerate(MAPS) if (p in held) == flag]
            for name, flag in (('train', False), ('heldout', True))}


def independent_probability(logits, emitted):
    """Scalar float64 log-sum-exp reference, with explicit map/code loops."""
    z = np.asarray(logits, dtype=np.float64)
    messages = np.asarray(emitted)
    require(z.shape == (49, 2, 6) and np.isfinite(z).all(), 'Logit schema/finite')
    require(messages.shape == (30, 16, 2) and np.issubdtype(messages.dtype, np.integer) and
            ((messages >= 0) & (messages < 7)).all(), 'Greedy message schema')
    logp = np.empty(z.shape, dtype=np.float64)
    for code in range(49):
        for goal in range(2):
            row = z[code, goal]
            peak = max(row)
            normalizer = math.log(math.fsum(math.exp(float(v-peak)) for v in row))
            logp[code, goal] = row-peak-normalizer
    probabilities = np.exp(logp)
    actions = z.argmax(2)
    counts = (z == z.max(2, keepdims=True)).sum(2)
    C, EG, U, N, logC, logEG, codesets = [], [], [], [], [], [], []
    all_joint_logp = []
    for mid, (food, water) in enumerate(MAPS):
        scores = [float(logp[c, 0, food]+logp[c, 1, water]) for c in range(49)]
        maximum = max(scores)
        logC.append(maximum); C.append(math.exp(maximum))
        codesets.append([c for c, v in enumerate(scores) if v == maximum])
        all_joint_logp.append(scores)
        unique = [(actions[c, 0] == food and actions[c, 1] == water) for c in range(49)]
        U.append(any(unique))
        chosen = [int(m[0]*7+m[1]) for m in messages[mid]]
        EG.append([math.exp(scores[c]) for c in chosen]); logEG.append([scores[c] for c in chosen])
        N.append([bool(unique[c]) for c in chosen])
        require(all(scores[c] <= maximum for c in chosen), 'One-code E_G exceeds C in log domain')
    return dict(logp=logp, probabilities=probabilities, actions=actions, unique_max_count=counts,
                C=np.array(C), E_G=np.array(EG), U=np.array(U), N=np.array(N),
                log_C=np.array(logC), log_E_G=np.array(logEG), max_code_ids=codesets,
                joint_log_probability=np.array(all_joint_logp))


def independent_summary(result, ids):
    if not ids: return None
    C, EG = float(result['C'][ids].mean()), float(result['E_G'][ids].mean())
    return dict(C=C, E_G=EG, U=float(result['U'][ids].mean()), N=float(result['N'][ids].mean()), C_minus_E_G=C-EG,
                C_sum=float(result['C'][ids].sum()), C_n=len(ids), E_G_sum=float(result['E_G'][ids].sum()), E_G_n=len(ids)*16,
                U_numerator=int(result['U'][ids].sum()), U_denominator=len(ids),
                N_numerator=int(result['N'][ids].sum()), N_denominator=len(ids)*16)


def source_record_check(source, entry, expected_photos):
    """Cross-check all old discrete observations before using new probabilities."""
    for key in ('seed', 'condition', 'update', 'scout'):
        require(source[key] == entry[key], 'Source index identity: ' + key)
    require(source['collector'] == 1-source['scout'] and source['photo_pairs'] == expected_photos,
            'Source collector/fixed validation photos')
    z = np.asarray(source['receiver']['logits'], dtype=np.float64)
    require(z.shape == (49, 2, 6) and np.isfinite(z).all() and
            np.array_equal(z.astype(np.float32).astype(np.float64), z), 'Exact source float32 logits')
    emitted = np.asarray(source['emitted'])
    require(np.array_equal(emitted, np.asarray(source['delivered'])), 'Source direct channel')
    calc = independent_probability(z, emitted)
    require((calc['unique_max_count'] == 1).all() and
            calc['unique_max_count'].tolist() == source['receiver']['unique_max_count'] and
            calc['actions'].tolist() == source['receiver']['actions'], 'Exact old actions/zero ties')
    ordered = np.sort(z.astype(np.float32), axis=-1)
    require(source['receiver']['top_two_margin'] == (ordered[..., -1]-ordered[..., -2]).tolist(), 'Original float32 margins')
    require((calc['probabilities'] > 0).all() and (np.exp(calc['joint_log_probability']) > 0).all(),
            'Registered whole-batch failure rule: probability/product underflow')
    prefix = source['condition'].split('_')[0]
    parts = partition(int(prefix[5:]) if prefix.startswith('split') else 0)
    require(len(source['analysis']['maps']) == 30 and entry['summaries'] == source['analysis']['summaries'],
            'Old map/index summary inventory')
    codes = emitted[..., 0]*7+emitted[..., 1]
    for mid, target in enumerate(MAPS):
        saved = source['analysis']['maps'][mid]
        joint = np.flatnonzero((calc['actions'] == target).all(1)).tolist()
        natural_actions = calc['actions'][codes[mid]]
        correct = natural_actions == target
        require(saved['map_id'] == mid and saved['locations'] == list(target) and
                saved['U_full49'] is bool(calc['U'][mid]) and saved['U_channel'] is bool(calc['U'][mid]) and
                saved['joint_code_ids'] == joint and saved['natural_both'] == calc['N'][mid].tolist() and
                saved['natural_actions'] == natural_actions.tolist() and
                saved['natural_correct_by_goal'] == correct.tolist(), 'Old map-level U/N/action evidence')
    summaries = {}
    for part, ids in dict(all=list(range(30)), **parts).items():
        saved = source['analysis']['summaries'][part]
        summaries[part] = independent_summary(calc, ids)
        if not ids:
            require(saved is None, 'Empty old partition'); continue
        for name, count, denom in (('U_full49', int(calc['U'][ids].sum()), len(ids)),
                                  ('N_both', int(calc['N'][ids].sum()), len(ids)*16)):
            require(saved[name] == dict(numerator=count, denominator=denom, rate=count/denom), 'Exact old integer summary')
    return calc, summaries


def compare_tree(expected, actual, label='root', tolerance=2e-12):
    """Continuous values use abs+relative tolerance; keys/integers remain exact."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and expected.keys() == actual.keys(), 'Dictionary schema: ' + label)
        for key in expected: compare_tree(expected[key], actual[key], label+'/'+str(key), tolerance)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, list) and len(expected) == len(actual), 'Sequence schema: ' + label)
        for i, (a, b) in enumerate(zip(expected, actual)): compare_tree(a, b, label+'/'+str(i), tolerance)
    elif isinstance(expected, (float, np.floating)):
        require(isinstance(actual, (float, int)) and math.isfinite(actual) and
                abs(expected-actual) <= tolerance*(1+abs(expected)), f'Continuous mismatch {label}: {expected} != {actual}')
    else:
        require(type(expected) is type(actual) and expected == actual, 'Exact discrete mismatch ' + label)


def independent_aggregate(records):
    """First average directions/splits inside seed; retain all four seeds."""
    buckets = defaultdict(list)
    for r in records:
        prefix, kind, gain = r['condition'].split('_')
        family = 'split' if prefix.startswith('split') else 'full'
        for part in ('train', 'heldout'):
            values = r['summaries'][part]
            if values is None: continue
            key = family, part, r['update'], r['seed'], kind, int(gain[4:])
            buckets[key].append(values)
    rows = []
    for (family, part, update, seed, kind, gain), values in sorted(buckets.items()):
        require(len(values) == (6 if family == 'split' else 2), 'Within-seed count')
        row = dict(family=family, partition=part, update=update, seed=seed, kind=kind, gain=gain,
                    **{m: math.fsum(v[m] for v in values)/len(values) for m in METRICS})
        for key in ('U_numerator', 'U_denominator', 'N_numerator', 'N_denominator'):
            if all(key in v for v in values): row[key] = sum(v[key] for v in values)
        rows.append(row)
    expected = set(product(('split',), ('train', 'heldout'), UPDATES, SEEDS, ('additive', 'joint'), (1, 3)))
    expected |= set(product(('full',), ('train',), UPDATES, SEEDS, ('additive', 'joint'), (1, 3)))
    require({tuple(r[k] for k in ('family', 'partition', 'update', 'seed', 'kind', 'gain')) for r in rows} == expected,
            'Complete seed-level grid')
    grouped = defaultdict(list)
    auc_grouped = defaultdict(dict)
    for r in rows:
        grouped[tuple(r[k] for k in ('family', 'partition', 'update', 'kind', 'gain'))].append(r)
        if r['update'] <= 1200:
            auc_grouped[tuple(r[k] for k in ('family', 'partition', 'seed', 'kind', 'gain'))][r['update']] = r
    means = []
    for key, vals in sorted(grouped.items()):
        vals.sort(key=lambda v: v['seed'])
        require([v['seed'] for v in vals] == list(SEEDS), 'Four seed mean')
        means.append(dict(zip(('family', 'partition', 'update', 'kind', 'gain'), key), seeds=list(SEEDS),
                    metrics={m: dict(seed_values=[v[m] for v in vals], mean=math.fsum(v[m] for v in vals)/4)
                             for m in METRICS}))
    auc = []
    for key, points in sorted(auc_grouped.items()):
        require(set(points) == set(EARLY), 'Fixed five-point AUC interval')
        auc.append(dict(zip(('family', 'partition', 'seed', 'kind', 'gain'), key),
            **{m+'_auc_0_1200': math.fsum((points[l][m]+points[r][m])*(r-l)/2
                        for l, r in zip(EARLY[:-1], EARLY[1:]))/1200 for m in METRICS}))
    time_seed, time_means = independent_time_changes(rows)
    auc_means = []
    auc_buckets = defaultdict(list)
    for row in auc: auc_buckets[tuple(row[k] for k in ('family', 'partition', 'kind', 'gain'))].append(row)
    for key, vals in sorted(auc_buckets.items()):
        vals.sort(key=lambda row: row['seed'])
        auc_means.append(dict(zip(('family', 'partition', 'kind', 'gain'), key), seeds=list(SEEDS),
             metrics={m+'_auc_0_1200': dict(seed_values=[v[m+'_auc_0_1200'] for v in vals],
                 mean=math.fsum(v[m+'_auc_0_1200'] for v in vals)/4) for m in METRICS}))
    return dict(seed_values=rows, means=means, early_auc_seed_values=auc, early_auc_means=auc_means,
                time_change_seed_values=time_seed, time_change_means=time_means,
                effects=dict(by_checkpoint=independent_effects(rows, ('family', 'partition', 'update'), METRICS),
                  early_auc=independent_effects(auc, ('family', 'partition'), tuple(m+'_auc_0_1200' for m in METRICS))))


def independent_time_changes(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ('family', 'partition', 'seed', 'kind', 'gain'))
        require(row['update'] not in groups[key], 'Duplicate time point')
        groups[key][row['update']] = row
    changes, pooled = [], defaultdict(list)
    for key, points in sorted(groups.items()):
        require(set(points) == set(UPDATES), 'Complete time series')
        for begin, end in ((0, 300), (300, 600), (600, 1200), (0, 2400)):
            row = dict(zip(('family', 'partition', 'seed', 'kind', 'gain'), key),
                       start_update=begin, end_update=end,
                       **{m: points[end][m]-points[begin][m] for m in METRICS})
            changes.append(row)
            pooled[tuple(row[k] for k in ('family', 'partition', 'start_update', 'end_update', 'kind', 'gain'))].append(row)
    means = []
    for key, vals in sorted(pooled.items()):
        vals.sort(key=lambda r: r['seed'])
        require([v['seed'] for v in vals] == list(SEEDS), 'Time contrast four seeds')
        means.append(dict(zip(('family', 'partition', 'start_update', 'end_update', 'kind', 'gain'), key),
                          seeds=list(SEEDS), metrics={m: dict(seed_values=[v[m] for v in vals],
                              mean=math.fsum(v[m] for v in vals)/4) for m in METRICS}))
    return changes, means


def independent_effects(rows, group_keys, metrics):
    groups = defaultdict(dict)
    for row in rows:
        group = tuple(row[k] for k in group_keys)
        cell = row['seed'], row['kind'], row['gain']
        require(cell not in groups[group], 'Duplicate effect cell')
        groups[group][cell] = row
    contrasts = {
        'lambda_at_gain1': (-1, 0, 1, 0), 'lambda_at_gain3': (0, -1, 0, 1),
        'gain_at_lambda0': (-1, 1, 0, 0), 'gain_at_lambda1': (0, 0, -1, 1),
        'interaction': (1, -1, -1, 1), 'diagonal_joint3_minus_additive1': (-1, 0, 0, 1)}
    answer = []
    for group, cells in sorted(groups.items()):
        require(set(cells) == set(product(SEEDS, ('additive', 'joint'), (1, 3))), 'Effect four-seed/four-cell grid')
        for metric in metrics:
            values = {name: [] for name in contrasts}
            for seed in SEEDS:
                four = [cells[seed, kind, gain][metric] for kind, gain in
                        (('additive', 1), ('additive', 3), ('joint', 1), ('joint', 3))]
                for name, coefficients in contrasts.items():
                    values[name].append(math.fsum(v*c for v, c in zip(four, coefficients)))
            answer.append(dict(zip(group_keys, group), metric=metric, seeds=list(SEEDS), effects={
                name: dict(seed_values=v, mean=math.fsum(v)/4, min=min(v), max=max(v)) for name, v in values.items()}))
    return answer


def audit_record(source, computed, output_record, source_entry, expected_summaries):
    for key in ('seed', 'condition', 'update', 'scout', 'collector', 'photo_pairs', 'emitted'):
        require(output_record[key] == source[key], 'New/source identity: ' + key)
    require(output_record['source_record_file'] == source_entry['file'] and
            output_record['source_record_sha256'] == source_entry['sha256'], 'New/source file binding')
    require(output_record['old_exact_check'] == dict(map_U_checked=30, photo_N_checked=480,
             receiver_actions_checked=98, original_float32_margins_exact=True), 'Old check receipt')
    logp = np.asarray(output_record['receiver_log_probabilities'], dtype=np.float64)
    prob = np.asarray(output_record['receiver_probabilities'], dtype=np.float64)
    require(logp.shape == prob.shape == (49, 2, 6) and np.isfinite(logp).all() and
            np.isfinite(prob).all() and (prob > 0).all() and (prob <= 1).all(), 'Output probability schema/range')
    np.testing.assert_allclose(logp, computed['logp'], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(prob, computed['probabilities'], rtol=2e-12, atol=2e-12)
    require(np.array_equal(np.exp(logp), prob), 'Saved probability is exponential of saved log probability')
    np.testing.assert_allclose(prob.sum(2), 1, rtol=0, atol=1e-14)
    require(len(output_record['maps']) == 30, 'Output map inventory')
    tied = 0
    prefix = source['condition'].split('_')[0]
    part = partition(int(prefix[5:]) if prefix.startswith('split') else 0)
    codes = np.asarray(source['emitted'])[..., 0]*7+np.asarray(source['emitted'])[..., 1]
    for mid, target in enumerate(MAPS):
        row = output_record['maps'][mid]
        # Exact set of maximum codes in the producer's validated float64 probabilities.
        products = prob[:, 0, target[0]] * prob[:, 1, target[1]]
        require((products > 0).all() and np.isfinite(products).all(), 'Product underflow/nonfinite')
        best = float(products.max())
        best_codes = np.flatnonzero(products == best).tolist()
        eg = products[codes[mid]]
        require(row['C'] == best and row['max_code_ids'] == best_codes and
                row['first_max_code_id'] == best_codes[0] and row['max_messages'] == [[c//7, c%7] for c in best_codes] and
                row['first_max_message'] == [best_codes[0]//7, best_codes[0]%7], 'Exact same-code product argmax/ties')
        require(row['map_id'] == mid and row['locations'] == list(target) and
                row['partition'] == ('train' if mid in part['train'] else 'heldout') and
                np.array_equal(eg, row['E_G']) and np.array_equal(best-eg, row['C_minus_E_G']) and
                (eg <= best).all(), 'Exact per-photo product/gap')
        require(row['U'] is bool(computed['U'][mid]) and row['N'] == computed['N'][mid].tolist() and
                row['N_count'] == int(computed['N'][mid].sum()), 'Exact per-map/photo U/N')
        np.testing.assert_allclose(best, computed['C'][mid], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(eg, computed['E_G'][mid], rtol=2e-12, atol=2e-12)
        tied += len(best_codes) > 1
    compare_tree({k: expected_summaries[k] for k in ('train', 'heldout')}, output_record['summaries'], 'Record summaries')
    return tied


def audit(output):
    """Run only after the parent has completed the pure probability producer."""
    output = Path(output).resolve()
    plan, result = read(output/'plan.json'), read(output/'execution/results.json')
    freeze = read(output/'freeze.json')
    require(result['status'] == 'complete' and result['plan_sha256'] == freeze['plan_sha256'] == sha(output/'plan.json'),
            'Complete new result/frozen plan required')
    require(plan['seeds'] == list(SEEDS) and plan['conditions'] == list(CONDITIONS) and plan['updates'] == list(UPDATES) and
            plan['early_updates'] == list(EARLY) and plan['time_pairs'] == [[0, 300], [300, 600], [600, 1200], [0, 2400]] and
            plan['metrics'] == list(METRICS) and plan['temperature'] == 1 and plan['dtype'] == 'float64' and
            plan['expected'] == dict(runs=64, records=1024, map_records=30720, photo_messages=491520), 'Fixed probability design')
    require(result['counts'] == dict(records=1024, maps=30720, photo_messages=491520, receiver_distributions=100352) and
            all(result[k] == 0 for k in ('model_imports', 'weight_loads', 'neural_forward_calls')),
            'Completed result scope/budget')
    require(not (output/'execution/failure.json').exists(), 'Producer failure evidence')
    hashes = {}
    def bind(path, digest=None):
        path = Path(path).resolve()
        actual = sha(path)
        if digest is not None: require(actual == digest, 'Input SHA changed: ' + str(path))
        hashes[str(path)] = actual
        return actual
    for name in ('plan.json', 'freeze.json', 'execution/results.json', 'execution/started.json'): bind(output/name)
    for path, digest in plan['source_files_sha256'].items(): bind(path, digest)
    for path, digest in plan['analysis_sources_sha256'].items():
        bind(path, digest); bind(output/'code_snapshot'/Path(path).name, digest)
    batch = Path(plan['batch']).resolve()
    require(batch == WORK/'research_program/policy_gain_study/results/gain_001', 'Fixed original batch')
    manifest = read(batch/'manifest.json')
    for path, digest in manifest['source_hashes'].items():
        bind(path, digest)
        if Path(path).suffix in ('.py', '.md'):
            bind(batch/'source_snapshot'/Path(path).relative_to(WORK), digest)
            bind(batch/'probe/code_snapshot'/Path(path).relative_to(WORK), digest)
    require(read(batch/'audit_execution_001/verification.json')['status'] == 'passed' and
            read(batch/'audit_results_002.json')['status'] == 'passed', 'Prior independent audits')
    probe_plan = read(batch/'probe/plan.json')
    probe_results = read(batch/'probe/execution/results.json')
    require(probe_results['status'] == 'complete' and probe_results['plan_sha256'] == sha(batch/'probe/plan.json') ==
            read(batch/'probe/freeze.json')['plan_sha256'], 'Prior probe frozen plan')
    for path, digest in probe_plan['source_files_sha256'].items(): bind(path, digest)
    photos = probe_plan['validation_photo_pairs']
    photo_entries = read(WORK/'redesign_v0.4/data/manifest.json')['images']
    pools = [[i for i, p in enumerate(photo_entries) if p['split'] == 'test' and p['category'] == kind]
             for kind in ('food', 'water')]
    require(len(pools[0]) == len(pools[1]) == 8 and photos == [list(x) for x in product(pools[0][4:], pools[1][4:])],
            'Fixed 16 photo pairs')
    identity = lambda r: tuple(r[k] for k in ('seed', 'condition', 'update', 'scout'))
    expected = set(product(SEEDS, CONDITIONS, UPDATES, range(2)))
    sources = {identity(r): r for r in plan['source_records']}
    old_index = {identity(r): r for r in probe_results['records']}
    new_index = {identity(r): r for r in result['records']}
    require(len(plan['source_records']) == len(probe_results['records']) == len(result['records']) == 1024 and
            set(sources) == set(old_index) == set(new_index) == expected, 'Complete unique three-way 1024 grid')
    runs = {(r['seed'], r['condition']): r for r in probe_plan['runs']}
    summaries, tied_maps = [], 0
    for n, key in enumerate(sorted(expected), 1):
        src, old_entry, new_entry = sources[key], old_index[key], new_index[key]
        source_path = (batch/'probe/execution'/old_entry['file']).resolve()
        require(source_path == Path(src['file']).resolve() and src['sha256'] == old_entry['sha256'], 'Old/new source index binding')
        bind(source_path, src['sha256'])
        source = read(source_path)
        checkpoint = runs[key[0], key[1]]['checkpoint_files'][str(key[2])]
        require(source['checkpoint_sha256'] == probe_plan['source_files_sha256'][checkpoint], 'Saved source checkpoint provenance')
        calc, expected_summaries = source_record_check(source, old_entry, photos)
        result_path = (output/'execution'/new_entry['file']).resolve()
        require(result_path.is_relative_to(output/'execution/records'), 'New record path containment')
        bind(result_path, new_entry['sha256'])
        saved = read(result_path)
        tied_maps += audit_record(source, calc, saved, src, expected_summaries)
        require(new_entry['summaries'] == saved['summaries'], 'New index/record summaries')
        summaries.append(dict(seed=key[0], condition=key[1], update=key[2], scout=key[3], summaries=expected_summaries))
        if n % 128 == 0: print(f'AUDIT {n}/1024 saved probability records', flush=True)
    require({str(p.resolve()) for p in (output/'execution/records').glob('*.json')} ==
            {str((output/'execution'/r['file']).resolve()) for r in result['records']}, 'No missing/extra new record files')
    aggregate = independent_aggregate(summaries)
    for key, expected_value in aggregate.items(): compare_tree(expected_value, result[key], key)
    old = read(plan['source_analysis'])['same_photo_trajectory']
    rowkey = lambda r: tuple(r[k] for k in ('family', 'partition', 'update', 'seed', 'kind', 'gain'))
    old_rows = {rowkey(r): r for r in old['seed_values']}
    require({rowkey(r) for r in result['seed_values']} == set(old_rows), 'Old seed curve grid')
    for row in result['seed_values']:
        prev = old_rows[rowkey(row)]
        require(row['U'] == prev['U_full49'] and row['N'] == prev['N_both'], 'Exact old seed U/N values')
        require(row['U_numerator']/row['U_denominator'] == sum(
            s['summaries'][row['partition']]['U_numerator'] for s in summaries if s['seed'] == row['seed'] and
            s['update'] == row['update'] and s['condition'].endswith(f'_{row["kind"]}_gain{row["gain"]}') and
            s['condition'].startswith('split' if row['family'] == 'split' else 'full'))/row['U_denominator'],
            'Exact seed integer U denominator')
    short = lambda r: tuple(r[k] for k in ('family', 'partition', 'seed', 'kind', 'gain'))
    old_auc = {short(r): r for r in old['early_auc_seed_values']}
    for row in result['early_auc_seed_values']:
        require(row['U_auc_0_1200'] == old_auc[short(row)]['U_auc_0_1200'] and
                row['N_auc_0_1200'] == old_auc[short(row)]['N_auc_0_1200'], 'Exact old U/N AUC anchor')
    require(result['old_summary_anchor'] == dict(seed_checkpoint_U_N_pairs=384, seed_AUC_U_N_pairs=48, exact=True),
            'Old aggregate check receipt')
    require(all(sha(path) == digest for path, digest in hashes.items()), 'Input/result mutated during audit')
    require(not ({'torch', 'camp', 'transformers', 'mlx'} & set(sys.modules)), 'Unexpected neural runtime import')
    return dict(status='passed', completed_at=datetime.now(timezone.utc).isoformat(), output=str(output),
        audit_sha256=sha(__file__), continuous_tolerance=dict(relative=2e-12, absolute=2e-12, normalization_absolute=1e-14),
        exact_maximum_scope='Strict products of saved float64 probabilities, after independent log-sum-exp numeric validation; no near-tie tolerance.',
        counts=dict(records=1024, maps_C=30720, photo_E_G=491520, receiver_distributions=100352,
          probability_scalars=602112, old_integer_map_U=30720, old_integer_photo_N=491520,
          seed_curve_rows=len(aggregate['seed_values']), four_seed_means=len(aggregate['means']),
          seed_AUC_rows=len(aggregate['early_auc_seed_values']), seed_time_changes=len(aggregate['time_change_seed_values']),
          time_change_means=len(aggregate['time_change_means']), checkpoint_effect_records=len(aggregate['effects']['by_checkpoint']),
          AUC_effect_records=len(aggregate['effects']['early_auc']), contrasts_per_record=6,
          maps_with_exact_probability_maximum_ties=tied_maps, source_and_record_hashes=len(hashes)),
        input_sha256=hashes, model_imports=0, neural_forward_calls=0,
        interpretation='Temperature-one analytic conditional reception probabilities at the old greedy sender code. Not stochastic-mode replay, calibration, a fresh experiment, or a language-formation claim.')


def run_audit(output, destination):
    require(read(Path(output)/'execution/results.json')['status'] == 'complete', 'Wait for completed main calculation')
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        receipt = audit(output); receipt['elapsed_seconds'] = time.monotonic()-started
        with (destination/'verification.json').open('x') as f: json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False)
        (destination/'独立核验.md').write_text('# 连续成功概率补充分析独立核验\n\n'
          '全量核验通过。仅读取保存的 JSON；没有权重加载、训练或神经网络前向。\n\n'
          '- 1024份记录、30720个地图C和491520个照片消息E_G均独立重算。\n'
          '- 原动作、最大值并列、逐地图U／照片N及整数分母保持精确；连续值容差为绝对与相对各2×10⁻¹²。\n'
          '- 同一码下两目标概率先相乘，再选最优码；最优码集合按已核验的实际float64概率严格检查，未放宽并列。\n'
          '- 四种子曲线、0–1200固定AUC、四个固定时段变化及六个四格对比全部通过。\n\n'
          'C是研究者选择最佳完整码的乐观参照；E_G以旧贪心发送码为条件。二者是温度1策略概率的解析计算，'
          '不等同原随机发送与接收的实际重放，不构成新训练或语言形成证据。\n')
        print(json.dumps(dict(status='passed', directory=str(destination), counts=receipt['counts']), ensure_ascii=False))
    except BaseException as error:
        with (destination/'failure.json').open('x') as f:
            json.dump(dict(status='failed', error=repr(error), traceback=traceback.format_exc(), automatic_retry=False), f,
                      ensure_ascii=False, indent=2)
        raise


class SyntheticTests(unittest.TestCase):
    @staticmethod
    def probabilities():
        p = np.empty((49, 2, 6), float)
        for code in range(49):
            for goal, target in ((0, 0), (1, 1)):
                wanted = (.9 if goal == 0 else .01) if code == 0 else ((.01 if goal == 0 else .9) if code == 1 else .001)
                p[code, goal, target] = wanted
                other = [x for x in range(6) if x != target]
                p[code, goal, other] = (1-wanted)*np.arange(5, 0, -1)/15
        return p

    def test_one_complete_code_not_two_marginal_codes(self):
        result = independent_probability(np.log(self.probabilities()), np.full((30, 16, 2), [0, 2]))
        self.assertAlmostEqual(result['C'][0], .009, places=14)
        self.assertAlmostEqual(result['E_G'][0, 0], .000001, places=14)
        self.assertLess(result['C'][0], .9*.9)
        self.assertFalse(result['U'][0]); self.assertFalse(result['N'][0].any())
        self.assertTrue((result['E_G'] <= result['C'][:, None]+1e-15).all())

    def test_uniform_probability(self):
        result = independent_probability(np.zeros((49, 2, 6)), np.zeros((30, 16, 2), int))
        np.testing.assert_allclose(result['C'], 1/36, atol=1e-15)
        np.testing.assert_allclose(result['E_G'], 1/36, atol=1e-15)
        self.assertTrue(all(ids == list(range(49)) for ids in result['max_code_ids']))

    def test_logit_row_shift_invariance(self):
        z = np.log(self.probabilities()); messages = np.zeros((30, 16, 2), int)
        a = independent_probability(z, messages)
        b = independent_probability(z+np.arange(98).reshape(49, 2, 1)*100, messages)
        np.testing.assert_allclose(a['C'], b['C'], rtol=2e-12, atol=1e-14)
        np.testing.assert_allclose(a['E_G'], b['E_G'], rtol=2e-12, atol=1e-14)

    def test_log_domain_keeps_optimal_code_when_products_underflow(self):
        z = np.zeros((49, 2, 6)); z[:, 0, 0] = -2000; z[:, 1, 1] = -2000
        z[3, 0, 0] = -1990
        r = independent_probability(z, np.zeros((30, 16, 2), int))
        self.assertEqual(r['C'][0], 0.)
        self.assertEqual(r['max_code_ids'][0], [3])
        self.assertTrue(np.isfinite(r['log_C']).all())

    def test_six_contrasts_and_missing_cell_rejection(self):
        rows = [dict(family='split', partition='heldout', seed=s, kind=k, gain=g,
                     C=v+(s-SEEDS[0])*.1) for s in SEEDS for k, g, v in
                (('additive', 1, 1.), ('additive', 3, 2.), ('joint', 1, 4.), ('joint', 3, 8.))]
        result = independent_effects(rows, ('family', 'partition'), ('C',))[0]['effects']
        expected = dict(lambda_at_gain1=3, lambda_at_gain3=6, gain_at_lambda0=1,
                        gain_at_lambda1=4, interaction=3, diagonal_joint3_minus_additive1=7)
        for k, val in expected.items(): self.assertAlmostEqual(result[k]['mean'], val)
        with self.assertRaises(AssertionError): independent_effects(rows[:-1], ('family', 'partition'), ('C',))

    def test_shape_nan_and_fractional_code_rejection(self):
        for z, m in ((np.zeros((48, 2, 6)), np.zeros((30, 16, 2), int)),
                     (np.full((49, 2, 6), np.nan), np.zeros((30, 16, 2), int)),
                     (np.zeros((49, 2, 6)), np.full((30, 16, 2), .5))):
            with self.assertRaises(AssertionError): independent_probability(z, m)

    def test_complete_seed_aggregation_and_auc(self):
        records = []
        for seed, condition, update, scout in product(SEEDS, CONDITIONS, UPDATES, (0, 1)):
            prefix, kind, gain = condition.split('_')
            split = int(prefix[5:]) if prefix.startswith('split') else 0
            base = .01*(seed-SEEDS[0])+.02*(kind == 'joint')+.03*(gain == 'gain3')+.001*split+.0001*scout
            value = base+.2*update/2400
            summary = {m: value*(i+1)/5 for i, m in enumerate(METRICS)}
            records.append(dict(seed=seed, condition=condition, update=update, scout=scout,
                summaries=dict(train=summary, heldout=summary if split else None)))
        result = independent_aggregate(records)
        self.assertEqual(len(result['seed_values']), 384)
        self.assertEqual(len(result['means']), 96)
        self.assertEqual(len(result['early_auc_seed_values']), 48)
        auc = next(r for r in result['early_auc_seed_values'] if r['seed'] == SEEDS[0] and
                   r['family'] == 'split' and r['partition'] == 'heldout' and r['kind'] == 'additive' and r['gain'] == 1)
        self.assertAlmostEqual(auc['C_auc_0_1200'], (.002+.00005+.05)/5, places=14)
        with self.assertRaises(AssertionError): independent_aggregate(records[:-1])

    def test_saved_record_exact_code_set_and_probability_corruption(self):
        z = np.log(self.probabilities()).astype(np.float32).astype(np.float64)
        emitted = np.full((30, 16, 2), [0, 2], dtype=int)
        calc = independent_probability(z, emitted)
        src = dict(seed=SEEDS[0], condition='split1_additive_gain1', update=0, scout=0, collector=1,
                   photo_pairs=[[a, b] for a, b in product(range(4), range(4, 8))], emitted=emitted.tolist())
        entry = dict(file='/synthetic/record.json', sha256='synthetic')
        parts = partition(1)
        summaries = {name: independent_summary(calc, ids) for name, ids in parts.items()}
        saved = dict(src, source_record_file=entry['file'], source_record_sha256=entry['sha256'],
                     old_exact_check=dict(map_U_checked=30, photo_N_checked=480, receiver_actions_checked=98,
                                          original_float32_margins_exact=True),
                     receiver_log_probabilities=calc['logp'].tolist(), receiver_probabilities=calc['probabilities'].tolist(),
                     summaries=summaries, maps=[])
        for mid, (food, water) in enumerate(MAPS):
            values = calc['probabilities'][:, 0, food]*calc['probabilities'][:, 1, water]
            maximum = float(values.max()); codes = np.flatnonzero(values == maximum).tolist()
            natural = values[np.full(16, 2)]
            saved['maps'].append(dict(map_id=mid, locations=[food, water], partition='train' if mid in parts['train'] else 'heldout',
                C=maximum, max_code_ids=codes, first_max_code_id=codes[0], max_messages=[[c//7, c%7] for c in codes],
                first_max_message=[codes[0]//7, codes[0]%7], E_G=natural.tolist(), C_minus_E_G=(maximum-natural).tolist(),
                U=bool(calc['U'][mid]), N=calc['N'][mid].tolist(), N_count=int(calc['N'][mid].sum())))
        audit_record(src, calc, saved, entry, summaries)
        bad = json.loads(json.dumps(saved)); bad['maps'][0]['max_code_ids'] = []
        with self.assertRaises(AssertionError): audit_record(src, calc, bad, entry, summaries)
        bad = json.loads(json.dumps(saved)); bad['receiver_probabilities'][0][0][0] += .01
        with self.assertRaises(AssertionError): audit_record(src, calc, bad, entry, summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--source', type=Path, default=ROOT/'results/probability_001')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.self_test: unittest.main(argv=[sys.argv[0]])
    else: run_audit(args.source, args.out or args.source/'audit_probability_001')
