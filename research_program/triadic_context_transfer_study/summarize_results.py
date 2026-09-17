"""Read-only streaming summary of the complete, already finished transfer batch.

No network parameters are loaded; no neural forward or new training occurs.
Raw row measurements remain exactly reconstructible from saved NPZ + dataset +
the bound metrics source. Outcomes are never used to select rows or seeds.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
import platform
import time
import numpy as np
from research_program.triadic_context_transfer_study import metrics

HERE = Path(__file__).resolve().parent
CHUNK = 16384
SEEDS, CONDITIONS, PARTS = metrics.SEEDS, metrics.CONDITIONS, metrics.PARTITIONS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x', encoding='utf8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def load_npz(path):
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def record_key(row):
    return (int(row['seed']), row['condition'], row['partition'], row['mode'],
            int(row['shift_index']), int(row['direction']))


def validate_grid(result, donor_counts):
    require(result['status'] == 'completed', 'Main execution is not completed')
    actual = result['records']
    lookup = {record_key(row): row for row in actual}
    expected = set()
    for seed in SEEDS:
        for condition in CONDITIONS:
            for part in PARTS:
                for direction in (0, 1):
                    for mode in metrics.CONTROL_MODES:
                        expected.add((seed, condition, part, mode, -1, direction))
                    for shift in range(donor_counts[part]):
                        for mode in metrics.REMOTE_MODES:
                            expected.add((seed, condition, part, mode, shift, direction))
    require(len(actual) == len(lookup) and set(lookup) == expected, 'Missing, duplicate, or unexpected intervention record')
    for key, row in lookup.items():
        silent = key[1].endswith('_silent')
        require(row['is_silent_alias'] is silent, 'Silent/live alias mismatch')
        require((row['path'] is None and row['data_sha256'] is None) if silent
                else (bool(row['path']) and bool(row['data_sha256'])), 'Unexpected data/reference representation')
        require(row['new_forward_worlds'] == (0 if silent else row['worlds']), 'Forward-world bookkeeping mismatch')
    refs = result['natural_references']
    natural = {(int(r['seed']), r['condition'], r['partition']): r for r in refs}
    require(len(refs) == len(natural) and set(natural) == {(s, c, p) for s in SEEDS for c in CONDITIONS for p in PARTS}, 'Incomplete natural references')
    return lookup, natural


class StratifiedAccumulator:
    """Use the official accumulator at fixed, prespecified marginal strata."""
    def __init__(self):
        self.overall = metrics.WeightedAccumulator()
        self.directions = {d: metrics.WeightedAccumulator() for d in (0, 1)}
        self.pairs = {(s, l): metrics.WeightedAccumulator() for s in range(3) for l in range(3) if s != l}

    def update(self, values, spec, selection, weights, direction):
        axis = spec['axis'][selection]
        self.overall.update(values, axis, weights)
        # Each displayed stratum has its own fixed conditional denominator.
        # The principal statistic always uses unmodified weights above.
        self.directions[direction].update(values, axis, weights * 2)
        for (sender, listener), accumulator in self.pairs.items():
            selected = (spec['sender'][selection] == sender) & (spec['listener'][selection] == listener)
            if selected.any():
                accumulator.update({key: np.asarray(value)[selected] for key, value in values.items()},
                    axis[selected], weights[selected] * 6)

    def finish(self):
        report = self.overall.finish()
        report['by_direction'] = {str(d): acc.finish() for d, acc in self.directions.items()}
        report['by_sender_listener'] = {f'{s}_{l}': acc.finish() for (s, l), acc in self.pairs.items()}
        report['stratum_definition'] = 'Direction conditions on one original endpoint (weight ×2); ordered S/L conditions on one of six fixed pairs (weight ×6). No outcome-conditioned stratum.'
        return report


def verify_sources(run):
    run = Path(run).resolve()
    plan, freeze, result = read(run / 'plan.json'), read(run / 'freeze.json'), read(run / 'execution/results.json')
    require(sha(run / 'plan.json') == freeze['plan_sha256'] == result['plan_sha256'], 'Frozen plan mismatch')
    sources = {str(run / filename): sha(run / filename) for filename in ('plan.json', 'freeze.json', 'execution/results.json')}
    for collection in ('sources_sha256', 'inputs_sha256'):
        for path, digest in plan[collection].items():
            require(sha(path) == digest, 'Frozen source/input changed: ' + path)
            sources[path] = digest
    # This summary is separately bound; it is not claimed to be a main frozen source.
    for path in (Path(__file__).resolve(), Path(metrics.__file__).resolve()):
        sources[str(path)] = sha(path)
    return plan, result, sources


def summarize(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    require(not out.exists(), 'Refuse to overwrite any previous summary')
    # Do not create a finished-looking output until the main completion gate holds.
    plan, result, sources = verify_sources(run)
    data_directory = Path(plan['dataset_directory'])
    manifest = read(data_directory / 'manifest.json')
    donor_counts = {p: manifest['partitions'][p]['n_donors'] for p in PARTS}
    lookup, natural = validate_grid(result, donor_counts)
    require(len(lookup) == 3072, 'Official batch must contain 3072 logical records')
    out.mkdir(parents=True)
    started = time.perf_counter()
    write(out / 'started.json', dict(at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        source_sha256=sources, runtime=dict(python=platform.python_version(), numpy=np.__version__),
        chunk_size=CHUNK, status='started_read_only_summary'))
    seen = set()
    aliases, actual, measured_rows = 0, 0, 0
    reconstruction = []
    policies = []

    def load_cell(record, n):
        nonlocal aliases, actual
        key = record_key(record)
        require(key not in seen and record['worlds'] == n, 'Duplicate cell or incompatible row count')
        seen.add(key)
        if record['is_silent_alias']:
            aliases += 1
            data = None
        else:
            path = Path(record['path'])
            require(sha(path) == record['data_sha256'], 'Intervention NPZ changed: ' + str(path))
            sources[str(path)] = record['data_sha256']
            data = load_npz(path)
            actual += 1
        reconstruction.append({key: record[key] for key in ('seed', 'condition', 'partition', 'mode',
            'shift_index', 'direction', 'worlds', 'is_silent_alias', 'path', 'data_sha256', 'natural_source')})
        return data

    try:
        for seed in SEEDS:
            for condition in CONDITIONS:
                information = condition.split('_')[0]
                live = condition.endswith('_live')
                policy = dict(seed=seed, condition=condition, partitions={})
                for part in PARTS:
                    part_record = manifest['partitions'][part]
                    for prefix in ('array', 'metadata'):
                        path = Path(part_record[prefix + '_path'])
                        require(sha(path) == part_record[prefix + '_sha256'], 'Prepared data changed')
                        sources[str(path)] = part_record[prefix + '_sha256']
                    spec = load_npz(part_record['array_path'])
                    ref = natural[seed, condition, part]
                    require(sha(ref['path']) == ref['data_sha256'], 'Natural reference changed')
                    sources[ref['path']] = ref['data_sha256']
                    pool = load_npz(ref['path'])
                    context = metrics.build_pool_context(pool, information)
                    n = len(spec['endpoint_indices'])
                    controls = {mode: StratifiedAccumulator() for mode in metrics.CONTROL_MODES}
                    remote = {layer: {mode: StratifiedAccumulator() for mode in (*metrics.REMOTE_MODES, 'contrast')}
                              for layer in ('eligible', 'all_other')}
                    for mode in metrics.CONTROL_MODES:
                        for direction in (0, 1):
                            row = lookup[seed, condition, part, mode, -1, direction]
                            data = load_cell(row, n)
                            for start in range(0, n, CHUNK):
                                selection = slice(start, min(start + CHUNK, n))
                                values = metrics.cell_values(spec, context, data, mode=mode, shift=-1,
                                    direction=direction, live=live, rows=selection)
                                weights = metrics.support_weights(spec, -1, 'control', rows=selection)
                                controls[mode].update(values, spec, selection, weights, direction)
                                measured_rows += len(weights)
                            del data
                    for shift in range(donor_counts[part]):
                        for direction in (0, 1):
                            same_data = load_cell(lookup[seed, condition, part, 'remote_same_both', shift, direction], n)
                            opposite_data = load_cell(lookup[seed, condition, part, 'remote_opposite_both', shift, direction], n)
                            for start in range(0, n, CHUNK):
                                selection = slice(start, min(start + CHUNK, n))
                                same = metrics.cell_values(spec, context, same_data, mode='remote_same_both',
                                    shift=shift, direction=direction, live=live, rows=selection)
                                opposite = metrics.cell_values(spec, context, opposite_data, mode='remote_opposite_both',
                                    shift=shift, direction=direction, live=live, rows=selection)
                                contrast = metrics.contrast_values(same, opposite)
                                for layer in ('eligible', 'all_other'):
                                    weights = metrics.support_weights(spec, shift, layer, rows=selection)
                                    for name, values in (('remote_same_both', same), ('remote_opposite_both', opposite), ('contrast', contrast)):
                                        remote[layer][name].update(values, spec, selection, weights, direction)
                                measured_rows += 2 * len(weights)
                            del same_data, opposite_data
                    cell = {layer: {mode: accumulator.finish() for mode, accumulator in reports.items()}
                            for layer, reports in remote.items()}
                    cell['controls'] = {mode: accumulator.finish() for mode, accumulator in controls.items()}
                    cell['natural_reference'] = ref
                    # The both-window endpoint exchange is a redundant marginal
                    # identity, not a second semantic success criterion.
                    for layer in ('eligible', 'all_other'):
                        same, opposite = cell[layer]['remote_same_both']['macro'], cell[layer]['remote_opposite_both']['macro']
                        require(abs(opposite['target_apt'] - same['current_apt']) <= 2e-11,
                                'Both-window endpoint-exchange marginal identity failed')
                        require(abs(opposite['current_apt'] - same['target_apt']) <= 2e-11,
                                'Both-window endpoint-exchange reverse marginal identity failed')
                        if not live:
                            require(abs(cell[layer]['contrast']['macro']['target_apt_gain']) <= 2e-12,
                                    'Silent transfer must be structural zero')
                    policy['partitions'][part] = cell
                    write(out / f'{seed}_{condition}_{part}.json', cell)
                    print(json.dumps(dict(stage='partition_summarized', seed=seed, condition=condition, partition=part,
                        actual_files=actual, aliases=aliases, elapsed_seconds=time.perf_counter() - started)), flush=True)
                    del spec, pool, context
                policies.append(policy)
        require(seen == set(lookup) and actual == 1536 and aliases == 1536, 'Incomplete actual/alias processing')
        require(measured_rows == 368271360, 'Logical row count differs from all live plus silent references')
        primary = metrics.primary_comparison(policies)
        # Verify bytes again after reading; no records are selectively omitted.
        for path, digest in sources.items():
            require(sha(path) == digest, 'Source changed during summary: ' + path)
        summary = dict(status='completed_read_only_summary', primary_comparison=primary, policies=policies,
            counts=dict(policies=len(policies), actual_npz=actual, silent_aliases=aliases,
                logical_intervention_rows=measured_rows, natural_files=64),
            scope='Complete four partitions and all 16 frozen policies, every donor and direction; eligible and all-other weights kept separate. One-listener directional aptitude is not the preceding study two-endpoint joint success.',
            identity_scope='Sham, silent zero, and local/both input or endpoint-exchange identities are implementation facts, not additional evidence of learned semantic transfer.',
            row_reconstruction='Every row value is reproducible with metrics.build_pool_context + metrics.cell_values and the bound dataset and NPZ references. No per-row outcome vector has been filtered; large derived vectors are not redundantly copied.',
            source_sha256=sources, elapsed_seconds=time.perf_counter() - started,
            neural_forward_calls=0, parameter_loads=0, training_calls=0)
        write(out / 'row_reconstruction.json', dict(records=reconstruction, metrics_sha256=sha(metrics.__file__),
            dataset_directory=str(data_directory), calls='cell_values(spec,build_pool_context(natural_pool,PL_or_LL),actual_data_or_None,mode=mode,shift=shift_index,direction=direction,live=not is_silent_alias,rows=slice)',
            truth_scope='Recipient current and counterfactual truth; donor same/opposite actual actions are separate covariates. No gate-conditioned denominators.'))
        write(out / 'summary.json', summary)
        write(out / 'completed.json', dict(status=summary['status'], elapsed_seconds=summary['elapsed_seconds'],
            summary_sha256=sha(out / 'summary.json'), counts=summary['counts']))
        return summary
    except BaseException as exc:
        write(out / 'failure.json', dict(status='failed', error=repr(exc), elapsed_seconds=time.perf_counter() - started,
            actual_npz_processed=actual, silent_aliases_processed=aliases, unique_records_seen=len(seen), source_sha256=sources))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    arguments = parser.parse_args()
    result = summarize(arguments.run, arguments.out)
    print(json.dumps(dict(status=result['status'], primary=result['primary_comparison']['primary'], counts=result['counts']), ensure_ascii=False))
