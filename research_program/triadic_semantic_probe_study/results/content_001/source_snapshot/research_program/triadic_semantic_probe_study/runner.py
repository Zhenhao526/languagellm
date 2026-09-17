"""Frozen diagnostic execution only: saved policies, no optimization or retries."""
import os
for _thread_key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_key] = '1'
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
import platform
import shutil
import time
import numpy as np
from research_program.triadic_semantic_probe_study import channel_intervention as ch, dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = HERE.parent / 'triadic_partner_ecology_study/results/partner_001'
DATA = HERE / 'dataset_001'
SEEDS = (49101, 49102, 49103, 49104)
CONDS = ('FI_silent', 'FI_live', 'PI_silent', 'PI_live')
PARTS = ('train', 'heldout_layouts')
STEPS = (0, 100, 500, 1500, 3000, 6000)
MODES = ('sham',) + tuple(f'{origin}_{window}' for origin in
    ('local_opposite', 'remote_same', 'remote_opposite') for window in ('w1', 'w2', 'both'))
BATCH = 1024
FIELDS = ('states', 'state_indices', 'messages', 'action_indices', 'action_probabilities',
          'greedy_reward', 'executed', 'satisfied')
CONFIG = dict(seeds=SEEDS, conditions=CONDS, partitions=PARTS, checkpoints=STEPS, modes=MODES,
              batch_size=BATCH, new_natural_worlds=3732480, new_intervention_worlds=2764800,
              new_natural_npz=160, new_intervention_npz=320, reused_endpoint_npz=32,
              reused_endpoint_worlds=746496, new_network_samples=50181120,
              training_updates=0, execution_order='seed, condition, checkpoint/partition, mode/direction')


def sha(path): return sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        f.write('\n')
def npz(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def collect_inputs():
    dm = read(DATA / 'manifest.json')
    assert dm['summary']['unique_cases_by_class'] == dict(content=120, descriptive=240, role=408)
    paths = [DATA / 'manifest.json', SOURCE / 'plan.json', SOURCE / 'freeze.json',
             SOURCE / 'execution/results.json', SOURCE / 'execution/status.json',
             SOURCE / 'audit_execution_001/verification.json']
    for name, info in dm['outputs'].items():
        assert sha(DATA / name) == info['sha256']
        paths.append(DATA / name)
    assert sha(HERE / 'dataset.py') == dm['dataset_source_sha256']
    for path, expected in dm['source_sha256'].items(): assert sha(path) == expected
    main = read(SOURCE / 'execution/results.json')
    assert sha(SOURCE / 'execution/results.json') == 'c86b9a432bf57700483d2dfd44a95f150cfbdfe0813feed6091967fc08047533'
    assert main['status'] == 'completed' and main['completed_run_count'] == 32
    assert read(SOURCE / 'audit_execution_001/verification.json')['status'] == 'passed'
    policies = []
    for seed in SEEDS:
        for condition in CONDS:
            directory = SOURCE / 'execution' / f'seed_{seed}_unique_{condition}'
            original = read(directory / 'result.json')
            assert original == next(r for r in main['runs'] if
                (r['seed'], r['ecology'], r['condition']) == (seed, 'unique', condition))
            paths.append(directory / 'result.json')
            checkpoints = {str(u): str(directory / f'checkpoint_{u:04d}.npz') for u in STEPS}
            expected_checkpoints = {str(row['update']): row['checkpoint_sha256'] for row in original['monitor']}
            for update, path in checkpoints.items(): assert sha(path) == expected_checkpoints[update]
            paths.extend(map(Path, checkpoints.values()))
            endpoints = {}
            for part in PARTS:
                score = original['final'][part]['natural']
                path = SOURCE / 'execution' / score['data_file']
                assert sha(path) == score['data_sha256']
                paths.append(path); endpoints[part] = str(path)
            policies.append(dict(seed=seed, condition=condition, checkpoints=checkpoints, endpoints=endpoints))
    return policies, {str(p): sha(p) for p in paths}


def source_files():
    for relative, expected in read(SOURCE / 'plan.json')['sources'].items(): assert sha(ROOT / relative) == expected
    files = [HERE / name for name in ('runner.py', 'channel_intervention.py', 'dataset.py', 'metrics.py',
              'plan.md', '设计独立审查.md', 'tests/test_channel_intervention.py', 'tests/test_dataset.py',
              'tests/test_metrics.py', 'tests/test_runner.py', 'channel_test_receipt_002.json',
              'dataset_test_preflight_001.json', 'metrics_synthetic_preflight.json', 'runner_test_receipt_001.json')]
    files += [Path(ch.core.__file__), Path(ch.core.base.__file__), Path(ch.core.base.env.__file__),
              HERE.parent / 'triadic_partner_ecology_study/metrics.py']
    return {str(p): sha(p) for p in files}


def prepare(out):
    out = Path(out).resolve()
    assert not out.exists()
    policies, inputs = collect_inputs()
    sources = source_files()
    out.mkdir(parents=True, exist_ok=False)
    for path in sources:
        relative = Path(path).relative_to(ROOT)
        target = out / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    plan = dict(status='prepared_without_forward', prepared_at=now(), config=CONFIG,
                runtime=dict(python=platform.python_version(), numpy=np.__version__),
                sources_sha256=sources, inputs_sha256=inputs, policies=policies,
                dataset_directory=str(DATA), source_directory=str(SOURCE))
    write(out / 'plan.json', plan); write(out / 'freeze.json', dict(plan_sha256=sha(out / 'plan.json')))
    return dict(status=plan['status'], policies=len(policies), plan_sha256=sha(out / 'plan.json'))


def verify(out):
    out = Path(out).resolve(); plan = read(out / 'plan.json')
    assert sha(out / 'plan.json') == read(out / 'freeze.json')['plan_sha256']
    assert plan['config'] == json.loads(json.dumps(CONFIG))
    assert plan['runtime'] == dict(python=platform.python_version(), numpy=np.__version__)
    assert source_files() == plan['sources_sha256']
    for p, h in plan['sources_sha256'].items(): assert sha(out / 'source_snapshot' / Path(p).relative_to(ROOT)) == h
    for p, h in plan['inputs_sha256'].items(): assert sha(p) == h
    return plan


def natural_evaluation(networks, x, states, live):
    n = len(x)
    data = dict(states=states, state_indices=np.arange(n, dtype=np.int64),
                messages=np.empty((n, 2, 3, 4), dtype=np.int8),
                action_indices=np.empty((n, 3), dtype=np.int16),
                action_probabilities=np.empty((n, 3, 17), dtype=np.float64))
    for start in range(0, n, BATCH):
        sl = slice(start, min(start+BATCH, n))
        trace = ch.core.rollout(networks, x[sl], live)
        probabilities, _ = ch.core.base.policy_distribution(trace['action_logits'])
        data['messages'][sl] = trace['messages']; data['action_probabilities'][sl] = probabilities
        data['action_indices'][sl] = np.argmax(probabilities, axis=-1)
    data.update(ch.settle(states, data['action_indices']))
    return data


def donor_selection(spec, rows, part, direction, mode):
    if mode == 'sham':
        return np.full(len(rows), PARTS.index(part), dtype=np.int8), spec['endpoint_indices'][rows, direction]
    endpoint = direction if mode.startswith('remote_same') else 1-direction
    if mode.startswith('local_'):
        return np.full(len(rows), PARTS.index(part), dtype=np.int8), spec['endpoint_indices'][rows, endpoint]
    assert mode.startswith('remote_')
    return spec['remote_donor_partition'][rows], spec['remote_donor_endpoint_indices'][rows, endpoint]


def run_intervention(networks, x, pool, spec, rows, part, direction, mode, live):
    n = len(rows); sender = spec['sender'][rows]
    recipients = spec['endpoint_indices'][rows, direction]
    dp, di = donor_selection(spec, rows, part, direction, mode)
    packets = np.empty((n, 2, 4), dtype=np.int8)
    for part_id, name in enumerate(PARTS):
        chosen = np.flatnonzero(dp == part_id)
        packets[chosen] = pool[name]['messages'][di[chosen], :, sender[chosen], :]
    windows = (0, 1) if mode == 'sham' or mode.endswith('_both') else (0,) if mode.endswith('_w1') else (1,)
    data = dict(dataset_rows=rows, recipient_indices=recipients, donor_partition=dp, donor_indices=di,
                donor_packets=packets, messages=np.empty((n, 2, 3, 4), dtype=np.int8),
                action_indices=np.empty((n, 3), dtype=np.int16), action_probabilities=np.empty((n, 3, 17)))
    batches = []; max_identity_error = 0.
    for start in range(0, n, BATCH):
        end = min(n, start+BATCH); sl = slice(start, end); ids = recipients[sl]
        trace = ch.intervene(networks, x[ids], pool[part]['messages'][ids], sender[sl], packets[sl], windows, live)
        for k in ('messages', 'action_indices', 'action_probabilities'): data[k][sl] = trace[k]
        own = sender[sl]; local_rows = np.arange(end-start)
        assert np.array_equal(trace['messages'][local_rows, 1, own], pool[part]['messages'][ids, 1, own])
        if mode == 'sham' or not live:
            assert np.array_equal(trace['messages'], pool[part]['messages'][ids])
            assert np.array_equal(trace['action_indices'], pool[part]['action_indices'][ids])
            error = float(np.max(np.abs(trace['action_probabilities'] - pool[part]['action_probabilities'][ids])))
            assert error <= 2e-12; max_identity_error = max(max_identity_error, error)
        if live and mode == 'local_opposite_both':
            target_ids = di[sl]
            expected_inputs = np.concatenate((x[target_ids], ch.core.routed_window(pool[part]['messages'][target_ids, 0], True),
                                              ch.core.routed_window(pool[part]['messages'][target_ids, 1], True)), axis=-1)
            non_sender = np.arange(3)[None, :] != own[:, None]
            assert np.array_equal(trace['action_inputs'][non_sender], expected_inputs[non_sender])
            error = float(np.max(np.abs(trace['action_probabilities'][non_sender] - pool[part]['action_probabilities'][target_ids][non_sender])))
            assert error <= 2e-12; max_identity_error = max(max_identity_error, error)
        batches.append(dict(start=start, end=end,
            action_inputs_sha256=ch.core.array_sha(trace['action_inputs']),
            first_routes_sha256=ch.core.array_sha(trace['first_routes']), second_routes_sha256=ch.core.array_sha(trace['second_routes'])))
    data.update(ch.settle(pool[part]['states'][recipients], data['action_indices']))
    return data, batches, max_identity_error


def execute(out):
    out = Path(out).resolve(); plan = verify(out)
    execution = out / 'execution'; execution.mkdir(exist_ok=False)
    started = time.perf_counter()
    write(execution / 'started.json', dict(started_at=now(), pid=os.getpid(), plan_sha256=sha(out/'plan.json')))
    natural_records, intervention_records = [], []
    try:
        specs = {p: npz(DATA / f'{p}.npz') for p in PARTS}
        spaces = read(DATA / 'endpoint_index_spaces.json')
        first = plan['policies'][0]
        states = {p: npz(first['endpoints'][p])['states'] for p in PARTS}
        features = {}
        for p in PARTS:
            assert np.array_equal(states[p], dataset.endpoint_states(spaces[p], np.arange(len(states[p]))))
            features[p] = {full: ch.observations(states[p], full) for full in (False, True)}
        feature_hashes = {p: {str(full): ch.core.array_sha(x) for full, x in f.items()} for p, f in features.items()}
        write(execution / 'feature_hashes.json', feature_hashes)
        for policy in plan['policies']:
            seed, condition = policy['seed'], policy['condition']
            full, live = condition.startswith('FI_'), condition.endswith('_live')
            directory = execution / f'seed_{seed}_{condition}'; directory.mkdir()
            for checkpoint in STEPS[:-1]:
                networks = ch.core.load_networks(policy['checkpoints'][str(checkpoint)])
                for part in PARTS:
                    data = natural_evaluation(networks, features[part][full], states[part], live)
                    path = directory / f'natural_{checkpoint:04d}_{part}.npz'
                    np.savez_compressed(path, **data)
                    natural_records.append(dict(kind='natural', seed=seed, condition=condition, checkpoint=checkpoint,
                        partition=part, path=str(path), data_sha256=sha(path), worlds=len(states[part]), reused_saved_endpoint=False))
            pool = {p: npz(policy['endpoints'][p]) for p in PARTS}
            for part in PARTS:
                assert np.array_equal(pool[part]['states'], states[part])
                assert np.array_equal(pool[part]['state_indices'], np.arange(len(states[part])))
                natural_records.append(dict(kind='natural', seed=seed, condition=condition, checkpoint=6000,
                    partition=part, path=policy['endpoints'][part], data_sha256=sha(policy['endpoints'][part]),
                    worlds=len(states[part]), reused_saved_endpoint=True))
            if not full:
                networks = ch.core.load_networks(policy['checkpoints']['6000'])
                for part in PARTS:
                    rows = np.flatnonzero(specs[part]['content_within_axis_weight'] > 0)
                    for mode in MODES:
                        for direction in (0, 1):
                            data, batches, err = run_intervention(networks, features[part][False], pool,
                                specs[part], rows, part, direction, mode, live)
                            path = directory / f'{mode}_{part}_d{direction}.npz'
                            np.savez_compressed(path, **data)
                            intervention_records.append(dict(kind='intervention', seed=seed, condition=condition,
                                checkpoint=6000, partition=part, mode=mode, direction=direction, path=str(path),
                                data_sha256=sha(path), worlds=len(rows), reused_saved_endpoint=False,
                                routing_batches=batches, max_structural_identity_probability_error=err))
            write(directory / 'completed.json', dict(seed=seed, condition=condition, completed_at=now()))
            print(json.dumps(dict(completed_policy=dict(seed=seed, condition=condition), elapsed_seconds=time.perf_counter()-started)), flush=True)
        assert len(natural_records) == 192 and len(intervention_records) == 320
        counts = dict(new_natural_worlds=sum(r['worlds'] for r in natural_records if not r['reused_saved_endpoint']),
                      reused_endpoint_worlds=sum(r['worlds'] for r in natural_records if r['reused_saved_endpoint']),
                      new_intervention_worlds=sum(r['worlds'] for r in intervention_records))
        for key, value in counts.items(): assert value == CONFIG[key]
        verify(out)
        result = dict(status='completed', completed_at=now(), elapsed_seconds=time.perf_counter()-started,
            config=CONFIG, plan_sha256=sha(out/'plan.json'), dataset_directory=str(DATA),
            policies=plan['policies'], natural_records=natural_records, intervention_records=intervention_records,
            counts=counts, new_network_samples=CONFIG['new_network_samples'], model_parameter_loads=88,
            statistical_scope='Development diagnostics of four paired training initializations; no optimization',
            feature_hashes=feature_hashes)
        write(execution / 'results.json', result); write(execution / 'status.json', dict(status='completed', completed_at=now()))
        return dict(status='completed', counts=counts, elapsed_seconds=result['elapsed_seconds'])
    except BaseException as error:
        write(execution / 'failure.json', dict(status='failed', failed_at=now(), error_type=type(error).__name__, error=str(error),
                                             elapsed_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'execute')); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    answer = prepare(args.out) if args.command == 'prepare' else execute(args.out) if args.command == 'execute' else verify(args.out)
    print(json.dumps(answer, ensure_ascii=False))
