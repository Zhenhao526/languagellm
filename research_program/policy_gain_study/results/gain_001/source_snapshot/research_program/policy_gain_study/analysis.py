"""Read-only complete-grid policy-gain analysis. No Torch/model imports.

J: two actual endpoint queries in one world. U: existence of one complete code
for both targets. N: one emitted code for a fixed map/photo pair, reused twice.
Training seed, not split/direction/photo, is the independent comparison unit.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parent
WORK = STUDY.parents[1]
sys.path.insert(0, str(STUDY.parent))
import v08_formation_trajectory as measurement
import v08_receiver_coverage as endpoint

SEEDS = (28101, 28102, 28103, 28104)
KINDS = ('additive', 'joint')
GAINS = (1, 3)
UPDATES = (0, 100, 300, 600, 1200, 1800, 2100, 2400)
MODES = ('normal', 'shuffle', 'blank', 'stochastic', 'erase_memory')
MAPS = measurement.MAPS
CONDITIONS = {}
for split, kind, gain in product((1, 2, 3, 0), KINDS, GAINS):
    base = f'split{split}_{kind}' if split else f'full_{kind}'
    CONDITIONS[f'{base}_gain{gain}'] = dict(base_condition=base,
        plan=measurement.expected_plan(base), policy_gain=gain)

require = measurement.require
read = measurement.read_json
sha = measurement.sha
now = measurement.now
write_new = measurement.write_new


def complete_inventory(batch):
    """Require exactly the registered grid, frozen sources and completed files."""
    batch = Path(batch).resolve()
    status, manifest = read(batch / 'status.json'), read(batch / 'manifest.json')
    require(status['status'] == 'complete' and status['completed_runs'] == status['expected_runs'] == 64,
            'All 64 social runs must finish before analysis or probe preparation')
    require(manifest['seeds'] == list(SEEDS) and manifest['conditions'] == CONDITIONS and
            manifest['updates'] == 2400 and manifest['batch'] == 512 and manifest['eval_n'] == 9600 and
            manifest['expected_social_runs'] == 64 and manifest['native_reward_changed_by_gain'] is False,
            'Registered factorial grid/budget changed')
    expected = set(product(SEEDS, CONDITIONS))
    indexed = {(r['seed'], r['condition']): r for r in status['runs']}
    require(len(status['runs']) == len(indexed) == 64 and set(indexed) == expected, 'Completed grid differs')
    require({p.parent.name for p in batch.glob('s*/result.json')} ==
            {f's{s}_{c}' for s, c in expected}, 'Unexpected/missing social run')
    hashes = {str(batch / 'manifest.json'): sha(batch / 'manifest.json'),
              str(batch / 'status.json'): sha(batch / 'status.json')}
    for path, digest in manifest['source_hashes'].items():
        path = Path(path).resolve()
        require(sha(path) == digest, f'Frozen source changed: {path}')
        hashes[str(path)] = digest
        if path.suffix in ('.py', '.md'):
            require(sha(batch / 'source_snapshot' / path.relative_to(WORK)) == digest,
                    f'Training source snapshot changed: {path}')
    for path in (Path(__file__).resolve(), STUDY / 'probe.py',
                 Path(measurement.__file__).resolve(), Path(endpoint.__file__).resolve()):
        require(str(path) in manifest['source_hashes'], f'Measurement source not frozen before training: {path}')
    controls = read(batch / 'individual_controls/summary.json')
    gate = read(batch / 'social_launch_gate.json')
    require(controls['complete'] and controls['passed'] and controls['runs'] == 4 and
            {r['seed'] for r in controls['results']} == set(SEEDS) and
            all(r['passed'] for r in controls['results']) and gate['all_passed'] and
            gate['personal_runs'] == 4 and gate['social_runs_planned'] == 64 and
            gate['diagnostic_weights_transferred'] is False and
            gate['personal_summary_sha256'] == sha(batch / 'individual_controls/summary.json'), 'Personal gate differs')
    for path in (batch / 'individual_controls/summary.json', batch / 'social_launch_gate.json'):
        hashes[str(path)] = sha(path)
    runs = []
    for seed, condition in product(SEEDS, CONDITIONS):
        directory = batch / f's{seed}_{condition}'
        cfg, result = read(directory / 'config.json'), read(directory / 'result.json')
        design = CONDITIONS[condition]
        train, held = measurement.map_partition(design['plan']['split'])
        for doc in (cfg, result):
            require(doc['seed'] == seed and doc['condition'] == condition and doc['plan'] == design['plan'] and
                    doc['policy_gain'] == design['policy_gain'] and doc['updates'] == 2400 and doc['batch'] == 512,
                    f'Run design differs: {directory}')
        require(cfg['eval_n'] == 9600 and cfg['checkpoint_eval_n'] == 1200 and
                cfg['checkpoints'] == list(UPDATES) and cfg['sites'] == 6 and cfg['history_dim'] == 18 and
                cfg['train_map_ids'] == train and cfg['heldout_map_ids'] == held and
                cfg['map_table'] == [list(m) for m in MAPS] and cfg['trace_schema'] == 'paired_world_v1' and
                cfg['initial_sha256'] == result['initial_sha256'] and result['frozen_projection_verified'] and
                result['final_sha256'] == indexed[seed, condition]['final_sha256'], f'Run metadata differs: {directory}')
        prepared = batch / f'prepared_{seed}.pt'
        require(Path(cfg['prepared_source']['path']).resolve() == prepared and
                sha(prepared) == cfg['prepared_source']['sha256'], 'Prepared checkpoint changed')
        for name, digest in cfg['source_hashes'].items():
            require(sha(WORK / 'redesign_v0.8' / name) == digest, 'Base source changed')
        files = [directory / n for n in ('config.json', 'result.json', 'curve.json', 'training.jsonl',
                 'training_schedule.json', 'initial.pt', 'final.pt', 'final_optimizer.pt')]
        files += [directory / f'checkpoint_{u:04d}.pt' for u in UPDATES]
        files += [directory / f'final_{mode}.npz' for mode in MODES] + [prepared]
        for path in files:
            require(path.is_file(), f'Incomplete run file: {path}')
            hashes.setdefault(str(path), sha(path))
        runs.append(dict(seed=seed, condition=condition, policy_gain=design['policy_gain'],
            plan=design['plan'], directory=str(directory), train_map_ids=train, heldout_map_ids=held,
            config=cfg, result=result, prepared_file=str(prepared), final_file=str(directory / 'final.pt'),
            checkpoint_files={str(u): str(directory / f'checkpoint_{u:04d}.pt') for u in UPDATES}))
    for seed in SEEDS:
        for field in ('initial_sha256', 'trainable_initial_sha256', 'receiver_initial_sha256'):
            require(len({r['config'][field] for r in runs if r['seed'] == seed}) == 1,
                    f'Initial parameters not paired for {seed}: {field}')
    return dict(batch=str(batch), manifest=manifest, runs=runs, source_files_sha256=hashes)


def paired_factorial(rows, group_keys, metrics):
    """rows already average splits/directions within seed; no pseudoreplication."""
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        cell = row['seed'], row['kind'], row['gain']
        require(cell not in groups[key], 'Duplicate factorial cell')
        groups[key][cell] = row
    output = []
    for key, cells in sorted(groups.items()):
        require(set(cells) == set(product(SEEDS, KINDS, GAINS)), 'Incomplete four-seed factorial cells')
        for metric in metrics:
            effects = defaultdict(list)
            for seed in SEEDS:
                a1, a3, j1, j3 = [cells[seed, k, g][metric] for k, g in
                                   [('additive', 1), ('additive', 3), ('joint', 1), ('joint', 3)]]
                for name, value in [('lambda_at_gain1', j1-a1), ('lambda_at_gain3', j3-a3),
                                    ('gain_at_lambda0', a3-a1), ('gain_at_lambda1', j3-j1),
                                    ('interaction', (j3-a3)-(j1-a1)),
                                    ('diagonal_joint3_minus_additive1', j3-a1)]:
                    effects[name].append(value)
            output.append(dict(zip(group_keys, key), metric=metric, seeds=list(SEEDS), effects={
                name: dict(seed_values=v, mean=sum(v)/4, min=min(v), max=max(v)) for name, v in effects.items()}))
    return output


def check_stats(stats, arrays, mask):
    """Independent native reward/J settlement, not equality of lambda-specific R."""
    import numpy as np
    success, goals, reward = arrays['successes'][mask], arrays['goals'][mask], arrays['reward'][mask]
    n = len(reward)
    require(stats['n'] == n and stats['decisions'] == n*2, 'Stats denominator differs')
    if not n:
        require(stats['both_accuracy'] is None and stats['mean_reward'] is None, 'Empty partition has a score')
        return
    by_resource = np.take_along_axis(success, np.argsort(goals, axis=1), axis=1).astype(int)
    counts = Counter(map(tuple, by_resource))
    wanted = dict(single_correct=int(success.sum()), both_correct=int(success.all(1).sum()),
        food_correct=int(by_resource[:, 0].sum()), water_correct=int(by_resource[:, 1].sum()),
        positive_rewards=int((reward > 0).sum()),
        outcome_counts={f'{a}{b}': counts[a, b] for a, b in product(range(2), repeat=2)})
    require(all(stats[k] == v for k, v in wanted.items()), 'Stored integer outcome counts differ')
    for name, value in [('mean_reward', reward.astype(float).mean()), ('reward_sum', reward.astype(float).sum()),
                        ('reward_variance', reward.astype(float).var()), ('single_accuracy', success.mean()),
                        ('both_accuracy', success.all(1).mean())]:
        require(abs(stats[name]-value) < 1e-7, f'Stored metric differs: {name}')


def settle_arrays(arrays, mode, complementarity):
    import numpy as np
    n = 9600
    shapes = dict(scout=(n,), episode=(n,), positions=(n, 2), photo_ids=(n, 2), goals=(n, 2),
        menu=(n, 2, 6), inventory=(n, 2), history=(n, 18), sent=(n, 2), delivered=(n, 2),
        action=(n, 2), place=(n, 2), successes=(n, 2), reward=(n,))
    require(set(arrays) == set(shapes) and all(arrays[k].shape == v for k, v in shapes.items()), 'Trace shape/schema')
    for k in ('scout', 'episode', 'positions', 'photo_ids', 'goals', 'menu', 'sent', 'delivered', 'action', 'place'):
        require(np.issubdtype(arrays[k].dtype, np.integer), 'Noninteger trace identity/action')
    require(np.isin(arrays['scout'], [0, 1]).all(), 'Invalid scout')
    require(np.array_equal(np.sort(arrays['goals'], 1), np.tile([0, 1], (n, 1))) and
            np.array_equal(np.sort(arrays['menu'], 2), np.tile(np.arange(6), (n, 2, 1))), 'Query/menu differs')
    require((arrays['inventory'] == 0).all() and (arrays['history'] == 0).all(), 'Receiver context differs')
    require(all(((arrays[k] >= 0) & (arrays[k] < 7)).all() for k in ('sent', 'delivered')) and
            ((arrays['action'] >= 0) & (arrays['action'] < 6)).all(), 'Invalid code/action')
    require(all(tuple(p) in MAPS for p in arrays['positions']), 'Invalid physical map')
    if mode == 'blank':
        require((arrays['delivered'] == 0).all(), 'Blank channel differs')
    elif mode != 'shuffle':
        require(np.array_equal(arrays['sent'], arrays['delivered']), 'Direct delivery differs')
    for scout in range(2):
        ix = arrays['scout'] == scout
        require(np.array_equal(np.sort(arrays['episode'][ix]), np.arange(n//2)), 'Direction episode identities')
        if mode == 'shuffle':
            for first in range(2):
                selected = ix & (arrays['goals'][:, 0] == first)
                require(Counter(map(tuple, arrays['sent'][selected])) ==
                        Counter(map(tuple, arrays['delivered'][selected])), 'Shuffle crosses declared strata')
    place = np.take_along_axis(arrays['menu'], arrays['action'][..., None], 2).squeeze(2)
    success = place == np.take_along_axis(arrays['positions'], arrays['goals'], 1)
    reward = ((1-complementarity)*success.mean(1)+complementarity*success.all(1)).astype(np.float32)
    require(np.array_equal(place, arrays['place']) and np.array_equal(success, arrays['successes']) and
            np.array_equal(reward, arrays['reward']), 'Physical settlement/native reward differs')
    return np.asarray([MAPS.index(tuple(p)) for p in arrays['positions']])


def training_summary(path, gain, kind):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    require(len(rows) == 2400 and [r['update'] for r in rows] == list(range(1, 2401)), 'Incomplete training curve')
    require([r['batch_identity'] for r in rows] == list(range(2400)), 'Training identities differ')
    norms, losses = [], []
    for row in rows:
        def finite(value):
            if isinstance(value, dict): return all(finite(v) for v in value.values())
            if isinstance(value, list): return all(finite(v) for v in value)
            return math.isfinite(value) if isinstance(value, (float, int)) else True
        require(finite(row), 'Nonfinite training observation')
        require(row['policy_gain'] == gain and row['entropy_weight'] == (.02 if row['update'] <= 2100 else 0),
                'Training gain/entropy schedule differs')
        counts = row['outcome_counts']; total = sum(counts.values())
        require(total == 512, 'Training batch denominator differs')
        single = (counts['01']+counts['10']+2*counts['11'])/1024
        both = counts['11']/512
        native = single if kind == 'additive' else both
        require(abs(row['single_accuracy']-single) < 1e-7 and abs(row['both_accuracy']-both) < 1e-7 and
                abs(row['reward']-native) < 1e-7, 'Native training reward/J differs')
        require(len(row['agents']) == 2 and all(len(a['components']) == 2 for a in row['agents']), 'Independent role loss count')
        for agent in row['agents']:
            norms.append(agent['gradient_norm']); losses.append(agent['loss'])
            require(agent['gradient_norm'] >= 0, 'Negative gradient norm')
            for c in agent['components']:
                require(abs(c['weighted_policy_loss']-gain*c['policy_loss']) < 1e-5 * (1+abs(c['weighted_policy_loss'])),
                        'Recorded applied policy gain differs')
            expected_loss = sum(c['weighted_policy_loss']+c['value_loss']-row['entropy_weight']*c['entropy']
                                for c in agent['components'])/2
            require(abs(agent['loss']-expected_loss) < 1e-5*(1+abs(expected_loss)), 'Total role-averaged loss differs')
    return dict(world_hashes=[r['world_sha256'] for r in rows], updates=2400,
        native_reward_mean=sum(r['reward'] for r in rows)/2400,
        J_mean=sum(r['both_accuracy'] for r in rows)/2400,
        clip_events=sum(x > 2 for x in norms), gradient_observations=len(norms),
        mean_preclip_total_gradient_norm=sum(norms)/len(norms), max_preclip_total_gradient_norm=max(norms),
        mean_total_loss=sum(losses)/len(losses))


def aggregate_seed_rows(rows, value_key):
    """Balanced split aggregation; full stays separate, preserving 4 actual seeds."""
    groups = defaultdict(list)
    for row in rows:
        prefix, kind, gain = row['condition'].split('_')
        family = 'split' if prefix.startswith('split') else 'full'
        key = family, row['partition'], row.get('mode', 'normal'), row.get('update', 2400), row['seed'], kind, int(gain[4:])
        groups[key].append(row[value_key])
    output = []
    for (family, part, mode, update, seed, kind, gain), vals in sorted(groups.items()):
        require(len(vals) == (3 if family == 'split' else 1), 'Incomplete within-seed split aggregation')
        output.append(dict(family=family, partition=part, mode=mode, update=update, seed=seed,
                           kind=kind, gain=gain, **{value_key: sum(vals)/len(vals)}))
    return output


def analyze(batch, probe_dir):
    import numpy as np
    inventory = complete_inventory(batch)
    probe_dir = Path(probe_dir).resolve()
    probe_plan = read(probe_dir / 'plan.json')
    require(sha(probe_dir / 'plan.json') == read(probe_dir / 'freeze.json')['plan_sha256'] and
            Path(probe_plan['batch']) == Path(inventory['batch']), 'Probe plan/batch differs')
    for path, digest in probe_plan['source_files_sha256'].items():
        require(sha(path) == digest, f'Probe input changed: {path}')
    probed = read(probe_dir / 'execution/results.json')
    require(probed['status'] == 'complete' and probed['counts']['directions'] == 1024 and
            probed['counts']['final_anchors'] == 128 and probed['counts']['receiver_maximum_ties'] == 0 and
            probed['plan_sha256'] == sha(probe_dir / 'plan.json'), 'Full no-tie probe required')
    recs = {}
    for entry in probed['records']:
        path = probe_dir / 'execution' / entry['file']
        require(sha(path) == entry['sha256'], 'Probe record hash differs')
        key = entry['seed'], entry['condition'], entry['update'], entry['scout']
        require(key not in recs, 'Duplicate probe record')
        record = read(path)
        require(tuple(record[k] for k in ('seed', 'condition', 'update', 'scout')) == key and
                record['collector'] == 1-record['scout'], 'Probe record identity differs')
        checked_receiver = measurement.lookup_from_logits(record['receiver']['logits'])
        require(all(record['receiver'][k] == v for k, v in checked_receiver.items()), 'Probe decoder differs from logits')
        train, held = measurement.map_partition(CONDITIONS[record['condition']]['plan']['split'])
        checked_analysis = measurement.analyze_arrays(record['receiver']['actions'], record['emitted'],
            record['delivered'], train, held, False)
        require(record['analysis'] == checked_analysis and entry['summaries'] == checked_analysis['summaries'],
                'Stored probe measurements differ from per-photo/per-code arrays')
        recs[key] = record
    require(set(recs) == set(product(SEEDS, CONDITIONS, UPDATES, range(2))), 'Probe record grid differs')
    import importlib.util
    spec = importlib.util.spec_from_file_location('policy_gain_probe_measurement', STUDY / 'probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summaries, effects = module.summarize(probed['records'])
    require(probed['summaries'] == summaries and probed['effects'] == effects, 'Probe aggregation differs')
    results, curves, training, pairing = [], [], [], {}
    for run in inventory['runs']:
        seed, condition = run['seed'], run['condition']
        directory = Path(run['directory']); lam = run['plan']['complementarity']
        tables = {scout: {tuple(code): tuple(actions) for code, actions in zip(product(range(7), repeat=2),
            recs[seed, condition, 2400, scout]['receiver']['actions'])} for scout in range(2)}
        world_reference = None
        for mode in MODES:
            with np.load(directory / f'final_{mode}.npz', allow_pickle=False) as saved:
                arrays = {k: saved[k] for k in saved.files}
            mapids = settle_arrays(arrays, mode, lam)
            external = hashlib.sha256()
            for scout in range(2):
                ix = arrays['scout'] == scout
                for k in ('positions', 'photo_ids', 'goals', 'menu'):
                    external.update(np.ascontiguousarray(arrays[k][ix]).tobytes())
            actual_world_sha = external.hexdigest()
            require(actual_world_sha == run['result']['scores'][mode]['world_sha256'], 'External world digest differs')
            if world_reference is None: world_reference = actual_world_sha
            require(actual_world_sha == world_reference, 'Evaluation modes do not share external worlds')
            if mode == 'normal':
                endpoint.audit_normal_arrays(arrays, tables, False)
            if mode != 'stochastic':
                for scout in range(2):
                    ix = arrays['scout'] == scout
                    table = np.asarray(list(tables[scout].values()))
                    codes = arrays['delivered'][ix, 0]*7+arrays['delivered'][ix, 1]
                    require(np.array_equal(np.take_along_axis(table[codes], arrays['goals'][ix], 1), arrays['place'][ix]),
                            f'Actual {mode} greedy actions differ from endpoint receiver codebook')
            stats = run['result']['scores'][mode]
            check_stats(stats, arrays, np.ones(9600, dtype=bool))
            for part, ids in [('train', run['train_map_ids']), ('heldout', run['heldout_map_ids'])]:
                mask = np.isin(mapids, ids)
                check_stats(stats['map_groups']['seen' if part == 'train' else 'unseen'], arrays, mask)
                if mask.any():
                    results.append(dict(seed=seed, condition=condition, mode=mode, partition=part,
                        J=float(arrays['successes'][mask].all(1).mean()), n=int(mask.sum()),
                        single_accuracy=float(arrays['successes'][mask].astype(float).mean()),
                        joint_correct=int(arrays['successes'][mask].all(1).sum()),
                        native_reward=float(arrays['reward'][mask].astype(float).mean())))
        key = seed, run['plan']['split']
        pair = pairing.setdefault(key, dict(endpoint_sha=world_reference, training_hashes=None))
        require(pair['endpoint_sha'] == world_reference, 'Endpoint worlds not factorial-paired')
        kind = run['plan']['reward_kind']
        tr = training_summary(directory / 'training.jsonl', run['policy_gain'], kind)
        if pair['training_hashes'] is None: pair['training_hashes'] = tr['world_hashes']
        require(pair['training_hashes'] == tr['world_hashes'], 'Training worlds not factorial-paired')
        training.append(dict(seed=seed, condition=condition, **{k: v for k, v in tr.items() if k != 'world_hashes'}))
        curve = read(directory / 'curve.json')
        require([x['update'] for x in curve] == list(UPDATES), 'Checkpoint curve grid differs')
        for checkpoint in curve:
            require(set(checkpoint['scores']) == set(MODES), 'Checkpoint evaluation modes missing')
            for mode in MODES:
                for part, key in [('train', 'seen'), ('heldout', 'unseen')]:
                    value = checkpoint['scores'][mode]['map_groups'][key]
                    expected_n = (len(run['train_map_ids']) if part == 'train' else len(run['heldout_map_ids']))*40
                    require(value['n'] == expected_n, 'Checkpoint partition denominator differs')
                    if expected_n:
                        require(value['both_accuracy'] == value['both_correct']/value['n'], 'Checkpoint J fraction differs')
                        curves.append(dict(seed=seed, condition=condition, update=checkpoint['update'], mode=mode,
                            partition=part, J=value['both_accuracy'], single_accuracy=value['single_accuracy'], n=value['n']))
    endpoint_seeds = aggregate_seed_rows(results, 'J')
    curve_seeds = aggregate_seed_rows(curves, 'J')
    for path, digest in inventory['source_files_sha256'].items():
        require(sha(path) == digest, 'Input changed while analyzing')
    return dict(status='complete', created_at=now(), batch=inventory['batch'], seeds=list(SEEDS),
        conditions=CONDITIONS, inputs_sha256=inventory['source_files_sha256'],
        probe_result_sha256=sha(probe_dir / 'execution/results.json'),
        scope='Fixed new seed factorial; old development photos; 4 independent seeds, not 64 independent repetitions.',
        endpoints=results, endpoint_seed_values=endpoint_seeds,
        endpoint_single_seed_values=aggregate_seed_rows(results, 'single_accuracy'),
        endpoint_effects=paired_factorial(endpoint_seeds, ('family', 'partition', 'mode', 'update'), ('J',)),
        checkpoint_curve_seed_values=curve_seeds,
        checkpoint_single_seed_values=aggregate_seed_rows(curves, 'single_accuracy'),
        checkpoint_curve_effects=paired_factorial(curve_seeds, ('family', 'partition', 'mode', 'update'), ('J',)),
        same_photo_trajectory=probed['summaries'], same_photo_effects=probed['effects'], training=training,
        checks=dict(complete_runs=64, endpoint_worlds=64*9600*5, normal_worlds_anchored=64*9600,
            training_updates=64*2400, training_world_hash_pairs=16, initial_parameter_seed_groups=4),
        limitations=['Policy gain scales REINFORCE only; entropy/value/native R are unscaled, but actual gradients and clipping may change.',
            'J samples random pairs from the old test pool of 8 food and 8 water photographs (64 possible pairs); N uses 16 fixed old validation pairs; do not merge denominators.',
            'Checkpoint J reads the saved 1200-world score summaries; those checkpoint evaluation raw worlds were not stored.',
            'No blocked training arm in this study; evaluation blank/shuffle effects are inference-time interventions.',
            'Four paired seed values describe this fixed experiment; no inference from photos/splits as independent training repetitions.'])


def markdown(result):
    lines = ['# 策略增益研究：完整结果分析', '', '64 个社会运行完整，4 个新独立训练种子。各格先平均三个划分，完整训练参照单列。',
             'J 是实际同一世界两个查询都正确；U 是存在同一完整码可同时正确；N 是固定同照片自然消息双目标正确。', '',
             '| 范围 | 数据 | 指标 | λ效应 α1 | λ效应 α3 | 交互 |', '|---|---|---|---:|---:|---:|']
    for row in result['endpoint_effects']:
        if row['mode'] != 'normal': continue
        e = row['effects']
        lines.append(f"| {row['family']} | {row['partition']} | J | {e['lambda_at_gain1']['mean']:.2%} | "
                     f"{e['lambda_at_gain3']['mean']:.2%} | {e['interaction']['mean']:.2%} |")
    lines += ['', '完整 JSON 保存全部四种子的效应、所有检查点及五种评价模式。交互为 (λ1−λ0)α3−(λ1−λ0)α1。', '']
    lines += ['- '+text for text in result['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='new output prefix')
    args = parser.parse_args()
    json_path, md_path = args.out.with_suffix('.json'), args.out.with_suffix('.md')
    require(not json_path.exists() and not md_path.exists(), 'Never overwrite analysis outputs')
    result = analyze(args.batch, args.probe)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    write_new(json_path, result)
    with md_path.open('x') as stream: stream.write(markdown(result))
    print(json.dumps(dict(status='complete', output=str(json_path), sha256=sha(json_path))))


if __name__ == '__main__': main()
