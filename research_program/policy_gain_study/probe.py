"""Frozen same-photo U/N measurements for all 64 policy-gain runs.

Import/prepare/verify use no neural runtime. Only explicit execute loads cached
visual features and frozen small CPU interfaces; no backbone or training.
Exact maximum ties stop the whole probe. Never select/resume successful cases.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
import analysis as a

WORK, SEEDS, CONDITIONS, UPDATES = a.WORK, a.SEEDS, a.CONDITIONS, a.UPDATES
require, read, sha, write_new, now = a.require, a.read, a.sha, a.write_new, a.now
measurement, endpoint = a.measurement, a.endpoint


def prepare(output, batch):
    require('torch' not in sys.modules and 'camp' not in sys.modules, 'Prepare in a clean no-model process')
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite a probe preparation')
    inventory = a.complete_inventory(batch)
    manifest = read(WORK / 'redesign_v0.4/data/manifest.json')['images']
    pools = [[i for i, row in enumerate(manifest) if row['split'] == 'test' and row['category'] == kind]
             for kind in ('food', 'water')]
    require(all(len(pool) == 8 for pool in pools), 'Old held-out photo pool changed')
    photos = [list(pair) for pair in product(pools[0][4:8], pools[1][4:8])]
    expected = dict(runs=64, checkpoint_files=512, directions=1024, receiver_inputs=100352,
                    natural_messages=491520, messages_per_direction=480, endpoint_anchors=128)
    plan = dict(schema_version=1, design='policy_gain_fixed_same_photo_formation', created_at=now(),
        timing='Measurement code and contrasts frozen in the training manifest; checkpoint byte binding prepared only after all 64 runs finish.',
        batch=inventory['batch'], runs=inventory['runs'], seeds=list(SEEDS), conditions=CONDITIONS,
        updates=list(UPDATES), map_table=[list(m) for m in a.MAPS], validation_photo_pairs=photos,
        expected=expected, source_files_sha256=inventory['source_files_sha256'],
        source_runtime=inventory['manifest'], prepare_runtime=measurement.runtime(),
        menu_rule=measurement.menu_gather_source_check(WORK / 'redesign_v0.8/camp.py'),
        receiver_context=dict(inventory=[0, 0], history=[0]*18, menu=list(range(6)), goals=[[1, 0], [0, 1]]),
        sender_context=dict(visible_goal=[0, 0], inventory=[0, 0], greedy=True, numpy_seed=810071),
        main_process_summary=dict(metric=['U_full49', 'N_both'], updates=[0, 100, 300, 600, 1200],
            formula='trapezoidal integral divided by 1200; average splits/directions within each seed first'),
        execution=dict(device='cpu', torch_threads=1, no_training=True, no_backbone=True,
            weight_loading='weights_only=True', no_automatic_retry=True, no_resume=True),
        prepare_weight_loads=0, prepare_forward_calls=0)
    output.mkdir(parents=True, exist_ok=False)
    for path in inventory['manifest']['source_hashes']:
        source = Path(path)
        if source.suffix in ('.py', '.md'):
            dest = output / 'code_snapshot' / source.relative_to(WORK)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    write_new(output / 'plan.json', plan)
    write_new(output / 'freeze.json', dict(plan_sha256=sha(output / 'plan.json'), status='prepared_not_executed'))
    return dict(status='prepared_not_executed', output=str(output), **expected)


def verify(output):
    output = Path(output).resolve()
    require(sha(output / 'plan.json') == read(output / 'freeze.json')['plan_sha256'], 'Probe plan changed')
    plan = read(output / 'plan.json')
    for path, digest in plan['source_files_sha256'].items():
        require(sha(path) == digest, f'Frozen probe source/input changed: {path}')
    for path, digest in plan['source_runtime']['source_hashes'].items():
        source = Path(path)
        if source.suffix in ('.py', '.md'):
            require(sha(output / 'code_snapshot' / source.relative_to(WORK)) == digest, 'Probe snapshot changed')
    return plan


def anchor_endpoint(records, arrays, complementarity):
    """All actual normal worlds; optional photo overlap is counted, never assumed."""
    import numpy as np
    require(len(records) == 2 and {r['scout'] for r in records} == {0, 1} and
            all(r['update'] == 2400 for r in records), 'Endpoint direction grid')
    tables = {r['scout']: {tuple(code): tuple(action) for code, action in zip(product(range(7), repeat=2),
                r['receiver']['actions'])} for r in records}
    a.settle_arrays(arrays, 'normal', complementarity)
    checked = endpoint.audit_normal_arrays(arrays, tables, False)
    output = {}
    for record in records:
        scout = record['scout']; ix = np.flatnonzero(arrays['scout'] == scout)
        photos = {tuple(p): i for i, p in enumerate(record['photo_pairs'])}
        matches, contexts = 0, set()
        for i in ix:
            pair = tuple(arrays['photo_ids'][i])
            if pair in photos:
                mid, pid = int(checked['map_ids'][i]), photos[pair]
                require(record['emitted'][mid][pid] == arrays['sent'][i].tolist(),
                        'Frozen sender message differs for the same saved endpoint photo/map')
                matches += 1; contexts.add((mid, pid))
        require(matches > 0, 'No endpoint overlap with fixed validation photos')
        output[scout] = dict(actual_worlds=len(ix), physical_actions_checked=len(ix)*2,
            both_correct=int(checked['joint'][ix].sum()), same_photo_message_rows_checked=matches,
            distinct_same_photo_contexts_checked=len(contexts),
            scope='All actual receiver actions; sender identity only for recorded matching photo/map contexts.')
    return output


def normalized_auc(points, metric):
    """Fixed early interval only; no best-checkpoint or metric-dependent stopping."""
    required = (0, 100, 300, 600, 1200)
    require(set(points) == set(required), 'Early AUC needs exactly five fixed checkpoints')
    return sum((points[left][metric]+points[right][metric])*.5*(right-left)
               for left, right in zip(required[:-1], required[1:]))/1200


def summarize(records):
    buckets = defaultdict(list)
    metrics = ('U_full49', 'N_both', 'N_single', 'natural_failure_U0', 'natural_failure_U1',
               'optimal_uniform_goal_full49')
    for row in records:
        prefix, kind, gain = row['condition'].split('_')
        family = 'split' if prefix.startswith('split') else 'full'
        for part in ('train', 'heldout'):
            values = row['summaries'][part]
            if values is not None:
                key = family, part, row['update'], row['seed'], kind, int(gain[4:])
                buckets[key].append(values)
    seed_rows = []
    for (family, part, update, seed, kind, gain), values in sorted(buckets.items()):
        require(len(values) == (6 if family == 'split' else 2), 'Incomplete within-seed split/direction cells')
        data = {metric: sum(v[metric]['rate'] for v in values)/len(values) for metric in metrics}
        for category in measurement.CATEGORIES:
            data[category] = sum(v['coverage_categories'][category]['rate'] for v in values)/len(values)
        failures = sum(v['fraction_failures_U0']['denominator'] for v in values)
        data['fraction_failures_U0'] = sum(v['fraction_failures_U0']['numerator'] for v in values)/failures if failures else None
        seed_rows.append(dict(family=family, partition=part, update=update, seed=seed, kind=kind, gain=gain, **data))
    grouped = defaultdict(list)
    for row in seed_rows:
        key = row['family'], row['partition'], row['update'], row['kind'], row['gain']
        grouped[key].append(row)
    means = []
    for (family, part, update, kind, gain), rows in sorted(grouped.items()):
        require({r['seed'] for r in rows} == set(SEEDS), 'Incomplete seed set')
        rows.sort(key=lambda r: r['seed'])
        fields = list(metrics)+list(measurement.CATEGORIES)+['fraction_failures_U0']
        means.append(dict(family=family, partition=part, update=update, kind=kind, gain=gain, seeds=list(SEEDS),
            metrics={m: dict(seed_values=[r[m] for r in rows],
                mean=sum(r[m] for r in rows)/4 if all(r[m] is not None for r in rows) else None) for m in fields}))
    auc_buckets = defaultdict(dict)
    for row in seed_rows:
        if row['update'] <= 1200:
            key = row['family'], row['partition'], row['seed'], row['kind'], row['gain']
            auc_buckets[key][row['update']] = row
    auc = [dict(family=f, partition=p, seed=s, kind=k, gain=g,
                U_auc_0_1200=normalized_auc(points, 'U_full49'), N_auc_0_1200=normalized_auc(points, 'N_both'))
           for (f, p, s, k, g), points in sorted(auc_buckets.items())]
    effects = dict(by_checkpoint=a.paired_factorial(seed_rows, ('family', 'partition', 'update'), metrics),
        early_auc=a.paired_factorial(auc, ('family', 'partition'), ('U_auc_0_1200', 'N_auc_0_1200')))
    return dict(seed_values=seed_rows, means=means, early_auc_seed_values=auc), effects


def execute(output):
    output = Path(output).resolve()
    plan = verify(output)
    directory = output / 'execution'
    directory.mkdir(exist_ok=False)
    (directory / 'records').mkdir()
    started = time.monotonic(); context = {}
    write_new(directory / 'started.json', dict(started_at=now(), runtime=measurement.runtime(),
                                              plan_sha256=sha(output / 'plan.json')))
    try:
        import importlib
        import numpy as np
        import torch
        torch.set_num_threads(1)
        require(platform.python_version() == plan['source_runtime']['python'] and
                str(torch.__version__) == plan['source_runtime']['torch'] and
                str(np.__version__) == plan['source_runtime']['numpy'], 'Use the frozen training runtime')
        require('camp' not in sys.modules, 'Execute in a clean process')
        sys.path.insert(0, str(WORK / 'redesign_v0.8'))
        camp = importlib.import_module('camp')
        for name, relative in [('camp', 'redesign_v0.8/camp.py'), ('agents', 'redesign_v0.4/agents.py'),
                               ('run_pilot', 'redesign_v0.4/run_pilot.py'), ('resource_env', 'redesign_v0.4/resource_env.py')]:
            require(Path(sys.modules[name].__file__).resolve() == WORK / relative, 'Imported wrong runtime module')
        require(camp.MAPS.tolist() == plan['map_table'], 'Runtime map table differs')
        bank = camp.ImageBank()
        photos = np.asarray(plan['validation_photo_pairs'], dtype=np.int64)
        require(photos.tolist() == [list(p) for p in product(bank.pools['test', 0][4:8], bank.pools['test', 1][4:8])],
                'Runtime validation photos changed')
        codes = np.asarray(list(product(range(7), repeat=2)), dtype=np.int64)
        message_inputs = np.repeat(codes, 2, axis=0)
        goals = np.eye(2, dtype=np.float32)[np.tile(np.arange(2), 49)]
        maps = np.repeat(np.arange(30), 16); pids = np.tile(np.arange(16), 30)
        index, initial = [], {}
        with torch.no_grad():
            for run in plan['runs']:
                seed, condition = run['seed'], run['condition']
                prepared = torch.load(run['prepared_file'], weights_only=True, map_location='cpu')
                agents = camp.remake_agents(seed, prepared, 7, 2, representation='identity')
                frozen = [{k: v.clone() for k, v in agent.state_dict().items()
                           if k.startswith('project.') or k == 'input_transform'} for agent in agents]
                banks = camp.projected_banks(agents, bank)
                final = torch.load(run['final_file'], weights_only=True, map_location='cpu')
                for update in UPDATES:
                    checkpoint = run['checkpoint_files'][str(update)]
                    context = dict(seed=seed, condition=condition, update=update, checkpoint=checkpoint)
                    states = torch.load(checkpoint, weights_only=True, map_location='cpu')
                    require(len(states) == len(final) == len(agents) == 2, 'Need two independent interfaces')
                    for who, agent in enumerate(agents):
                        if update == 2400:
                            require(states[who].keys() == final[who].keys() and
                                    all(torch.equal(v, final[who][k]) for k, v in states[who].items()), 'Final state differs')
                        agent.load_state_dict(states[who], strict=True)
                        agent.eval(); agent.requires_grad_(False)
                        require(all(torch.equal(v, agent.state_dict()[k]) for k, v in frozen[who].items()),
                                'Frozen visual projection/transform changed')
                    current = []
                    for scout in range(2):
                        context['scout'] = scout
                        collector = 1-scout
                        logits, _ = agents[collector].receive(torch.from_numpy(message_inputs), torch.from_numpy(goals),
                            torch.zeros(98, 2), torch.zeros(98, 18), torch.arange(6).repeat(98, 1))
                        logits = logits.numpy().reshape(49, 2, 6)
                        receiver = dict(logits=logits.tolist(), **measurement.lookup_from_logits(logits))
                        latent = agents[scout].observe(camp.scene_visual(camp.MAPS[maps], photos[pids], banks[scout]))
                        emitted, _, _, _ = agents[scout].send(latent, torch.zeros(480, 2), torch.zeros(480, 2),
                            np.random.default_rng(810071), True)
                        emitted = emitted.numpy().reshape(30, 16, 2)
                        record = dict(seed=seed, condition=condition, update=update, scout=scout, collector=collector,
                            checkpoint_sha256=plan['source_files_sha256'][checkpoint], photo_pairs=photos.tolist(),
                            receiver=receiver, emitted=emitted.tolist(), delivered=emitted.tolist(),
                            analysis=measurement.analyze_arrays(receiver['actions'], emitted, emitted,
                                run['train_map_ids'], run['heldout_map_ids'], False))
                        if update == 0:
                            key = seed, scout
                            if key in initial: measurement.update0_cache_verify(record, initial[key])
                            else: initial[key] = record
                        current.append(record)
                    if update == 2400:
                        with np.load(Path(run['directory']) / 'final_normal.npz', allow_pickle=False) as saved:
                            arrays = {k: saved[k] for k in saved.files}
                        anchors = anchor_endpoint(current, arrays, run['plan']['complementarity'])
                        for record in current: record['endpoint_anchor'] = anchors[record['scout']]
                    for record in current:
                        scout = record['scout']
                        path = f'records/s{seed}_{condition}_u{update:04d}_s{scout}.json'
                        write_new(directory / path, record)
                        index.append(dict(seed=seed, condition=condition, update=update, scout=scout,
                            file=path, sha256=sha(directory / path), summaries=record['analysis']['summaries'],
                            endpoint_anchor=record.get('endpoint_anchor')))
                print(f'Completed {seed} {condition}: {len(index)}/1024 directions', flush=True)
        require(len(index) == 1024 and {(r['seed'], r['condition'], r['update'], r['scout']) for r in index} ==
                set(product(SEEDS, CONDITIONS, UPDATES, range(2))), 'Incomplete probe grid')
        require(sum(r['endpoint_anchor'] is not None for r in index) == 128, 'Incomplete endpoint anchors')
        summaries, effects = summarize(index)
        verify(output)
        result = dict(status='complete', completed_at=now(), elapsed_seconds=time.monotonic()-started,
            runtime=measurement.runtime(), plan_sha256=sha(output / 'plan.json'), records=index,
            summaries=summaries, effects=effects,
            counts=dict(directions=1024, receiver_inputs=100352, natural_messages=491520, final_anchors=128,
                initial_condition_comparisons=120, receiver_maximum_ties=0),
            interpretation='U is a same-complete-code existential receiver property; N uses one frozen map/photo emission for both queries. Neither alone establishes compositional language.')
        write_new(directory / 'results.json', result)
        return dict(status='complete', output=str(directory / 'results.json'), **result['counts'])
    except Exception as error:
        failure = dict(status='failed', failed_at=now(), elapsed_seconds=time.monotonic()-started,
                       context=context, error=repr(error), traceback=traceback.format_exc())
        if isinstance(error, measurement.TieError): failure['tie'] = error.details
        write_new(directory / 'failure.json', failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'execute'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--batch', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        require(args.batch is not None, 'prepare needs --batch')
        result = prepare(args.out, args.batch)
    elif args.command == 'verify':
        plan = verify(args.out); result = dict(status='verified', **plan['expected'])
    else: result = execute(args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
