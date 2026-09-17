"""Independent, read-only execution audit. Never trains or calls a model.

NumPy reconstructs fixtures and physical settlement. Torch is used only to read
weights_only CPU checkpoint tensors; no project module or model is imported.
Audit output is exclusive and the complete 64-run status is required first.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import permutations, product
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np

STUDY = Path(__file__).resolve().parent
WORK = STUDY.parents[1]
SEEDS = (28101, 28102, 28103, 28104)
CHECKPOINTS = (0, 100, 300, 600, 1200, 1800, 2100, 2400)
MODES = ('normal', 'shuffle', 'blank', 'stochastic', 'erase_memory')
MAPS = np.array(list(permutations(range(6), 2)), dtype=np.int64)
MATCHINGS = {1: ((0, 1), (2, 3), (4, 5)), 2: ((0, 2), (1, 4), (3, 5)),
             3: ((0, 3), (1, 5), (2, 4))}
WORLD_KEYS = ('positions', 'photo_ids', 'goals', 'menu')


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finite(value, label):
    if isinstance(value, dict):
        for k, v in value.items(): finite(v, label + '/' + str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value): finite(v, label + '/' + str(i))
    elif isinstance(value, (float, np.floating)):
        require(math.isfinite(value), 'Nonfinite: ' + label)


def near(a, b, label, tolerance=1e-10):
    require(a is not None and math.isfinite(a) and abs(a-b) <= tolerance * (1+abs(b)),
            f'{label}: {a} != {b}')


def partition(split):
    pairs = {p for a, b in MATCHINGS.get(split, ()) for p in ((a, b), (b, a))}
    held = [i for i, p in enumerate(MAPS) if tuple(p) in pairs]
    return [i for i in range(30) if i not in held], held


def design(split, kind):
    return dict(split=split, representation='identity', schedule='direct', vocab=7,
                length=2, known=False, blocked=False, reward_kind=kind,
                complementarity=0. if kind == 'additive' else 1.)


def grid():
    return {f'{"split"+str(s) if s else "full"}_{k}_gain{g}':
            dict(base_condition=f'{"split"+str(s) if s else "full"}_{k}',
                 plan=design(s, k), policy_gain=g)
            for s, k, g in product((1, 2, 3, 0), ('additive', 'joint'), (1, 3))}


def photo_pools():
    entries = read(WORK / 'redesign_v0.4/data/manifest.json')['images']
    pools = {(s, k): np.array([i for i, e in enumerate(entries)
             if e['split'] == s and e['category'] == kind], dtype=np.int64)
             for s in ('train', 'test') for k, kind in enumerate(('food', 'water'))}
    require(all(len(p) >= 4 for p in pools.values()), 'Photo pools')
    require(not ({e['sha256'] for e in entries if e['split'] == 'train'} &
                 {e['sha256'] for e in entries if e['split'] == 'test'}), 'Photo byte split overlap')
    return pools


def fixture_pair(seed, n, pools, training=False, map_pool=None):
    """Independent external RNG transcription; no learned parameters involved."""
    rng = np.random.default_rng(seed)
    n2 = n // 2
    rows = []
    for scout in range(2):
        if training:
            pos = MAPS[rng.choice(np.array(map_pool, dtype=np.int64), n2)].copy()
            first = rng.integers(2, size=n2)
        else:
            require(n2 % 60 == 0, 'Balanced evaluation fixture')
            cells = np.tile(np.array(list(product(range(30), range(2)))), (n2//60, 1))
            cells = cells[rng.permutation(n2)]
            pos, first = MAPS[cells[:, 0]].copy(), cells[:, 1]
        photos = np.column_stack([rng.choice(pools['train' if training else 'test', k], n2)
                                  for k in range(2)])
        rows.append(dict(positions=pos, photo_ids=photos,
            goals=np.column_stack((first, 1-first)), menu=np.argsort(rng.random((n2, 2, 6)), axis=2)))
    return rows


def world_sha(rows):
    h = hashlib.sha256()
    for row in rows:
        for key in WORLD_KEYS:
            h.update(np.ascontiguousarray(row[key]).tobytes())
    return h.hexdigest()


def metrics_from_counts(stats, expected_n, lam, label):
    """Validate all scalar metrics from the four stored food/water outcomes."""
    finite(stats, label)
    counts = stats['outcome_counts']
    require(set(counts) == {'00', '01', '10', '11'} and
            all(type(v) is int and v >= 0 for v in counts.values()) and
            sum(counts.values()) == expected_n, label + ' outcome counts')
    n = expected_n
    require(stats['n'] == n and stats['decisions'] == n*2, label + ' denominator')
    correct = counts['01'] + counts['10'] + 2*counts['11']
    exact = dict(single_correct=correct, both_correct=counts['11'],
                 food_correct=counts['10']+counts['11'], water_correct=counts['01']+counts['11'],
                 positive_rewards=n-counts['00'] if lam == 0 else counts['11'])
    for k, v in exact.items(): require(stats[k] == v, label + '/' + k)
    values = np.array([0, (1-lam)/2, (1-lam)/2, 1.], dtype=float)
    weights = np.array([counts[k] for k in ('00', '01', '10', '11')])
    total = float(values @ weights)
    near(stats['reward_sum'], total, label + '/reward_sum')
    if not n:
        require(all(stats[k] is None for k in ('mean_reward', 'reward_variance',
                    'single_accuracy', 'both_accuracy')), label + ' empty metrics')
    else:
        mean = total / n
        for k, v in dict(mean_reward=mean, reward_variance=float(((values-mean)**2)@weights/n),
                         single_accuracy=correct/(2*n), both_accuracy=counts['11']/n).items():
            near(stats[k], v, label + '/' + k)


def audit_score(stats, n, split, lam, whash, label, actual=None):
    metrics_from_counts(stats, n, lam, label)
    require(stats['episodes'] == n and stats['horizon'] == 1 and stats['world_sha256'] == whash,
            label + ' world metadata')
    train, held = partition(split)
    subsets = [('all', stats, np.ones(n, bool))]
    for who in range(2):
        row = stats['direction_metrics'][who]
        metrics_from_counts(row, n//2, lam, label + '/direction')
        near(stats['direction_means'][who], row['mean_reward'], label + '/direction mean')
        subsets.append(('direction', row, np.repeat(np.arange(2), n//2) == who))
    for group, ids in (('seen', train), ('unseen', held)):
        row = stats['map_groups'][group]
        metrics_from_counts(row, n*len(ids)//30, lam, label + '/' + group)
        if actual is not None:
            pos = actual['positions']
            mids = pos[:, 0]*5+pos[:, 1]-(pos[:, 1] > pos[:, 0])
            subsets.append((group, row, np.isin(mids, ids)))
    require(stats['maps_were_withheld'] is (split != 0), label + ' split flag')
    for key in ('00', '01', '10', '11'):
        require(stats['outcome_counts'][key] == sum(d['outcome_counts'][key] for d in stats['direction_metrics']) ==
                sum(stats['map_groups'][g]['outcome_counts'][key] for g in ('seen', 'unseen')), label + ' aggregation')
    if actual is not None:
        resource = np.take_along_axis(actual['successes'], np.argsort(actual['goals'], axis=1), axis=1)
        codes = resource[:, 0].astype(int)*2 + resource[:, 1].astype(int)
        for name, row, mask in subsets:
            require(row['outcome_counts'] == {k: int((codes[mask] == i).sum())
                    for i, k in enumerate(('00', '01', '10', '11'))}, label + '/' + name + ' raw outcomes')


def audit_array(a, external, mode, lam, seed):
    n = len(a['reward'])
    sizes = dict(scout=(n,), episode=(n,), positions=(n, 2), photo_ids=(n, 2), goals=(n, 2),
                 menu=(n, 2, 6), inventory=(n, 2), history=(n, 18), sent=(n, 2), delivered=(n, 2),
                 action=(n, 2), place=(n, 2), successes=(n, 2), reward=(n,))
    require(set(a) == set(sizes), 'Array schema')
    for key, size in sizes.items():
        require(a[key].shape == size and np.isfinite(a[key]).all(), 'Array shape/finite ' + key)
        if key not in ('history', 'successes', 'reward'):
            require(np.issubdtype(a[key].dtype, np.integer), 'Integer field ' + key)
    require(np.array_equal(a['scout'], np.repeat(np.arange(2), n//2)) and
            np.array_equal(a['episode'], np.tile(np.arange(n//2), 2)), 'Exact direction/episode order')
    for key in WORLD_KEYS:
        require(np.array_equal(a[key], np.concatenate([r[key] for r in external])), 'Exogenous fixture: ' + key)
    require((a['inventory'] == 0).all() and (a['history'] == 0).all(), 'Receiver has extra state')
    require(all(((a[k] >= 0) & (a[k] < 7)).all() for k in ('sent', 'delivered')) and
            ((a['action'] >= 0) & (a['action'] < 6)).all(), 'Legal code/menu index')
    place = np.take_along_axis(a['menu'], a['action'][..., None], axis=2)[..., 0]
    success = place == np.take_along_axis(a['positions'], a['goals'], axis=1)
    reward = ((1-lam)*success.mean(1)+lam*success.all(1)).astype(np.float32)
    require(np.array_equal(place, a['place']) and np.array_equal(success, a['successes']) and
            np.array_equal(reward, a['reward']), 'Actual action physical settlement')
    delivered = a['sent'].copy()
    if mode == 'blank': delivered[:] = 0
    elif mode == 'shuffle':
        for scout in range(2):
            rng = np.random.default_rng(seed + 10001 + scout)
            for first in range(2):
                rows = np.flatnonzero((a['scout'] == scout) & (a['goals'][:, 0] == first))
                delivered[rows] = a['sent'][rng.permutation(rows)]
    require(np.array_equal(delivered, a['delivered']), 'Exact channel intervention')


def state_hash(states, selected='all'):
    h = hashlib.sha256()
    for who, state in enumerate(states):
        for key, value in sorted(state.items()):
            if selected == 'trainable' and (key.startswith('project.') or key == 'input_transform'): continue
            if selected == 'receiver' and not key.startswith(('receive_embedding.', 'actor.', 'receive_value.')): continue
            h.update(f'{who}/{key}'.encode())
            h.update(value.numpy().tobytes())
    return h.hexdigest()


def load_weights(path, counts):
    import torch
    obj = torch.load(path, weights_only=True, map_location='cpu')
    def visit(v):
        if isinstance(v, torch.Tensor):
            require(torch.isfinite(v).all().item(), 'Nonfinite tensor ' + str(path))
            counts['static_tensors'] += 1
        elif isinstance(v, dict):
            for x in v.values(): visit(x)
        elif isinstance(v, (list, tuple)):
            for x in v: visit(x)
        else: finite(v, str(path))
    visit(obj)
    counts['weight_files'] += 1
    return obj


def exact_states(a, b, label):
    require(len(a) == len(b), label + ' agents')
    for x, y in zip(a, b):
        require(x.keys() == y.keys() and all(np.array_equal(x[k].numpy(), y[k].numpy()) for k in x), label)


def audit_training(path, seed, split, lam, gain, pools, reference=None, updates=2400, batch=512):
    train, _ = partition(split)
    boundary = updates-min(300, updates//6)
    hashes, clip = [], 0
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    require(len(rows) == updates, 'Training update budget ' + str(path))
    for ix, r in enumerate(rows):
        finite(r, str(path))
        require(r['update'] == ix+1 and r['batch_identity'] == ix and r['active_sites'] == 6 and
                r['allowed_sites'] == list(range(6)) and r['map_pool'] == train and r['policy_gain'] == gain,
                'Training identity/access/gain')
        beta = .02 if ix < boundary else 0.
        require(r['entropy_weight'] == beta, 'Entropy schedule')
        wh = reference[ix] if reference is not None else world_sha(
            fixture_pair(seed*100000+60000000+ix+1, batch, pools, True, train))
        require(r['world_sha256'] == wh, 'Training external world hash')
        hashes.append(wh)
        c = r['outcome_counts']
        require(set(c) == {'00', '01', '10', '11'} and all(type(v) is int and v >= 0 for v in c.values())
                and sum(c.values()) == batch, 'Training outcome denominator')
        single = (c['01']+c['10']+2*c['11'])/(batch*2)
        both = c['11']/batch
        rew = (1-lam)*single+lam*both
        variance = sum(c[k]*(v-rew)**2 for k, v in zip(('00', '01', '10', '11'),
                          (0, (1-lam)/2, (1-lam)/2, 1)))/batch
        for key, val in (('single_accuracy', single), ('both_accuracy', both), ('reward', rew), ('reward_variance', variance)):
            near(r[key], val, key)
        require(r['positive_rewards'] == (batch-c['00'] if lam == 0 else c['11']), 'Positive rewards')
        require(len(r['agents']) == 2, 'Two independent agents')
        for agent in r['agents']:
            require(len(agent['components']) == 2 and agent['gradient_norm'] >= 0, 'Role loss/gradient')
            loss = []
            for comp in agent['components']:
                weighted = float(np.float32(gain)*np.float32(comp['policy_loss']))
                require(weighted == comp['weighted_policy_loss'] and comp['value_loss'] >= 0 and
                        0 <= comp['entropy'] <= 2*math.log(7)+1e-5, 'Applied gain/value/entropy')
                loss.append(weighted+comp['value_loss']-beta*comp['entropy'])
            near(agent['loss'], sum(loss)/2, 'Role-averaged combined loss',
                 tolerance=2e-6*(1+sum(abs(x) for x in loss)))
            clip += agent['gradient_norm'] > 2
    return hashes, dict(updates=updates, worlds=updates*batch, choices=updates*batch*2,
                       gradient_observations=updates*2, preclip_norm_above_two=clip)


def audit(batch):
    batch = Path(batch).resolve()
    status = read(batch/'status.json')
    expected = grid()
    require(status['status'] == 'complete' and status['completed_runs'] == status['expected_runs'] == 64,
            'Wait for completed 64-run status; do not audit partial data')
    manifest = read(batch/'manifest.json')
    require(manifest['seeds'] == list(SEEDS) and manifest['conditions'] == expected and
            manifest['updates'] == 2400 and manifest['batch'] == 512 and manifest['eval_n'] == 9600 and
            manifest['expected_social_runs'] == 64 and manifest['expected_social_updates'] == 153600 and
            manifest['expected_social_worlds'] == 78643200 and manifest['expected_social_choices'] == 157286400 and
            manifest['native_reward_changed_by_gain'] is False, 'Manifest design/budget')
    wanted = set(product(SEEDS, expected))
    indexed = {(r['seed'], r['condition']): r for r in status['runs']}
    require(len(status['runs']) == len(indexed) == 64 and set(indexed) == wanted, 'Status grid')
    require({p.name for p in batch.glob('s[0-9]*') if p.is_dir()} == {f's{s}_{c}' for s, c in wanted}, 'Run directory grid')
    require(not list(batch.rglob('*FAILED*')) and not (batch/'failure.json').exists(), 'Failure evidence present')
    hashes, counts = {}, Counter()
    def remember(path):
        path = Path(path).resolve()
        digest = sha(path); hashes[str(path)] = digest
        return digest
    for name in ('manifest.json', 'status.json', 'social_launch_gate.json', 'individual_controls/summary.json'):
        remember(batch/name)
    for path, digest in manifest['source_hashes'].items():
        src = Path(path).resolve()
        require(remember(src) == digest, 'Source SHA changed: ' + path)
        counts['frozen_source_files'] += 1
        if src.suffix in ('.py', '.md'):
            require(remember(batch/'source_snapshot'/src.relative_to(WORK)) == digest, 'Source snapshot changed')
            counts['source_snapshots'] += 1
    preflight_path = STUDY/'preflight_result.json'
    require(remember(preflight_path) == manifest['preflight_sha256'], 'Preflight receipt changed')
    preflight = read(preflight_path)
    require(preflight['status'] == 'passed' and preflight['runner_sha256'] == remember(STUDY/'runner.py') and
            preflight['micro_runs'] == 4 and preflight['total_training_updates'] == 8 and
            preflight['formal_study_runs_started'] == 0 and
            all(x['status'] == 'exact_match' for x in preflight['bridge_comparisons']), 'Preflight scope/status')
    require(read(Path(preflight['development_directory'])/'result.json') == preflight, 'Development receipt')
    development = Path(preflight['development_directory'])
    require(remember(development/'plan.json') == preflight['development_plan_sha256'] and
            remember(development/'runner.py') == preflight['runner_sha256'] and
            remember(development/'preflight.py') == preflight['preflight_sha256'] == sha(STUDY/'preflight.py'),
            'Development source/plan snapshots')
    for path in development.rglob('*'):
        if path.is_file(): remember(path)
    gate, personal = read(batch/'social_launch_gate.json'), read(batch/'individual_controls/summary.json')
    require(personal['complete'] is True and personal['passed'] is True and personal['runs'] == 4 and
            len(personal['results']) == 4 and {r['seed'] for r in personal['results']} == set(SEEDS) and
            gate['all_passed'] is True and gate['personal_runs'] == 4 and gate['social_runs_planned'] == 64 and
            gate['diagnostic_weights_transferred'] is False and
            gate['personal_summary_sha256'] == remember(batch/'individual_controls/summary.json'), 'Personal launch gate')
    pools = photo_pools()
    initial_by_seed, prep_by_seed = {}, {}
    for seed in SEEDS:
        prep_path = batch/f'prepared_{seed}.pt'; remember(prep_path)
        prep_by_seed[seed] = load_weights(prep_path, counts)
        rec = read(batch/f'preparation_{seed}.json'); remember(batch/f'preparation_{seed}.json')
        require(rec['seed'] == seed and len(rec['two_sites']) == len(rec['six_sites']) == 2 and
                all(x >= .9 for x in rec['six_sites']) and
                all(r['agent'] == i and r['updates'] == 200 and r['choices'] == 12800 and
                    r['heldout_need_sensitive_choice'] >= .9 for i, r in enumerate(rec['two_sites'])), 'Preparation gate')
        directory = batch/'individual_controls'/f's{seed}_identity'
        cfg, result = read(directory/'config.json'), read(directory/'result.json')
        finite(result, 'Personal result')
        require(result == next(r for r in personal['results'] if r['seed'] == seed) and
                result['complete'] is True and result['passed'] is True and
                result['frozen_unused_parameters_verified'] is True and cfg['updates'] == 2400 and
                cfg['batch'] == 512 and cfg['eval_n_per_agent'] == 9600 and cfg['gate'] == .9,
                'Individual diagnostic setup/result')
        require(cfg['prepared_source'] == dict(path=str(prep_path), sha256=sha(prep_path)), 'Personal prepared source')
        initial = load_weights(directory/'initial.pt', counts)
        final = load_weights(directory/'final.pt', counts)
        initial_by_seed[seed] = initial['agents']
        require(state_hash(initial['agents']) == cfg['initial_sha256'] and
                state_hash(initial['agents'], 'trainable') == cfg['common_initial_sha256'], 'Personal initial SHA')
        for who in range(2):
            for key, tensor in initial['agents'][who].items():
                if not key.startswith(('memory.', 'slot_phi.')):
                    require(np.array_equal(tensor.numpy(), final['agents'][who][key].numpy()), 'Diagnostic frozen tensors')
            require(any(not np.array_equal(t.numpy(), final['agents'][who][k].numpy())
                        for k, t in initial['agents'][who].items() if k.startswith(('memory.', 'slot_phi.'))),
                    'Diagnostic fitted parameters unexpectedly unchanged')
        for mode in ('normal', 'erase_memory'):
            with np.load(directory/f'final_{mode}.npz', allow_pickle=False) as saved:
                a = {k: saved[k] for k in saved.files}
            require(len(a['reward']) == 19200 and np.array_equal(a['agent'], np.repeat(np.arange(2), 9600)),
                    'Personal endpoint denominator')
            place = a['menu'][np.arange(19200), a['action']]
            reward = place == a['positions'][np.arange(19200), a['goals']]
            require(np.array_equal(place, a['place']) and np.array_equal(reward, a['reward']), 'Personal raw settlement')
            for who in range(2):
                correct = int(reward[a['agent'] == who].sum())
                require(result['scores'][mode]['per_agent'][who] == dict(correct=correct, n=9600, accuracy=correct/9600),
                        'Personal endpoint score')
                if mode == 'normal': require(correct*10 >= 9600*9, 'Personal observed gate')
            near(result['scores'][mode]['mean_accuracy'], float(reward.mean()), 'Personal mean')
        training = [json.loads(s) for s in (directory/'training.jsonl').read_text().splitlines()]
        require(len(training) == 2400 and [r['update'] for r in training] == list(range(1, 2401)), 'Personal training budget')
        finite(training, 'Personal training')
        for r in training:
            require([a['agent'] for a in r['agents']] == [0, 1] and
                    all(0 <= a['reward'] <= 1 and a['gradient_norm'] >= 0 for a in r['agents']), 'Personal training values')
        load_weights(directory/'final_optimizer.pt', counts)
        for path in directory.iterdir():
            if path.is_file(): remember(path)
        counts['personal_controls'] += 1
    references, rows = {}, []
    endpoint_worlds = {s: fixture_pair(s+69200000, 9600, pools) for s in SEEDS}
    curve_worlds = {s: world_sha(fixture_pair(s+69100000, 1200, pools)) for s in SEEDS}
    for seed in SEEDS:
        for condition, spec in expected.items():
            directory = batch/f's{seed}_{condition}'
            cfg, result = read(directory/'config.json'), read(directory/'result.json')
            split, lam, gain = spec['plan']['split'], spec['plan']['complementarity'], spec['policy_gain']
            for doc in (cfg, result):
                finite(doc, condition)
                require(doc['seed'] == seed and doc['condition'] == condition and doc['plan'] == spec['plan'] and
                        doc['policy_gain'] == gain and doc['updates'] == 2400 and doc['batch'] == 512, 'Run design')
            train, held = partition(split)
            require(cfg['checkpoints'] == list(CHECKPOINTS) and cfg['eval_n'] == 9600 and cfg['checkpoint_eval_n'] == 1200 and
                    cfg['sites'] == 6 and cfg['history_dim'] == 18 and cfg['train_map_ids'] == train and cfg['heldout_map_ids'] == held and
                    cfg['map_table'] == MAPS.tolist() and cfg['learning_rate'] == .0007 and cfg['entropy_coefficient'] == .02 and
                    cfg['entropy_off_after'] == 2100 and cfg['gamma'] == 1 and cfg['training_seed_offset'] == 60000000 and
                    cfg['choices_per_world'] == 2 and cfg['trace_schema'] == 'paired_world_v1', 'Run configuration')
            require(cfg['prepared_source'] == dict(path=str(batch/f'prepared_{seed}.pt'), sha256=sha(batch/f'prepared_{seed}.pt')),
                    'Social prepared source')
            for name, digest in cfg['source_hashes'].items():
                require(digest == manifest['source_hashes'][str(WORK/'redesign_v0.8'/name)], 'Social base source')
            initial = load_weights(directory/'initial.pt', counts)
            final = load_weights(directory/'final.pt', counts)
            exact_states(initial, initial_by_seed[seed], 'Social initial equals diagnostic initial, not fitted diagnostic final')
            for selected, key in (('all', 'initial_sha256'), ('trainable', 'trainable_initial_sha256'), ('receiver', 'receiver_initial_sha256')):
                require(state_hash(initial, selected) == cfg[key], 'Independent initial SHA')
            require(state_hash(final) == result['final_sha256'] == indexed[seed, condition]['final_sha256'] and
                    cfg['initial_sha256'] == result['initial_sha256'] and result['frozen_projection_verified'] is True,
                    'Independent final SHA/frozen claim')
            for who in range(2):
                require(cfg['trainable_parameters'][who] == sum(t.numel() for k, t in initial[who].items()
                        if not k.startswith('project.') and k != 'input_transform'), 'Trainable parameter count')
                for k, t in initial[who].items():
                    if k.startswith('project.'):
                        require(np.array_equal(t.numpy(), prep_by_seed[seed][who][k].numpy()), 'Private prepared projection')
                require(np.array_equal(initial[who]['input_transform'].numpy(), np.eye(6, dtype=np.float32)) and
                        cfg['input_transforms'][who] == np.eye(6).tolist(), 'Identity input transform')
            for update in CHECKPOINTS:
                state = load_weights(directory/f'checkpoint_{update:04d}.pt', counts)
                for who in range(2):
                    require(state[who].keys() == initial[who].keys(), 'Checkpoint schema')
                    for k, t in initial[who].items():
                        if k.startswith('project.') or k == 'input_transform':
                            require(np.array_equal(t.numpy(), state[who][k].numpy()) and
                                    np.array_equal(t.numpy(), final[who][k].numpy()), 'Frozen checkpoint tensor')
                if update == 0: exact_states(state, initial, 'Initial checkpoint')
                if update == 2400: exact_states(state, final, 'Final checkpoint')
                counts['social_checkpoints'] += 1
            opt = load_weights(directory/'final_optimizer.pt', counts)
            require(len(opt) == 2, 'Independent Adam states')
            for who, optimizer in enumerate(opt):
                params = [t for k, t in initial[who].items() if not k.startswith('project.') and k != 'input_transform']
                require(len(optimizer['param_groups']) == 1 and len(optimizer['state']) == len(params), 'Adam state completeness')
                group = optimizer['param_groups'][0]
                require(group['lr'] == .0007 and group['betas'] == (.9, .999) and group['eps'] == 1e-8 and
                        group['weight_decay'] == 0 and group['params'] == list(range(len(params))), 'Adam configuration')
                for entry in optimizer['state'].values():
                    require(entry['step'].item() == 2400 and (entry['exp_avg_sq'] >= 0).all().item(), 'Adam update count/moment')
            schedule = read(directory/'training_schedule.json')
            require(schedule['levels'] == [6]*2400 and schedule['batch_identities'] == list(range(2400)), 'Direct schedule')
            refkey = seed, split
            whashes, training = audit_training(directory/'training.jsonl', seed, split, lam, gain, pools, references.get(refkey))
            references.setdefault(refkey, whashes)
            curve = read(directory/'curve.json')
            require([r['update'] for r in curve] == list(CHECKPOINTS), 'Eight curve checkpoints')
            for record in curve:
                require(set(record['scores']) == set(MODES), 'Checkpoint mode inventory')
                for mode in MODES:
                    audit_score(record['scores'][mode], 1200, split, lam, curve_worlds[seed], 'Curve')
                    counts['checkpoint_score_records'] += 1
            require(set(result['scores']) == set(MODES), 'Final mode inventory')
            sent_reference = None
            for mode in MODES:
                with np.load(directory/f'final_{mode}.npz', allow_pickle=False) as saved:
                    a = {k: saved[k] for k in saved.files}
                require(len(a['reward']) == 9600, 'Final world budget')
                audit_array(a, endpoint_worlds[seed], mode, lam, seed+69200000)
                audit_score(result['scores'][mode], 9600, split, lam, world_sha(endpoint_worlds[seed]), 'Endpoint', a)
                if mode == 'normal': sent_reference = a['sent'].copy()
                elif mode in ('shuffle', 'blank'):
                    require(np.array_equal(a['sent'], sent_reference), 'Same greedy sender for channel interventions')
                counts['endpoint_arrays'] += 1; counts['endpoint_worlds'] += 9600
            for path in directory.iterdir():
                if path.is_file(): remember(path)
            counts['social_runs'] += 1
            rows.append(dict(seed=seed, condition=condition, initial_sha256=cfg['initial_sha256'],
                             final_sha256=result['final_sha256'], training=training,
                             endpoint_normal_J=result['scores']['normal']['both_accuracy']))
            print(f'AUDIT {counts["social_runs"]}/64 {directory.name}', flush=True)
    require(counts['social_runs'] == 64 and counts['endpoint_arrays'] == 320 and
            counts['social_checkpoints'] == 512 and len(references) == 16, 'Final audit budget')
    # Detect mutation while auditing, without reading or modifying a live trainer.
    require(all(sha(path) == digest for path, digest in hashes.items()), 'Source/result changed during audit')
    return dict(status='passed', completed_at=datetime.now(timezone.utc).isoformat(),
        batch=str(batch), audit_sha256=sha(__file__), counts=dict(counts), runs=rows,
        independent_fixture_reconstructions=38400, paired_four_cell_training_world_checks=153600,
        social_training_worlds=78643200, social_training_choices=157286400,
        source_and_record_sha256=hashes,
        limitations=['No model forward pass or rerun of training; raw endpoint actions are physically resettled.',
          'Checkpoint curves have no saved per-world action arrays: counts, aggregation, budgets and external-world hashes are checked.',
          'Training losses are checked against saved float32 components; gradients are checked for finiteness/counts, not recomputed.',
          'Private projection and independent initial state are verified from tensors and frozen source; runtime data-pointer assertions are not rerun.',
          'Formal runs remain four training seeds; splits, modes and checkpoints are within-seed repeated measurements.'])


def self_test():
    """Uses already saved two-update development records, never neural execution."""
    pools = photo_pools()
    base = STUDY/'development/preflight_001'
    for kind in ('additive', 'joint'):
        directory = base/('new_'+kind); result = read(directory/'result.json')
        lam = result['plan']['complementarity']
        audit_training(directory/'training.jsonl', 27101, 1, lam, 1, pools, updates=2, batch=16)
        for mode in MODES:
            with np.load(directory/f'final_{mode}.npz', allow_pickle=False) as saved:
                a = {k: saved[k] for k in saved.files}
            external = fixture_pair(27101+69200000, 120, pools)
            audit_array(a, external, mode, lam, 27101+69200000)
            audit_score(result['scores'][mode], 120, 1, lam, world_sha(external), 'Self test', a)
        bad = dict(a, reward=a['reward'].copy()); bad['reward'][0] += .25
        try: audit_array(bad, external, mode, lam, 27101+69200000)
        except AssertionError: pass
        else: raise AssertionError('Corrupt reward accepted')
    print('PASS: two recorded development runs, ten endpoint arrays, four training rows; two corrupt rewards rejected. No model imported.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=STUDY/'results/gain_001')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    out = args.out or args.batch/'independent_execution_audit'
    require(read(args.batch/'status.json')['status'] == 'complete', 'Training not complete')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        result = audit(args.batch)
        result['elapsed_seconds'] = time.monotonic()-started
        with (out/'verification.json').open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
        text = ('# 策略增益实验独立执行核验\n\n全部 64 组原始记录核验通过。没有训练、神经网络前向或改动原结果。\n\n'
          '- 核对来源与快照、预检、四组个人准备和门槛；社会初值等于个人诊断训练前初值，冻结私人投影保持。\n'
          '- 独立重建 38,400 个外生训练批次，核对同种子同划分四格的全部 153,600 个 world SHA。\n'
          '- 全部 512 个检查点及初终权重、优化器、损失记录、有限值和训练预算通过。\n'
          '- 全部 320 份终点数组、3,072,000 个世界从私人菜单实际动作重新结算；五模式和四格使用相同外生世界。\n\n'
          '核验边界：曲线检查点未保存逐世界动作，仅验证计数自洽、汇总、分母及外生世界哈希。训练损失从保存的分量核对，不重算反向梯度。'
          '结果分析仍以四个训练种子为比较单位；本审计不作语言形成或因果机制判断。\n')
        (out/'独立核验.md').write_text(text)
        print(json.dumps(dict(status='passed', output=str(out), counts=result['counts']), ensure_ascii=False))
    except BaseException as error:
        with (out/'failure.json').open('x') as f:
            json.dump(dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
                           automatic_retry=False), f, ensure_ascii=False, indent=2)
        raise


if __name__ == '__main__':
    main()
