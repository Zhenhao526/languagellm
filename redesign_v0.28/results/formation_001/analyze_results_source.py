"""NumPy-only, saved-raw analysis for the v28 formation replication.

No model, production metric/world module, images, or network is accessed. All
private evaluation and social protocol NPZs are rescored. Saved training traces
are checked for schema, world alignment, actions/rewards and logged hashes; this
does not replay unsaved training updates or establish model forward provenance.
Checkpoints and the run matrix come from invocation/config, including dev40.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MAPS = np.asarray(list(itertools.permutations(range(6), 2)), dtype=np.int64)
MATCHINGS = (((0, 1), (2, 3), (4, 5)), ((0, 2), (1, 4), (3, 5)),
             ((0, 3), (1, 5), (2, 4)))
WORLD_KEYS = ('map_id', 'photo_ids', 'positions', 'shown')
MASKS = ('pooled', 'food_only', 'water_only')
ARMS = ('old_old', 'all_old', 'all_all')
SCOPES = ('old', 'all')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def arrays_sha(arrays):
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class Audit:
    def __init__(self):
        self.checks = 0
        self.counts = Counter()
        self.files = {}
        self.max_metric_error = 0.0
        self.context = ''

    def require(self, value, label):
        self.checks += 1
        if not bool(value):
            raise ValueError(f'{self.context}: {label}')

    def exact(self, got, expected, label):
        if isinstance(got, np.ndarray) or isinstance(expected, np.ndarray):
            self.require(np.array_equal(got, expected), label)
        else:
            self.require(got == expected, label)

    def close(self, got, expected, label):
        if isinstance(expected, dict):
            self.exact(set(got), set(expected), label + ' keys')
            for key in expected:
                self.close(got[key], expected[key], label + '/' + key)
        elif isinstance(expected, list):
            self.exact(len(got), len(expected), label + ' length')
            for i, (a, b) in enumerate(zip(got, expected)):
                self.close(a, b, f'{label}/{i}')
        elif isinstance(expected, bool) or isinstance(expected, str) or expected is None:
            self.exact(got, expected, label)
        elif isinstance(expected, int):
            self.exact(got, expected, label)
        else:
            a, b = np.asarray(got), np.asarray(expected)
            self.exact(a.shape, b.shape, label + ' shape')
            self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + ' finite')
            error = float(np.max(np.abs(a - b))) if a.size else 0.0
            self.max_metric_error = max(self.max_metric_error, error)
            self.require(np.allclose(a, b, rtol=1e-9, atol=1e-10), label + ' numerical agreement')

    def load_npz(self, path, out, completion_files):
        self.context = str(path.relative_to(out))
        rel = str(path.relative_to(out))
        self.require(rel not in self.files, 'NPZ processed exactly once')
        self.require(rel in completion_files, 'NPZ present in training completion manifest')
        digest = sha(path)
        self.exact(digest, completion_files[rel], 'raw file SHA256')
        with np.load(path, allow_pickle=False) as f:
            self.exact(len(f.files), len(set(f.files)), 'unique NPZ keys')
            result = {key: f[key] for key in f.files}
        self.require(bool(result), 'nonempty NPZ')
        for key, a in result.items():
            self.require(a.dtype.kind in 'biuf', key + ' real numeric nonobject dtype')
            self.require(np.isfinite(a).all(), key + ' finite values')
        self.files[rel] = digest
        self.counts['npz_files'] += 1
        return result


def group_maps(partition):
    if partition not in (1, 2, 3):
        raise ValueError('partition must be 1, 2, or 3')
    pairs = lambda matching: {pair for a, b in matching for pair in ((a, b), (b, a))}
    added_pairs = pairs(MATCHINGS[partition - 1])
    sealed_pairs = pairs(MATCHINGS[partition % 3])
    added = np.asarray([i for i, m in enumerate(MAPS) if tuple(m) in added_pairs])
    sealed = np.asarray([i for i, m in enumerate(MAPS) if tuple(m) in sealed_pairs])
    new = np.concatenate((added, sealed))
    return dict(old=np.asarray([i for i in range(30) if i not in set(new)]),
                added=added, sealed=sealed, common30=np.arange(30), new12=new)


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    exp = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def score(raw, partition, private):
    """Greedy J and exact mathematical categorical Q, independently rescored.

    Social J uses saved sequential-greedy tokens, NEVER joint-table argmax.
    Shuffle draws one complete greedy code from the entire pooled test support,
    independently of the target; it does not rebuild a histogram per subgroup.
    """
    pos = raw['positions']
    n = len(pos)
    extra = {}
    if private:
        actions = np.argmax(raw['logits'], axis=2)
        prob = softmax(raw['logits'])
        q = prob[np.arange(n), 0, pos[:, 0]] * prob[np.arange(n), 1, pos[:, 1]]
    else:
        codes = raw['tokens'][:, 0] * 7 + raw['tokens'][:, 1]
        decoder = np.argmax(raw['receiver_logits'], axis=2)
        actions = decoder[codes]
        sender, receiver = softmax(raw['sender_log_probs']), softmax(raw['receiver_logits'])
        q = np.asarray([math.fsum(float(sender[i, code] * receiver[code, 0, f]
                                           * receiver[code, 1, w]) for code in range(49))
                        for i, (f, w) in enumerate(pos)])
        frequencies = np.bincount(codes, minlength=49) / n
        extra['blank_J'] = np.all(decoder[0] == pos, axis=1).astype(float)
        extra['shuffle_J'] = np.asarray([sum(float(frequencies[code]) for code in range(49)
                                            if tuple(decoder[code]) == tuple(target)) for target in pos])
    correct = actions == pos
    result = {}
    for name, maps in group_maps(partition).items():
        selected = np.isin(raw['map_id'], maps)
        result[name] = {}
        for label, mask in (('pooled', selected), ('food_only', selected & (raw['shown'] == 0)),
                            ('water_only', selected & (raw['shown'] == 1))):
            ix = np.flatnonzero(mask)
            count = len(ix)
            if not count:
                raise ValueError(f'empty evaluation group {name}/{label}')
            good = correct[ix]
            metrics = dict(n=count, J=int(np.all(good, axis=1).sum()) / count,
                           food=int(good[:, 0].sum()) / count,
                           water=int(good[:, 1].sum()) / count,
                           Q=math.fsum(float(x) for x in q[ix]) / count)
            metrics.update({key: math.fsum(float(x) for x in values[ix]) / count
                            for key, values in extra.items()})
            result[name][label] = metrics
    return result


def integer_array(audit, a, shape, low, high, label):
    audit.exact(a.shape, shape, label + ' shape')
    audit.require(a.dtype.kind in 'iu', label + ' integer dtype')
    audit.require(((a >= low) & (a < high)).all(), label + ' domain')


def check_world(audit, raw, expected, label):
    for key in WORLD_KEYS:
        audit.require(key in raw, label + '/' + key + ' required')
        audit.exact(raw[key].dtype, expected[key].dtype, label + '/' + key + ' dtype')
        audit.exact(raw[key], expected[key], label + '/' + key + ' complete world/order')


def reconstruct_tables(pools):
    """Independent rectangular Cartesian table, no sqrt/support assumption."""
    result = {}
    for split in ('train', 'test'):
        food, water = (sorted(pools[f'{split}_{k}']) for k in (0, 1))
        rows = [(mid, f, w, shown) for shown in (0, 1) for mid in range(30)
                for f in food for w in water]
        mids = np.asarray([r[0] for r in rows], dtype=np.int64)
        result[split] = dict(map_id=mids, positions=MAPS[mids],
                             photo_ids=np.asarray([[r[1], r[2]] for r in rows], dtype=np.int64),
                             shown=np.asarray([r[3] for r in rows], dtype=np.int64))
    return result


def validate_protocol(audit, raw, world, private):
    required = set(WORLD_KEYS) | ({'h', 'logits'} if private else
                                 {'sender_log_probs', 'tokens', 'receiver_logits'})
    audit.exact(set(raw), required, 'evaluation schema')
    check_world(audit, raw, world, 'evaluation')
    n = len(world['map_id'])
    if private:
        audit.exact(raw['h'].shape, (n, 96), 'private h shape')
        audit.exact(raw['logits'].shape, (n, 2, 6), 'private logits shape')
        floats = ('h', 'logits')
    else:
        integer_array(audit, raw['tokens'], (n, 2), 0, 7, 'saved sequential greedy tokens')
        audit.exact(raw['sender_log_probs'].shape, (n, 49), 'all message log probabilities')
        audit.exact(raw['receiver_logits'].shape, (49, 2, 6), 'complete receiver table')
        normalization = np.exp(raw['sender_log_probs'].astype(np.float64)).sum(1)
        audit.require(np.max(np.abs(normalization - 1)) < 2e-6, 'sender complete-code normalization')
        # A tolerance is needed when recovering prefix marginals from float32
        # sums. Exact first-index ties and neural provenance are forward-audit work.
        sp = softmax(raw['sender_log_probs']).reshape(n, 7, 7)
        first = sp.sum(2)
        token = raw['tokens']
        audit.require(np.max(first.max(1) - first[np.arange(n), token[:, 0]]) < 2e-6,
                      'saved first token compatible with prefix-marginal greedy')
        conditional = raw['sender_log_probs'].reshape(n, 7, 7)[np.arange(n), token[:, 0]]
        audit.require(np.max(conditional.max(1) - conditional[np.arange(n), token[:, 1]]) < 2e-6,
                      'saved second token compatible with chosen-prefix greedy')
        floats = ('sender_log_probs', 'receiver_logits')
    for key in floats:
        audit.exact(raw[key].dtype, np.dtype('float32'), key + ' float32')


def numeric_finite(value):
    if isinstance(value, dict):
        return all(numeric_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(numeric_finite(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def check_train_trace(audit, raw, private, scope, part, table, logrow):
    worlds = {k[7:]: v for k, v in raw.items() if k.startswith('world__')}
    traces = {k[7:]: v for k, v in raw.items() if k.startswith('trace__')}
    audit.exact(set(raw), {'world__' + k for k in worlds} | {'trace__' + k for k in traces},
                'training trace namespace')
    audit.exact(arrays_sha(worlds), logrow['world_sha256'], 'saved fixture log hash')
    audit.exact(arrays_sha(traces), logrow['trace_sha256'], 'saved trace log hash')
    audit.exact(set(worlds), ({'indices', 'uniforms', 'goals'} if private else
                            {f'd{d}__{k}' for d in (0, 1) for k in ('indices', 'uniforms', 'goals')}),
                'saved fixture required schema')
    for direction in ((None,) if private else (0, 1)):
        prefix = '' if private else f'd{direction}__'
        f = {k[len(prefix):]: v for k, v in worlds.items() if k.startswith(prefix)}
        t = {k[len(prefix):]: v for k, v in traces.items() if k.startswith(prefix)}
        n = 512 if private else 256
        integer_array(audit, f['indices'], (n,), 0, len(table['map_id']), 'training indices')
        integer_array(audit, f['goals'], (n,), 0, 2, 'training goals')
        audit.exact(f['uniforms'].shape, (n, 1 if private else 4), 'training uniform shape')
        audit.exact(f['uniforms'].dtype, np.dtype('float32'), 'training uniform dtype')
        audit.require(((f['uniforms'] >= 0) & (f['uniforms'] <= 1)).all(), 'uniform domain')
        ix = f['indices']
        audit.exact(ix[n // 2:], ix[:n // 2] + len(table['map_id']) // 2, 'paired masks share world')
        audit.exact(f['goals'][:n // 2], f['goals'][n // 2:], 'paired masks share goal')
        mids = table['map_id'][ix]
        allowed = group_maps(part)['old'] if scope == 'old' else np.arange(30)
        audit.require(np.isin(mids, allowed).all(), 'saved training support')
        pos = table['positions'][ix]
        if private:
            required = {'h0', 'raw_h1', 'hfinal', 'full_logits', 'goalselected_logits',
                        'selected_probabilities', 'action_uniform', 'action', 'reward', 'selected_target'}
            audit.exact(set(t), required, 'private training trace schema')
            for key in ('h0', 'raw_h1', 'hfinal'):
                audit.exact(t[key].shape, (n, 96), key + ' training state shape')
            audit.exact(t['full_logits'].shape, (n, 2, 6), 'training full private logits')
            audit.exact(t['goalselected_logits'], t['full_logits'][np.arange(n), f['goals']], 'selected logits')
            audit.exact(t['selected_probabilities'].shape, (n, 6), 'private action probabilities')
            audit.exact(t['action_uniform'], f['uniforms'], 'private uniforms retained')
            integer_array(audit, t['action'], (n,), 0, 6, 'private sampled action')
            target = pos[np.arange(n), f['goals']]
            audit.exact(t['selected_target'], target, 'private selected target from actual world')
            reward = (t['action'] == target).astype(np.float32)
            audit.exact(t['reward'], reward, 'private reward independently settled')
            audit.close(logrow['components']['mean_reward'], float(reward.mean()), 'private logged reward')
            audit.counts['saved_private_training_actions_settled'] += n
        else:
            required = {'h', 'uniforms', 'messages', 'first_logits', 'second_logits', 'token_probabilities',
                        'action_logits', 'action_probabilities', 'actions', 'positions', 'success', 'reward',
                        'advantage', 'sender_logp', 'receiver_logp', 'sender_entropy', 'receiver_entropy',
                        'sender_loss', 'receiver_loss', 'entropy_weight', 'baseline'}
            audit.exact(set(t), required, 'social training trace schema')
            audit.exact(t['h'].shape, (n, 96), 'social frozen state trace')
            audit.exact(t['uniforms'], f['uniforms'], 'social uniforms retained')
            integer_array(audit, t['messages'], (n, 2), 0, 7, 'sampled message')
            integer_array(audit, t['actions'], (n, 2), 0, 6, 'sampled actions')
            audit.exact(t['positions'], pos, 'social target from actual world')
            for key, shape in (('first_logits', (n, 7)), ('second_logits', (n, 7)),
                               ('token_probabilities', (n, 2, 7)), ('action_logits', (n, 2, 6)),
                               ('action_probabilities', (n, 2, 6))):
                audit.exact(t[key].shape, shape, key + ' shape')
            success = (t['actions'] == pos).astype(np.float32)
            reward = .25 * success.sum(1) + .5 * success.prod(1)
            audit.exact(t['success'], success, 'social successes independently settled')
            audit.exact(t['reward'], reward, 'mixed team reward independently settled')
            audit.exact(t['baseline'], np.asarray(.5), 'fixed baseline')
            audit.exact(t['advantage'], reward - .5, 'reward minus baseline')
            audit.counts['saved_social_training_worlds_settled'] += n
            audit.counts['saved_social_training_actions_settled'] += n * 2
        probs = (t['selected_probabilities'],) if private else (t['token_probabilities'], t['action_probabilities'])
        for prob in probs:
            audit.require(((prob >= 0) & (prob <= 1)).all(), 'saved training probability range')
            audit.require(np.max(np.abs(prob.astype(np.float64).sum(-1) - 1)) < 2e-6,
                          'saved training probability normalization')


def average(values):
    first = values[0]
    if isinstance(first, dict):
        return {key: average([v[key] for v in values]) for key in first}
    return math.fsum(float(v) for v in values) / len(values)


def auc(points):
    times = [p['update'] for p in points]
    def integrate(values):
        if isinstance(values[0], dict):
            return {key: integrate([v[key] for v in values]) for key in values[0] if key != 'n'}
        return math.fsum((times[i + 1] - times[i]) * (values[i + 1] + values[i]) / 2
                         for i in range(len(times) - 1)) / (times[-1] - times[0])
    return integrate([p['scores'] for p in points])


def summarize(rows, seeds, parts, times):
    source_rows = []
    conditions = ('private_old', 'private_all') + ARMS
    for seed, condition in itertools.product(seeds, conditions):
        selected = [r for r in rows if r['seed'] == seed and r['condition'] == condition]
        if len(selected) != len(parts) * 2:
            raise ValueError('each source cell must include every partition and both directions')
        curve = [dict(update=t, scores=average([r['curve'][i]['scores'] for r in selected]))
                 for i, t in enumerate(times)]
        source_rows.append(dict(seed=seed, condition=condition, partitions=len(parts), directions=2,
                                curve=curve, scores=curve[-1]['scores'], auc=auc(curve)))
    aggregate = {}
    for condition in conditions:
        selected = [r for r in source_rows if r['condition'] == condition]
        curve = [dict(update=t, scores=average([r['curve'][i]['scores'] for r in selected]))
                 for i, t in enumerate(times)]
        aggregate[condition] = dict(independent_sources=len(seeds), curve=curve,
                                    scores=curve[-1]['scores'], auc=auc(curve))
    primary, capability = [], []
    for seed in seeds:
        values = {r['condition']: r for r in source_rows if r['seed'] == seed}
        get = lambda c, endpoint: values[c][endpoint]['new12']['pooled']['J']
        primary.append(dict(seed=seed, A=get('old_old', 'scores'), B=get('all_old', 'scores'),
                            C=get('all_all', 'scores'), difference=get('all_old', 'scores') - get('old_old', 'scores'),
                            auc_difference=get('all_old', 'auc') - get('old_old', 'auc')))
        private = values['private_all']['scores']['new12']
        mask_j = {key: private[key]['J'] for key in MASKS[1:]}
        capability.append(dict(seed=seed, mask_J=mask_j, passed=all(v >= .8 for v in mask_j.values())))
    differences = [r['difference'] for r in primary]
    return dict(rows=rows, seed_rows=source_rows, aggregate=aggregate, primary=primary,
                primary_mean=math.fsum(differences) / len(seeds),
                primary_range=[min(differences), max(differences)],
                primary_auc_mean=math.fsum(r['auc_difference'] for r in primary) / len(seeds),
                capability=dict(passed=all(r['passed'] for r in capability), source_rows=capability,
                                threshold=.8, filter_applied=False, social_matrix_proceeds_regardless=True))


def analyze(out, audit):
    invocation = read(out / 'invocation.json')
    completion = read(out / 'training_complete.json')
    audit.context = 'batch metadata'
    audit.exact(completion['status'], 'complete', 'training must be complete before analysis')
    seeds, parts, updates = invocation['seeds'], invocation['partitions'], invocation['updates']
    audit.require(len(seeds) > 0 and len(seeds) == len(set(seeds)), 'distinct source seeds')
    audit.require(len(parts) > 0 and len(parts) == len(set(parts)) and set(parts) <= {1, 2, 3}, 'partitions')
    audit.require(isinstance(updates, int) and updates > 0, 'positive declared budget')
    audit.exact(invocation['private_scopes'], list(SCOPES), 'complete private scopes')
    audit.exact(invocation['social_arms'], list(ARMS), 'complete A/B/C arms')
    audit.exact(completion['formal'], invocation['formal'], 'formal flag')
    for key in ('source_hashes', 'input_hashes'):
        audit.exact(completion[key], invocation[key], 'launch/completion ' + key)
        for path, expected_sha in invocation[key].items():
            audit.exact(sha(path), expected_sha, key + ' unchanged current bytes')
            if key == 'source_hashes':
                frozen = out / 'frozen_sources' / Path(path).relative_to(PROJECT)
                audit.exact(sha(frozen), expected_sha, 'frozen source archive bytes')
    files = completion['files']
    # All saved numeric arrays, including caches, must retain their original
    # completion-manifest hashes. Checkpoint tensor/forward checks are external.
    expected_npz = {key for key in files if key.endswith('.npz')}
    actual_npz = {str(path.relative_to(out)) for path in out.rglob('*.npz')}
    audit.exact(actual_npz, expected_npz, 'all NPZ inventory, no missing or extra raw')
    for relative, expected_sha in files.items():
        if Path(relative).suffix in ('.json', '.jsonl', '.npy'):
            audit.exact(sha(out / relative), expected_sha, 'metadata/cache SHA256 ' + relative)
            if relative.endswith('.npy'):
                a = np.load(out / relative, allow_pickle=False)
                split = Path(relative).name.split('_')[0]
                expected_n = invocation[split + '_worlds']
                audit.exact(a.shape, (expected_n, 96), 'all encoded cache shapes')
                audit.exact(a.dtype, np.dtype('float32'), 'all encoded cache dtype')
                audit.require(np.isfinite(a).all(), 'all encoded cache finite')
                audit.counts['cache_arrays_checked'] += 1
    pools = read(out / 'image_pools.json')
    audit.exact(set(pools), {'train_0', 'train_1', 'test_0', 'test_1'}, 'photo pool keys')
    audit.exact([len(pools[k]) for k in ('train_0', 'train_1', 'test_0', 'test_1')], [6, 2, 3, 1],
                'fixed rectangular photo support')
    ids = [i for values in pools.values() for i in values]
    audit.require(all(isinstance(i, int) for i in ids), 'integer photo IDs')
    audit.exact(sorted(ids), list(range(12)), 'disjoint resource/train/test photo IDs')
    tables = reconstruct_tables(pools)
    for split in ('train', 'test'):
        raw = audit.load_npz(out / f'{split}_worlds.npz', out, files)
        audit.exact(set(raw), set(WORLD_KEYS), 'world table schema')
        check_world(audit, raw, tables[split], split + ' independently constructed world')
        audit.exact(len(raw['map_id']), invocation[split + '_worlds'], split + ' row count')
    normalization = audit.load_npz(out / 'normalization.npz', out, files)
    audit.exact(set(normalization), {'center', 'scale'}, 'normalization schema')
    audit.exact(normalization['center'].shape, (1024,), 'normalization mean width')
    audit.exact(normalization['center'].dtype, np.dtype('float32'), 'normalization mean dtype')
    audit.exact(normalization['scale'].shape, (), 'normalization scalar scale')
    audit.require(float(normalization['scale']) > 0, 'positive normalization scale')
    descriptors = []
    for seed, p, d, scope in itertools.product(seeds, parts, (0, 1), SCOPES):
        descriptors.append(('private', f's{seed}_p{p}_d{d}_{scope}', seed, p, d, scope))
    for seed, p, arm in itertools.product(seeds, parts, ARMS):
        descriptors.append(('social', f's{seed}_p{p}_{arm}', seed, p, None, arm))
    for phase in ('private', 'social'):
        audit.exact({p.name for p in (out / phase).iterdir() if p.is_dir()},
                    {name for ph, name, *_ in descriptors if ph == phase}, phase + ' exact run matrix')
    rows, common_times = [], None
    for phase, name, seed, part, direction, condition in descriptors:
        folder = out / phase / name
        audit.context = str(folder.relative_to(out))
        cfg, curve, result = (read(folder / key) for key in ('config.json', 'curve.json', 'result.json'))
        audit.exact((cfg['seed'], cfg['partition'], cfg['updates']), (seed, part, updates), 'run identity/budget')
        private = phase == 'private'
        if private:
            audit.exact((cfg['direction'], cfg['scope']), (direction, condition), 'private identity')
            audit.exact(result['selected_goal_actions'], updates * 512, 'private budget units')
        else:
            audit.exact(cfg['arm'], condition, 'social arm')
            audit.exact((result['messages'], result['actions']), (updates * 512, updates * 1024), 'social budget units')
        audit.exact(result['status'], 'complete', 'run completed')
        audit.exact(result['updates'], updates, 'result update count')
        audit.exact(result['frozen_verified'], True, 'producer frozen flag, external audit still required')
        times = cfg['checkpoints']
        audit.require(len(times) >= 2 and times == sorted(set(times)) and times[0] == 0 and times[-1] == updates,
                      'ordered complete checkpoint schedule')
        audit.exact([r['update'] for r in curve], times, 'curve/config checkpoint agreement')
        if common_times is None:
            common_times = times
        audit.exact(times, common_times, 'paired schedules identical')
        logs = [json.loads(line) for line in (folder / 'training.jsonl').read_text().splitlines()]
        audit.exact([r['update'] for r in logs], list(range(1, updates + 1)), 'complete ordered training log')
        for log in logs:
            audit.require(numeric_finite(log), 'finite logged numbers')
            audit.require(all(isinstance(log[k], str) and len(log[k]) == 64
                              and set(log[k]) <= set('0123456789abcdef')
                              for k in ('world_sha256', 'trace_sha256')), 'logged hash syntax')
        audit.counts[phase + '_logged_updates_checked'] += len(logs)
        per_direction = {d: [] for d in ([direction] if private else (0, 1))}
        expected_local_npz = set()
        for point in curve:
            step = point['update']
            got_scores = []
            for d in per_direction:
                filename = f'evaluation_{step:04d}.npz' if private else f'protocol_{step:04d}_d{d}.npz'
                expected_local_npz.add(filename)
                raw = audit.load_npz(folder / filename, out, files)
                validate_protocol(audit, raw, tables['test'], private)
                got = score(raw, part, private)
                saved = point['scores'] if private else point['scores'][d]
                audit.close(got, saved, 'independent evaluation metrics')
                per_direction[d].append(dict(update=step, scores=got))
                got_scores.append(got)
                audit.counts[phase + '_evaluation_tables'] += 1
                audit.counts[phase + '_evaluation_world_rows'] += len(raw['map_id'])
                if step == updates:
                    audit.counts[phase + '_endpoint_world_rows'] += len(raw['map_id'])
            if step == updates:
                audit.close(got_scores[0] if private else got_scores, result['scores'], 'endpoint result metrics')
        for d, points in per_direction.items():
            rows.append(dict(seed=seed, partition=part, direction=d,
                             condition='private_' + condition if private else condition,
                             raw_folder=str(folder), curve=points, scores=points[-1]['scores']))
        for step in (1, 2101):
            if step > updates:
                continue
            name = f'train_{step:04d}.npz'
            expected_local_npz.add(name)
            raw = audit.load_npz(folder / name, out, files)
            scope = condition if private else condition.split('_')[1]
            check_train_trace(audit, raw, private, scope, part, tables['train'], logs[step - 1])
            audit.counts[phase + '_training_trace_files'] += 1
        audit.exact({p.name for p in folder.glob('*.npz')}, expected_local_npz, 'run raw inventory matches checkpoint/trace schedule')
        audit.counts[phase + '_fits'] += 1
    audit.exact(set(audit.files), expected_npz, 'every saved NPZ parsed and checked exactly once')
    summary = summarize(rows, seeds, parts, common_times)
    audit.close(summary['capability'], read(out / 'private_applicability.json'), 'private gate independently recomputed without filtering')
    private_fits, social_runs = len(seeds) * len(parts) * 4, len(seeds) * len(parts) * 3
    budgets = dict(private_fits=private_fits, social_runs=social_runs, private_updates=private_fits * updates,
                   private_actions=private_fits * updates * 512, pair_updates=social_runs * updates,
                   messages=social_runs * updates * 512, actions=social_runs * updates * 1024,
                   initial_preparation_updates=len(seeds) * 400,
                   initial_preparation_actions=len(seeds) * 400 * 64)
    for key, expected in budgets.items():
        audit.exact(completion[key], expected, 'complete matrix budget/' + key)
    summary.update(status='complete', formal=invocation['formal'], seeds=seeds, partitions=parts,
                   times=common_times, checkpoint_times=common_times, endpoint_update=updates, independent_sources=len(seeds),
                   budgets=budgets, evaluation_support=dict(train_worlds=len(tables['train']['map_id']),
                   test_worlds=len(tables['test']['map_id']), train_photo_pairs=12, test_photo_pairs=3),
                   definitions=dict(J='Saved sequential greedy message followed by greedy two-goal receiver; both goals correct.',
                   Q='Float64 normalized mathematical categorical policy: sum over 49 whole codes of sender probability times both receiver target probabilities; not an exact PRNG-bin simulator.',
                   blank_J='Receiver actions for constant complete code (0,0).',
                   shuffle_J='Independent complete greedy code draw from pooled test all30 x both masks x all test photo pairs. Same global histogram for every reported subgroup.',
                   aggregation='Equal means: both masks within pooled score; 3 partitions x 2 people/directions within source; source means across 4 sources (actual dev matrix from invocation). Rows/photos/partitions/directions are not independent seeds.',
                   n='Within one evaluation-table support; averaging n does not create additional independent observations.',
                   AUC='Trapezoidal mean over saved config checkpoints, divided by endpoint minus initial update; auxiliary only.',
                   gate='Each private_all source must reach new12 J >= .80 in each mask after averaging partitions and both people; no filtering or additional training.'),
                   scope='Development formation replication on a fixed small image pool; no untouched independent material confirmation or cross-round single-factor causal claim.')
    return summary


def self_test():
    """Analytic known outcomes, non-square support and sequential-greedy trap."""
    audit = Audit()
    pools = dict(train_0=list(range(6)), train_1=[6, 7], test_0=[8, 9, 10], test_1=[11])
    tables = reconstruct_tables(pools)
    audit.exact([len(tables[k]['map_id']) for k in ('train', 'test')], [720, 180], 'rectangular table sizes')
    world = tables['test']; n = len(world['map_id'])
    private = dict(world, h=np.zeros((n, 96), np.float32), logits=np.zeros((n, 2, 6), np.float32))
    uniform = score(private, 1, True)
    audit.close(uniform['common30']['pooled']['Q'], 1 / 36, 'uniform private Q')
    audit.exact(uniform['common30']['pooled']['J'], 0.0, 'same-place greedy actions cannot solve distinct resources')
    for i, (f, w) in enumerate(world['positions']):
        private['logits'][i, 0, f] = 1000
        private['logits'][i, 1, w] = 1000
    audit.exact(score(private, 1, True)['common30']['pooled']['J'], 1.0, 'perfect private J')
    code = world['map_id']
    logits = np.full((49, 2, 6), -1000., dtype=np.float32)
    for i, (f, w) in enumerate(MAPS):
        logits[i, 0, f] = 0
        logits[i, 1, w] = 0
    logits[30:, :, 0] = 0
    lp = np.full((n, 49), -1000., dtype=np.float32)
    lp[np.arange(n), code] = 0
    raw = dict(world, tokens=np.stack((code // 7, code % 7), 1), receiver_logits=logits, sender_log_probs=lp)
    scores = score(raw, 1, False)
    audit.exact(scores['common30']['pooled']['J'], 1.0, 'perfect whole-map greedy J')
    audit.exact(scores['common30']['pooled']['Q'], 1.0, 'perfect whole-map categorical Q')
    audit.close(scores['common30']['pooled']['shuffle_J'], 1 / 30, 'global whole-map shuffle')
    audit.close(scores['common30']['pooled']['blank_J'], 1 / 30, 'constant legal map code')
    # Prefix0 has mass .6 spread over seven continuations; prefix1 has .39
    # almost entirely on code7. Greedy tokens00 differ from joint argmax7.
    prob = np.full(49, .01 / 35)
    prob[:7] = .6 / 7
    prob[7:14] = .39 / 1000 / 6
    prob[7] = .39 * 999 / 1000
    raw['sender_log_probs'] = np.tile(np.log(prob).astype(np.float32), (n, 1))
    raw['tokens'] = np.zeros((n, 2), np.int64)
    validate_protocol(audit, raw, world, False)
    audit.exact(int(raw['sender_log_probs'][0].argmax()), 7, 'joint argmax differs in toy')
    actual = score(raw, 1, False)
    target0 = world['map_id'] == 0
    subset_j = actual['added']['pooled']['J']
    audit.close(subset_j, float(target0[np.isin(world['map_id'], group_maps(1)['added'])].mean()), 'score retains sequential greedy code00')
    corrupt = dict(raw, tokens=np.full((n, 2), 7, np.int64))
    try:
        validate_protocol(Audit(), corrupt, world, False)
    except ValueError:
        audit.require(True, 'invalid token rejected')
    else:
        raise AssertionError('invalid token not rejected')
    points = [dict(update=t, scores={'J': v, 'n': 180}) for t, v in ((0, 0.), (1, 1.), (4, 1.))]
    audit.close(auc(points)['J'], .875, 'irregular checkpoint AUC')
    audit.exact(set(auc(points)), {'J'}, 'support counts are not AUC metrics')
    return dict(passed=True, checks=audit.checks, source_sha256=sha(__file__), model_calls=0,
                analytical_cases=['rectangular720/180', 'uniformQ1/36', 'perfectwholemapJQ1',
                                  'completecodeshuffle1/30', 'sequential_vs_joint_argmax', 'invalid_token', 'irregular_AUC'])


def markdown(summary, audit):
    lines = ['# v0.28 独立原始表统计', '',
             '本文件仅汇总保存的评价与训练证据，不加载模型或图片。完整模型前向与训练配置核验由另一个审计提供。', '',
             f"实际矩阵：{len(summary['seeds'])} 个来源，划分 {summary['partitions']}，终点 {summary['endpoint_update']} 更新。主量为测试 new12 的 B（all_old）减 A（old_old）自然双目标 J。", '',
             '| 来源 | A J (%) | B J (%) | C J (%) | B−A (百分点) | B−A AUC (百分点) |',
             '|---|---:|---:|---:|---:|---:|']
    for r in summary['primary']:
        lines.append(f"| {r['seed']} | {r['A']*100:.4f} | {r['B']*100:.4f} | {r['C']*100:.4f} | {r['difference']*100:+.4f} | {r['auc_difference']*100:+.4f} |")
    lines += ['', f"主均值差 {summary['primary_mean']*100:+.4f} 个百分点；范围 {summary['primary_range'][0]*100:+.4f} 至 {summary['primary_range'][1]*100:+.4f}。",
              '', '所有臂、old18/added6/sealed6/new12/common30、两种末帧 mask、J/Q、常码与打乱参照及全部过程值见 analysis.json。来源内先等权平均划分和方向，再平均来源；不把图片、世界行或方向当独立重复。', '',
              f"私人 all 的 new12 两 mask 80%门槛：{'全部来源通过' if summary['capability']['passed'] else '未全部通过'}；未筛除来源，完整社会矩阵保留。", '',
              'Q 对全部49码按数学 categorical 概率求期望；J 使用保存的逐 token 贪心消息，不能换成整码概率 argmax。shuffle_J 从完整测试表的贪心整码频率独立抽取，其全局频率对各子组相同；blank_J 固定为00码。', '',
              f"原始 NPZ 全部解析且哈希一致：{audit.counts['npz_files']} 份；私人评价 {audit.counts['private_evaluation_world_rows']} 行，社会评价 {audit.counts['social_evaluation_world_rows']} 行（社会每行两种需求行动）。全部保存检查点均重算，生产摘要最大绝对差 {audit.max_metric_error:.3g}。", '',
              '已保存训练片段按实际索引重结算动作后果并核对日志哈希；未重播其余训练梯度、优化器或模型前向。该小图池属于完整形成开发复现，不能据此宣布独立材料确认、领域创新或语言知识的必要性。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    if args.out is None:
        parser.error('--out is required unless --self-test is used')
    out = args.out.resolve()
    if not (out / 'training_complete.json').exists():
        parser.error('training_complete.json is required; no interim scientific analysis is performed')
    names = ('analysis.json', 'raw_validation.json', '独立统计摘要.md', 'analyze_results_source.py')
    old = [out / name for name in names if (out / name).exists()]
    if old:
        history = out / 'analysis_history' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        history.mkdir(parents=True)
        for path in old:
            shutil.copy2(path, history / path.name)
    shutil.copy2(__file__, out / 'analyze_results_source.py')
    audit = Audit(); started = time.monotonic()
    validation = dict(passed=False, created_utc=datetime.now(timezone.utc).isoformat(),
                      analysis_source_sha256=sha(__file__), invocation_sha256=sha(out / 'invocation.json'),
                      training_complete_sha256=sha(out / 'training_complete.json'), production_modules_imported=False,
                      model_calls=0, network_calls=0, pixel_reads=0,
                      bounds='All saved raw tables and their statistics; no model-forward or complete-training replay claim.')
    try:
        summary = analyze(out, audit)
        summary['analysis_source_sha256'] = validation['analysis_source_sha256']
        summary['training_complete_sha256'] = validation['training_complete_sha256']
        write(out / 'analysis.json', summary)
        (out / '独立统计摘要.md').write_text(markdown(summary, audit))
        validation.update(passed=True, analysis_sha256=sha(out / 'analysis.json'))
    except Exception as exc:
        validation.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        validation.update(checks=audit.checks, counts=dict(audit.counts), checked_raw_sha256=audit.files,
                          maximum_metric_absolute_difference=audit.max_metric_error,
                          elapsed_seconds=time.monotonic() - started)
        write(out / 'raw_validation.json', validation)
    print(json.dumps(dict(status='complete', checks=audit.checks, counts=dict(audit.counts),
                          primary_mean=summary['primary_mean'], private_gate=summary['capability']['passed'],
                          analysis=str(out / 'analysis.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
