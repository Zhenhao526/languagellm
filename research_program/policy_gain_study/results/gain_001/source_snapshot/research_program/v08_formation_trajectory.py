"""Post-hoc v0.8 same-photo N/U trajectory. Import/prepare never import Torch.

Execute alone restores frozen small interfaces and cached visual features on CPU;
it never constructs the DINO backbone, trains, or modifies a source run. Exact
receiver maximum ties stop the whole probe. No menu fallback or selective retry.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
from itertools import permutations, product
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

WORK = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = WORK / 'redesign_v0.8/results/complementarity_001'
SEEDS = (27101, 27102, 27103, 27104)
UPDATES = (0, 100, 300, 600, 1200, 1800, 2100, 2400)
KINDS = ('additive', 'mixed', 'joint')
CONDITIONS = tuple(f'split{s}_{k}' for s in (1, 2, 3) for k in KINDS) + tuple(
    f'{prefix}_{k}' for prefix in ('full', 'blocked') for k in KINDS)
MAPS = tuple(permutations(range(6), 2))
MATCHINGS = {1: ((0, 1), (2, 3), (4, 5)), 2: ((0, 2), (1, 4), (3, 5)),
             3: ((0, 3), (1, 5), (2, 4))}
CODE_FILES = (
    'research_program/v08_formation_trajectory.py',
    'research_program/v08_formation_trajectory_tests.py',
    'research_program/v08_formation_trajectory_预审.md',
    'redesign_v0.8/camp.py', 'redesign_v0.8/run_experiment.py',
    'redesign_v0.8/analyze_protocols.py', 'redesign_v0.8/固定执行方案.md',
    'redesign_v0.4/agents.py', 'redesign_v0.4/run_pilot.py', 'redesign_v0.4/resource_env.py',
)
CATEGORIES = ('same_code_both_correct', 'both_marginals_no_joint_code',
              'exactly_one_marginal', 'neither_marginal')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        handle.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def runtime():
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages={key: importlib.metadata.version(key) for key in ('numpy', 'torch')
                          if importlib.util.find_spec(key) is not None})


def proportion(n, d):
    return dict(numerator=int(n), denominator=int(d), rate=float(n / d) if d else None)


def menu_gather_source_check(path):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CampAgent')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'receive')
    refs = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == 'menu']
    expr = next(n.value for n in fn.body if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == 'logits' for t in n.targets))
    require(len(refs) == 1 and isinstance(expr, ast.Call) and
            isinstance(expr.func, ast.Attribute) and expr.func.attr == 'gather' and
            len(expr.args) == 2 and isinstance(expr.args[0], ast.Constant) and expr.args[0].value == 1 and
            isinstance(expr.args[1], ast.Name) and expr.args[1].id == 'menu', 'Menu is not only final gather')
    require(isinstance(fn.body[-1], ast.Return) and isinstance(fn.body[-1].value, ast.Tuple) and
            isinstance(fn.body[-1].value.elts[0], ast.Name) and fn.body[-1].value.elts[0].id == 'logits',
            'Changed returned logits')
    return dict(receive_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
                proof='A unique maximum remains the same physical site under all 720 permutations.',
                tie_rule='Stop the entire probe; save the tied logits; do not collapse or enumerate a fallback.')


def expected_plan(condition):
    prefix, kind = condition.split('_')
    return dict(split=int(prefix[-1]) if prefix.startswith('split') else 0, representation='identity',
                schedule='direct', vocab=7, length=2, known=False, blocked=prefix == 'blocked',
                reward_kind=kind, complementarity={'additive': 0., 'mixed': .5, 'joint': 1.}[kind])


def map_partition(split):
    held = {i for i, pair in enumerate(MAPS) if split and tuple(sorted(pair)) in MATCHINGS[split]}
    return [i for i in range(30) if i not in held], sorted(held)


def build_plan(batch=DEFAULT_BATCH):
    """Hash only; no Torch import, state deserialization or inference."""
    require('torch' not in sys.modules and 'camp' not in sys.modules, 'Prepare in a clean no-model process')
    batch = Path(batch).resolve()
    protocol_file = batch / 'protocol_analysis.json'
    protocol = read_json(protocol_file)
    require(protocol['status'] == 'complete' and protocol['expected_run_count'] == 60 and
            protocol['expected_seeds'] == list(SEEDS) and not protocol['missing_runs'], 'Protocol incomplete')
    indexed = {(r['seed'], r['condition']): r for r in protocol['runs']}
    require(len(indexed) == len(protocol['runs']) == 60 and set(indexed) == set(product(SEEDS, CONDITIONS)),
            'Protocol must contain all 60 unique runs')
    hashes = {}

    def freeze(path):
        path = Path(path).resolve()
        require(path.is_file(), f'Missing file: {path}')
        if str(path) not in hashes:
            hashes[str(path)] = sha(path)
        return str(path)

    freeze(protocol_file)
    for relative in CODE_FILES:
        freeze(WORK / relative)
    for short, relative in [('camp', 'redesign_v0.8/camp.py'), ('analysis', 'redesign_v0.8/analyze_protocols.py'),
                            ('features', 'redesign_v0.4/data/features.npz'),
                            ('photo_manifest', 'redesign_v0.4/data/manifest.json')]:
        require(hashes[freeze(WORK / relative)] == protocol['fingerprints'][short], f'Protocol source mismatch: {short}')
    invocation = batch / 'invocation_1789472349972302000.json'
    source_invocation = read_json(freeze(invocation))
    require(source_invocation['seeds'] == list(SEEDS) and source_invocation['conditions'] == list(CONDITIONS),
            'Changed invocation grid')
    for source, digest in source_invocation['hashes'].items():
        require(hashes[freeze(source)] == digest, f'Changed training source: {source}')
    audit = read_json(freeze(batch / 'audit_execution.json'))
    require(audit['status'] == 'passed' and audit['completed_runs'] == 60 and not audit['pending'],
            'Source execution audit not complete')
    entries = read_json(WORK / 'redesign_v0.4/data/manifest.json')['images']
    pools = [[i for i, e in enumerate(entries) if e['split'] == 'test' and e['category'] == kind]
             for kind in ('food', 'water')]
    require(all(len(p) == 8 for p in pools), 'Unexpected held-out photo pool')
    photos = [list(pair) for pair in product(pools[0][4:8], pools[1][4:8])]
    require(protocol['photo_splits']['validation'] == photos, 'Validation photo index mismatch')
    runs = []
    for seed, condition in product(SEEDS, CONDITIONS):
        old = indexed[seed, condition]
        directory = batch / f's{seed}_{condition}'
        config_file = freeze(directory / 'config.json')
        config = read_json(config_file)
        result = read_json(freeze(directory / 'result.json'))
        plan = expected_plan(condition)
        train, held = map_partition(plan['split'])
        require(config['seed'] == old['seed'] == seed and config['condition'] == old['condition'] == condition and
                config['plan'] == old['plan'] == plan and config['checkpoints'] == list(UPDATES) and
                config['updates'] == 2400 and config['sites'] == 6 and config['history_dim'] == 18 and
                config['map_table'] == [list(m) for m in MAPS] and
                config['train_map_ids'] == old['train_map_ids'] == train and
                config['heldout_map_ids'] == old['heldout_map_ids'] == held,
                f'Unexpected source config: {directory}')
        require(result['seed'] == seed and result['condition'] == condition and
                result['frozen_projection_verified'], 'Incomplete/invalid result')
        for name, digest in config['source_hashes'].items():
            require(hashes[freeze(WORK / 'redesign_v0.8' / name)] == digest, 'Changed run source')
        prepared = freeze(batch / f'prepared_{seed}.pt')
        final = freeze(directory / 'final.pt')
        require(Path(old['prepared_checkpoint']).resolve() == Path(prepared) and
                Path(old['final_checkpoint']).resolve() == Path(final), 'Anchor path mismatch')
        require(hashes[prepared] == old['prepared_sha256'] == config['prepared_source']['sha256'] and
                hashes[final] == old['final_sha256'], 'Anchor checkpoint mismatch')
        checkpoints = {str(u): freeze(directory / f'checkpoint_{u:04d}.pt') for u in UPDATES}
        require(len(old['directions']) == 2 and {d['scout'] for d in old['directions']} == {0, 1}, 'Direction grid')
        for direction in old['directions']:
            require(direction['collector'] == 1 - direction['scout'] and
                    direction['phases']['validation']['photo_pairs'] == photos, 'Direction/photo mismatch')
            require(direction['menu_audit']['all_menu_permutations_physically_equivalent'],
                    'Final source is menu-dependent; this strict no-tie plan cannot simplify it')
        runs.append(dict(seed=seed, condition=condition, plan=plan, train_map_ids=train, heldout_map_ids=held,
                         config_file=config_file, prepared_file=prepared, final_file=final, checkpoint_files=checkpoints))
    return dict(schema_version=1, design='post_hoc_v08_same_photo_formation_trajectory',
                timing='Designed after intermediate/final training evidence was visible; exploratory, not preregistered.',
                created_at=now(), batch=str(batch), source_protocol=str(protocol_file), runs=runs,
                seeds=list(SEEDS), conditions=list(CONDITIONS), updates=list(UPDATES), map_table=[list(m) for m in MAPS],
                validation_photo_pairs=photos, expected=dict(runs=60, checkpoint_files=480, directions=960,
                    receiver_inputs=94080, natural_messages=460800, messages_per_direction=480, endpoint_anchors=120),
                receiver_context=dict(inventory=[0, 0], history=[0] * 18, menu=list(range(6)), goals=[[1, 0], [0, 1]]),
                sender_context=dict(visible_goal=[0, 0], inventory=[0, 0], greedy=True, numpy_seed=810071),
                menu_rule=menu_gather_source_check(WORK / 'redesign_v0.8/camp.py'),
                source_files_sha256=hashes, source_runtime=source_invocation, prepare_runtime=runtime(),
                execution=dict(device='cpu', torch_threads=1, weight_loading='weights_only=True',
                    no_training=True, no_backbone=True, no_automatic_retry=True, no_resume=True),
                prepare_weight_loads=0, prepare_forward_calls=0)


def prepare(output, batch=DEFAULT_BATCH):
    output = Path(output).resolve()
    require(not output.exists(), 'Output exists; never overwrite an old preparation')
    plan = build_plan(batch)
    output.mkdir(parents=True, exist_ok=False)
    for relative in CODE_FILES:
        target = output / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(WORK / relative, target)
    write_new(output / 'plan.json', plan)
    write_new(output / 'freeze.json', dict(plan_sha256=sha(output / 'plan.json'), status='prepared_not_executed'))
    return dict(status='prepared_not_executed', output=str(output), **plan['expected'])


def verify(output):
    output = Path(output).resolve()
    require(sha(output / 'plan.json') == read_json(output / 'freeze.json')['plan_sha256'], 'Plan changed')
    plan = read_json(output / 'plan.json')
    for source, digest in plan['source_files_sha256'].items():
        require(sha(source) == digest, f'Frozen source changed: {source}')
    for relative in CODE_FILES:
        require(sha(output / 'code_snapshot' / relative) == plan['source_files_sha256'][str(WORK / relative)],
                f'Source snapshot changed: {relative}')
    return plan


class TieError(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__('Exact receiver maximum tie: stop the entire probe without a menu fallback')


def lookup_from_logits(logits):
    import numpy as np
    logits = np.asarray(logits)
    require(logits.shape == (49, 2, 6) and np.isfinite(logits).all(), 'Invalid receiver logits')
    counts = (logits == logits.max(-1, keepdims=True)).sum(-1)
    if not np.all(counts == 1):
        raise TieError(dict(logits=logits.tolist(), unique_max_count=counts.tolist(),
                            tied_code_goal=np.argwhere(counts != 1).tolist()))
    ordered = np.sort(logits, axis=-1)
    return dict(actions=logits.argmax(-1).astype(np.int64).tolist(), unique_max_count=counts.tolist(),
                top_two_margin=(ordered[..., -1] - ordered[..., -2]).tolist())


def analyze_arrays(actions, emitted, delivered, train_ids, heldout_ids, blocked=False):
    """Pure NumPy metrics; U has 30 map units, N has 30 x 16 photo-map units."""
    import numpy as np
    actions, emitted, delivered = map(np.asarray, (actions, emitted, delivered))
    require(actions.shape == (49, 2) and np.issubdtype(actions.dtype, np.integer) and
            ((actions >= 0) & (actions < 6)).all(), 'Invalid actions')
    for messages in (emitted, delivered):
        require(messages.shape == (30, 16, 2) and np.issubdtype(messages.dtype, np.integer) and
                ((messages >= 0) & (messages < 7)).all(), 'Invalid messages')
    require(np.array_equal(delivered, np.zeros_like(emitted) if blocked else emitted), 'Channel violated')
    require(len(set(train_ids)) == len(train_ids) and len(set(heldout_ids)) == len(heldout_ids) and
            not set(train_ids) & set(heldout_ids) and set(train_ids) | set(heldout_ids) == set(range(30)), 'Map partition')
    codes = delivered[..., 0] * 7 + delivered[..., 1]
    natural = actions[codes] == np.asarray(MAPS)[:, None, :]
    maps = []
    for i, target in enumerate(MAPS):
        correct = actions == np.asarray(target)[None, :]
        joint_codes = np.flatnonzero(correct.all(1)).tolist()
        f, w = map(bool, correct.any(0))
        cls = CATEGORIES[0] if joint_codes else CATEGORIES[1] if f and w else CATEGORIES[2] if f or w else CATEGORIES[3]
        u = bool(joint_codes)
        allowed_u = bool(correct[0].all()) if blocked else u
        allowed_max = int(correct[0].sum()) if blocked else int(correct.sum(1).max())
        n = natural[i].all(1)
        require(not n.any() or allowed_u, 'Natural success outside permitted code coverage')
        maps.append(dict(map_id=i, locations=list(target), partition='train' if i in train_ids else 'heldout',
                         category=cls, food_marginal=f, water_marginal=w, joint_code_ids=joint_codes,
                         U_full49=u, U_channel=allowed_u, optimal_goal_sum_full49=int(correct.sum(1).max()),
                         optimal_goal_sum_channel=allowed_max,
                         natural_actions=actions[codes[i]].tolist(), natural_correct_by_goal=natural[i].tolist(),
                         natural_both=n.tolist(), natural_class=[
                             'N_success' if x else 'N_failure_U1' if allowed_u else 'N_failure_U0' for x in n]))
    summaries = {}
    for name, ids in [('all', list(range(30))), ('train', train_ids), ('heldout', heldout_ids)]:
        if not ids:
            summaries[name] = None
            continue
        rows = [maps[i] for i in ids]
        n = natural[ids].all(-1)
        count_u0 = sum(sum(c == 'N_failure_U0' for c in r['natural_class']) for r in rows)
        count_u1 = sum(sum(c == 'N_failure_U1' for c in r['natural_class']) for r in rows)
        failures = int((~n).sum())
        summaries[name] = dict(
            U_full49=proportion(sum(r['U_full49'] for r in rows), len(ids)),
            U_channel=proportion(sum(r['U_channel'] for r in rows), len(ids)),
            N_both=proportion(n.sum(), n.size), N_single=proportion(natural[ids].sum(), natural[ids].size),
            natural_failure_U0=proportion(count_u0, n.size), natural_failure_U1=proportion(count_u1, n.size),
            fraction_failures_U0=proportion(count_u0, failures),
            optimal_uniform_goal_full49=proportion(sum(r['optimal_goal_sum_full49'] for r in rows), 2 * len(ids)),
            optimal_uniform_goal_channel=proportion(sum(r['optimal_goal_sum_channel'] for r in rows), 2 * len(ids)),
            coverage_categories={c: proportion(sum(r['category'] == c for r in rows), len(ids)) for c in CATEGORIES})
        require(int(n.sum()) + count_u0 + count_u1 == n.size, 'N classification partition')
    return dict(maps=maps, summaries=summaries)


def message_counts(messages):
    return [dict(message=list(m), count=n) for m, n in sorted(Counter(map(tuple, messages)).items())]


def anchor_direction(record, source):
    """Full final physical table and map-frequency anchors, not old per-photo identity."""
    import numpy as np
    actions = np.asarray(record['receiver']['actions'], dtype=np.int64)
    require(source['scout'] == record['scout'] and source['collector'] == record['collector'], 'Anchor direction')
    table = source['receiver_decoder_table']
    require(len(table) == 49, 'Final decoder table length')
    for i, (a, row) in enumerate(zip(actions, table)):
        counts = [[720 if j == int(site) else 0 for j in range(6)] for site in a]
        require(row == dict(message=[i // 7, i % 7], actions_by_goal=a.tolist(),
                            menu_action_counts_by_goal=counts), f'Final decoder row mismatch: {i}')
    physical = np.repeat(actions[:, :, None], 720, axis=2)
    audit = source['menu_audit']
    require(audit['all_menu_permutations_physically_equivalent'] and audit['menus_used_after_check'] == 1 and
            audit['cases'] == 49 * 2 * 720 and audit['physical_action_sha256'] == hashlib.sha256(physical.tobytes()).hexdigest(),
            'Final physical decoder hash mismatch')
    phase = source['phases']['validation']
    require(record['photo_pairs'] == phase['photo_pairs'], 'Final photos mismatch')
    require(len(phase['codebook']) == 60, 'Expected 30 maps x two duplicated hidden sender-goal queries')
    by_key = {(r['map_id'], r['sender_goal']): r for r in phase['codebook']}
    require(len(by_key) == 60, 'Duplicate source codebook key')
    for m, target in enumerate(MAPS):
        correct = np.asarray(record['analysis']['maps'][m]['natural_correct_by_goal'])
        for g in range(2):
            expected = dict(map_id=m, food_location=target[0], water_location=target[1], sender_goal=g,
                            emitted_messages=message_counts(record['emitted'][m]),
                            delivered_messages=message_counts(record['delivered'][m]),
                            native=proportion(correct[:, g].sum(), 16),
                            switched=proportion(correct[:, 1-g].sum(), 16),
                            both=proportion(correct.all(1).sum(), 16))
            require(by_key[m, g] == expected, f'Final codebook mismatch: map {m}, sender goal {g}')
    return dict(decoder_rows_matched=49, complete_720_menu_hash_matched=True, codebook_rows_matched=60,
                source_per_photo_match_available=False,
                limitation='Source saves per-map frequency, not the correspondence between individual photos and messages.')


def update0_cache_verify(record, reference):
    require(record['update'] == reference['update'] == 0 and record['seed'] == reference['seed'] and
            record['scout'] == reference['scout'] and record['collector'] == reference['collector'], 'Initial identity')
    for key in ('receiver', 'emitted', 'photo_pairs'):
        require(record[key] == reference[key], f'Initial condition mismatch: {key}')
    # Delivery is deliberately excluded: the blocked channel replaces emissions by zero.
    return True


def summarize(records):
    """Average splits and directions within seed, then retain all four seed values."""
    buckets = defaultdict(list)
    for r in records:
        prefix, kind = r['condition'].split('_')
        family = 'split' if prefix.startswith('split') else prefix
        for part, values in r['summaries'].items():
            if values is not None:
                buckets[family, kind, r['update'], part, r['seed']].append(values)
    grouped = defaultdict(dict)
    for (family, kind, update, part, seed), values in sorted(buckets.items()):
        require(len(values) == (6 if family == 'split' else 2), 'Incomplete within-seed directions/splits')
        metrics = {k: sum(v[k]['rate'] for v in values) / len(values)
                   for k in values[0] if k not in ('coverage_categories', 'fraction_failures_U0')}
        for category in CATEGORIES:
            metrics[category] = sum(v['coverage_categories'][category]['rate'] for v in values) / len(values)
        # Conditional failure fraction pools equal-denominator within-seed cases before division.
        bad = sum(v['fraction_failures_U0']['numerator'] for v in values)
        failures = sum(v['fraction_failures_U0']['denominator'] for v in values)
        metrics['fraction_failures_U0'] = bad / failures if failures else None
        grouped[family, kind, update, part][seed] = metrics
    result = []
    for (family, kind, update, part), seeds in sorted(grouped.items()):
        require(set(seeds) == set(SEEDS), 'Incomplete independent seed set')
        values = {metric: [seeds[s][metric] for s in SEEDS] for metric in seeds[SEEDS[0]]}
        result.append(dict(family=family, reward_kind=kind, update=update, partition=part, seeds=list(SEEDS),
                           metrics={m: dict(seed_values=v, mean=sum(v)/4 if all(x is not None for x in v) else None)
                                    for m, v in values.items()}))
    return result


def execute(output):
    """Only caller of Torch/model APIs. Root explicitly starts this after review."""
    output = Path(output).resolve()
    plan = verify(output)
    run_dir = output / 'execution'
    run_dir.mkdir(exist_ok=False)  # no overwrite, selective resume, or automatic retry
    (run_dir / 'records').mkdir()
    started = time.monotonic()
    write_new(run_dir / 'started.json', dict(started_at=now(), runtime=runtime(), plan_sha256=sha(output / 'plan.json')))
    context = {}
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1)
        require(platform.python_version() == plan['source_runtime']['python'] and
                torch.__version__ == plan['source_runtime']['torch'], 'Use the frozen original Python/Torch runtime')
        require('camp' not in sys.modules, 'Run execute in a fresh process')
        sys.path.insert(0, str(WORK / 'redesign_v0.8'))
        camp = importlib.import_module('camp')
        for module, relative in [('camp', 'redesign_v0.8/camp.py'), ('agents', 'redesign_v0.4/agents.py'),
                                 ('run_pilot', 'redesign_v0.4/run_pilot.py'), ('resource_env', 'redesign_v0.4/resource_env.py')]:
            require(Path(sys.modules[module].__file__).resolve() == WORK / relative, 'Imported wrong source module')
        require(camp.MAPS.tolist() == plan['map_table'], 'Runtime maps changed')
        bank = camp.ImageBank()  # cached features only; no DINO backbone
        photos = np.asarray(plan['validation_photo_pairs'], dtype=np.int64)
        require(photos.tolist() == [list(p) for p in product(bank.pools['test', 0][4:8], bank.pools['test', 1][4:8])],
                'Runtime photo indexing changed')
        protocol = read_json(plan['source_protocol'])
        anchors = {(r['seed'], r['condition']): r for r in protocol['runs']}
        index, init_reference = [], {}
        codes = np.asarray(list(product(range(7), repeat=2)), dtype=np.int64)
        goal_indices = np.tile(np.arange(2), 49)
        message_inputs = np.repeat(codes, 2, axis=0)
        map_indices = np.repeat(np.arange(30), 16)
        photo_indices = np.tile(np.arange(16), 30)
        with torch.no_grad():
            for run in plan['runs']:
                seed, condition = run['seed'], run['condition']
                prepared = torch.load(run['prepared_file'], weights_only=True, map_location='cpu')
                agents = camp.remake_agents(seed, prepared, 7, 2, representation='identity')
                projection = [{k: v.clone() for k, v in a.project.state_dict().items()} for a in agents]
                transforms = [a.input_transform.clone() for a in agents]
                projected = camp.projected_banks(agents, bank)
                final = torch.load(run['final_file'], weights_only=True, map_location='cpu')
                for update in UPDATES:
                    checkpoint = run['checkpoint_files'][str(update)]
                    context = dict(seed=seed, condition=condition, update=update, checkpoint=checkpoint)
                    states = torch.load(checkpoint, weights_only=True, map_location='cpu')
                    require(len(states) == len(agents) == len(final) == 2, 'Expected two independent interfaces')
                    for i, (agent, state) in enumerate(zip(agents, states)):
                        if update == 2400:
                            require(state.keys() == final[i].keys() and all(torch.equal(v, final[i][k]) for k, v in state.items()),
                                    'checkpoint2400 differs from final.pt')
                        agent.load_state_dict(state, strict=True)
                        agent.eval()
                        agent.requires_grad_(False)
                        require(torch.equal(agent.input_transform, transforms[i]) and all(
                            torch.equal(v, projection[i][k]) for k, v in agent.project.state_dict().items()),
                            'Private projection/transform changed; cached projections would be invalid')
                    for scout in range(2):
                        collector = 1 - scout
                        context['scout'] = scout
                        logits, _ = agents[collector].receive(torch.from_numpy(message_inputs),
                            torch.from_numpy(np.eye(2, dtype=np.float32)[goal_indices]), torch.zeros(98, 2),
                            torch.zeros(98, 18), torch.arange(6).repeat(98, 1))
                        logits = logits.numpy().reshape(49, 2, 6)
                        receiver = dict(logits=logits.tolist(), **lookup_from_logits(logits))
                        hidden = agents[scout].observe(camp.scene_visual(camp.MAPS[map_indices],
                            photos[photo_indices], projected[scout]))
                        emitted, _, _, _ = agents[scout].send(hidden, torch.zeros(480, 2), torch.zeros(480, 2),
                            np.random.default_rng(810071), True)
                        emitted = emitted.numpy().reshape(30, 16, 2)
                        delivered = np.zeros_like(emitted) if run['plan']['blocked'] else emitted.copy()
                        record = dict(seed=seed, condition=condition, update=update, scout=scout, collector=collector,
                            checkpoint_sha256=plan['source_files_sha256'][checkpoint],
                            photo_pairs=photos.tolist(), receiver=receiver, emitted=emitted.tolist(), delivered=delivered.tolist(),
                            analysis=analyze_arrays(receiver['actions'], emitted, delivered,
                                run['train_map_ids'], run['heldout_map_ids'], run['plan']['blocked']))
                        if update == 0:
                            key = seed, scout
                            if key in init_reference:
                                update0_cache_verify(record, init_reference[key])
                            else:
                                init_reference[key] = record
                        if update == 2400:
                            source = next(d for d in anchors[seed, condition]['directions'] if d['scout'] == scout)
                            record['endpoint_anchor'] = anchor_direction(record, source)
                        relative = f'records/s{seed}_{condition}_u{update:04d}_s{scout}.json'
                        write_new(run_dir / relative, record)
                        index.append(dict(seed=seed, condition=condition, update=update, scout=scout,
                            file=relative, sha256=sha(run_dir / relative), summaries=record['analysis']['summaries'],
                            endpoint_anchor=record.get('endpoint_anchor')))
                print(f'Completed {seed} {condition}: {len(index)}/960 directions', flush=True)
        expected = set(product(SEEDS, CONDITIONS, UPDATES, range(2)))
        require(len(index) == 960 and {(r['seed'], r['condition'], r['update'], r['scout']) for r in index} == expected,
                'Incomplete trajectory grid')
        require(sum(r['endpoint_anchor'] is not None for r in index) == 120, 'Incomplete endpoint anchors')
        verify(output)  # ensure source bytes stayed frozen throughout this run
        result = dict(status='complete', completed_at=now(), elapsed_seconds=time.monotonic()-started,
            runtime=runtime(), plan_sha256=sha(output / 'plan.json'), records=index, summaries=summarize(index),
            counts=dict(directions=960, receiver_inputs=94080, natural_messages=460800, final_anchors=120,
                        initial_condition_comparisons=112, receiver_maximum_ties=0),
            interpretation='Post-hoc frozen greedy same-photo N/U trajectory; seed is the independent training unit.',
            source_per_photo_anchor_unavailable=True)
        write_new(run_dir / 'results.json', result)
        return dict(status='complete', output=str(run_dir / 'results.json'), **result['counts'])
    except Exception as error:
        failure = dict(status='failed', failed_at=now(), elapsed_seconds=time.monotonic()-started,
                       context=context, error=repr(error), traceback=traceback.format_exc())
        if isinstance(error, TieError):
            failure['tie'] = error.details
        write_new(run_dir / 'failure.json', failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'execute'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--batch', type=Path, default=DEFAULT_BATCH)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.out, args.batch)
    elif args.command == 'verify':
        plan = verify(args.out)
        result = dict(status='verified', files=len(plan['source_files_sha256']), **plan['expected'])
    else:
        result = execute(args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
