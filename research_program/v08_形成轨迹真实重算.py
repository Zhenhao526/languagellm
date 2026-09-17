"""Independent arithmetic audit of completed v0.8 probe artifacts; no Torch."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research_program/v08_formation_trajectory_001'
TEST = ROOT / 'research_program/v08_形成轨迹独立核验.py'
spec = importlib.util.spec_from_file_location('independent_formation_reference', TEST)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
MAPS = reference.MAPS
SEEDS = (27101, 27102, 27103, 27104)
CATEGORIES = ('same_code_both_correct', 'both_marginals_no_joint_code', 'exactly_one_marginal', 'neither_marginal')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise AssertionError(message)


def prop(n, d):
    return dict(numerator=int(n), denominator=int(d), rate=float(n / d) if d else None)


def audit():
    started = time.perf_counter()
    execution = OUT / 'execution'
    require(not (execution / 'failure.json').exists(), 'Probe failure must be reported, not retried')
    result = read(execution / 'results.json')
    plan = read(OUT / 'plan.json')
    require(result['status'] == 'complete', 'Wait for complete probe')
    require(sha(OUT / 'plan.json') == read(OUT / 'freeze.json')['plan_sha256'] == result['plan_sha256'], 'Plan SHA')
    for path, digest in plan['source_files_sha256'].items():
        require(sha(path) == digest, 'Frozen source: ' + path)
    snapshots = list((OUT / 'code_snapshot').rglob('*'))
    snapshots = [p for p in snapshots if p.is_file()]
    for path in snapshots:
        original = str(ROOT / path.relative_to(OUT / 'code_snapshot'))
        require(sha(path) == plan['source_files_sha256'][original], 'Snapshot digest')
    protocol = read(plan['source_protocol'])
    sources = {(r['seed'], r['condition']): r for r in protocol['runs']}
    planned = {(r['seed'], r['condition']): r for r in plan['runs']}
    expected = set(product(SEEDS, plan['conditions'], plan['updates'], (0, 1)))
    index = result['records']
    require(len(index) == 960 and {(r['seed'], r['condition'], r['update'], r['scout']) for r in index} == expected, 'Exact direction grid')
    require(len(list((execution / 'records').glob('*.json'))) == 960, 'No extra/missing record files')
    initial, initial_checks, anchors = {}, 0, 0
    summary_buckets = defaultdict(list)
    map_rows_checked = 0
    min_margin = float('inf')
    all_counts, per_group_time = [], []
    record_hashes = {}
    for ix in index:
        path = execution / ix['file']
        require(path.resolve().parent == (execution / 'records').resolve(), 'Record path')
        digest = sha(path); require(digest == ix['sha256'], 'Record SHA')
        record_hashes[ix['file']] = digest
        record = read(path)
        for key in ('seed', 'condition', 'update', 'scout'):
            require(record[key] == ix[key], 'Record identity')
        seed, name, update, scout = (record[k] for k in ('seed', 'condition', 'update', 'scout'))
        run = planned[seed, name]
        require(record['collector'] == 1 - scout, 'Opposite collector')
        require(record['checkpoint_sha256'] == plan['source_files_sha256'][run['checkpoint_files'][str(update)]], 'Checkpoint record SHA')
        require(record['photo_pairs'] == plan['validation_photo_pairs'], 'Same ordered photo pairs at all times')
        actions, margins = reference.unique_lookup_reference(record['receiver']['logits'])
        require(actions.tolist() == record['receiver']['actions'], 'Saved argmax actions')
        require(record['receiver']['unique_max_count'] == [[1, 1]] * 49, 'Unique maxima')
        require(np.asarray(margins, np.float32).tolist() == record['receiver']['top_two_margin'], 'Float32 top-two margins')
        min_margin = min(min_margin, float(margins.min()))
        emitted = np.array(record['emitted']); delivered = np.array(record['delivered'])
        blocked = run['plan']['blocked']
        simple = reference.natural_counts_reference(actions, emitted, delivered, blocked)
        legal_actions = actions[:1] if blocked else actions
        maps = []
        for m, target in enumerate(MAPS):
            a = record['analysis']['maps'][m]
            truth = actions == np.array(target)
            codes = [i for i, row in enumerate(actions) if tuple(row) == target]
            food, water = bool(truth[:, 0].any()), bool(truth[:, 1].any())
            category = CATEGORIES[0] if codes else CATEGORIES[1] if food and water else CATEGORIES[2] if food or water else CATEGORIES[3]
            natural_actions = [actions[int(code[0]) * 7 + int(code[1])].tolist() for code in delivered[m]]
            correct = [[x == t for x, t in zip(row, target)] for row in natural_actions]
            both = [all(row) for row in correct]
            channel_u = target in {tuple(row) for row in legal_actions}
            classes = ['N_success' if ok else 'N_failure_U1' if channel_u else 'N_failure_U0' for ok in both]
            expected_row = dict(map_id=m, locations=list(target), partition='train' if m in run['train_map_ids'] else 'heldout',
                category=category, food_marginal=food, water_marginal=water, joint_code_ids=codes,
                U_full49=bool(codes), U_channel=channel_u, optimal_goal_sum_full49=int(truth.sum(1).max()),
                optimal_goal_sum_channel=int((legal_actions == np.array(target)).sum(1).max()),
                natural_actions=natural_actions, natural_correct_by_goal=correct, natural_both=both, natural_class=classes)
            require(a == expected_row, 'Independent per-map record arithmetic')
            require(sum(both) == simple[m]['N'], 'Independent natural count reference')
            maps.append(expected_row); map_rows_checked += 1
        summaries = {}
        for part, ids in [('all', list(range(30))), ('train', run['train_map_ids']), ('heldout', run['heldout_map_ids'])]:
            if not ids:
                summaries[part] = None; continue
            rows = [maps[i] for i in ids]; denominator = 16 * len(ids)
            success = sum(sum(r['natural_both']) for r in rows)
            f0 = sum(sum(v == 'N_failure_U0' for v in r['natural_class']) for r in rows)
            f1 = denominator - success - f0
            summaries[part] = dict(U_full49=prop(sum(r['U_full49'] for r in rows), len(ids)),
                U_channel=prop(sum(r['U_channel'] for r in rows), len(ids)), N_both=prop(success, denominator),
                N_single=prop(sum(sum(sum(x) for x in r['natural_correct_by_goal']) for r in rows), 2 * denominator),
                natural_failure_U0=prop(f0, denominator), natural_failure_U1=prop(f1, denominator),
                fraction_failures_U0=prop(f0, denominator - success),
                optimal_uniform_goal_full49=prop(sum(r['optimal_goal_sum_full49'] for r in rows), 2 * len(ids)),
                optimal_uniform_goal_channel=prop(sum(r['optimal_goal_sum_channel'] for r in rows), 2 * len(ids)),
                coverage_categories={c: prop(sum(r['category'] == c for r in rows), len(ids)) for c in CATEGORIES})
        require(record['analysis']['summaries'] == ix['summaries'] == summaries, 'Per-direction summary arithmetic')
        prefix, kind = name.split('_'); family = 'split' if prefix.startswith('split') else prefix
        for part, values in summaries.items():
            if values is not None:
                summary_buckets[family, kind, update, part, seed].append(values)
        if update == 0:
            key = seed, scout
            comparison = {k: record[k] for k in ('receiver', 'emitted', 'photo_pairs')}
            if key in initial:
                require(comparison == initial[key], 'Across-condition initial identity'); initial_checks += 1
            else:
                initial[key] = comparison
        if update == 2400:
            old = next(d for d in sources[seed, name]['directions'] if d['scout'] == scout)
            require(old['collector'] == 1 - scout and old['phases']['validation']['photo_pairs'] == record['photo_pairs'], 'Endpoint direction/photo anchor')
            for code, row in enumerate(old['receiver_decoder_table']):
                require(row == dict(message=[code // 7, code % 7], actions_by_goal=actions[code].tolist(),
                    menu_action_counts_by_goal=[[int(site == j) * 720 for j in range(6)] for site in actions[code]]), 'Endpoint decoder table')
            physical = np.broadcast_to(actions[:, :, None], (49, 2, 720)).copy()
            require(old['menu_audit']['physical_action_sha256'] == hashlib.sha256(physical.tobytes()).hexdigest(), 'Endpoint full-menu physical SHA')
            for row in old['phases']['validation']['codebook']:
                m, g = row['map_id'], row['sender_goal']; cr = maps[m]['natural_correct_by_goal']
                em = [dict(message=list(code), count=n) for code, n in sorted(Counter(map(tuple, record['emitted'][m])).items())]
                dm = [dict(message=list(code), count=n) for code, n in sorted(Counter(map(tuple, record['delivered'][m])).items())]
                require(row == dict(map_id=m, food_location=MAPS[m][0], water_location=MAPS[m][1], sender_goal=g,
                    emitted_messages=em, delivered_messages=dm, native=prop(sum(x[g] for x in cr), 16),
                    switched=prop(sum(x[1-g] for x in cr), 16), both=prop(sum(all(x) for x in cr), 16)), 'Endpoint codebook anchor')
            require(record['endpoint_anchor'] == ix['endpoint_anchor'] and record['endpoint_anchor']['decoder_rows_matched'] == 49 and record['endpoint_anchor']['codebook_rows_matched'] == 60, 'Recorded anchor count')
            anchors += 1
    independently_grouped = defaultdict(dict)
    for (family, kind, update, part, seed), values in summary_buckets.items():
        require(len(values) == (6 if family == 'split' else 2), 'Within-seed scope')
        metrics = {}
        for metric in values[0]:
            if metric == 'coverage_categories':
                for cat in CATEGORIES:
                    metrics[cat] = sum(v[metric][cat]['rate'] for v in values) / len(values)
            elif metric == 'fraction_failures_U0':
                n = sum(v[metric]['numerator'] for v in values); d = sum(v[metric]['denominator'] for v in values)
                metrics[metric] = n / d if d else None
            else:
                metrics[metric] = sum(v[metric]['rate'] for v in values) / len(values)
        independently_grouped[family, kind, update, part][seed] = metrics
    require(len(result['summaries']) == len(independently_grouped) == 168, '168 distinct reported groups')
    metric_comparisons = 0
    for row in result['summaries']:
        key = row['family'], row['reward_kind'], row['update'], row['partition']
        seed_metrics = independently_grouped[key]
        require(set(seed_metrics) == set(SEEDS) and row['seeds'] == list(SEEDS), 'Four seed identities')
        expected_metrics = {}
        for name in seed_metrics[SEEDS[0]]:
            v = [seed_metrics[s][name] for s in SEEDS]
            expected_metrics[name] = dict(seed_values=v, mean=sum(v) / 4 if all(x is not None for x in v) else None)
        require(row['metrics'] == expected_metrics, 'Independent four-seed aggregation')
        metric_comparisons += len(expected_metrics)
    require(initial_checks == 112 and anchors == 120 and map_rows_checked == 28800, 'All anchor/map checks')
    require(not any(k in sys.modules for k in ('torch', 'camp', 'mlx', 'mlx_lm')), 'No neural runtime imported')
    return dict(status='passed_independent_completed_probe_audit', verified_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter() - started, probe_directory=str(OUT),
        independent_script_sha256=sha(__file__), independent_reference_sha256=sha(TEST),
        source_result_sha256=sha(execution / 'results.json'), plan_sha256=sha(OUT / 'plan.json'),
        source_files_verified=len(plan['source_files_sha256']), source_snapshots_verified=len(snapshots),
        record_digests_verified=len(record_hashes), record_sha256=record_hashes,
        directions_recomputed=960, map_records_recomputed=map_rows_checked, same_photo_natural_messages_recomputed=460800,
        greedy_receiver_inputs_recomputed=94080, receiver_maximum_ties=0, minimum_top_two_margin=min_margin,
        independent_endpoint_anchors=anchors, independent_update0_comparisons=initial_checks,
        four_seed_summary_groups_recomputed=168, summary_metrics_recomputed=metric_comparisons,
        source_per_photo_endpoint_anchor_available=False, summary=result['summaries'],
        new_neural_forward_calls=0, new_training_updates=0, audited_summary_function_called=False,
        scope=['Post-hoc frozen-checkpoint probe; all saved tables/messages recalculated without neural inference.',
               'This audit hashes checkpoint bytes but does not independently deserialize or re-run the 480 neural checkpoints.',
               'Endpoint anchors compare complete decoder functions and per-map message frequencies, not unavailable old photo-to-message identities.',
               'Four seeds remain independent; eight checkpoints do not identify first language emergence or an ecological causal mechanism.'])


if __name__ == '__main__':
    destination = ROOT / 'research_program/v08_形成轨迹独立核验_真实.json'
    require(not destination.exists(), 'Preserve prior independent audit')
    result = audit()
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('record_sha256', 'summary')}, ensure_ascii=False))
