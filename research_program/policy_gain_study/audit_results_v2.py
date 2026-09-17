"""Independent numeric audit. Reads saved JSON/NPZ only; no neural runtime.

Does not import the study runner, probe, analysis, or their measurement functions.
Run only after the entire study, probe and analysis are complete. Source records
are never changed; new audit output is created exclusively.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
import sys
import traceback

import numpy as np

SEEDS = (28101, 28102, 28103, 28104)
TIMES = (0, 100, 300, 600, 1200, 1800, 2100, 2400)
KINDS, GAINS = ('additive', 'joint'), (1, 3)
CONDITIONS = tuple(f'{"full" if s == 0 else "split"+str(s)}_{k}_gain{g}'
                   for s, k, g in product((1, 2, 3, 0), KINDS, GAINS))
MAPS = tuple(permutations(range(6), 2))
MATCHINGS = {1: {(0, 1), (2, 3), (4, 5)}, 2: {(0, 2), (1, 4), (3, 5)}, 3: {(0, 3), (1, 5), (2, 4)}}
CATEGORIES = ('same_code_both_correct', 'both_marginals_no_joint_code', 'exactly_one_marginal', 'neither_marginal')
METRICS = ('U_full49', 'N_both', 'N_single', 'natural_failure_U0', 'natural_failure_U1', 'optimal_uniform_goal_full49')
MODES = ('normal', 'shuffle', 'blank', 'stochastic', 'erase_memory')
CONTRASTS = dict(lambda_at_gain1=(-1, 0, 1, 0), lambda_at_gain3=(0, -1, 0, 1),
                 gain_at_lambda0=(-1, 1, 0, 0), gain_at_lambda1=(0, 0, -1, 1),
                 interaction=(1, -1, -1, 1), diagonal_joint3_minus_additive1=(-1, 0, 0, 1))


def check(value, label):
    if not value:
        raise AssertionError(label)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def equal(actual, expected, label='value'):
    """Structural exactness; floats allow 1e-12 independent reduction order."""
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and set(actual) == set(expected), label+' keys')
        for k in expected:
            equal(actual[k], expected[k], f'{label}.{k}')
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), label+' length')
        for i, v in enumerate(expected):
            equal(actual[i], v, f'{label}[{i}]')
    elif isinstance(expected, float):
        check(isinstance(actual, (int, float)) and np.isfinite(actual) and abs(actual-expected) <= 1e-12, label)
    else:
        check(actual == expected, label)


def fraction(n, d):
    return dict(numerator=int(n), denominator=int(d), rate=float(n/d) if d else None)


def split_info(condition):
    prefix, kind, gain = condition.split('_')
    split = int(prefix[-1]) if prefix.startswith('split') else 0
    held = [i for i, pair in enumerate(MAPS) if split and tuple(sorted(pair)) in MATCHINGS[split]]
    return ('split' if split else 'full'), kind, int(gain[4:]), [i for i in range(30) if i not in held], held


def physical_lookup(logits):
    logits = np.asarray(logits)
    check(logits.shape == (49, 2, 6) and np.isfinite(logits).all(), 'receiver finite logits')
    maximum = np.max(logits, axis=2)
    ties = np.count_nonzero(logits == maximum[:, :, None], axis=2)
    check(np.array_equal(ties, np.ones((49, 2), dtype=int)), 'receiver maximum must be unique')
    actions = np.argmax(logits, axis=2).astype(np.int64)
    second = np.partition(logits, 4, axis=2)[:, :, 4]
    # Source Torch interfaces produce float32 logits and float32 subtraction;
    # JSON preserves their values but Python's default subtraction is float64.
    return actions, ties, (maximum-second).astype(np.float32)


def measure(actions, emitted, condition):
    """Independent set-based code reachability and integer natural counts."""
    actions, emitted = np.asarray(actions), np.asarray(emitted)
    check(actions.shape == (49, 2) and emitted.shape == (30, 16, 2), 'lookup/message shape')
    check(np.issubdtype(emitted.dtype, np.integer) and ((emitted >= 0) & (emitted < 7)).all(), 'message alphabet')
    _, _, _, train, held = split_info(condition)
    pairs = [tuple(map(int, a)) for a in actions]
    food_sites = {a[0] for a in pairs}
    water_sites = {a[1] for a in pairs}
    rows = []
    for mid, world in enumerate(MAPS):
        good_codes = [c for c, pair in enumerate(pairs) if pair == world]
        f, w = world[0] in food_sites, world[1] in water_sites
        u = len(good_codes) > 0
        category = CATEGORIES[0] if u else CATEGORIES[1] if f and w else CATEGORIES[2] if f or w else CATEGORIES[3]
        emitted_codes = [int(m[0])*7+int(m[1]) for m in emitted[mid]]
        natural_actions = [list(pairs[c]) for c in emitted_codes]
        correct = [[a[0] == world[0], a[1] == world[1]] for a in natural_actions]
        both = [all(x) for x in correct]
        optimal = max(int(a[0] == world[0])+int(a[1] == world[1]) for a in pairs)
        check(not any(both) or u, 'N cannot exceed full code reachability')
        rows.append(dict(map_id=mid, locations=list(world), partition='train' if mid in train else 'heldout',
            category=category, food_marginal=f, water_marginal=w, joint_code_ids=good_codes,
            U_full49=u, U_channel=u, optimal_goal_sum_full49=optimal, optimal_goal_sum_channel=optimal,
            natural_actions=natural_actions, natural_correct_by_goal=correct, natural_both=both,
            natural_class=['N_success' if ok else 'N_failure_U1' if u else 'N_failure_U0' for ok in both]))
    summaries = {}
    for part, ids in [('all', list(range(30))), ('train', train), ('heldout', held)]:
        if not ids:
            summaries[part] = None
            continue
        selected = [rows[i] for i in ids]
        maps, count = len(ids), 16*len(ids)
        n = sum(sum(r['natural_both']) for r in selected)
        single = sum(sum(sum(c) for c in r['natural_correct_by_goal']) for r in selected)
        uncovered = sum(16 for r in selected if not r['U_full49'])
        unused = count-n-uncovered
        u = sum(r['U_full49'] for r in selected)
        optimal = sum(r['optimal_goal_sum_full49'] for r in selected)
        summaries[part] = dict(U_full49=fraction(u, maps), U_channel=fraction(u, maps),
            N_both=fraction(n, count), N_single=fraction(single, 2*count),
            natural_failure_U0=fraction(uncovered, count), natural_failure_U1=fraction(unused, count),
            fraction_failures_U0=fraction(uncovered, count-n), optimal_uniform_goal_full49=fraction(optimal, 2*maps),
            optimal_uniform_goal_channel=fraction(optimal, 2*maps),
            coverage_categories={c: fraction(sum(r['category'] == c for r in selected), maps) for c in CATEGORIES})
    return dict(maps=rows, summaries=summaries)


def factorial(rows, keys, metrics):
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in keys)
        cell = row['seed'], row['kind'], row['gain']
        check(cell not in groups[key], 'duplicate factorial cell')
        groups[key][cell] = row
    result = []
    for key, cells in sorted(groups.items()):
        check(set(cells) == set(product(SEEDS, KINDS, GAINS)), 'all 16 factorial seed cells required')
        for metric in metrics:
            data = np.asarray([[cells[s,k,g][metric] for k,g in [('additive',1),('additive',3),('joint',1),('joint',3)]]
                               for s in SEEDS])
            effects = {}
            for label, weight in CONTRASTS.items():
                values = data @ np.asarray(weight)
                effects[label] = dict(seed_values=values.tolist(), mean=float(values.mean()),
                                      min=float(values.min()), max=float(values.max()))
            result.append(dict(zip(keys, key), metric=metric, seeds=list(SEEDS), effects=effects))
    return result


def trajectory_summaries(records):
    groups = defaultdict(list)
    for row in records:
        family, kind, gain, _, _ = split_info(row['condition'])
        for part in ('train', 'heldout'):
            s = row['summaries'][part]
            if s is not None:
                groups[family, part, row['update'], row['seed'], kind, gain].append(s)
    seed_rows = []
    fields = list(METRICS)+list(CATEGORIES)+['fraction_failures_U0']
    for (f,p,t,s,k,g), items in sorted(groups.items()):
        check(len(items) == (6 if f == 'split' else 2), 'within-seed count')
        values = {m: float(np.mean([x[m]['rate'] for x in items])) for m in METRICS}
        values.update({c: float(np.mean([x['coverage_categories'][c]['rate'] for x in items])) for c in CATEGORIES})
        den = sum(x['fraction_failures_U0']['denominator'] for x in items)
        values['fraction_failures_U0'] = sum(x['fraction_failures_U0']['numerator'] for x in items)/den if den else None
        seed_rows.append(dict(family=f, partition=p, update=t, seed=s, kind=k, gain=g, **values))
    grouped, early = defaultdict(dict), defaultdict(dict)
    for row in seed_rows:
        key = row['family'], row['partition'], row['update'], row['kind'], row['gain']
        grouped[key][row['seed']] = row
        if row['update'] <= 1200:
            early[row['family'],row['partition'],row['seed'],row['kind'],row['gain']][row['update']] = row
    means = []
    for (f,p,t,k,g), cells in sorted(grouped.items()):
        check(set(cells) == set(SEEDS), 'independent four-seed mean')
        vals = {m: [cells[s][m] for s in SEEDS] for m in fields}
        means.append(dict(family=f, partition=p, update=t, kind=k, gain=g, seeds=list(SEEDS), metrics={
            m: dict(seed_values=v, mean=float(np.mean(v)) if all(x is not None for x in v) else None) for m,v in vals.items()}))
    auc = []
    # Independent fixed trapezoid quadrature weights for 0,100,300,600,1200.
    times, weights = (0,100,300,600,1200), np.asarray([50.,150.,250.,450.,300.])/1200
    for (f,p,s,k,g), cells in sorted(early.items()):
        check(set(cells) == set(times), 'all five fixed AUC checkpoints')
        auc.append(dict(family=f, partition=p, seed=s, kind=k, gain=g,
            U_auc_0_1200=float(weights @ np.asarray([cells[t]['U_full49'] for t in times])),
            N_auc_0_1200=float(weights @ np.asarray([cells[t]['N_both'] for t in times]))))
    effects = dict(by_checkpoint=factorial(seed_rows, ('family','partition','update'), METRICS),
                   early_auc=factorial(auc, ('family','partition'), ('U_auc_0_1200','N_auc_0_1200')))
    return dict(seed_values=seed_rows, means=means, early_auc_seed_values=auc), effects


def endpoint_seed_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        f,k,g,_,_ = split_info(row['condition'])
        groups[f,row['partition'],row['mode'],2400,row['seed'],k,g].append(row['J'])
    out = []
    for (f,p,m,t,s,k,g), vals in sorted(groups.items()):
        check(len(vals) == (3 if f == 'split' else 1), 'J split count')
        out.append(dict(family=f, partition=p, mode=m, update=t, seed=s, kind=k, gain=g,J=float(np.mean(vals))))
    return out


def audit(batch, probe_dir, analysis_file):
    check('torch' not in sys.modules and 'camp' not in sys.modules, 'Numeric audit must not load models')
    batch, probe_dir, analysis_file = Path(batch).resolve(), Path(probe_dir).resolve(), Path(analysis_file).resolve()
    manifest, status = read(batch/'manifest.json'), read(batch/'status.json')
    check(status['status'] == 'complete' and status['completed_runs'] == 64, 'Wait for all social runs')
    check(manifest['seeds'] == list(SEEDS) and set(manifest['conditions']) == set(CONDITIONS), 'Frozen matrix')
    plan, frozen = read(probe_dir/'plan.json'), read(probe_dir/'freeze.json')
    check(sha(probe_dir/'plan.json') == frozen['plan_sha256'], 'Probe plan SHA')
    result, analyzed = read(probe_dir/'execution/results.json'), read(analysis_file)
    check(result['status'] == analyzed['status'] == 'complete', 'Require complete probe and analysis')
    check(result['plan_sha256'] == frozen['plan_sha256'], 'Results plan binding')
    check(analyzed['probe_result_sha256'] == sha(probe_dir/'execution/results.json'), 'Analysis/probe binding')
    source_hashes = dict(plan['source_files_sha256'])
    for path, digest in manifest['source_hashes'].items():
        check(path not in source_hashes or source_hashes[path] == digest, 'Conflicting frozen source')
        source_hashes[path] = digest
    for path, digest in analyzed['inputs_sha256'].items():
        check(path not in source_hashes or source_hashes[path] == digest, 'Conflicting analysis source')
        source_hashes[path] = digest
    for path, digest in source_hashes.items():
        check(sha(path) == digest, 'Frozen source changed: '+path)
    check(len(result['records']) == 1024, 'Complete 1024 record list')
    photos = plan['validation_photo_pairs']
    check(len(photos) == 16 and len({tuple(p) for p in photos}) == 16, '16 distinct fixed photo pairs')
    entries=read(Path(__file__).resolve().parents[2]/'redesign_v0.4/data/manifest.json')['images']
    pools=[[i for i,e in enumerate(entries) if e['split']=='test' and e['category']==name] for name in ('food','water')]
    check(all(len(p)==8 for p in pools),'Original test photo pool')
    equal(photos,[list(p) for p in product(pools[0][4:8],pools[1][4:8])],'Exact planned validation photo subset')
    decoded, summaries, initial, endpoint_records = {}, [], {}, {}
    record_hashes = {}
    for entry in result['records']:
        path = probe_dir/'execution'/entry['file']
        digest = sha(path)
        check(digest == entry['sha256'], 'Record changed')
        record_hashes[str(path)] = digest
        r = read(path)
        key = r['seed'], r['condition'], r['update'], r['scout']
        check(key not in decoded, 'Duplicate record')
        equal([entry[k] for k in ('seed','condition','update','scout')], list(key), 'Index identity')
        equal(entry['endpoint_anchor'],r.get('endpoint_anchor'),'Index endpoint anchor')
        equal(r['photo_pairs'], photos, 'Constant photo definition')
        check(r['collector'] == 1-r['scout'], 'Direction identities')
        actions, counts, margins = physical_lookup(r['receiver']['logits'])
        equal(r['receiver']['actions'], actions.tolist(), 'Saved physical actions')
        equal(r['receiver']['unique_max_count'], counts.tolist(), 'Saved maximum counts')
        equal(r['receiver']['top_two_margin'], margins.tolist(), 'Saved margins')
        equal(r['delivered'], r['emitted'], 'All new gain arms have open channels')
        measurement = measure(actions, r['emitted'], r['condition'])
        equal(r['analysis'], measurement, 'Independent map/N/U calculation')
        equal(entry['summaries'], measurement['summaries'], 'Record-index summary')
        summaries.append(dict(seed=r['seed'], condition=r['condition'],update=r['update'],scout=r['scout'],
                              summaries=measurement['summaries']))
        decoded[key] = actions
        if r['update'] == 0:
            ik = r['seed'],r['scout']
            shared = {k: r[k] for k in ('receiver','emitted','photo_pairs')}
            if ik in initial:
                equal(shared, initial[ik], 'All 16 initial conditions equal')
            else:
                initial[ik] = shared
        if r['update'] == 2400:
            endpoint_records[r['seed'],r['condition'],r['scout']] = r
    check(set(decoded) == set(product(SEEDS,CONDITIONS,TIMES,range(2))), 'Full record grid')
    expected_summaries, expected_effects = trajectory_summaries(summaries)
    equal(result['summaries'], expected_summaries, 'Probe seed means/AUC')
    equal(result['effects'], expected_effects, 'Probe paired contrasts')
    equal(analyzed['same_photo_trajectory'], expected_summaries, 'Analysis same-photo summaries')
    equal(analyzed['same_photo_effects'], expected_effects, 'Analysis same-photo contrasts')

    endpoints, npz_hashes = [], {}
    map_lookup = {m:i for i,m in enumerate(MAPS)}
    normal_anchors, same_photo_rows = 0, 0
    for seed, condition in product(SEEDS, CONDITIONS):
        _,kind,_,train,held = split_info(condition)
        for mode in MODES:
            path = batch/f's{seed}_{condition}'/f'final_{mode}.npz'
            npz_hashes[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            check(len(arrays['scout']) == 9600, 'Endpoint world denominator')
            positions, goals, menu, action = (arrays[k] for k in ('positions','goals','menu','action'))
            check(positions.shape == goals.shape == action.shape == (9600,2) and menu.shape == (9600,2,6), 'Endpoint shapes')
            check(np.array_equal(np.sort(goals,axis=1), np.tile([0,1],(9600,1))), 'Two distinct private goals')
            check(np.array_equal(np.sort(menu,axis=2), np.tile(np.arange(6),(9600,2,1))), 'Complete local menus')
            check(np.issubdtype(action.dtype,np.integer) and ((action>=0)&(action<6)).all(), 'Legal action indices')
            places = np.take_along_axis(menu, action[:,:,None], axis=2)[:,:,0]
            targets = np.take_along_axis(positions, goals, axis=1)
            success = places == targets
            check(np.array_equal(places, arrays['place']) and np.array_equal(success,arrays['successes']), 'Independent physical settlement')
            reward = success.all(1).astype(float) if kind == 'joint' else success.mean(1)
            check(np.array_equal(reward, arrays['reward']), 'Native reward is unscaled by policy gain')
            mid = np.asarray([map_lookup[tuple(x)] for x in positions])
            if mode == 'normal':
                for scout in range(2):
                    ix = np.flatnonzero(arrays['scout'] == scout)
                    check(len(ix) == 4800, 'Balanced endpoint directions')
                    rec = endpoint_records[seed,condition,scout]
                    table = decoded[seed,condition,2400,scout]
                    codes = arrays['delivered'][ix,0]*7+arrays['delivered'][ix,1]
                    physical = np.take_along_axis(table[codes],goals[ix],axis=1)
                    check(np.array_equal(physical,places[ix]), 'All actual normal actions match saved greedy logits')
                    photo_lookup = {tuple(p): i for i,p in enumerate(photos)}
                    matches, contexts = 0,set()
                    for i in ix:
                        pair = tuple(arrays['photo_ids'][i])
                        if pair in photo_lookup:
                            pi = photo_lookup[pair]
                            check(rec['emitted'][mid[i]][pi] == arrays['sent'][i].tolist(), 'Same-photo endpoint message anchor')
                            matches += 1
                            contexts.add((int(mid[i]),pi))
                    check(matches > 0, 'Endpoint photo overlap required')
                    same_photo_rows += matches
                    expected_anchor = dict(actual_worlds=4800, physical_actions_checked=9600,
                        both_correct=int(success[ix].all(1).sum()), same_photo_message_rows_checked=matches,
                        distinct_same_photo_contexts_checked=len(contexts),
                        scope='All actual receiver actions; sender identity only for recorded matching photo/map contexts.')
                    equal(rec['endpoint_anchor'],expected_anchor,'Saved endpoint anchor')
                    normal_anchors += 1
            for part, ids in [('train',train),('heldout',held)]:
                mask = np.isin(mid,ids)
                if not mask.any():
                    continue
                check(int(mask.sum()) == len(ids)*320, 'Balanced final map support')
                endpoints.append(dict(seed=seed,condition=condition,mode=mode,partition=part,
                    J=float(success[mask].all(1).mean()),n=int(mask.sum()),single_accuracy=float(success[mask].mean()),joint_correct=int(success[mask].all(1).sum()),
                    native_reward=float(reward[mask].mean())))
    equal(analyzed['endpoints'],endpoints,'Independent all-mode J/native endpoint recount')
    endpoint_seeds = endpoint_seed_rows(endpoints)
    equal(analyzed['endpoint_seed_values'],endpoint_seeds,'J four-seed aggregation')
    endpoint_effects = factorial(endpoint_seeds,('family','partition','mode','update'),('J',))
    equal(analyzed['endpoint_effects'],endpoint_effects,'J paired interactions')
    equal(result['counts'],dict(directions=1024,receiver_inputs=100352,natural_messages=491520,
        final_anchors=128,initial_condition_comparisons=120,receiver_maximum_ties=0),'Probe total counts')
    check('torch' not in sys.modules and 'camp' not in sys.modules, 'No neural runtime imported')
    primary_j = next(r for r in endpoint_effects if r['family']=='split' and r['partition']=='heldout' and r['mode']=='normal')
    primary_process = [r for r in expected_effects['early_auc'] if r['family']=='split' and r['partition']=='heldout']
    return dict(status='passed',audited_at=datetime.now(timezone.utc).isoformat(),
        audit_script_sha256=sha(__file__), source_files_sha256=source_hashes,
        record_files_sha256=record_hashes, endpoint_files_sha256=npz_hashes,
        probe_result_sha256=sha(probe_dir/'execution/results.json'), analysis_sha256=sha(analysis_file),
        counts=dict(records=1024,maps=30720,receiver_logit_vectors=100352,same_photo_messages=491520,
                    initial_comparisons=120,endpoint_worlds=64*5*9600,normal_anchors=normal_anchors,
                    matched_endpoint_photo_rows=same_photo_rows), primary_J2400=primary_j,primary_process_AUC=primary_process,
        weights_loaded=0,neural_forward_calls=0,
        limitations=['Saved logits are recomputed numerically, not independently regenerated from weights.',
                     'No new visual data; J and fixed-photo N remain distinct denominators.',
                     'Full training/optimizer replay and checkpoint-curve raw evaluation reconstruction are outside this audit.'])


def self_test():
    actions = np.asarray([(min(f,5),min(w,5)) for f,w in product(range(7),repeat=2)])
    messages = np.tile(np.asarray(MAPS)[:,None,:],(1,16,1))
    good = measure(actions,messages,'split1_additive_gain1')
    check(good['summaries']['heldout']['U_full49']==fraction(6,6),'synthetic complete U')
    check(good['summaries']['heldout']['N_both']==fraction(96,96),'synthetic complete N')
    constant = np.tile([0,1],(30,16,1))
    fixed = measure(actions,constant,'full_joint_gain3')
    check(fixed['summaries']['all']['N_both']==fraction(16,480),'constant code works on one map')
    check(fixed['summaries']['all']['natural_failure_U1']==fraction(464,480),'available but unused classification')
    missing = np.tile([0,0],(49,1)); missing[1]=[1,0]; missing[2]=[0,2]
    marginal = measure(missing,np.zeros((30,16,2),int),'split2_joint_gain1')
    check(marginal['maps'][MAPS.index((1,2))]['category']=='both_marginals_no_joint_code','marginal vs joint')
    z = np.zeros((49,2,6));z[:,:,0]=1
    a,_,_ = physical_lookup(z)
    check(np.all(a==0),'unique synthetic lookup')
    for bad in ('tie','nan','inf'):
        copy=z.copy();copy[0,0,1]={'tie':1,'nan':np.nan,'inf':np.inf}[bad]
        try:
            physical_lookup(copy)
        except AssertionError:
            pass
        else:
            raise AssertionError('Must reject '+bad)
    rows=[]
    for s,k,g in product(SEEDS,KINDS,GAINS):
        value={('additive',1):.1,('additive',3):.2,('joint',1):.3,('joint',3):.5}[k,g]
        rows.append(dict(seed=s,kind=k,gain=g,Y=value))
    interaction=factorial(rows,(),('Y',))[0]['effects']['interaction']
    equal(interaction['seed_values'],[.1]*4,'factorial contrast synthetic truth')
    check(abs(np.asarray([50,150,250,450,300])@np.asarray([0,100,300,600,1200])/1200-600)<1e-12,'linear AUC quadrature')
    check('torch' not in sys.modules and 'camp' not in sys.modules,'self-test has no neural imports')
    return dict(status='synthetic_passed',weights_loaded=0,neural_forward_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--batch',type=Path)
    parser.add_argument('--probe',type=Path)
    parser.add_argument('--analysis',type=Path)
    parser.add_argument('--out',type=Path,help='new audit output prefix')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),indent=2));return
    check(all((args.batch,args.probe,args.analysis,args.out)),'Specify complete batch/probe/analysis and a new output prefix')
    target=args.out.with_suffix('.json');md=args.out.with_suffix('.md')
    check(not target.exists() and not md.exists(),'Never overwrite an existing audit')
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=audit(args.batch,args.probe,args.analysis)
    except Exception as error:
        failure=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),
                     audit_script_sha256=sha(__file__),weights_loaded=0,neural_forward_calls=0)
        with target.open('x') as stream:json.dump(failure,stream,ensure_ascii=False,indent=2)
        raise
    with target.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    with md.open('x') as stream:
        stream.write('# 独立结果数学核验\n\n全部数值核对通过。直接从保存的 logits、消息及五种终点 NPZ 重算；没有导入正式测量函数、加载权重或执行新前向。\n\n')
        stream.write('核验计数：`'+json.dumps(result['counts'],ensure_ascii=False)+'`。\n\n')
        stream.write('主 J2400 交互四种子：`'+str(result['primary_J2400']['effects']['interaction']['seed_values'])+'`。完整AUC与对照保留在JSON。\n\n')
        stream.write('这是保存结果的独立数学重算，不是权重再推断或完整优化器训练重放。J和N分母保持分开；通过审计不等于语言形成证据。\n')
    print(json.dumps(dict(status='passed',output=str(target),counts=result['counts']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
