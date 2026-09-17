"""Post-hoc, read-only receiver coverage for complete v0.8 results.

No policy/trainer/Torch imports. Uses saved 49-code tables and (with NumPy)
final_normal.npz only. Never writes under redesign_v0.8. All 60 runs are required.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / 'redesign_v0.8/results/complementarity_001'
SEEDS = (27101, 27102, 27103, 27104)
KINDS = {'additive': 0., 'mixed': .5, 'joint': 1.}
MAPS = list(permutations(range(6), 2))
MATCHINGS = {1: ((0,1),(2,3),(4,5)), 2: ((0,2),(1,4),(3,5)), 3: ((0,3),(1,5),(2,4))}
SOURCE_SHA = {
    'analyze_protocols.py': '33ef44b59d2ae7042e02af1750404bf8dd1054b8d1ff299c3453d14ad63fd8a5',
    'run_experiment.py': '485221ac4c90f69b88047f380edfaf4922973cff986c8bfa5c47e700cab13d62',
    '固定执行方案.md': 'cf3d92f5eb9755a83b11dcbd3f1792f5474203ce445624fc7b347ccf499996b4',
    'camp.py': '21ad5ff2eecf871ff2e8e17e5623875f55d701e66302b741a46658c0cec81854',
}


def require(value, text):
    if not value:
        raise AssertionError(text)


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rate(n, d): return n / d if d else None
def project_path(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT/path).resolve()


def expected_conditions():
    return {f'split{s}_{k}': (s,k,False) for s in MATCHINGS for k in KINDS} | {
        f'{prefix}_{k}': (0,k,blocked) for prefix,blocked in (('full',False),('blocked',True)) for k in KINDS}


def decoder_table(direction):
    ma = direction['menu_audit']
    require(ma['enumerated_messages'] == 49 and ma['goals'] == 2 and ma['menus'] == 720 and ma['cases'] == 70560,
            'Incomplete full-code/menu enumeration')
    require(ma['all_menu_permutations_physically_equivalent'] is True and ma['menus_used_after_check'] == 1,
            'Menu non-invariance: stop the entire analysis. Independent query menus require a separately specified estimand.')
    rows = direction['receiver_decoder_table']
    require(len(rows) == 49 and {tuple(r['message']) for r in rows} == set(product(range(7), repeat=2)), 'Incomplete 49-code table')
    result = {}
    for row in rows:
        acts = tuple(row['actions_by_goal'])
        require(len(acts) == 2 and all(type(x) is int and 0 <= x < 6 for x in acts), 'Invalid physical decoder action')
        for goal in (0,1):
            require(row['menu_action_counts_by_goal'][goal] == [720 if x == acts[goal] else 0 for x in range(6)],
                    'Stored menu action counts are not invariant')
        result[tuple(row['message'])] = acts
    return result


def map_coverage(target, decoder):
    joint = target in decoder.values()
    food = any(actions[0] == target[0] for actions in decoder.values())
    water = any(actions[1] == target[1] for actions in decoder.values())
    kind = ('joint' if joint else 'both_marginals_no_joint' if food and water else
            'food_only' if food else 'water_only' if water else 'neither')
    return dict(U=int(joint), food_marginal_reachable=int(food), water_marginal_reachable=int(water),
                marginal_class=kind, **{k:int(kind == k) for k in ('both_marginals_no_joint','food_only','water_only','neither')})


def protocol_map_rows(phase, decoder, blocked):
    """Collapse only the producer's explicitly redundant hidden-goal axis."""
    require(len(phase['photo_pairs']) == len({tuple(p) for p in phase['photo_pairs']}) == 16, 'Expected16 unique photo pairs')
    require(len(phase['codebook']) == 60, 'Expected30 maps times2 compatibility indices')
    unchanged = phase['cross_goal']['message_unchanged_across_sender_goal']
    require(unchanged['numerator'] == unchanged['denominator'] == 480 and unchanged['rate'] == 1,
            'Producer did not report exact duplicated hidden-goal messages')
    grouped = defaultdict(list)
    for row in phase['codebook']:
        map_id = row['map_id']; target = (row['food_location'],row['water_location'])
        require(type(map_id) is int and 0 <= map_id < 30 and target == MAPS[map_id], 'Map identity mismatch')
        require(row['sender_goal'] in (0,1), 'Unknown compatibility index')
        frequencies = {tuple(r['message']): r['count'] for r in row['delivered_messages']}
        emitted = {tuple(r['message']): r['count'] for r in row['emitted_messages']}
        require(len(frequencies) == len(row['delivered_messages']) and sum(frequencies.values()) == 16 and
                all(type(n) is int and n > 0 and code in decoder for code,n in frequencies.items()), 'Invalid message frequencies')
        require(len(emitted) == len(row['emitted_messages']) and sum(emitted.values()) == 16 and
                all(code in decoder and type(n) is int and n > 0 for code,n in emitted.items()),
                'Invalid emitted frequencies')
        require(frequencies == ({(0,0):16} if blocked else emitted), 'Protocol actual channel delivery differs')
        counts = {'both':sum(n for code,n in frequencies.items() if decoder[code] == target),
                  'native':sum(n for code,n in frequencies.items() if decoder[code][row['sender_goal']] == target[row['sender_goal']]),
                  'switched':sum(n for code,n in frequencies.items() if decoder[code][1-row['sender_goal']] == target[1-row['sender_goal']])}
        for key,n in counts.items():
            require(row[key] == {'numerator':n,'denominator':16,'rate':n/16}, 'Codebook score not reproduced from actual delivered code')
        grouped[map_id].append((row,frequencies))
    require(set(grouped) == set(range(30)), 'Missing map codebook')
    result = []
    for map_id in range(30):
        pair = sorted(grouped[map_id],key=lambda r:r[0]['sender_goal'])
        require(len(pair) == 2 and [r[0]['sender_goal'] for r in pair] == [0,1] and pair[0][1] == pair[1][1],
                'Compatibility indices are not duplicate frequency observations')
        require({tuple(r['message']):r['count'] for r in pair[0][0]['emitted_messages']} ==
                {tuple(r['message']):r['count'] for r in pair[1][0]['emitted_messages']},
                'Emitted frequency differs across hidden compatibility indices, even if blocked delivery is constant')
        require(pair[0][0]['both'] == pair[1][0]['both'], 'Compatibility indices have different joint success')
        target = MAPS[map_id]
        coverage = map_coverage(target, decoder)
        u = coverage['U']
        allowed_u = decoder[(0,0)] == target if blocked else u
        n = pair[0][0]['both']['numerator']
        require(u or n == 0, 'Natural success outside full receiver coverage')
        require(allowed_u or n == 0, 'Natural success outside actually allowed channel')
        allowed_class = map_coverage(target,{(0,0):decoder[(0,0)]})['marginal_class'] if blocked else coverage['marginal_class']
        result.append(dict(map_id=map_id, target=list(target), **coverage, U_channel_allowed=int(allowed_u),
            channel_allowed_marginal_class=allowed_class,
            protocol_N=n, protocol_n=16, protocol_failure_no_code=0 if u else 16,
            protocol_failure_no_allowed_code=0 if allowed_u else 16))
    require(sum(r['protocol_N'] for r in result)*2 == phase['cross_goal']['same_message_correct_for_both_goals']['numerator'],
            'Natural all-map aggregate does not match duplicated producer count')
    return result


def audit_normal_arrays(a, decoders, blocked, expected_n=9600):
    """Replay one stored greedy endpoint world and its two queries; no network."""
    import numpy as np
    n = expected_n
    shapes = {'scout':(n,), 'episode':(n,), 'positions':(n,2), 'photo_ids':(n,2), 'goals':(n,2), 'menu':(n,2,6),
              'inventory':(n,2), 'history':(n,18), 'sent':(n,2), 'delivered':(n,2),
              'action':(n,2), 'place':(n,2), 'successes':(n,2), 'reward':(n,)}
    require(all(k in a and a[k].shape == shape for k,shape in shapes.items()), 'Paired-world endpoint shapes differ')
    require(all(np.issubdtype(a[k].dtype,np.integer) for k in ('scout','episode','positions','photo_ids','goals','menu','sent','delivered','action','place')),
            'Expected integer identity/message/action arrays')
    for scout in (0,1):
        require(np.array_equal(np.sort(a['episode'][a['scout'] == scout]),np.arange(n//2)), 'Actual per-direction world IDs differ')
    require(np.isin(a['scout'],[0,1]).all() and np.array_equal(np.sort(a['goals'],axis=1),np.tile([0,1],(n,1))), 'Two resource queries required')
    require(np.array_equal(np.sort(a['menu'],axis=2),np.tile(np.arange(6),(n,2,1))), 'Both independent private menus must be complete')
    require(((a['positions'] >= 0)&(a['positions'] < 6)).all() and (a['positions'][:,0] != a['positions'][:,1]).all(), 'Invalid map positions')
    require((a['inventory'] == 0).all() and (a['history'] == 0).all(), 'Endpoint receiver context is not frozen empty context')
    require(all(((a[k] >= 0)&(a[k] < 7)).all() for k in ('sent','delivered')) and
            ((a['action'] >= 0)&(a['action'] < 6)).all(), 'Message/action range differs')
    require(np.array_equal(a['delivered'],np.zeros_like(a['sent']) if blocked else a['sent']), 'Not actual normal-mode delivery')
    places = np.take_along_axis(a['menu'],a['action'][...,None],axis=2).squeeze(2)
    wanted = np.take_along_axis(a['positions'],a['goals'],axis=1)
    successes = places == wanted
    require(np.array_equal(a['place'],places) and np.array_equal(a['successes'],successes), 'Stored two-action settlement differs')
    joint = successes.all(axis=1)
    u = np.empty(n,dtype=bool); allowed_u = np.empty(n,dtype=bool)
    for scout in (0,1):
        ix = a['scout'] == scout
        table = np.asarray([decoders[scout][code] for code in product(range(7),repeat=2)],dtype=int)
        codes = a['delivered'][ix,0]*7+a['delivered'][ix,1]
        expected_places = np.take_along_axis(table[codes],a['goals'][ix],axis=1)
        require(np.array_equal(places[ix],expected_places), 'Saved final decisions differ from protocol greedy decoder; do not silently mix functions')
        u[ix] = np.asarray([tuple(p) in decoders[scout].values() for p in a['positions'][ix]])
        allowed_u[ix] = np.asarray([tuple(p) == decoders[scout][(0,0)] for p in a['positions'][ix]]) if blocked else u[ix]
    require(not np.any(joint & ~u) and not np.any(joint & ~allowed_u), 'Endpoint success outside usable receiver coverage')
    map_ids = a['positions'][:,0]*5+a['positions'][:,1]-(a['positions'][:,1]>a['positions'][:,0])
    return dict(joint=joint, successes=successes, U=u, U_channel_allowed=allowed_u, map_ids=map_ids)


COUNT_FIELDS = ('map_units','U','U_channel_allowed','protocol_N','protocol_n','protocol_failure_no_code',
                'protocol_failure_no_allowed_code','endpoint_J','endpoint_n','endpoint_failure_no_code','endpoint_failure_no_allowed_code',
                'food_marginal_reachable','water_marginal_reachable','both_marginals_no_joint','food_only','water_only','neither')


def totals(rows):
    r = {k:sum(x[k] for x in rows) for k in COUNT_FIELDS}
    r.update(U_rate=rate(r['U'],r['map_units']), U_channel_allowed_rate=rate(r['U_channel_allowed'],r['map_units']),
        both_marginals_no_joint_rate=rate(r['both_marginals_no_joint'],r['map_units']),
        protocol_N_rate=rate(r['protocol_N'],r['protocol_n']), endpoint_J_rate=rate(r['endpoint_J'],r['endpoint_n']),
        protocol_failure_no_code_share=rate(r['protocol_failure_no_code'],r['protocol_n']-r['protocol_N']),
        endpoint_failure_no_code_share=rate(r['endpoint_failure_no_code'],r['endpoint_n']-r['endpoint_J']),
        endpoint_failure_no_allowed_code_share=rate(r['endpoint_failure_no_allowed_code'],r['endpoint_n']-r['endpoint_J']))
    return r


def analyze(batch):
    batch = Path(batch).resolve()
    path = batch/'protocol_analysis.json'
    require(path.exists(), 'Complete protocol_analysis.json is not available; no final output is written')
    p = read(path); conditions = expected_conditions()
    require(p['status'] == 'complete' and not p['missing_runs'] and p['expected_seeds'] == list(SEEDS) and
            p['expected_run_count'] == len(p['runs']) == 60 and
            {(r['seed'],r['condition']) for r in p['runs']} == set(product(SEEDS,conditions)), 'All fixed60 runs required; never analyze selected completed conditions')
    require(p['fingerprints']['analysis'] == SOURCE_SHA['analyze_protocols.py'] and p['fingerprints']['camp'] == SOURCE_SHA['camp.py'],
            'Protocol producer schema/source changed; explicit review required')
    for name,digest in SOURCE_SHA.items():
        require(sha(ROOT/'redesign_v0.8'/name) == digest, 'Previously reviewed source changed: '+name)
    photo_manifest = ROOT/'redesign_v0.4/data/manifest.json'
    require(p['fingerprints']['photo_manifest'] == sha(photo_manifest), 'Protocol photo manifest changed')
    photos = read(photo_manifest)['images']
    import numpy as np
    direction_rows, source_files, endpoint_rows, map_level_records = [], [path], [], []
    prepared_hashes = {}
    for run in p['runs']:
        seed, name = run['seed'], run['condition']; split,kind,blocked = conditions[name]
        folder = batch/f's{seed}_{name}'; cfg, result = read(folder/'config.json'), read(folder/'result.json')
        expected_plan = dict(split=split,representation='identity',schedule='direct',vocab=7,length=2,known=False,
                             blocked=blocked,reward_kind=kind,complementarity=KINDS[kind])
        require(cfg['plan'] == result['plan'] == run['plan'] == expected_plan and cfg['seed'] == result['seed'] == seed and
                cfg['condition'] == result['condition'] == name, 'Run identity differs')
        require(cfg['updates'] == result['updates'] == 2400 and cfg['batch'] == result['batch'] == 512 and
                cfg['eval_n'] == 9600 and cfg['choices_per_world'] == 2 and cfg['trace_schema'] == 'paired_world_v1', 'Budget/schema differs')
        require(cfg['map_table'] == [list(x) for x in MAPS], 'Map table differs')
        for source_name in ('camp.py','run_experiment.py','固定执行方案.md'):
            require(cfg['source_hashes'][source_name] == SOURCE_SHA[source_name], 'Run config source differs')
        final_path = folder/'final.pt'; prepared_path = batch/f'prepared_{seed}.pt'
        require(project_path(run['final_checkpoint']) == final_path and
                project_path(run['prepared_checkpoint']) == prepared_path and
                project_path(cfg['prepared_source']['path']) == prepared_path, 'Checkpoint source path anchor differs')
        require(sha(final_path) == run['final_sha256'], 'Protocol final checkpoint byte hash differs')
        if seed not in prepared_hashes: prepared_hashes[seed] = sha(prepared_path)
        require(prepared_hashes[seed] == run['prepared_sha256'] == cfg['prepared_source']['sha256'], 'Prepared checkpoint byte hash differs')
        source_files.append(final_path)
        held = {MAPS.index(pair) for edge in MATCHINGS.get(split,()) for pair in (edge,edge[::-1])}
        train = set(range(30))-held
        require(set(run['heldout_map_ids']) == set(cfg['heldout_map_ids']) == held and
                set(run['train_map_ids']) == set(cfg['train_map_ids']) == train, 'Support differs')
        require(len(run['directions']) == 2 and {d['scout'] for d in run['directions']} == {0,1}, 'Two directions required')
        decoders = {d['scout']:decoder_table(d) for d in run['directions']}
        normal = folder/'final_normal.npz'; source_files += [folder/'config.json',folder/'result.json',normal]
        with np.load(normal,allow_pickle=False) as z:
            a = {k:z[k] for k in z.files}
        evaluated = audit_normal_arrays(a,decoders,blocked)
        for resource,category in enumerate(('food','water')):
            ids = a['photo_ids'][:,resource]
            require(((ids >= 0)&(ids < len(photos))).all() and all(photos[int(i)]['split'] == 'test' and
                    photos[int(i)]['category'] == category for i in np.unique(ids)), 'Actual photo IDs do not match old test categories')
        require(Counter(zip(evaluated['map_ids'].tolist(),a['goals'][:,0].tolist(),a['scout'].tolist())) ==
                Counter({(m,g,d):80 for m,g,d in product(range(30),range(2),range(2))}), 'Endpoint grid is not9600 balanced worlds')
        by_resource = np.take_along_axis(evaluated['successes'],np.argsort(a['goals'],axis=1),axis=1).astype(int)
        expected_reward = (1-KINDS[kind])*by_resource.mean(axis=1)+KINDS[kind]*by_resource.prod(axis=1)
        require(np.array_equal(a['reward'],expected_reward.astype(a['reward'].dtype)), 'Lambda utility mismatch')
        for subset, ids in [('seen',train),('unseen',held)]:
            ix = np.isin(evaluated['map_ids'],list(ids)); stated = result['scores']['normal']['map_groups'][subset]
            require(stated['n'] == int(ix.sum()) and stated['both_correct'] == int(evaluated['joint'][ix].sum()) and
                    stated['both_accuracy'] == rate(stated['both_correct'],stated['n']), 'Primary endpoint J differs from recorded exact counts')
        require(result['scores']['normal']['both_correct'] == int(evaluated['joint'].sum()), 'All-world primary J differs')
        family = 'split' if split else 'blocked' if blocked else 'full'
        for d in run['directions']:
            require(d['collector'] == 1-d['scout'], 'Recipient identity differs')
            phase = d['phases']['validation']
            require(phase['photo_pairs'] == p['photo_splits']['validation'], 'Protocol validation photos differ')
            maps = protocol_map_rows(phase,decoders[d['scout']],blocked)
            for m in maps:
                ix = (a['scout'] == d['scout']) & (evaluated['map_ids'] == m['map_id'])
                require(int(ix.sum()) == 160, 'Each direction/map needs160 actual endpoint worlds')
                map_level_records.append(dict(seed=seed,condition=name,family=family,reward_kind=kind,split=split,
                    scout=d['scout'],subset='heldout' if m['map_id'] in held else 'train' if split else 'all',
                    **m,endpoint_J=int(evaluated['joint'][ix].sum()),endpoint_n=160))
            subsets = [('train',train),('heldout',held)] if split else [('all',train)]
            for subset,ids in subsets:
                selected = [m for m in maps if m['map_id'] in ids]
                old = phase['cross_goal_groups']['seen' if subset in ('train','all') else 'unseen']['same_message_correct_for_both_goals']
                require(old['numerator'] == 2*sum(m['protocol_N'] for m in selected) and old['denominator'] == 32*len(ids),
                        'Protocol subset counts differ after removing duplicate index')
                ix = (a['scout'] == d['scout']) & np.isin(evaluated['map_ids'],list(ids))
                counts = {k:sum(m[k] for m in selected) for k in ('U','U_channel_allowed','protocol_N','protocol_n',
                    'protocol_failure_no_code','protocol_failure_no_allowed_code','food_marginal_reachable',
                    'water_marginal_reachable','both_marginals_no_joint','food_only','water_only','neither')}
                counts.update(map_units=len(ids),endpoint_J=int(evaluated['joint'][ix].sum()),endpoint_n=int(ix.sum()),
                    endpoint_failure_no_code=int((~evaluated['U'][ix]).sum()),
                    endpoint_failure_no_allowed_code=int((~evaluated['U_channel_allowed'][ix]).sum()))
                direction_rows.append(dict(seed=seed,condition=name,family=family,reward_kind=kind,split=split,
                                            scout=d['scout'],subset=subset,**counts))
        endpoint_rows.append(dict(seed=seed,condition=name,worlds=9600,choices=19200,
                                 J=int(evaluated['joint'].sum()),paired_decisions_match_decoder=True))
    groups = []
    for family in ('split','full','blocked'):
        for kind in KINDS:
            for subset in (('train','heldout') if family == 'split' else ('all',)):
                selected = [r for r in direction_rows if (r['family'],r['reward_kind'],r['subset']) == (family,kind,subset)]
                per_seed = [dict(seed=s,**totals([r for r in selected if r['seed'] == s])) for s in SEEDS]
                pooled = totals(selected)
                for field in ('U_rate','U_channel_allowed_rate','protocol_N_rate','endpoint_J_rate'):
                    require(abs(mean(r[field] for r in per_seed)-pooled[field]) < 1e-12, 'Equal seed weighting differs')
                groups.append(dict(family=family,reward_kind=kind,subset=subset,independent_seed_n=4,
                    per_seed=per_seed,pooled_counts=pooled,
                    seed_means={field:mean(r[field] for r in per_seed) for field in ('U_rate','U_channel_allowed_rate','protocol_N_rate','endpoint_J_rate')},
                    failure_share_note='Pooled failure shares are count-weighted descriptions; see per-seed failure shares, including None for no failures.'))
    require(not any(k in sys.modules for k in ('torch','mlx','mlx.core','mlx_lm')), 'Policy runtime unexpectedly imported')
    require(len(map_level_records) == 3600 and all(g['pooled_counts']['map_units'] == sum(g['pooled_counts'][k]
            for k in ('U','both_marginals_no_joint','food_only','water_only','neither')) for g in groups), 'Marginal classification is not exhaustive')
    return dict(status='complete_readonly_posthoc_analysis',generated_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(__file__),batch=str(batch),reviewed_producer_sha256=SOURCE_SHA,
        source_sha256={str(f):sha(f) for f in source_files},prepared_checkpoint_byte_sha256=prepared_hashes,
        independent_seeds=list(SEEDS),social_runs=60,
        endpoint_worlds_checked=576000,endpoint_choices_checked=1152000,decoder_directions_checked=120,
        protocol_duplicate_index_removed=True,groups=groups,direction_subset_counts=direction_rows,map_level_records=map_level_records,
        endpoint_runs=endpoint_rows,new_model_calls=0,new_training_updates=0,
        scope=['Endpoint J is the predetermined paired-world behavioral outcome; U and U=0 failure decomposition are post-hoc additions.',
            'Protocol N uses16 validation photo pairs per map; the two legacy sender_goal indices are duplicate compatibility axes, not two actual sends.',
            'Protocol codebooks preserve per-map frequencies, not each photo pair to message assignment. They cannot reconstruct photo-matched natural/stitch interventions.',
            'Actual endpoint NPZ preserves each world photo_ids, one delivered message, two goal orders/menus/actions; this audit aligns J and U on those same saved worlds.',
            'Only final_normal.npz is replayed; five-mode execution audit, neural checkpoint evaluation and training/optimizer replay are not repeated.',
            'Checkpoint files are hashed as bytes only, never deserialized or loaded into a neural model.',
            'Both query menus are independently generated. Binary coverage results require stored physical action invariance for all720 menus; any exception stops the entire analysis.',
            'For blocked runs, U over49 codes is counterfactual receiver capacity; only the [0,0] channel-allowed column is actionable.',
            'Four independent pair seeds; split/direction/world/photo/query/recoding counts do not enlarge independent n.',
            'No claim that improved J establishes compositional structure or a unique internal mechanism.'])


def markdown(r):
    rows = ['# v0.8 终点共同成功与接收覆盖：事后只读分解','',
        f"生成时间：{r['generated_utc']}。完整60运行、4种子；只读576000个正常终点世界（1152000次选择）及120张接收表，新模型调用0。",'',
        'J是既定真实双查询终点指标；U及失败分解是本次事后添加。协议N来自另16验证照片对，去除了旧sender_goal兼容重复轴，不与J逐照片相减。', '',
        '| 范围 | 回报 | 支持 | 全49码U | 协议N | 真实终点J | J失败中U=0 | 通道实际允许U |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    pct = lambda x:'未定义' if x is None else f'{100*x:.2f}%'
    for g in r['groups']:
        v=g['pooled_counts'];rows.append(f"| {g['family']} | {g['reward_kind']} | {g['subset']} | {pct(v['U_rate'])} | {pct(v['protocol_N_rate'])} | {pct(v['endpoint_J_rate'])} | {pct(v['endpoint_failure_no_code_share'])} | {pct(v['U_channel_allowed_rate'])} |")
    rows += ['', '上表成功率与覆盖率等价于先在种子内平均三划分／两方向，再平均四种子；失败占比为合并失败分母的描述，逐种子分子／分母另存JSON。full/blocked各只有一个split0，不复制为三组。', '',
        'blocked的全49码U包含实际不会投递的码，不能称为可用通道上界；需同时查看仅[0,0]的通道允许U与相应失败分类。', '',
        '全部720菜单的保存物理动作计数等价后才使用二元同码U。两需求必须使用同一码；不按目标、照片、菜单分别选码。若任何方向不等价，本程序拒绝生成完整汇总，不剔除该方向。', '',
        '协议JSON不保存逐照片消息对应；真实终点NPZ保存照片、实际单条消息与两次行动，本次由它们重放J并对齐U。没有模型推断、参数更新或五模式全审计。U是原贪心接收函数的码空间存在性，不说明发送者实际能找到该码或学出词典。', '',
        'λ改变部分成功效用和学习信号；三组都以两资源全成功为最优。不得以跨λ不同定义的R代替共同J，不把J改善直接解释为句法或组合结构。四新种子与旧照片仍属于本系统的探索。']
    rows += ['', '| 范围 | 回报 | 支持 | 同码可达 | 两边际可达但无同码 | 仅食物 | 仅水 | 两边际皆不可达 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for g in r['groups']:
        v=g['pooled_counts'];counts=' | '.join(f"{v[k]}/{v['map_units']}" for k in ('U','both_marginals_no_joint','food_only','water_only','neither'))
        rows.append(f"| {g['family']} | {g['reward_kind']} | {g['subset']} | {counts} |")
    rows += ['', '边际表使用全部49码：食物与水分别可能由不同码正确，不等于存在一条共同码。五类互斥完备；全部3600个地图／方向／运行单元及通道允许码分类均存JSON，blocked仍须按仅零码的实际通道解释。']
    return '\n'.join(rows)+'\n'


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,default=DEFAULT_BATCH)
    parser.add_argument('--out',type=Path,required=True,help='New research_program/v08_* prefix; JSON and Markdown')
    args=parser.parse_args(); out=args.out.resolve()
    require(out.parent == ROOT/'research_program' and out.name.startswith('v08_'), 'Outputs must be new research_program/v08_* files')
    jp,mp=out.with_suffix('.json'),out.with_suffix('.md')
    require(not jp.exists() and not mp.exists(), 'Preserve existing output')
    result=analyze(args.batch)
    with jp.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    with mp.open('x',encoding='utf-8') as f:f.write(markdown(result))
    print(json.dumps({'status':result['status'],'runs':60,'new_model_calls':0,'out':str(out)},ensure_ascii=False))
