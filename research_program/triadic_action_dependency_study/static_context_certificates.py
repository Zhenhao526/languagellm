"""Finite information-necessity certificates, reading frozen preparation only.

No model files/results are opened. Demand/owner stay fixed while the two
LL-hidden sites exchange resources. Counts enumerate unordered layout pairs;
expanded witnesses overlap and are NOT an estimate of a no-communication rate.
"""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import argparse
import json
import time
from . import environment as env

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PARTITIONS = ('train', 'new_needs', 'new_layouts', 'new_needs_and_layouts')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def text_sha(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':')).encode()).hexdigest()


def unique_plan(needs, layout):
    plans = env.full_success_plans(needs, layout)
    require(len(plans) == 1, 'Expected exactly one native full-success plan')
    return plans[0]


def actions_for_plan(plan):
    i, j, site, destination = plan
    actions = {a: {'kind': 'wait'} for a in env.AGENTS}
    for who, other in ((i, j), (j, i)):
        actions[env.AGENTS[who]] = dict(kind='transport', site=env.SITES[site],
            destination=env.DESTINATIONS[destination], partner=env.AGENTS[other])
    return actions


def make_example(spec, condition, need_index):
    needs = tuple(spec['needs'][need_index])
    owner = tuple(condition['private_sites'])
    layouts = tuple(tuple(condition[k]) for k in ('layout_before', 'layout_after'))
    listener = condition['listener_index']; agent = env.AGENTS[listener]
    states = [env.State(needs, layout, owner) for layout in layouts]
    plans = [unique_plan(s.needs, s.layout) for s in states]
    actions = [actions_for_plan(p) for p in plans]
    ll = [env.observe(s, agent, information='LL') for s in states]
    pl = [env.observe(s, agent, information='PL') for s in states]
    correct = [a[agent] for a in actions]
    require(ll[0] == ll[1], 'Official LL observations differ')
    require(pl[0] != pl[1], 'Official PL observations are identical')
    require(all(a['kind'] == 'transport' for a in correct), 'Listener can escape by waiting')
    require(correct[0]['site'] != correct[1]['site'], 'Required transport sites did not change')
    require(correct[0]['partner'] == correct[1]['partner'], 'Required partner changed')
    require(correct[0]['destination'] == correct[1]['destination'], 'Destination changed')
    native = [env.settle(s, a) for s, a in zip(states, actions)]
    require(all(r['reward'] == 1 for r in native), 'Witness actions are not full-success')
    wait_rewards = []
    for s, a in zip(states, actions):
        waiting = dict(a); waiting[agent] = {'kind': 'wait'}
        wait_rewards.append(env.settle(s, waiting)['reward'])
    require(wait_rewards == [0., 0.], 'Listener wait should break the only required pair')
    nl, no = len(spec['layouts']), len(spec['private_sites'])
    owner_index = condition['owner_index']
    ids = [(need_index * nl + layout_index) * no + owner_index
           for layout_index in condition['layout_indices']]
    return dict(need_index=need_index, needs=list(needs),
        need_views={a: env.need_view(needs[i]) for i, a in enumerate(env.AGENTS)},
        listener=agent, listener_index=listener, private_sites=list(owner),
        target_material_id=condition['target_material_id'],
        target_material=list(env.MATERIALS[condition['target_material_id']]),
        layout_indices=condition['layout_indices'], layouts=[list(x) for x in layouts],
        layout_site_materials=[{site: list(env.MATERIALS[layout[k]])
                               for k, site in enumerate(env.SITES)} for layout in layouts],
        state_indices=ids, hidden_sites=condition['hidden_sites'],
        correct_full_plans=[list(p) for p in plans], correct_actions=actions,
        listener_correct_actions=correct,
        listener_correct_action_indices=[env.all_actions(agent).index(a) for a in correct],
        listener_success_projection_cardinalities=[1, 1],
        listener_success_projections_disjoint=True, listener_wait_is_full_success=False,
        full_native_rewards=[r['reward'] for r in native],
        listener_wait_with_other_actions_unchanged_rewards=wait_rewards,
        LL_observations=ll, PL_observations=pl,
        LL_observation_sha256=[text_sha(o) for o in ll], PL_observation_sha256=[text_sha(o) for o in pl])


def analyze_partition(spec):
    needs = [tuple(n) for n in spec['needs']]
    layouts = [tuple(x) for x in spec['layouts']]
    owners = [tuple(x) for x in spec['private_sites']]
    require(needs == sorted(needs) and len(needs) == len(set(needs)), 'Need order must be unique lexicographic')
    require(layouts == sorted(layouts) and len(layouts) == len(set(layouts)), 'Layout order must be lexicographic')
    lookup = {layout: i for i, layout in enumerate(layouts)}
    groups = {(listener, material): [] for listener in range(3) for material in range(4)}
    for index, need in enumerate(needs):
        i, j, material, destination = unique_plan(need, (0, 1, 2, 3))
        for listener in (i, j):
            groups[listener, material].append(index)
    require(sum(map(len, groups.values())) == 2 * len(needs), 'Each demand table must have two actual participating actors')
    conditions = []
    pure_backgrounds = set()
    missing_partner_layouts = 0
    for layout_index, before in enumerate(layouts):
        for owner_index, owner in enumerate(owners):
            for listener in range(3):
                visible = (0, owner[listener])
                hidden = tuple(s for s in range(4) if s not in visible)
                require(len(hidden) == 2 and 0 not in hidden, 'Not exactly two nonpublic hidden sites')
                after = list(before)
                x, y = hidden; after[x], after[y] = after[y], after[x]
                after = tuple(after)
                if after not in lookup:
                    missing_partner_layouts += 1
                    continue
                if not before < after:
                    continue
                other_index = lookup[after]
                pure_backgrounds.add((layout_index, other_index, owner_index, listener))
                for material in sorted((before[x], before[y])):
                    indices = groups[listener, material]
                    row = dict(layout_indices=[layout_index, other_index],
                        layout_before=list(before), layout_after=list(after), owner_index=owner_index,
                        private_sites=list(owner), listener_index=listener, listener=env.AGENTS[listener],
                        visible_sites=list(visible), hidden_sites=list(hidden),
                        target_material_id=material, target_material=list(env.MATERIALS[material]),
                        target_sites=[before.index(material), after.index(material)],
                        eligible_need_group=f'{env.AGENTS[listener]}/material_{material}',
                        eligible_need_count=len(indices))
                    # All eligible needs are enumerated in the separate group list.
                    # Actual observe/settle methods are checked on one member per
                    # background condition; independence of the relation from the
                    # unchanged own need makes the remaining checks factorized.
                    if indices:
                        representative = make_example(spec, row, indices[0])
                        row.update(representative_need_index=indices[0],
                                   representative_LL_equal=True, representative_PL_different=True,
                                   representative_native_reward=[1., 1.],
                                   representative_LL_sha256=representative['LL_observation_sha256'][0])
                    conditions.append(row)
    conditions.sort(key=lambda r: (r['layout_before'], r['layout_after'], r['private_sites'],
                                   r['listener_index'], r['target_material_id']))
    counts = Counter()
    for row in conditions:
        counts['background_target_material_conditions'] += 1
        counts['expanded_unordered_need_layout_pair_witnesses'] += row['eligible_need_count']
        counts['conditions_with_eligible_need'] += int(row['eligible_need_count'] > 0)
        counts['by_listener_' + row['listener']] += row['eligible_need_count']
        counts['by_material_' + str(row['target_material_id'])] += row['eligible_need_count']
    candidates = [row for row in conditions if row['eligible_need_count']]
    first = None
    if candidates:
        choice = min(candidates, key=lambda r: (needs[r['representative_need_index']],
            r['layout_before'], r['layout_after'], r['private_sites'], r['listener_index'], r['target_material_id']))
        first = make_example(spec, choice, choice['representative_need_index'])
    return dict(partition=spec['partition'], demand_tables=len(needs), layouts=len(layouts), owners=len(owners),
        full_partition_world_count=spec['world_count'],
        layout_pair_owner_listener_backgrounds=len(pure_backgrounds),
        attempted_ordered_layout_owner_listener_checks=len(layouts)*len(owners)*3,
        ordered_checks_whose_swapped_layout_is_outside_partition=missing_partner_layouts,
        counts=dict(counts), endpoint_occurrences=2*counts['expanded_unordered_need_layout_pair_witnesses'],
        no_witness=not bool(candidates),
        first_lexicographic_instance_order=['needs', 'layout_before', 'layout_after', 'private_sites', 'listener_index', 'target_material_id'],
        first_lexicographic_instance=first,
        need_groups={f'{env.AGENTS[l]}/material_{m}':dict(listener_index=l, material_id=m,
            need_count=len(indices), need_indices=indices) for (l, m), indices in groups.items()},
        background_conditions=conditions,
        enumeration_scope='Every same-partition hidden-two-site swap, every owner/listener, both hidden target materials, and all eligible need indices. Need groups are exact canonical-native-plan enumerations; permutation preserves the material acceptance relation. Official LL/PL observe and native settle checked on one representative need per background condition, not separately on every expanded witness.',
        counting_scope='Unordered layout-pair witnesses with fixed need/owner/listener/material. Repeated endpoint worlds and listeners can overlap. These counts are not a success-rate denominator.')


def source_binding(prepared_path):
    prepared_path = Path(prepared_path).resolve()
    prepared = json.loads(prepared_path.read_text())
    plan_path, freeze_path = prepared_path.parent/'plan.json', prepared_path.parent/'freeze.json'
    plan, freeze = json.loads(plan_path.read_text()), json.loads(freeze_path.read_text())
    require(digest(prepared_path) == plan['prepared_sha256'] == freeze['prepared_sha256'], 'Frozen preparation hash mismatch')
    require(digest(plan_path) == freeze['plan_sha256'], 'Frozen plan hash mismatch')
    sources = {str(p.relative_to(ROOT)): digest(p) for p in (Path(env.__file__).resolve(), HERE/'dataset.py')}
    require(all(plan['sources'].get(k) == v for k, v in sources.items()), 'Environment/dataset differ from frozen plan')
    for relative, value in sources.items():
        require(digest(prepared_path.parent/'source_snapshot'/relative) == value, 'Frozen snapshot mismatch')
    require(set(prepared['partitions']) == set(PARTITIONS), 'Wrong four partitions')
    return prepared, dict(prepared_path=str(prepared_path), prepared_sha256=digest(prepared_path),
        main_plan_sha256=digest(plan_path), frozen_sources=sources,
        static_script_sha256=digest(__file__), read_files_exclude_model_results=True)


def write_report(out, result):
    path=out/'certificates.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 物资局部信息必要性证书','',
        '这是主训练启动后补充的纯静态枚举。仅读取已冻结的准备表和环境源码，未读取模型成绩、权重或通信输出；没有前向、训练或新策略搜索。', '',
        '四个分区均找到证书：三人需求、私有地点归属不变，只交换听者在 LL 中看不到的两个地点。听者的完整官方 LL 观察逐字段相同，PL 观察不同；两端均要求该听者真实运输，唯一正确运输地点发生改变。', '',
        '| 分区 | 需求表 | 布局 | 布局对×归属×听者 | 再分目标物资的背景条件 | 展开需求后的无向见证数 |',
        '|---|---:|---:|---:|---:|---:|']
    for part in PARTITIONS:
        row=result['partitions'][part]; c=row['counts']
        lines.append(f"| {part} | {row['demand_tables']} | {row['layouts']} | {row['layout_pair_owner_listener_backgrounds']} | {c['background_target_material_conditions']} | {c['expanded_unordered_need_layout_pair_witnesses']} |")
    lines += ['', '“无向见证”以需求表、无序布局对、归属、听者、目标物资标识。不同见证可以共享端点世界；上表不是互斥世界分割，也不能相除得到总体无通信成功率。', '', '## 每个分区的首个字典序实例','']
    for part in PARTITIONS:
        e=result['partitions'][part]['first_lexicographic_instance']
        lines += [f'### {part}', '']
        if e is None:
            lines += ['本分区没有符合条件的见证。',''];continue
        a,b=e['listener_correct_actions']; ll=e['LL_observations'][0]
        lines += [f"需求编号 `{e['needs']}`，听者 **{e['listener']}**，归属 `{e['private_sites']}`。布局按 S0–S3 顺序由 `{e['layouts'][0]}` 变为 `{e['layouts'][1]}`；仅交换地点 `{e['hidden_sites']}`。材料编号 0/1/2/3 分别为短木／长木／短纤维／长纤维。",
            '', f"两端正确行动分别为 `{a}` 和 `{b}`；只改变位置。搭档和目的地不变，两个正确方案的原生 R 均为 1。听者改为等待而伙伴保持正确行动时，两端 R 均为 0；更强的排除依据是两端都只有一个满分方案、且都包含该听者。", '',
            f"LL 官方观察的 SHA 相同：`{e['LL_observation_sha256'][0]}`。PL 的两份 SHA 不同。全部观察 JSON、三人正确动作及原状态索引见 [certificates.json](certificates.json)。", '']
    lines += ['## 证书支持什么', '',
        '令两个世界为 s、s′，听者为 i。枚举验证 O_LL,i(s)=O_LL,i(s′)，而两端唯一满分行动在 i 处的投影分别为两个不同的运输动作。因此，只依赖自身 LL 观察、且没有获得能区别这两个世界的他方信息的同一个确定策略，无法在两端都选对。即使另两人任意选择动作，也不能使听者的同一动作同时属于两个唯一满分方案。', '',
        '如果部署含随机性，可固定与当期私人事实无关的外生随机数再作同一论证；相同观察只能产生相同条件动作分布，不能在两个不同动作上都以概率 1 正确。这里没有推导原训练分布下最优随机团队的总体值。', '',
        'PL 将四处物资位置直接提供给听者，所以这组观察等同障碍消失；这不保证 PL 策略会用这些信息，也不消除它对其他人需求的未知。LL 的通信可能传递所需位置，也可能传递足以推断位置的材料事实；本证书不区分具体消息内容。', '',
        '本次没有改变需求、归属、可用动作或单世界满分答案数量；两端均为唯一满分方案。然而 PL/LL 的信息缺口本来就不同，此证书不是“纯计算难度恒定”的证明，也不是协议成分复用或语言形成证据。', '',
        '## 核验范围', '',
        '全部需求支持逐项用 env.full_success_plans 确认唯一正确材料、搭档和目的地；背景条件全部枚举。每一背景条件对其第一个适用需求直接调用官方 observe 与 settle 验证。其余适用需求的计数通过显式保存的 need_indices 分组展开：需求不变保证 own_need 字段不变，所见地点材料不变保证 LL 可见世界字段不变；物资置换将同一唯一材料搬到另一隐藏地点。这是确定的因子分解，没有把未逐项调用 observe 的展开记录描述为逐项调用。', '',
        f"本次用时 {result['elapsed_seconds']:.6f} 秒；0 模型前向、0 参数加载、0 训练。结果 JSON SHA：`{digest(path)}`。来源哈希绑定运行中的原准备包，执行前后均核对；未读取 execution/ 下任何成绩。"]
    (out/'说明.md').write_text('\n'.join(lines)+'\n')
    (out/'receipt.json').write_text(json.dumps(dict(status='completed',
        output_sha256={n:digest(out/n) for n in ('certificates.json','说明.md')},
        source_binding=result['source_binding'],model_forward_calls=0,training_updates=0),ensure_ascii=False,indent=2)+'\n')


def execute(prepared_path, out):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:
        prepared,binding=source_binding(prepared_path)
        partitions={p:analyze_partition(prepared['partitions'][p]) for p in PARTITIONS}
        _,end_binding=source_binding(prepared_path)
        require(binding==end_binding,'Sources changed during static enumeration')
        result=dict(schema='action_dependency_context_certificates_v1',status='completed_static_enumeration',
            created_at=datetime.now(timezone.utc).isoformat(),source_binding=binding,
            partitions=partitions,all_four_partitions_have_witness=all(not r['no_witness'] for r in partitions.values()),
            elapsed_seconds=time.perf_counter()-started,model_forward_calls=0,parameter_loads=0,
            training_updates=0,policy_optimization_calls=0,
            inference_scope='Necessary-information witness only. Not a global no-communication optimum, matched-difficulty proof, performance measurement, or semantic communication claim.')
        write_report(out,result)
        print(json.dumps(dict(status='completed',output=str(out),elapsed_seconds=result['elapsed_seconds'],
            counts={p:r['counts'] for p,r in partitions.items()},sha256=digest(out/'certificates.json')),ensure_ascii=False))
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps(dict(status='failed',error=repr(error),
            created_at=datetime.now(timezone.utc).isoformat(),model_forward_calls=0,training_updates=0),ensure_ascii=False,indent=2)+'\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    execute(args.prepared,args.out)
