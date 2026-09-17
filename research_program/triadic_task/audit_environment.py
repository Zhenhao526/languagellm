"""Independent, model-free audit of the implemented one-round triadic world.

Writes new audit outputs only. No policy, weights, training, or dialogue is run.
The reference acceptance rule uses binary material factors, not environment.accepts.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
from random import Random
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research_program.triadic_task import environment as env
from research_program.ecology_draft_counts import score as prior_score

HERE = Path(__file__).resolve().parent
NAMES = ('A', 'B', 'C')
SITES = ('S0', 'S1', 'S2', 'S3')
DESTS = ('L', 'R')
LAYOUTS = tuple(permutations(range(4)))
PRIVATE = tuple(permutations((1, 2, 3)))
PAIRS = tuple(combinations(range(3), 2))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accepts(need, material, destination):
    factor, destination_mode = divmod(need, 3)
    material_ok = (material // 2 == factor if factor < 2
                   else material % 2 == factor - 2)
    return material_ok and (destination_mode == 2 or destination == destination_mode)


def reference_actions(who):
    return (None,) + tuple((site, dest, partner)
                           for site, dest, partner in product(range(4), range(2), range(3))
                           if partner != who)


def to_actions(joint):
    return {NAMES[i]: ({'kind': 'wait'} if action is None else
                      {'kind': 'transport', 'site': SITES[action[0]],
                       'destination': DESTS[action[1]], 'partner': NAMES[action[2]]})
            for i, action in enumerate(joint)}


def reference(needs, layout, joint, matching):
    active = [i for i, action in enumerate(joint) if action is not None]
    execution, satisfied = [False] * 3, [False] * 3
    if len(active) <= 2:
        for i in active:
            site, dest, partner = joint[i]
            execution[i] = not matching or joint[partner] == (site, dest, i)
            satisfied[i] = execution[i] and accepts(needs[i], layout[site], dest)
    units = sum(satisfied)
    return {'reward': units / 2, 'satisfied_units': units, 'full_success': units == 2,
            'individual_feedback': {name: {'executed': execution[i],
                                         'own_need_satisfied': satisfied[i],
                                         'team_reward': units / 2}
                                    for i, name in enumerate(NAMES)},
            'researcher': {'overload': len(active) > 2,
                           'active_agents': [NAMES[i] for i in active],
                           'matching_required': bool(matching)}}


def need_view(need):
    factor, destination_mode = divmod(need, 3)
    return {'factor': 'kind' if factor < 2 else 'length',
            'value': ('wood', 'fiber', 'short', 'long')[factor],
            'destinations': [DESTS[d] for d in range(2)
                             if destination_mode == 2 or d == destination_mode]}


def reference_observation(needs, layout, private, who, shared, full=False):
    visible = range(4) if full else (0, private[who])
    result = {'self': NAMES[who], 'public_site': 'S0',
              'private_view_owners': {SITES[site]: NAMES[i] for i, site in enumerate(private)},
              'own_need': need_view(needs[who]),
              'visible_materials': [{'site': SITES[site],
                                     'kind': ('wood', 'fiber')[layout[site] // 2],
                                     'length': ('short', 'long')[layout[site] % 2]}
                                    for site in visible]}
    if shared or full:
        result['shared_needs'] = {NAMES[i]: need_view(need) for i, need in enumerate(needs)}
    if full:
        result['information_control'] = 'full_information'
    return result


class UnreadableState:
    """A zero-physics structure must never inspect either semantic field."""
    @property
    def needs(self):
        raise AssertionError('A physically impossible D1 action read needs')

    @property
    def layout(self):
        raise AssertionError('A physically impossible D1 action read material layout')


def audit():
    started = time.monotonic()
    sources = {str(path): sha(path) for path in
               (Path(env.__file__), HERE / 'test_environment.py',
                ROOT / 'research_program/ecology_draft_counts.py', Path(__file__))}
    assert env.AGENTS == NAMES and env.SITES == SITES and env.DESTINATIONS == DESTS
    degree_counts, selected = Counter(), []
    for needs in product(range(12), repeat=3):
        edges = tuple((i, j) for i, j in PAIRS
                      if any(accepts(needs[i], material, dest) and accepts(needs[j], material, dest)
                             for material, dest in product(range(4), range(2))))
        assert env.compatible_pairs(needs) == edges
        degree_counts[len(edges)] += 1
        if len(edges) >= 2:
            selected.append(needs)
    assert degree_counts == {0: 120, 1: 612, 2: 576, 3: 420}
    assert tuple(selected) == env.support() and len(selected) == 996
    for need, material, dest in product(range(12), range(4), range(2)):
        assert env.accepts(need, material, dest) == accepts(need, material, dest)

    joints = tuple(product(*(reference_actions(i) for i in range(3))))
    matched, nonoverloaded, zero_count, overload_count = [], [], 0, 0
    zero_state = UnreadableState()
    for joint in joints:
        actions = to_actions(joint)
        active = [i for i, action in enumerate(joint) if action is not None]
        expected = reference((0, 0, 0), (0, 1, 2, 3), joint, True)
        if any(row['executed'] for row in expected['individual_feedback'].values()):
            matched.append((joint, actions))
        else:
            actual = env.settle(zero_state, actions, require_match=True)
            assert actual == expected
            zero_count += 1
        if len(active) <= 2:
            nonoverloaded.append((joint, actions))
        else:
            assert env.settle(zero_state, actions, require_match=False) == reference(
                (0, 0, 0), (0, 1, 2, 3), joint, False)
            overload_count += 1
    assert len(joints) == 4913 and len(matched) == 24 and zero_count == 4889
    assert len(nonoverloaded) == 817 and overload_count == 4096
    original_matched_actions = deepcopy(matched)

    d1_rewards = Counter()
    matching_checks, previous_score_checks, state_checks = 0, 0, 0
    for n_index, needs in enumerate(selected):
        # Reconstruct sets without importing the prior implementation's table.
        demands = tuple((frozenset(m for m in range(4) if accepts(n, m, n % 3 if n % 3 < 2 else 0)),
                         frozenset(d for d in range(2) if n % 3 == 2 or d == n % 3))
                        for n in needs)
        for layout in LAYOUTS:
            state = env.State(needs, layout)
            for joint, actions in matched:
                expected = reference(needs, layout, joint, True)
                actual = env.settle(state, actions, require_match=True)
                assert actual == expected, (needs, layout, joint, actual, expected)
                assert prior_score(demands, layout, joint, True) == expected['reward']
                previous_score_checks += 1
                d1_rewards[str(actual['reward'])] += 1
                expected_d0 = deepcopy(expected)
                expected_d0['researcher']['matching_required'] = False
                assert env.settle(state, actions, require_match=False) == expected_d0
                matching_checks += 2
            assert (state.needs, state.layout, state.private_sites) == (needs, layout, (1, 2, 3))
            state_checks += 1
        if (n_index + 1) % 249 == 0:
            print(f'matching support tables checked: {n_index + 1}/996', flush=True)
    assert matched == original_matched_actions
    assert state_checks == 23904 and matching_checks == 1147392 and previous_score_checks == 573696

    # The diagonal need stratum exhausts every individual need while crossing
    # every layout and every nonoverloaded joint action. It is not all 996 tables.
    d0_checks, d0_rewards = 0, Counter()
    for need, layout in product(range(12), LAYOUTS):
        needs = (need,) * 3
        state = env.State(needs, layout)
        for joint, actions in nonoverloaded:
            expected = reference(needs, layout, joint, False)
            actual = env.settle(state, actions, require_match=False)
            assert actual == expected
            d0_rewards[str(actual['reward'])] += 1
            d0_checks += 1
    assert d0_checks == 235296
    print('D0 action structure checks complete; checking observation boundaries', flush=True)

    local_count, full_count, private_worlds = 0, 0, 0
    i1_equivalence, i0_equivalence = {}, {}
    for needs, layout in product(selected, LAYOUTS):
        for private in PRIVATE:
            state = env.State(needs, layout, private)
            for who, name in enumerate(NAMES):
                for shared in (False, True):
                    actual = env.observe(state, name, shared_needs=shared)
                    expected = reference_observation(needs, layout, private, who, shared)
                    assert actual == expected
                    local_count += 1
                    if private == (1, 2, 3):
                        equivalence = i0_equivalence if shared else i1_equivalence
                        key = (who, needs if shared else needs[who], layout[0], layout[private[who]])
                        if key in equivalence:
                            assert equivalence[key] == actual
                        else:
                            equivalence[key] = actual
                full = env.observe(state, name, shared_needs=False, full_information=True)
                assert full == reference_observation(needs, layout, private, who, False, True)
                full_count += 1
            private_worlds += 1
    assert private_worlds == 143424 and local_count == 860544 and full_count == 430272
    assert len(i1_equivalence) == 432 and len(i0_equivalence) == 35856

    menu_orders = [list(range(offset, 17)) + list(range(offset)) for offset in range(17)]
    menu_orders.append(list(reversed(range(17))))
    menu_rng = Random(2026091502)
    for _ in range(100):
        order = list(range(17))
        menu_rng.shuffle(order)
        menu_orders.append(order)
    menu_checks = 0
    for who, name in enumerate(NAMES):
        independent = [to_actions(tuple(action if i == who else None for i in range(3)))[name]
                       for action in reference_actions(who)]
        assert env.all_actions(name) == independent
        for order in menu_orders:
            assert env.action_menu(name, order) == [
                {'id': i, 'action': independent[index]} for i, index in enumerate(order)]
            menu_checks += 1
    assert menu_checks == 354

    # Mutation and identity-order checks use heterogeneous needs and both modes.
    mutable_needs, mutable_layout, mutable_private = [0, 6, 9], [0, 1, 2, 3], [1, 2, 3]
    copied = env.State(mutable_needs, mutable_layout, mutable_private)
    input_lists = deepcopy((mutable_needs, mutable_layout, mutable_private))
    for joint, actions in matched:
        snapshot = deepcopy(actions)
        for matching in (False, True):
            expected = reference(copied.needs, copied.layout, joint, matching)
            for key_order in permutations(NAMES):
                assert env.settle(copied, {name: actions[name] for name in key_order},
                                  require_match=matching) == expected
        assert actions == snapshot
    assert (mutable_needs, mutable_layout, mutable_private) == input_lists
    mutable_needs[0], mutable_layout[0], mutable_private[0] = 11, 3, 3
    assert copied == env.State((0, 6, 9), (0, 1, 2, 3))
    view = env.observe(copied, 'A', shared_needs=True)
    view['own_need']['destinations'].append('INJECTED')
    view['shared_needs']['B']['value'] = 'INJECTED'
    view['visible_materials'][0]['kind'] = 'INJECTED'
    assert env.observe(copied, 'A', shared_needs=True) == reference_observation(
        copied.needs, copied.layout, copied.private_sites, 0, True)
    mutable_menu = env.action_menu('A', list(range(17)))
    mutable_menu[1]['action']['site'] = 'INJECTED'
    assert env.all_actions('A')[1]['site'] == 'S0'
    for invalid in (False, 0.0, '0'):
        try:
            env.State((invalid, 6, 9), (0, 1, 2, 3))
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid state id accepted')
    for operation in (lambda: env.all_actions('D'), lambda: env.action_menu('D', list(range(17)))):
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError('Unknown agent accepted')

    # IID is a source-level property of fixed support + fresh independent random
    # permutations/choice. Reproducibility checks cannot prove statistical IID.
    rng1, rng2 = Random(2026091503), Random(2026091503)
    selected_set, allocation_counts = set(selected), Counter()
    draw_count = 5000
    for _ in range(draw_count):
        state = env.draw_state(rng1, selected)
        assert state == env.draw_state(rng2, selected)
        assert state.needs in selected_set and state.layout in LAYOUTS and state.private_sites in PRIVATE
        assert all(type(v) is tuple for v in (state.needs, state.layout, state.private_sites))
        assert set(asdict(state)) == {'needs', 'layout', 'private_sites'}
        allocation_counts[str(state.private_sites)] += 1
    assert len(allocation_counts) == 6
    for path, original_hash in sources.items():
        assert sha(path) == original_hash, f'Source changed during audit: {path}'
    assert 'torch' not in sys.modules and 'mlx' not in sys.modules
    return {
        'status': 'passed_independent_environment_audit',
        'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'elapsed_seconds': round(time.monotonic() - started, 3),
        'source_sha256': sources,
        'demand_table_counts_by_compatible_edges': dict(sorted(degree_counts.items())),
        'selected_demand_tables': 996, 'canonical_states': state_checks,
        'semantic_states_all_private_assignments': private_worlds,
        'local_action_count': 17, 'joint_action_structures': len(joints),
        'matching_action_structures': len(matched),
        'd1_identically_zero_structures_checked_without_state_reads': zero_count,
        'overload_structures_checked_in_both_modes_without_state_reads': overload_count,
        'd1_complete_canonical_state_matching_action_checks': previous_score_checks,
        'd0_complete_canonical_state_matching_action_checks': previous_score_checks,
        'prior_independent_score_agreements': previous_score_checks,
        'd1_matching_cases_by_team_reward': dict(sorted(d1_rewards.items())),
        'd0_additional_all_nonoverloaded_action_checks': d0_checks,
        'd0_additional_scope': '12 diagonal need triples, all within the 996 support, x 24 layouts x 817 nonoverloaded structures; not all heterogeneous triples x all actions',
        'd0_additional_cases_by_team_reward': dict(sorted(d0_rewards.items())),
        'local_observation_checks': local_count,
        'full_information_observation_checks': full_count,
        'canonical_i1_observation_equivalence_classes_all_three_agents': len(i1_equivalence),
        'canonical_i0_observation_equivalence_classes_all_three_agents': len(i0_equivalence),
        'complete_menu_permutations_checked': menu_checks,
        'action_input_order_checks': 24 * 2 * 6,
        'input_nonmutation_and_returned_view_isolation': 'passed',
        'mutable_state_input_copy_and_invalid_id_regressions': 'passed',
        'reproducible_state_draws': draw_count,
        'private_allocation_sample_counts': dict(sorted(allocation_counts.items())),
        'iid_source_review': 'With fixed uniform 996 support, shuffle(layout), shuffle(private_sites), choice(support) are fresh draws from the supplied RNG. Under the ideal uniform independent RNG interpretation each world has probability 1/(996*24*6). The finite reproducibility sample is not a statistical proof of IID.',
        'model_calls': 0, 'training_updates': 0,
        'limits': [
            'No agent, conversation, learned convention, visual perception, or language formation was evaluated.',
            'Complete D1 verification is factored: all state-dependent matching cases plus all state-independent physical-zero structures, not a scalar loop over all 23904*4913 combinations.',
            'D0 unmatched cases are exhaustively checked over all own-need values in diagonal triples and all layouts/actions, but not the complete heterogeneous need-table Cartesian product.',
            'The public owner map and common team reward are declared information channels. Exact schema tests rule out extra raw hidden fields, not inference through these declared channels.',
            'Full settle results and sufficient-information witnesses are researcher objects; a future runner must pass only own individual_feedback and permitted observations to policies.',
            'draw_state accepts caller-provided support. A future runner must freeze the 996 support, keep RNG state/seed private, and avoid sharing raw State objects.',
            'The menu permutation generator is caller-owned. This audit checks complete menus for supplied orders, not future independent menu randomization.',
            'Kind and length are currently explicit symbolic observations. No visual recognition transfer has been established.',
            'D0 retains a shared two-person capacity constraint; it removes matching dependence, not all action dependence.',
        ],
    }


def render(result):
    hashes = '\n'.join(f'- `{Path(path).relative_to(ROOT)}`：`{digest}`'
                       for path, digest in result['source_sha256'].items())
    return f'''# 三人环境独立核验

核验时间：{result['checked_at']}。结果通过；未修改环境或原测试，模型调用和训练更新均为 0。本报告核验一轮任务的环境接口与结算，不是主体实验或语言形成结果。

## 实际覆盖

- 独立按种类、长短和目的地重建全部 1,728 张需求表，兼容边数 0/1/2/3 的数量为 120/612/576/420；当前支持集为后两层合计 996 张。
- 996 张需求表 × 24 个物资布局 = 23,904 个规范状态。对每个状态的全部 24 个匹配动作结构，逐项检查 D1 和 D0 的执行、个体满足、共同奖励与研究者反馈，共 1,147,392 次实际结算；D1 的 573,696 个奖励与此前独立 `ecology_draft_counts.score` 全部一致。
- 在全部 4,913 种联合动作结构中，D1 其余 4,889 种都不能产生物理执行。用读取需求/布局即报错的哨兵状态实际调用结算，全部返回物理零且未读取状态；其中 4,096 种三人超载结构在 D0 也同样为零。这是对全部规范状态的因子化穷举核验，没有声称执行了 23,904 × 4,913 次标量结算。
- D0 另对 12 种同需求三人组合 × 24 布局 × 全部 817 个未超载动作结构执行 235,296 次核验。它覆盖每个人的全部需求和动作，包括非匹配各自运输，但不等同于穷举全部异质需求表与任意联合动作。
- 六种私有地点分配全部纳入，合计 143,424 个语义状态；逐人检查 I0/I1 共 860,544 个局部观察及 430,272 个完整信息观察。规范地点分配下，隐藏世界变化不改变合法相同观察：三人合计 I1 有 432 个观察类，I0 有 35,856 个观察类。
- 354 份重新排序的菜单均恰含全部 17 个动作；菜单函数不接收状态，未按目标匹配、地点可见性或伙伴动作删减选项。288 次提交字典顺序交换结算一致。结算不修改输入，返回观察/菜单不与状态共享可变内容；可变序列输入复制、拒绝布尔/非整数状态、拒绝未知主体均通过。

匹配结构在 D1 下的共同奖励计数：{json.dumps(result['d1_matching_cases_by_team_reward'], ensure_ascii=False)}。半分情形中，匹配成功的双方均执行运输，只有满足自身需求者记成功；这与菜单合法、物理执行、目标满足三个概念的分离一致。

## 观察边界与采样含义

I1 只含本人的需求、公共地点和本人私有地点的物资；I0 额外公开三人的需求。完整信息控制明确标记，并公开四地物资和所有需求。地点观察者归属是公开字段，共同奖励是已声明的反馈通道，因此不能把这里的隐私检查解释为主体无法从任何现象推断隐藏信息。

`draw_state` 在给定支持集上使用传入随机发生器依次重新打乱四类物资、三个私有地点并选择需求表，不读取旧世界或行动。固定均匀支持集和理想独立均匀随机抽样下，每个语义状态概率为 1/143,424。5,000 次配对抽样验证了合法性、六种位置分配及同种子可复现；这不是用有限样本证明统计独立。状态和实际观察均不包含种子、随机发生器状态或抽样编号。未来调用者仍须固定支持集并保持随机来源私密。

## 下轮接入前的边界

1. `settle` 的整份返回值、原始 `State` 和全信息见证属于研究者层。主体只能收到合法观察和自己的 `individual_feedback`，不可直接收到他人的逐人反馈或研究者对象。
2. 当前检查保证给定排列下菜单完整；排列是否每人独立、是否与目标无关，要在未来 runner 审计。相同声明规则并不构成观察泄漏，但公共身份和地点标签可被主体用于约定。
3. D0 仍有最多两人行动的共同容量，不能称为完全独立行动条件。单轮物资不做跨轮消耗，后续任务必须按预定新世界解释，不能悄然改成持续库存。
4. 当前输入直接给出种类/长短，没有验证新视觉主体可识别这些属性。也尚未测无通信策略、自然或符号消息的因果作用、任务学习或约定形成。此次结果仅允许下一步接入主体前使用这个环境。

## 源文件 SHA-256

{hashes}

审计开始和结束时以上哈希一致。可复核代码：`audit_environment.py`；精确计数：`audit_environment.json`。
'''


def main():
    output_json, output_md = HERE / 'audit_environment.json', HERE / 'audit_environment.md'
    if output_json.exists() or output_md.exists():
        raise FileExistsError('Audit outputs already exist; refusing to overwrite')
    result = audit()
    with output_json.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    with output_md.open('x', encoding='utf-8') as handle:
        handle.write(render(result))
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                      'json': str(output_json), 'report': str(output_md)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
