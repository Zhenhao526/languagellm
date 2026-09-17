"""Post hoc static D1 game check. No policy/model loading or parameter updates."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations, permutations, product
import json
from pathlib import Path

from research_program.triadic_task import environment as env

HERE = Path(__file__).resolve().parent
AGENTS = ('A', 'B', 'C')
SITES = ('S0', 'S1', 'S2', 'S3')
DESTS = ('L', 'R')
MATERIALS = (('wood', 'short'), ('wood', 'long'), ('fiber', 'short'), ('fiber', 'long'))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def actions_for(agent):
    return [{'kind': 'wait'}] + [{'kind': 'transport', 'site': s, 'destination': d, 'partner': p}
        for s, d, p in product(SITES, DESTS, [a for a in AGENTS if a != agent])]


def independent_execution(actions):
    active = [a for a in AGENTS if actions[a]['kind'] == 'transport']
    if len(active) > 2:
        return False, 'three_active_overload'
    if len(active) != 2:
        return False, 'fewer_than_two_active'
    a, b = active
    left, right = actions[a], actions[b]
    matched = left['partner'] == b and right['partner'] == a and all(left[k] == right[k] for k in ('site', 'destination'))
    return matched, 'matched' if matched else 'two_active_mismatch'


def acceptance_sets():
    result = []
    for (factor, value), ds in product((('kind', 'wood'), ('kind', 'fiber'), ('length', 'short'), ('length', 'long')),
                                      (('L',), ('R',), ('L', 'R'))):
        col = 0 if factor == 'kind' else 1
        result.append({(m, d) for m, attrs in enumerate(MATERIALS) for d in ds if attrs[col] == value})
    return result


def main():
    out = HERE / 'enumeration_001'
    if out.exists():
        raise FileExistsError('Refuse to overwrite earlier static evidence')
    constant_path = HERE.parent / 'results/learning_001/constant_action_check.json'
    sources = [Path(__file__), Path(env.__file__), HERE.parent / 'plan.md', constant_path]
    source_sha = {str(p): digest(p) for p in sources}
    menus = {a: actions_for(a) for a in AGENTS}
    assert all(len(menus[a]) == 17 and menus[a] == env.all_actions(a) for a in AGENTS)
    accepted = acceptance_sets()
    support = [n for n in product(range(12), repeat=3)
               if sum(bool(accepted[n[i]] & accepted[n[j]]) for i, j in combinations(range(3), 2)) >= 2]
    assert len(support) == 996
    profiles = []
    for (a, b), s, d in product(combinations(AGENTS, 2), SITES, DESTS):
        acts = {who: {'kind': 'wait'} for who in AGENTS}
        acts[a] = {'kind': 'transport', 'site': s, 'destination': d, 'partner': b}
        acts[b] = {'kind': 'transport', 'site': s, 'destination': d, 'partner': a}
        assert independent_execution(acts) == (True, 'matched')
        profiles.append({'id': a+b+':'+s+':'+d, 'actions': acts,
                         'indices': [menus[who].index(acts[who]) for who in AGENTS]})
    assert len(profiles) == len({tuple(p['indices']) for p in profiles}) == 24

    deviations = []
    reasons = Counter()
    for profile in profiles:
        for who in AGENTS:
            for index, replacement in enumerate(menus[who]):
                if replacement == profile['actions'][who]:
                    continue
                acts = dict(profile['actions'])
                acts[who] = replacement
                executed, reason = independent_execution(acts)
                assert executed is False
                reasons[reason] += 1
                deviations.append({'profile_id': profile['id'], 'agent': who, 'replacement_index': index,
                                   'replacement_action': replacement, 'independent_executed': False,
                                   'independent_reason': reason})
    assert len(deviations) == 24*3*16 == 1152
    assert reasons == {'three_active_overload': 384, 'fewer_than_two_active': 48, 'two_active_mismatch': 720}

    # One complete traversal: all 23,904 canonical worlds x all 24 constant
    # matching profiles. Actor private-site allocations do not enter settle.
    hist = {p['id']: Counter() for p in profiles}
    representatives = {p['id']: {} for p in profiles}
    env_checks = 0
    for needs, layout in product(support, permutations(range(4))):
        state = env.State(needs, layout, (1, 2, 3))
        for profile in profiles:
            units = sum((layout[SITES.index(act['site'])], act['destination']) in accepted[needs[AGENTS.index(a)]]
                        for a, act in profile['actions'].items() if act['kind'] == 'transport')
            expected = units/2
            actual = env.settle(state, profile['actions'], require_match=True)
            assert actual['reward'] == expected and actual['satisfied_units'] == units
            assert actual['full_success'] == (units == 2)
            assert sum(f['executed'] for f in actual['individual_feedback'].values()) == 2
            hist[profile['id']][str(expected)] += 1
            representatives[profile['id']].setdefault(str(expected), {
                'needs': list(needs), 'layout': list(layout), 'private_sites': [1, 2, 3]})
            env_checks += 1
    assert env_checks == 23904*24 == 573696
    assert all(h == {'0.0': 10320, '0.5': 9984, '1.0': 3600} for h in hist.values())

    profile_map = {p['id']: p for p in profiles}
    deviation_env_checks = 0
    for row in deviations:
        profile = profile_map[row['profile_id']]
        assert set(representatives[profile['id']]) == {'0.0', '0.5', '1.0'}
        checked = []
        for base_reward, fields in representatives[profile['id']].items():
            state = env.State(**fields)
            actions = dict(profile['actions'])
            actions[row['agent']] = row['replacement_action']
            actual = env.settle(state, actions, require_match=True)
            assert actual['reward'] == 0 and actual['satisfied_units'] == 0 and not actual['full_success']
            assert not any(f['executed'] for f in actual['individual_feedback'].values())
            checked.append({'original_reward': float(base_reward), 'deviation_reward': 0,
                            'all_execution_flags_false': True})
            deviation_env_checks += 1
        row['environment_representative_checks'] = checked
    assert deviation_env_checks == 3456
    observed = json.loads(constant_path.read_text())
    observed_links = []
    for seed in observed['seeds']:
        assert seed['distinct_joint_argmax_actions'] == 1 and seed['worlds'] == 143424
        matches = [p for p in profiles if p['indices'] == seed['action_indices'][0]]
        assert len(matches) == 1
        observed_links.append({'seed': seed['seed'], 'profile_id': matches[0]['id'],
            'constancy_scope': 'Reported by existing constant_action_check.json; raw saved policy arrays not reaudited here.',
            'static_predicted_full_domain_reward_counts': {k: 6*v for k, v in hist[matches[0]['id']].items()}})
    assert {str(p): digest(p) for p in sources} == source_sha
    counts = {'0.0': 10320, '0.5': 9984, '1.0': 3600}
    result = {
        'status': 'passed_static_post_hoc_check', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': source_sha, 'matching_profiles': 24, 'deviations_per_profile': 48,
        'total_single_actor_deviations': 1152, 'all_deviations_zero_execution_and_reward': True,
        'deviation_reasons': dict(reasons), 'independent_support_needs': 996,
        'canonical_worlds': 23904, 'full_worlds_including_six_private_allocations': 143424,
        'all_constant_profiles_environment_checks': env_checks,
        'deviation_environment_checks_on_0_half_1_representatives': deviation_env_checks,
        'constant_profile_counts_all_equal': True, 'each_constant_profile_canonical_counts': counts,
        'each_constant_profile_full_domain_counts': {k: v*6 for k, v in counts.items()},
        'each_constant_profile_fullsuccess_fraction': {'numerator': 3600, 'denominator': 23904, 'value': 3600/23904},
        'each_constant_profile_native_R_fraction': {'numerator': 8592, 'denominator': 23904, 'value': 8592/23904},
        'profile_counts': [{'profile': p, 'canonical_counts': dict(hist[p['id']]),
                            'representatives': representatives[p['id']]} for p in profiles],
        'observed_constant_profiles': observed_links,
        'interpretation': 'Pointwise common-payoff pure-action unilateral-deviation property, not a causal test of training, entropy, logits or parameter landscape.',
        'new_model_forward_calls': 0, 'parameter_updates': 0, 'network_requests': 0,
    }
    out.mkdir()
    save(out / 'results.json', result)
    save(out / 'deviations.json', deviations)
    save(out / 'manifest.json', {'source_sha256': source_sha, 'output_sha256': {
        name: digest(out / name) for name in ('results.json', 'deviations.json')}})
    print(json.dumps({k: result[k] for k in ('status', 'total_single_actor_deviations', 'deviation_reasons',
        'all_constant_profiles_environment_checks', 'each_constant_profile_canonical_counts',
        'each_constant_profile_fullsuccess_fraction', 'each_constant_profile_native_R_fraction')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
