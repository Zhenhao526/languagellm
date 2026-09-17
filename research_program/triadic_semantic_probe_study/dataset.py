"""Static semantic-pair preparation. No models, trained actions or messages read.

All source demand pairs are retained as metadata. Only the unique ecology is
expanded to endpoint indices; probe weights are not the original world weights.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, permutations, product
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'research_program/triadic_partner_ecology_study'
SOURCE = OLD/'next_semantic_static_001'
PREPARED = OLD/'results/partner_001/prepared.json'
AGENTS = ('A', 'B', 'C')
AXES = ('kind_wood_fiber', 'length_short_long', 'destination_L_R')
PARTITIONS = ('train', 'heldout_layouts')
CLASSES = ('descriptive', 'content', 'role')
ROLE_NAMES = ('always_wait', 'always_participate', 'wait_or_participate')
PAIRS = tuple(combinations(range(3), 2))
RESOURCES = ({0, 1}, {2, 3}, {0, 2}, {1, 3})
DESTINATIONS = ({0}, {1}, {0, 1})


def require(condition, message):
    if not condition: raise AssertionError(message)


def sha(path): return sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def json_bytes(value): return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n').encode()
def array_sha(value): return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def action_menu(agent):
    who = AGENTS.index(agent)
    return [dict(kind='wait')]+[dict(kind='transport', site=f'S{site}', destination='LR'[dest], partner=AGENTS[peer])
        for site, dest, peer in product(range(4), range(2), [a for a in range(3) if a != who])]


def structural_plans():
    menus = {a: action_menu(a) for a in AGENTS}
    plans = []
    for i, j in PAIRS:
        for site, dest in product(range(4), range(2)):
            actions = {a: dict(kind='wait') for a in AGENTS}
            for who, peer in ((i, j), (j, i)):
                actions[AGENTS[who]] = dict(kind='transport', site=f'S{site}', destination='LR'[dest], partner=AGENTS[peer])
            plans.append(dict(pair=AGENTS[i]+AGENTS[j], actions=actions,
                indices=[menus[a].index(actions[a]) for a in AGENTS]))
    return plans


def winning_plan_ids(needs, layout=(0, 1, 2, 3)):
    """Independent full-success acceptance formula for all24 matched plans."""
    result = []
    for k, (i, j) in enumerate(PAIRS):
        for site, dest in product(range(4), range(2)):
            if all(layout[site] in RESOURCES[needs[a]//3] and dest in DESTINATIONS[needs[a] % 3] for a in (i, j)):
                result.append(k*8+site*2+dest)
    return result


def full_action_set(needs, listener, layout=(0, 1, 2, 3)):
    plans = structural_plans()
    return {plans[k]['indices'][listener] for k in winning_plan_ids(needs, layout)}


def bitmask(actions): return sum(1 << int(i) for i in set(actions))
def mask_members(mask): return {i for i in range(17) if int(mask) & (1 << i)}
def role(actions): return 'always_wait' if actions == {0} else 'always_participate' if 0 not in actions else 'wait_or_participate'


def partner_set(listener, actions):
    peers = [a for a in range(3) if a != listener]
    return sorted({peers[(a-1) % 2] for a in actions if a})


def action_permutation(layout):
    """Canonical material==site to actual location, independently for each actor."""
    return np.asarray([0]+[1+list(layout).index(site)*4+dest*2+peer
        for site, dest, peer in product(range(4), range(2), range(2))], dtype=np.int16)


def endpoint_states(index_space, indices):
    """Reconstruct research-only packed states; input index shape is preserved."""
    indices = np.asarray(indices)
    require(indices.dtype.kind in ('i', 'u') and np.all(indices >= 0)
        and np.all(indices < index_space['endpoint_world_count']), 'Invalid endpoint indices')
    owner_count, layout_count = len(index_space['private_sites']), len(index_space['layouts'])
    ni = indices//(layout_count*owner_count)
    li = (indices//owner_count) % layout_count
    oi = indices % owner_count
    return np.concatenate((np.asarray(index_space['needs'], dtype=np.int16)[ni],
        np.asarray(index_space['layouts'], dtype=np.int16)[li],
        np.asarray(index_space['private_sites'], dtype=np.int16)[oi]), axis=-1)


def flip_candidates(need):
    resource, dest = divmod(need, 3)
    yield ('kind_wood_fiber' if resource < 2 else 'length_short_long'), 3*(resource ^ 1)+dest
    if dest != 2: yield 'destination_L_R', 3*resource+(1-dest)


def layout_donor_mapping(layouts):
    """Deterministic bipartite matching, all four materials changed, same partition.

    The returned proposal is static and is not an enabled probe intervention.
    On failure, reachable alternating vertices give a Hall-deficient subset.
    """
    layouts = [tuple(x) for x in layouts]
    adjacency = [[j for j, right in enumerate(layouts) if all(a != b for a, b in zip(left, right))]
        for left in layouts]
    right_match = [-1]*len(layouts)
    def augment(left, visited):
        for right in adjacency[left]:
            if right in visited: continue
            visited.add(right)
            if right_match[right] == -1 or augment(right_match[right], visited):
                right_match[right] = left
                return True
        return False
    for left in range(len(layouts)): augment(left, set())
    mapping = [-1]*len(layouts)
    for right, left in enumerate(right_match):
        if left != -1: mapping[left] = right
    full = all(i != -1 for i in mapping)
    witness = None
    if not full:
        reachable_left = {i for i, v in enumerate(mapping) if v == -1}; reachable_right = set()
        changed = True
        while changed:
            old = (len(reachable_left), len(reachable_right))
            reachable_right.update(j for i in reachable_left for j in adjacency[i])
            reachable_left.update(right_match[j] for j in reachable_right if right_match[j] != -1)
            changed = old != (len(reachable_left), len(reachable_right))
        require(len(reachable_right) < len(reachable_left), 'Expected Hall deficiency')
        witness = dict(left_indices=sorted(reachable_left), neighbor_right_indices=sorted(reachable_right))
    if full:
        require(sorted(mapping) == list(range(len(layouts))) and all(j in adjacency[i] for i, j in enumerate(mapping)), 'Invalid layout bijection')
    return dict(status='perfect_matching_exists' if full else 'no_perfect_matching',
        layout_count=len(layouts), adjacency=adjacency, donor_layout_indices=mapping if full else None,
        matched_edges=sum(v != -1 for v in mapping), Hall_deficiency=witness,
        deterministic_rule='Original frozen layout index order on both sides; depth-first augmenting paths, ascending neighbor index.',
        all_four_sites_change=full, private_owner_mapping='identity',
        enabled_in_main_probe=False, proposal_only=True)


def load_sources(source_dir=SOURCE, prepared_path=PREPARED):
    source_dir, prepared_path = Path(source_dir), Path(prepared_path)
    result = read(source_dir/'results.json')
    require(result['status'] == 'completed_static_only', 'Incomplete source static enumeration')
    sources = {str(source_dir/'results.json'): sha(source_dir/'results.json')}
    for path, digest in result['sources_sha256'].items():
        require(sha(path) == digest, 'Original static source changed '+path); sources[path] = digest
    for name, digest in result['outputs_sha256'].items():
        path = source_dir/name
        require(sha(path) == digest, 'Original static output changed '+name); sources[str(path)] = digest
    frozen = read(prepared_path.parent/'freeze.json')
    require(sha(prepared_path) == frozen['prepared_sha256'], 'Original prepared endpoint index source changed')
    for path in (prepared_path, prepared_path.parent/'freeze.json'):
        sources[str(path)] = sha(path)
    prepared = read(prepared_path)
    original = read(source_dir/'structural_plans_and_action_index.json')
    require(original['all_24_plans'] == structural_plans(), 'Full physical plans mismatch')
    require(original['agent_action_menus'] == {a: action_menu(a) for a in AGENTS}, 'Full17 menus mismatch')
    return read(source_dir/'demand_pairs.json'), prepared, sources, original


def build_dataset(source_dir=SOURCE, prepared_path=PREPARED):
    pairs, prepared, sources, original = load_sources(source_dir, prepared_path)
    require(len(pairs) == 2244, 'All2244 source pairs required')
    domains = {e: {tuple(n) for n in prepared['partitions'][e]['train']['needs']} for e in ('unique', 'multiple')}
    expected = set()
    for ecology, domain in domains.items():
        for needs in domain:
            for sender in range(3):
                for axis, change in flip_candidates(needs[sender]):
                    other = list(needs); other[sender] = change; other = tuple(other)
                    if other in domain:
                        left, right = sorted((needs, other)); expected.add((ecology, axis, sender, left, right))
    actual = {(p['ecology'], p['axis'], AGENTS.index(p['changed_person']), tuple(p['needs_before']), tuple(p['needs_after'])) for p in pairs}
    require(len(actual) == len(pairs) and actual == expected, 'Source pair inclusion/completeness mismatch')
    need_lookup = {e: {tuple(n): i for i, n in enumerate(prepared['partitions'][e]['train']['needs'])} for e in domains}
    for e, part in product(domains, PARTITIONS):
        require(prepared['partitions'][e][part]['needs'] == prepared['partitions'][e]['train']['needs'], 'Need index order differs across partitions')
    cases = []
    for pair_index, pair in enumerate(pairs):
        sender = AGENTS.index(pair['changed_person']); ecology = pair['ecology']
        endpoint_needs = [pair['needs_before'], pair['needs_after']]
        plans = [winning_plan_ids(n) for n in endpoint_needs]
        require(plans == [pair['full_plan_ids_before'], pair['full_plan_ids_after']] and all(plans), 'All24 independent success plans mismatch')
        require({l['listener'] for l in pair['listeners']} == set(AGENTS)-{AGENTS[sender]}, 'Missing listener')
        for listener_data in pair['listeners']:
            listener = AGENTS.index(listener_data['listener'])
            action_sets = [full_action_set(n, listener) for n in endpoint_needs]
            require([sorted(x) for x in action_sets] == [listener_data['successful_actions_before'], listener_data['successful_actions_after']], 'Listener full-action projection differs')
            roles = [role(x) for x in action_sets]
            disjoint = not (action_sets[0] & action_sets[1])
            switch = ecology == 'unique' and set(roles) == {'always_wait', 'always_participate'}
            require(disjoint == listener_data['full_success_action_sets_disjoint'] and switch == listener_data['unique_participation_wait_switch'], 'Saved set/role label differs')
            require(roles == [listener_data['role_before'], listener_data['role_after']], 'Saved role label differs')
            require(endpoint_needs[0][listener] == endpoint_needs[1][listener] and listener != sender, 'Listener local demand changed')
            group = 'role' if switch else 'content' if disjoint else 'descriptive'
            require(not disjoint or ecology == 'unique', 'Unexpected multiple necessary-listener case')
            if group == 'content': require(roles == ['always_participate']*2, 'Content would force participation change')
            partners = [partner_set(listener, x) for x in action_sets]
            if group == 'content': require(partners[0] == partners[1] == [sender], 'Content does not hold same sender-partner')
            cases.append(dict(case_index=len(cases), case_id=pair['pair_id']+'_'+AGENTS[listener],
                source_pair_index=pair_index, pair_id=pair['pair_id'], ecology=ecology,
                axis=pair['axis'], axis_index=AXES.index(pair['axis']), sender=sender, listener=listener,
                sender_name=AGENTS[sender], listener_name=AGENTS[listener],
                needs=endpoint_needs, need_indices=[need_lookup[ecology][tuple(n)] for n in endpoint_needs],
                canonical_success_action_masks=[bitmask(x) for x in action_sets],
                canonical_success_actions=[sorted(x) for x in action_sets], roles=roles,
                role_indices=[ROLE_NAMES.index(x) for x in roles], listener_partners=partners,
                endpoint_compatible_pairs=[pair['compatible_pairs_before'], pair['compatible_pairs_after']],
                classification=group, classification_index=CLASSES.index(group),
                disjoint=disjoint, role_switch=switch, full_plan_ids=plans,
                original_endpoint_need_weights=pair['endpoint_target_need_weights'],
                PI_listener_observation_identical=True, expanded=ecology == 'unique'))
    require(len(cases) == 4488, 'All4488 listener metadata required')
    unique_cases = [c for c in cases if c['ecology'] == 'unique']
    require(Counter(c['classification'] for c in unique_cases) == {'content': 120, 'role': 408, 'descriptive': 240}, 'Wrong unique case classification')
    strata = Counter((c['classification'], c['axis'], c['sender'], c['listener']) for c in unique_cases)
    expected_counts = {'content': (24, 24, 72), 'role': (204, 204, 0), 'descriptive': (24, 24, 192)}
    for group, counts in expected_counts.items():
        for axis, expected_count in zip(AXES, counts):
            require(sum(c['classification'] == group and c['axis'] == axis for c in unique_cases) == expected_count, 'Axis count differs')
            if group != 'descriptive' and expected_count:
                require(all(strata[group, axis, s, l] == expected_count//6 for s, l in permutations(range(3), 2)), 'Ordered sender/listener strata unbalanced')
    for c in cases:
        for group in ('content', 'role'):
            c[group+'_within_axis_weight_fraction'] = str(Fraction(1, 6*strata[group, c['axis'], c['sender'], c['listener']])) if c['ecology'] == 'unique' and c['classification'] == group else '0'
    arrays_by_part, index_spaces, donor = {}, {}, {}
    layout_lookup = {tuple(layout): (pi, li) for pi, part in enumerate(PARTITIONS)
        for li, layout in enumerate(prepared['partitions']['unique'][part]['layouts'])}
    require(set(layout_lookup) == set(permutations(range(4))), 'Original partition union must have all24 layouts')
    remote_flows = {}
    for part in PARTITIONS:
        spec = prepared['partitions']['unique'][part]
        layouts, owners = spec['layouts'], spec['private_sites']
        require(len(layouts) == (18 if part == 'train' else 6) and set(map(tuple, owners)) == set(permutations((1, 2, 3))), 'Wrong layout/owner support')
        physical = len(layouts)*len(owners); n = len(unique_cases)*physical
        case_indices = np.repeat(np.asarray([c['case_index'] for c in unique_cases], dtype=np.int32), physical)
        li = np.tile(np.repeat(np.arange(len(layouts), dtype=np.int16), len(owners)), len(unique_cases))
        oi = np.tile(np.arange(len(owners), dtype=np.int8), len(unique_cases)*len(layouts))
        ni = np.repeat(np.asarray([c['need_indices'] for c in unique_cases], dtype=np.int16), physical, axis=0)
        endpoints = (ni.astype(np.int64)*len(layouts)+li[:, None])*len(owners)+oi[:, None]
        masks = np.empty((n, 2), dtype=np.uint32)
        content = np.zeros(n, dtype=np.float64); role_weight = np.zeros(n, dtype=np.float64)
        denominators = np.zeros(n, dtype=np.int64)
        for ci, c in enumerate(unique_cases):
            start = ci*physical
            if c['classification'] in ('content', 'role'):
                denominator = 6*strata[c['classification'], c['axis'], c['sender'], c['listener']]*physical
                denominators[start:start+physical] = denominator
                (content if c['classification'] == 'content' else role_weight)[start:start+physical] = 1/denominator
            for layout_index, layout in enumerate(layouts):
                transform = action_permutation(layout)
                actual_masks = [bitmask(transform[list(x)]) for x in c['canonical_success_actions']]
                block = start+layout_index*len(owners)
                masks[block:block+len(owners)] = actual_masks
        repeated = lambda key, dtype: np.repeat(np.asarray([c[key] for c in unique_cases], dtype=dtype), physical, axis=0)
        listener_partners = [[x[0] if len(x) == 1 else -1 for x in c['listener_partners']] for c in unique_cases]
        arrays = dict(case_index=case_indices, source_pair_index=repeated('source_pair_index', np.int16),
            endpoint_indices=endpoints, success_action_masks=masks, need_indices=ni,
            axis=repeated('axis_index', np.int8), sender=repeated('sender', np.int8), listener=repeated('listener', np.int8),
            classification=repeated('classification_index', np.int8), roles=repeated('role_indices', np.int8),
            listener_partners=np.repeat(np.asarray(listener_partners, dtype=np.int8), physical, axis=0),
            layout_indices=li, owner_indices=oi, content_within_axis_weight=content,
            role_within_axis_weight=role_weight, active_weight_denominator=denominators)
        remote_locations = [layout_lookup[tuple((material+1) % 4 for material in layout)] for layout in layouts]
        remote_part = np.asarray([x[0] for x in remote_locations], dtype=np.int8)[li]
        remote_layout = np.asarray([x[1] for x in remote_locations], dtype=np.int16)[li]
        remote_endpoints = np.empty_like(endpoints)
        for donor_pi, donor_part in enumerate(PARTITIONS):
            selected = remote_part == donor_pi
            donor_layout_count = len(prepared['partitions']['unique'][donor_part]['layouts'])
            remote_endpoints[selected] = (ni[selected].astype(np.int64)*donor_layout_count+remote_layout[selected, None])*len(owners)+oi[selected, None]
        arrays.update(remote_donor_partition=remote_part, remote_donor_layout_indices=remote_layout,
            remote_donor_endpoint_indices=remote_endpoints)
        remote_flows[part] = {donor_part: dict(layouts=sum(dp == dpi for dp, dl in remote_locations),
            physical_layout_owner_combinations=sum(dp == dpi for dp, dl in remote_locations)*len(owners),
            all_unique_listener_rows=int(np.count_nonzero(remote_part == dpi)),
            content_rows=int(np.count_nonzero((remote_part == dpi) & (content > 0))),
            role_rows=int(np.count_nonzero((remote_part == dpi) & (role_weight > 0))),
            per_axis_probe_probability_fraction=str(Fraction(sum(dp == dpi for dp, dl in remote_locations), len(layouts))))
            for dpi, donor_part in enumerate(PARTITIONS)}
        require(endpoints.min() >= 0 and endpoints.max() < spec['world_count'], 'Bad original endpoint index')
        require(np.all(masks > 0) and np.all(masks < 1 << 17), 'Empty/illegal action mask')
        for group, weights, axes in (('content', content, range(3)), ('role', role_weight, range(2))):
            for axis in axes:
                require(abs(float(weights[arrays['axis'] == axis].sum())-1.) <= 1e-12, 'Per-axis weights not normalized')
        arrays_by_part[part] = arrays
        index_spaces[part] = dict(needs=spec['needs'], layouts=layouts, private_sites=owners,
            endpoint_world_count=spec['world_count'], pair_listener_physical_rows=n, physical_multiplier=physical,
            index_formula='((need_index * layout_count) + layout_index) * owner_count + owner_index',
            row_order='unique listener cases in cases.json order, then frozen layout index, then frozen private-owner index')
        donor[part] = layout_donor_mapping(layouts)
    require(set(map(tuple, index_spaces['train']['layouts'])).isdisjoint(map(tuple, index_spaces['heldout_layouts']['layouts'])), 'Layout overlap')
    require(all(sha(p) == digest for p, digest in sources.items()), 'Source changed while preparing')
    return dict(schema='single_sender_need_counterfactual_dataset_v1', source_sha256=sources,
        source_pairs=pairs, cases=cases, unique_case_indices=[c['case_index'] for c in unique_cases],
        partitions=arrays_by_part, endpoint_index_spaces=index_spaces, structural_plans_and_actions=original,
        layout_donor_mapping_proposal=donor,
        remote_background_mapping=dict(rule='At each site, donor_material=(recipient_material+1) mod4; demands and private-site owners unchanged.',
            all24_layout_bijection=True, all_four_site_materials_change=True,
            donor_partition_codes=dict(enumerate(PARTITIONS)), recipient_to_donor_flows=remote_flows,
            may_cross_train_heldout_partition=True,
            scope='Natural diagnostic partitions remain unchanged. In remote interventions, heldout labels only the recipient layout; donor may be trained.',
            frozen_from_static_feasibility_not_policy_performance=True),
        summary=dict(source_demand_pairs=2244, all_listener_cases=4488, unique_listener_cases=768,
            multiple_metadata_only_listener_cases=3720, unique_cases_by_class=dict(Counter(c['classification'] for c in unique_cases)),
            unique_cases_by_class_axis={g: dict(zip(AXES, counts)) for g, counts in expected_counts.items()},
            partition_rows={p: len(a['case_index']) for p, a in arrays_by_part.items()},
            content_rows={p: int(np.count_nonzero(a['content_within_axis_weight'])) for p, a in arrays_by_part.items()},
            role_rows={p: int(np.count_nonzero(a['role_within_axis_weight'])) for p, a in arrays_by_part.items()}),
        weighting=dict(unit='unordered demand-endpoint pair × listener × unchanged layout × unchanged owner',
            content='For each axis independently, six ordered sender/listener strata equal, then cases/layouts/owners uniform. Each axis sums1, all three axes sum3. A later equal-axis macro divides by3.',
            role='Separate two-axis secondary: each kind/length axis sums1 under six ordered sender/listener strata, then cases/layouts/owners uniform. Equal-axis macro divides by2; never mixed into content primary.',
            descriptive='No primary probability weights; all intersecting and multiple cases retained as metadata.',
            direction='Rows are unordered two-endpoint cases. Two transplant directions are not doubled into independent cases.',
            not_original_world_distribution=True),
        boundaries=dict(analysis_only_labels_not_actor_inputs=True, observed_policy_used_for_selection=False,
            trained_actions_or_messages_read=False, neural_forward_calls=0, training_calls=0,
            one_sender_demand_changes_other_demands_layout_owner_fixed=True,
            role_switch_is_derived_not_fourth_manipulated_axis=True,
            same_listener_PI_observation_by_construction=True,
            FI_listener_observation_not_identical=True, within_partition_matching_proposal_not_enabled=True,
            remote_all24_plus1_mapping_prepared_not_executed=True))


def prepare(output, source_dir=SOURCE, prepared_path=PREPARED):
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite preparation')
    data = build_dataset(source_dir, prepared_path)
    output.mkdir(parents=True)
    outputs = {}
    for name, value in (('source_demand_pairs.json', data['source_pairs']), ('cases.json', data['cases']),
        ('endpoint_index_spaces.json', data['endpoint_index_spaces']),
        ('structural_plans_and_actions.json', data['structural_plans_and_actions'])):
        path = output/name; path.write_bytes(json_bytes(value)); outputs[name] = dict(sha256=sha(path))
    for part, arrays in data['partitions'].items():
        name = part+'.npz'; path = output/name
        np.savez_compressed(path, **arrays)
        outputs[name] = dict(sha256=sha(path), arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=array_sha(v)) for k, v in arrays.items()})
    manifest = {k: v for k, v in data.items() if k not in ('source_pairs', 'cases', 'partitions', 'endpoint_index_spaces', 'structural_plans_and_actions')}
    manifest.update(created_at=datetime.now(timezone.utc).isoformat(), status='prepared_static_no_model',
        dataset_source_sha256=sha(__file__), outputs=outputs)
    (output/'manifest.json').write_bytes(json_bytes(manifest))
    return dict(status=manifest['status'], output=str(output), manifest_sha256=sha(output/'manifest.json'), summary=data['summary'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare',))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out), ensure_ascii=False))
