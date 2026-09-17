"""Deterministic semantic-orbit split and official-observation-only features.

All targets, demand ids and semantic pair labels remain researcher-side.
"""
from collections import Counter
from copy import deepcopy
from itertools import combinations, permutations, product
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np

from . import environment as env

PARTITIONS = ('train', 'new_needs', 'new_layouts', 'new_needs_and_layouts')
INFORMATION = env.INFORMATION
FEATURES = 54
AXES = ('kind', 'length', 'destination')
CLASSES = ('descriptive', 'content', 'role')
NEED_SPLIT_PREFIX = 'action_dependency_need_split_v1'
LAYOUT_SPLIT_PREFIX = 'action_dependency_layout_split_v1'
MONITOR_PREFIX = 'action_dependency_monitor_background_v1'
ACTIONS = [env.all_actions(a) for a in env.AGENTS]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_text(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False)


def ranked_hash(prefix, value):
    return hashlib.sha256((prefix + '|' + canonical_text(value)).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform_need(need, kind_flip, length_flip, swap_axes, destination_flip):
    r, d = divmod(need, 3)
    materials = []
    for material in env.RESOURCE_ACCEPTANCE[r]:
        kind, length = (material // 2) ^ kind_flip, (material % 2) ^ length_flip
        if swap_axes:
            kind, length = length, kind
        materials.append(kind * 2 + length)
    r = env.RESOURCE_ACCEPTANCE.index(tuple(sorted(materials)))
    if destination_flip and d != 2:
        d = 1 - d
    return r * 3 + d


def demand_orbit(needs):
    result = set()
    for k, l, swap, d in product(range(2), repeat=4):
        transformed = tuple(transform_need(n, k, l, swap, d) for n in needs)
        for order in permutations(range(3)):
            result.add(tuple(transformed[i] for i in order))
    return tuple(sorted(result))


def demand_orbits():
    domain = set(env.support()); remaining = set(domain); rows = []
    while remaining:
        orbit = demand_orbit(min(remaining))
        require(set(orbit) <= domain, 'Symmetry transform leaves task support')
        require(set(orbit) <= remaining, 'Orbit overlaps an earlier orbit')
        remaining.difference_update(orbit)
        canonical = orbit[0]
        rows.append(dict(canonical=list(canonical), members=[list(n) for n in orbit],
                         size=len(orbit), ranking_sha256=ranked_hash(NEED_SPLIT_PREFIX, canonical)))
    rows.sort(key=lambda row: (row['ranking_sha256'], row['canonical']))
    held = math.ceil(len(rows) / 4)
    for rank, row in enumerate(rows):
        row.update(rank=rank, split='heldout' if rank < held else 'train')
    return rows


def flips(need):
    resource, destination = divmod(need, 3)
    if resource < 4:
        yield AXES[resource // 2], (resource ^ 1) * 3 + destination
    else:
        yield 'kind', (4 + ((resource - 4) ^ 2)) * 3 + destination
        yield 'length', (4 + ((resource - 4) ^ 1)) * 3 + destination
    if destination != 2:
        yield 'destination', resource * 3 + (1 - destination)


def plan_action_indices(plan):
    i, j, site, destination = plan
    result = [0, 0, 0]
    for who, partner in ((i, j), (j, i)):
        result[who] = 1 + site * 4 + destination * 2 + [a for a in range(3) if a != who].index(partner)
    return result


JOINT_ACTIONS = np.asarray([plan_action_indices((i, j, s, d))
    for i, j in combinations(range(3), 2) for s, d in product(range(4), range(2))], dtype=np.int64)


def content_pairs(spec):
    """All listener cases, including role/descriptive; canonical layout only.

    Later expansion uses both endpoint need indices with identical layout/owner.
    No policy-dependent filtering or new nuisance background is supplied here.
    """
    needs = [tuple(n) for n in spec['needs']]; lookup = {n: i for i, n in enumerate(needs)}
    full = {n: env.full_success_plans(n) for n in needs}
    require(all(len(p) == 1 for p in full.values()), 'Pair support is not single-plan')
    correct = {n: plan_action_indices(p[0]) for n, p in full.items()}
    rows, counts, strata = [], Counter(), Counter()
    demand_pair_count = Counter()
    for before in needs:
        for sender in range(3):
            for axis, changed in flips(before[sender]):
                after = list(before); after[sender] = changed; after = tuple(after)
                if after not in lookup or before >= after:
                    continue
                demand_pair_count[axis] += 1
                for listener in range(3):
                    if listener == sender:
                        continue
                    choices = [correct[before][listener], correct[after][listener]]
                    role = (choices[0] == 0) != (choices[1] == 0)
                    content = all(choices) and choices[0] != choices[1]
                    label = 'role' if role else 'content' if content else 'descriptive'
                    partners = [[a for a in range(3) if a != listener][(c - 1) % 2] if c else -1 for c in choices]
                    if content:
                        require(partners == [sender, sender], 'Content changed partner or did not concern sender')
                    counts[axis, label] += 1
                    strata[label, axis, sender, listener] += 1
                    rows.append(dict(case_id=f"{lookup[before]}_{lookup[after]}_{sender}_{listener}",
                        classification=label, classification_index=CLASSES.index(label), mode=label,
                        axis=axis, axis_index=AXES.index(axis), sender=sender, listener=listener,
                        endpoint_need_indices=[lookup[before], lookup[after]], needs=[list(before), list(after)],
                        correct_actions=[correct[before], correct[after]],
                        listener_correct_actions=choices, success_action_masks=[1 << c for c in choices],
                        listener_partners=partners, role_indices=[int(c != 0) for c in choices],
                        required_partner_unchanged=partners[0] == partners[1],
                        listener_PL_LL_observation_identical=True,
                        canonical_layout=[0, 1, 2, 3]))
    for row in rows:
        denominator = 6 * strata[row['classification'], row['axis'], row['sender'], row['listener']]
        row['within_axis_case_weight_denominator'] = denominator if row['classification'] != 'descriptive' else 0
    availability = {a: all(strata['content', a, s, l] > 0 for s in range(3) for l in range(3) if s != l) for a in AXES}
    return dict(rows=rows, summary=dict(unordered_demand_pairs_by_axis=dict(demand_pair_count),
        listener_counts={'/'.join(k): v for k, v in sorted(counts.items())},
        content_ordered_sender_listener={'/'.join(map(str, k[1:])): v for k, v in sorted(strata.items()) if k[0] == 'content'},
        role_ordered_sender_listener={'/'.join(map(str, k[1:])): v for k, v in sorted(strata.items()) if k[0] == 'role'},
        content_each_axis_all_six_sender_listener_strata_present=availability,
        physical_multiplier=len(spec['layouts']) * len(spec['private_sites']),
        content_physical_rows_by_axis={a: counts[a, 'content'] * len(spec['layouts']) * len(spec['private_sites']) for a in AXES}),
        weighting='Content: equal three axes; within each axis six ordered S/L equal, then all cases/layouts/owners uniform. Role remains separate; absent strata must be reported, never silently dropped.',
        actor_inputs=False, policy_success_selection=False)


def make_prepared():
    orbits = demand_orbits()
    needs = {split: sorted(n for row in orbits if row['split'] == split for n in row['members']) for split in ('train', 'heldout')}
    marginal_presence = {split: [sorted({n[a] for n in values}) for a in range(3)] for split, values in needs.items()}
    require(all(row == list(range(24)) for values in marginal_presence.values() for row in values), 'Some individual need absent from a need split')
    layout_ranks = sorted([dict(layout=list(p), ranking_sha256=ranked_hash(LAYOUT_SPLIT_PREFIX, p))
                           for p in permutations(range(4))], key=lambda r: (r['ranking_sha256'], r['layout']))
    for rank, row in enumerate(layout_ranks):
        row.update(rank=rank, split='heldout' if rank < 6 else 'train')
    layouts = {split: sorted(row['layout'] for row in layout_ranks if row['split'] == split) for split in ('train', 'heldout')}
    owners = [list(p) for p in permutations((1, 2, 3))]
    partitions = {}
    for part in PARTITIONS:
        ns = 'heldout' if part in ('new_needs', 'new_needs_and_layouts') else 'train'
        ls = 'heldout' if part in ('new_layouts', 'new_needs_and_layouts') else 'train'
        count = len(needs[ns]) * len(layouts[ls]) * len(owners)
        ranked_backgrounds = []
        for li, layout in enumerate(layouts[ls]):
            for oi, owner in enumerate(owners):
                payload = MONITOR_PREFIX + '|' + canonical_text(layout) + '|' + canonical_text(owner)
                ranked_backgrounds.append(dict(layout_index=li, owner_index=oi, layout=layout,
                    private_sites=owner, ranking_sha256=hashlib.sha256(payload.encode()).hexdigest()))
        ranked_backgrounds.sort(key=lambda row: (row['ranking_sha256'], row['layout'], row['private_sites']))
        backgrounds = []
        for row in ranked_backgrounds:
            if row['layout_index'] not in {x['layout_index'] for x in backgrounds}:
                backgrounds.append(row)
            if len(backgrounds) == 2:
                break
        require(len(backgrounds) == 2, 'Two distinct monitor layouts unavailable')
        monitor = sorted((ni * len(layouts[ls]) + row['layout_index']) * 6 + row['owner_index']
                         for ni in range(len(needs[ns])) for row in backgrounds)
        spec = dict(partition=part, needs=needs[ns], layouts=layouts[ls], private_sites=owners,
            world_count=count, monitor_indices=monitor, monitor_backgrounds=backgrounds,
            need_split=ns, layout_split=ls,
            weighting='uniform needs × layouts × owners',
            monitor_scope='All needs at exactly two fixed SHA-ranked backgrounds with distinct layouts. Equal worlds for task metrics; not full-layout target evaluation. Every semantic pair has both endpoints.',
            state_order='need-major, then layout, then owner')
        spec['semantic_pair_summary'] = content_pairs(spec)['summary']
        partitions[part] = spec
    return dict(schema='triadic_action_dependency_dataset_v1', partitions=partitions,
        orbit_count=len(orbits), need_orbits=orbits, layout_ranks=layout_ranks,
        split_hash_serialization='prefix + | + compact JSON integer array for canonical need/layout; SHA256 hex then lex canonical tie-break',
        need_split_prefix=NEED_SPLIT_PREFIX, layout_split_prefix=LAYOUT_SPLIT_PREFIX, monitor_prefix=MONITOR_PREFIX,
        monitor_hash_serialization='action_dependency_monitor_background_v1|compact_JSON_layout|compact_JSON_owner; sort digest,layout,owner; first two distinct layouts',
        monitor_design_change='Before freezing: replaced 1024 isolated SHA-ranked worlds with all needs at two fixed backgrounds so every semantic pair has both endpoints; no failed run or model output prompted this change.',
        full_need_count=sum(len(v) for v in needs.values()),
        need_counts={s: len(v) for s, v in needs.items()}, individual_need_presence=marginal_presence,
        heldout_scope='Unseen joint three-person need-relation orbits; all24 individual needs appear for every actor in both splits.',
        state_packing=['need_A', 'need_B', 'need_C', 'material_S0', 'material_S1', 'material_S2', 'material_S3', 'private_site_A', 'private_site_B', 'private_site_C'],
        feature_definition='54: three (known,kind acceptance2,length acceptance2,destination acceptance2) blocks; four (known,kind2,length2) materials; owner9,self3,FIflag1.',
        full_support_world_count=5376 * 144, neural_forward_calls=0, training_calls=0)


def pack_states(spec, indices=None):
    count = spec['world_count']
    ids = np.arange(count, dtype=np.int64) if indices is None else np.asarray(indices)
    require(ids.ndim == 1 and ids.dtype.kind in 'iu' and np.all((ids >= 0) & (ids < count)), 'Invalid state indices')
    nl, no = len(spec['layouts']), len(spec['private_sites'])
    return np.concatenate((np.asarray(spec['needs'], dtype=np.int16)[ids // (nl * no)],
        np.asarray(spec['layouts'], dtype=np.int16)[(ids // no) % nl],
        np.asarray(spec['private_sites'], dtype=np.int16)[ids % no]), axis=1)


def sample_indices(spec, uniforms):
    u = np.asarray(uniforms, dtype=np.float64)
    require(u.ndim == 2 and u.shape[1] == 3 and np.isfinite(u).all() and np.all((u >= 0) & (u < 1)), 'Expected B×3 independent world uniforms')
    n = np.floor(u[:, 0] * len(spec['needs'])).astype(np.int64)
    l = np.floor(u[:, 1] * len(spec['layouts'])).astype(np.int64)
    o = np.floor(u[:, 2] * len(spec['private_sites'])).astype(np.int64)
    return (n * len(spec['layouts']) + l) * len(spec['private_sites']) + o


def encode_observations(observations):
    encoded = np.zeros((len(observations), 3, FEATURES), dtype=np.float64)
    allowed = {'self', 'public_site', 'private_view_owners', 'own_need', 'visible_materials', 'shared_needs', 'information_control'}
    for b, views in enumerate(observations):
        require(set(views) == set(env.AGENTS), 'Missing actor observation')
        for i, agent in enumerate(env.AGENTS):
            view = views[agent]; x = encoded[b, i]
            require(set(view) <= allowed and view['self'] == agent and view['public_site'] == 'S0', 'Unexpected observation field')
            needs = deepcopy(view.get('shared_needs', {}))
            require(set(needs) <= set(env.AGENTS) and (agent not in needs or needs[agent] == view['own_need']), 'Invalid known needs')
            needs[agent] = view['own_need']
            for who, need in needs.items():
                require(set(need) == {'kinds', 'lengths', 'destinations'}, 'Unexpected need field')
                off = 7 * env.AGENTS.index(who); x[off] = 1
                for key, options, start in (('kinds', ('wood', 'fiber'), 1), ('lengths', ('short', 'long'), 3), ('destinations', ('L', 'R'), 5)):
                    values = need[key]
                    require(values and len(set(values)) == len(values) and set(values) <= set(options), 'Invalid semantic acceptance set')
                    for value in values:
                        x[off + start + options.index(value)] = 1
                require(not (len(need['kinds']) == len(need['lengths']) == 2), 'Unsupported accept-all resource requirement')
            seen = set()
            for material in view['visible_materials']:
                require(set(material) == {'site', 'kind', 'length'} and material['site'] in env.SITES and material['site'] not in seen, 'Invalid visible item')
                require(material['kind'] in ('wood', 'fiber') and material['length'] in ('short', 'long'), 'Invalid visible attributes')
                seen.add(material['site']); off = 21 + 5 * env.SITES.index(material['site'])
                x[off] = 1; x[off + 1 + (material['kind'] == 'fiber')] = 1; x[off + 3 + (material['length'] == 'long')] = 1
            owners = view['private_view_owners']
            require(set(owners) == {'S1', 'S2', 'S3'} and set(owners.values()) == set(env.AGENTS), 'Invalid private-site owners')
            for site, who in owners.items():
                x[41 + 3 * (env.SITES.index(site) - 1) + env.AGENTS.index(who)] = 1
            x[50 + i] = 1
            flag = view.get('information_control')
            require(flag in (None, 'full_information'), 'Unexpected information flag')
            x[53] = flag == 'full_information'
    return encoded


def reward_terms(states):
    result = np.zeros((len(states), 24), dtype=np.float64)
    if not states:
        return result
    packed_needs = np.asarray([s.needs for s in states], dtype=np.int16)
    packed_layout = np.asarray([s.layout for s in states], dtype=np.int16)
    acceptance = np.asarray([[[env.accepts(n, m, d) for d in range(2)] for m in range(4)] for n in range(24)])
    for t, row in enumerate(JOINT_ACTIONS):
        for who, action in enumerate(row):
            if action:
                site, dest = (action - 1) // 4, ((action - 1) % 4) // 2
                result[:, t] += 0.5 * acceptance[packed_needs[:, who], packed_layout[:, site], dest]
    return result


def make_arrays(spec, information=None, indices=None):
    require(information is None or information in INFORMATION, 'Unknown feature information condition')
    packed = pack_states(spec, indices)
    states = [env.State(tuple(row[:3]), tuple(row[3:7]), tuple(row[7:10])) for row in packed]
    result = dict(states=states, packed_states=packed, rewards=reward_terms(states), joint_actions=JOINT_ACTIONS.copy())
    modes = INFORMATION if information is None else (information,)
    for mode in modes:
        features = np.empty((len(states), 3, FEATURES), dtype=np.float64)
        for start in range(0, len(states), 1024):
            chunk = states[start:start + 1024]
            observations = [{a: env.observe(s, a, information=mode) for a in env.AGENTS} for s in chunk]
            features[start:start + len(chunk)] = encode_observations(observations)
        result['x_' + mode] = features
    return result


build_arrays = make_arrays


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Never overwrite preparation')
    prepared = make_prepared(); out.mkdir(parents=True)
    outputs = {}
    for name, value in [('prepared.json', prepared)] + [(part + '_semantic_pairs.json', content_pairs(spec)) for part, spec in prepared['partitions'].items()]:
        path = out / name; path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); outputs[name] = sha(path)
    receipt = dict(status='prepared_static_only', sources={str(Path(__file__).resolve()): sha(__file__), str(Path(env.__file__).resolve()): sha(env.__file__)},
        output_sha256=outputs, orbit_count=prepared['orbit_count'], need_counts=prepared['need_counts'],
        world_counts={p: s['world_count'] for p, s in prepared['partitions'].items()},
        all_content_strata_available=all(all(s['semantic_pair_summary']['content_each_axis_all_six_sender_listener_strata_present'].values()) for s in prepared['partitions'].values()),
        neural_forward_calls=0, training_calls=0)
    (out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare',)); parser.add_argument('--out', required=True)
    args = parser.parse_args(); prepare(args.out)
