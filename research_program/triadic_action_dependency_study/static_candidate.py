"""Exhaustive task-support design calculation; no policy or result inputs."""
from collections import Counter
from itertools import combinations, permutations, product
from pathlib import Path
from hashlib import sha256
import json

RESOURCE = (0b0011, 0b1100, 0b0101, 0b1010, 1, 2, 4, 8)
DEST = (1, 2, 3)
AXES = ('kind', 'length', 'destination')
PAIRS = tuple(combinations(range(3), 2))


def plans(needs):
    result = []
    for i, j in PAIRS:
        resources = RESOURCE[needs[i] // 3] & RESOURCE[needs[j] // 3]
        destinations = DEST[needs[i] % 3] & DEST[needs[j] % 3]
        for material, destination in product(range(4), range(2)):
            if (resources & (1 << material)) and (destinations & (1 << destination)):
                result.append((i, j, material, destination))
    return tuple(result)


def actions(plans_, listener):
    peers = [a for a in range(3) if a != listener]
    return frozenset(0 if listener not in (i, j) else
        1 + material * 4 + destination * 2 + peers.index(j if listener == i else i)
        for i, j, material, destination in plans_)


def flips(need):
    r, d = divmod(need, 3)
    if r < 4:
        yield AXES[r // 2], (r ^ 1) * 3 + d
    else:
        yield 'kind', (4 + ((r - 4) ^ 2)) * 3 + d
        yield 'length', (4 + ((r - 4) ^ 1)) * 3 + d
    if d != 2:
        yield 'destination', r * 3 + (1 - d)


def pair_summary(domain):
    full = {n: plans(n) for n in domain}
    counts, sizes, by_ordered, by_pair, role_edges = Counter(), Counter(), Counter(), Counter(), Counter()
    pair_counts, examples = Counter(), {}
    for before in sorted(domain):
        for sender in range(3):
            for axis, changed in flips(before[sender]):
                after = list(before); after[sender] = changed; after = tuple(after)
                if after not in domain or before >= after:
                    continue
                pair_counts[axis] += 1
                for listener in range(3):
                    if listener == sender:
                        continue
                    a, b = actions(full[before], listener), actions(full[after], listener)
                    disjoint = a.isdisjoint(b)
                    content = disjoint and 0 not in a and 0 not in b
                    role = disjoint and ((a == {0} and 0 not in b) or (b == {0} and 0 not in a))
                    label = 'content' if content else 'role' if role else 'descriptive'
                    counts[axis, label] += 1
                    if label == 'content':
                        sizes[axis, len(a), len(b)] += 1
                        by_ordered[axis, sender, listener] += 1
                        by_pair[axis, ''.join('ABC'[a] for a in sorted((sender, listener)))] += 1
                    if label == 'role' and len(full[before]) == len(full[after]) == 1:
                        role_edges[axis, ''.join('ABC'[a] for a in full[before][0][:2]),
                                   ''.join('ABC'[a] for a in full[after][0][:2])] += 1
                    examples.setdefault(axis + '/' + label, dict(before=before, after=after,
                        sender='ABC'[sender], listener='ABC'[listener],
                        listener_actions_before=sorted(a), listener_actions_after=sorted(b),
                        full_plans_before=full[before], full_plans_after=full[after]))
    return dict(unordered_demand_pairs_by_axis=dict(pair_counts),
        listener_counts={'/'.join(k): v for k, v in sorted(counts.items())},
        content_size_counts={'/'.join(map(str, k)): v for k, v in sorted(sizes.items())},
        content_ordered_sender_listener={'/'.join(map(str, k)): v for k, v in sorted(by_ordered.items())},
        content_pair_counts={'/'.join(k): v for k, v in sorted(by_pair.items())},
        role_edge_transitions={'/'.join(k): v for k, v in sorted(role_edges.items())}, examples=examples)


def transform_need(n, kind_flip=False, length_flip=False, destination_flip=False):
    r, d = divmod(n, 3)
    mask = RESOURCE[r]
    transformed = sum(1 << (m ^ (2 if kind_flip else 0) ^ (1 if length_flip else 0))
                      for m in range(4) if mask & (1 << m))
    r = RESOURCE.index(transformed)
    if destination_flip and d != 2:
        d = 1 - d
    return r * 3 + d


def main():
    old = {n for n in product(range(12), repeat=3) if plans(n)}
    new = {n for n in product(range(24), repeat=3) if len(plans(n)) == 1}
    old_pairs, new_pairs = pair_summary(old), pair_summary(new)
    assert set(old_pairs['content_size_counts']) == {'kind/1/1', 'length/1/1', 'destination/2/2'}
    assert set(new_pairs['content_size_counts']) == {'kind/1/1', 'length/1/1', 'destination/1/1'}
    for order in permutations(range(3)):
        assert {tuple(n[i] for i in order) for n in new} == new
    for k, l, d in product((False, True), repeat=3):
        assert {tuple(transform_need(v, k, l, d) for v in n) for n in new} == new
    target_counts = Counter((plans(n)[0][0], plans(n)[0][1], plans(n)[0][2], plans(n)[0][3]) for n in new)
    need_marginals = [Counter(n[i] for n in new) for i in range(3)]
    assert need_marginals[0] == need_marginals[1] == need_marginals[2]
    assert len(set(target_counts.values())) == 1 and len(target_counts) == 24
    role_exact_count = Counter(sum(v // 3 >= 4 for v in n) for n in new)
    report = dict(status='completed_static_only', audit_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        old_environment_sha256=sha256((Path(__file__).parents[1] / 'triadic_task/environment.py').read_bytes()).hexdigest(),
        source='Explicit finite predicates; no model actions/messages/performance loaded.',
        resource_acceptance_bitmasks=RESOURCE, destination_acceptance_bitmasks=DEST,
        old=dict(all_demand_tables=12**3, solvable_demand_tables=len(old), summary=old_pairs),
        candidate=dict(all_demand_tables=24**3, demand_tables=len(new),
            world_counts=dict(all_24_layouts_6_owners=len(new)*144,
                              train_18_layouts_6_owners=len(new)*108,
                              heldout_6_layouts_6_owners=len(new)*36),
            demand_tables_by_number_exact_resource_agents=dict(sorted(role_exact_count.items())),
            identical_agent_need_marginal_counts=dict(sorted(need_marginals[0].items())),
            full_target_counts={'/'.join(map(str, k)): v for k, v in sorted(target_counts.items())},
            all_agent_permutations_invariant=True, all_global_kind_length_destination_flips_invariant=True,
            every_world_has_one_full_success_joint_plan=True,
            layout_permutation_preserves_action_set_sizes=True,
            summary=new_pairs),
        calls=dict(training=0, neural_forward=0),
        scope='Candidate support, not a frozen experiment; singleton successful actions do not match all cognitive/information difficulty across axes.')
    out = Path(__file__).with_name('static_candidate_001.json')
    assert not out.exists()
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(old_solvable=len(old), new_domain=len(new),
        old_content_sizes=old_pairs['content_size_counts'], new_content_sizes=new_pairs['content_size_counts'],
        candidate_counts=new_pairs['listener_counts'], output=str(out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
