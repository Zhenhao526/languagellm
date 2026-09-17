"""Static cross-layout transfer coverage; never reads any policy output."""
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
from research_program.triadic_action_dependency_study import dataset, environment

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'research_program/triadic_action_dependency_study/results/context_001'
RANK_PREFIX = 'triadic_context_transfer_layout_rank_v1'


def sha(path): return sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def compact(x): return json.dumps(x, separators=(',', ':'))


def matching(layouts):
    n = len(layouts)
    adjacency = [[j for j in range(n) if all(a != b for a, b in zip(layouts[i], layouts[j]))] for i in range(n)]
    right = [-1] * n
    def augment(i, seen):
        for j in adjacency[i]:
            if j in seen: continue
            seen.add(j)
            if right[j] < 0 or augment(right[j], seen):
                right[j] = i; return True
        return False
    for i in range(n): augment(i, set())
    result = [-1] * n
    for j, i in enumerate(right):
        if i >= 0: result[i] = j
    witness = None
    if -1 in result:
        left, seen_right = {i for i, j in enumerate(result) if j < 0}, set()
        while True:
            old = len(left), len(seen_right)
            seen_right.update(j for i in left for j in adjacency[i])
            left.update(right[j] for j in seen_right if right[j] >= 0)
            if old == (len(left), len(seen_right)): break
        assert len(seen_right) < len(left)
        empty = [i for i, values in enumerate(adjacency) if not values]
        witness = dict(left_indices=[empty[0]] if empty else sorted(left),
                       donor_neighbor_indices=[] if empty else sorted(seen_right))
    return dict(adjacency=adjacency, maximum_matching=sum(j >= 0 for j in result),
        perfect_mapping=result if -1 not in result else None, Hall_witness=witness)


def mapping_options(layouts):
    n = len(layouts)
    ranks = sorted(range(n), key=lambda i: (sha256((RANK_PREFIX + '|' + compact(layouts[i])).encode()).hexdigest(), layouts[i]))
    shifts = []
    for shift in range(1, n):
        mapping = [-1] * n
        for rank, i in enumerate(ranks): mapping[i] = ranks[(rank + shift) % n]
        assert sorted(mapping) == list(range(n)) and all(i != j for i, j in enumerate(mapping))
        shifts.append(mapping)
    assert all({m[i] for m in shifts} == set(range(n)) - {i} for i in range(n))
    strict = matching(layouts)
    options = {'sha_cycle': {i: [shifts[0][i]] for i in range(n)},
               'all_other_layouts': {i: [m[i] for m in shifts] for i in range(n)}}
    if strict['perfect_mapping'] is not None:
        options['strict_all_four_change'] = {i: [j] for i, j in enumerate(strict['perfect_mapping'])}
    else:
        assert n == 6
        candidates = [m for m in permutations(range(n)) if all(i != j for i, j in enumerate(m))]
        def objective(m):
            changes = [sum(a != b for a, b in zip(layouts[i], layouts[j])) for i, j in enumerate(m)]
            return sum(changes), min(changes)
        best = sorted(candidates, key=lambda m: (-objective(m)[0], -objective(m)[1], m))[0]
        options['max_position_change_derangement'] = {i: [j] for i, j in enumerate(best)}
        strict['all_720_bijections_tested'] = len(list(permutations(range(n))))
        strict['nonself_bijections'] = len(candidates)
        strict['max_change_mapping'] = list(best)
        strict['max_change_objective_total_and_min'] = list(objective(best))
    return dict(rank_prefix=RANK_PREFIX, rank_indices=ranks, nonzero_shift_mappings=shifts,
                strict_matching=strict), options


def fraction(x): return dict(exact=str(x), value=float(x))


def coverage(spec, rows, donors):
    layouts, owners = spec['layouts'], spec['private_sites']
    pos = [[layout.index(m) for m in range(4)] for layout in layouts]
    groups = Counter()
    for row in rows:
        if row['classification'] != 'content': continue
        target = [environment.full_success_plans(n)[0] for n in row['needs']]
        groups[(row['axis'], row['sender'], row['listener'], target[0][2], target[1][2],
                row['within_axis_case_weight_denominator'])] += 1
    report = {}
    for axis in dataset.AXES:
        patterns, raw_patterns = defaultdict(Fraction), Counter()
        visible = defaultdict(Fraction)
        raw_total = 0
        for (a, sender, listener, m0, m1, denominator), count in groups.items():
            if a != axis: continue
            for i in range(len(layouts)):
                for j in donors[i]:
                    pattern = (pos[i][m0] != pos[j][m0], pos[i][m1] != pos[j][m1])
                    key = ''.join(str(int(x)) for x in pattern)
                    weight = Fraction(count, denominator * len(layouts) * len(donors[i]))
                    patterns[key] += weight
                    raw_patterns[key] += count * len(owners)
                    raw_total += count * len(owners)
                    for owner in owners:
                        seen_s = any(layouts[i][site] != layouts[j][site] for site in (0, owner[sender]))
                        seen_l = any(layouts[i][site] != layouts[j][site] for site in (0, owner[listener]))
                        for key_, value in (('sender_LL_material_view_unchanged', not seen_s),
                                            ('listener_LL_material_view_unchanged', not seen_l),
                                            ('both_sender_listener_LL_views_unchanged', not seen_s and not seen_l)):
                            if value: visible[key_] += weight / len(owners)
        assert sum(patterns.values()) == 1
        endpoint = [sum(v for k, v in patterns.items() if k[e] == '1') for e in range(2)]
        report[axis] = dict(case_recipient_owner_donor_rows=raw_total,
            raw_endpoint_change_pattern_counts={k: raw_patterns[k] for k in ('00', '01', '10', '11')},
            weighted_endpoint_change_pattern={k: fraction(patterns[k]) for k in ('00', '01', '10', '11')},
            both_targets_change=fraction(patterns['11']), neither_target_changes=fraction(patterns['00']),
            endpoint_target_site_change=[fraction(x) for x in endpoint],
            direction_mean_correct_donor_action_copy_excluded=fraction(sum(endpoint) / 2),
            LL_visible_context={k: fraction(visible[k]) for k in ('sender_LL_material_view_unchanged', 'listener_LL_material_view_unchanged', 'both_sender_listener_LL_views_unchanged')})
    return report


def eligible_certificate(spec, rows):
    layouts = spec['layouts']; pos = [[layout.index(m) for m in range(4)] for layout in layouts]
    table, axis_hist, raw_eligible = {}, defaultdict(Counter), Counter()
    for row in rows:
        if row['classification'] != 'content': continue
        materials = tuple(environment.full_success_plans(n)[0][2] for n in row['needs'])
        key = ','.join(map(str, materials))
        if key not in table:
            table[key] = [[j for j in range(len(layouts)) if j != i and all(pos[i][m] != pos[j][m] for m in materials)] for i in range(len(layouts))]
        for eligible in table[key]:
            axis_hist[row['axis']][len(eligible)] += 1
            raw_eligible[row['axis']] += len(eligible) * len(spec['private_sites'])
    return dict(all_cases_all_recipient_layouts_have_eligible=all(0 not in h for h in axis_hist.values()),
        eligible_by_target_material_pair_and_recipient_layout=table,
        axis_eligible_count_distribution={a: dict(minimum=min(h), maximum=max(h),
            case_recipient_layout_count=sum(h.values()), histogram=dict(sorted(h.items())),
            case_recipient_owner_eligible_donor_rows=raw_eligible[a]) for a, h in axis_hist.items()},
        original_cases_retained=True, every_recipient_background_retained=True,
        selection='Before policy reads: donor != recipient, same layout partition, both endpoint target material sites differ; owner unchanged.',
        correct_action_certificate='For either endpoint, matching partner and destination are unchanged across its two backgrounds while correct site differs. Copying the donor TRUE full action cannot equal the recipient TRUE full action.',
        actual_action_limitation='No policy output has been read. A donor incorrect actual action may still coincide with a recipient correct action; this certificate does not rule that out.',
        conditional_weights='Each axis / six ordered S-L / case / recipient layout-owner remains uniform in that hierarchy; within a case-recipient background, eligible donors are uniform. Do not pool eligible rows with uniform row weights.')


def main():
    freeze, plan, prepared = [read(SOURCE / name) for name in ('freeze.json', 'plan.json', 'prepared.json')]
    assert sha(SOURCE / 'plan.json') == freeze['plan_sha256']
    assert sha(SOURCE / 'prepared.json') == freeze['prepared_sha256'] == plan['prepared_sha256']
    sources = {str(SOURCE / name): sha(SOURCE / name) for name in ('freeze.json', 'plan.json', 'prepared.json')}
    for module in (dataset, environment):
        path = Path(module.__file__).resolve(); assert sha(path) == plan['sources'][str(path.relative_to(ROOT))]
        sources[str(path)] = sha(path)
    geometric = {}; reports = {}; budget = {}
    for part, spec in prepared['partitions'].items():
        layout_key = spec['layout_split']
        if layout_key not in geometric:
            geometry, options = mapping_options(spec['layouts'])
            geometric[layout_key] = dict(layouts=spec['layouts'], geometry=geometry, options=options)
        options = geometric[layout_key]['options']
        pairs = dataset.content_pairs(spec); rows = pairs['rows']
        counts = Counter(r['axis'] for r in rows if r['classification'] == 'content')
        case_worlds = sum(counts.values()) * len(spec['layouts']) * len(spec['private_sites'])
        reports[part] = dict(content_cases_by_axis=dict(counts), case_recipient_owner_rows=case_worlds,
            coverage_by_mapping={name: coverage(spec, rows, donors) for name, donors in options.items()},
            eligible_certificate=eligible_certificate(spec, rows))
        assert reports[part]['eligible_certificate']['all_cases_all_recipient_layouts_have_eligible']
        budget[part] = dict(content_cases=sum(counts.values()), recipient_backgrounds=len(spec['layouts']) * 6,
            donor_other_layouts=len(spec['layouts']) - 1, recipient_case_rows=case_worlds,
            all_other_remote_rows_per_policy=case_worlds * (len(spec['layouts']) - 1) * 2 * 2,
            sham_plus_local_rows_per_policy=case_worlds * 2 * 2)
    remote = sum(b['all_other_remote_rows_per_policy'] for b in budget.values())
    local = sum(b['sham_plus_local_rows_per_policy'] for b in budget.values())
    result = dict(status='completed_static_only', source_sha256=sources, static_source_sha256=sha(__file__),
        geometry=geometric, partitions=reports, budgets=dict(by_partition=budget,
            remote_all_other_rows_per_policy=remote, sham_plus_local_rows_per_policy=local,
            combined_rows_per_policy=remote + local, if_eight_live_policies_combined_rows=8 * (remote + local),
            unit='Recipient-case-background × donor-layout × direction × intervention mode; not independent teams or network-module calls.',
            source_natural_records='Already stored endpoints; no new inference required just to obtain original donor packets.',
            silent_reuse='Not decided or counted as new forwards here; must verify the planned intervention identity first.'),
        boundaries=dict(neural_forward_calls=0, training_calls=0, policy_output_files_read=0,
            fixed_pair_policy_filter=False, all_content_cases_retained=True,
            whole_packet_transfer_not_a_test_of_compositional_message_parts=True,
            not_final_intervention_dataset=True))
    out = Path(__file__).with_name('layout_static_001'); assert not out.exists(); out.mkdir()
    (out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for path, digest in sources.items(): assert sha(path) == digest
    print(json.dumps(dict(status=result['status'], budget=result['budgets'],
        all_other_coverage={p: {a: x['both_targets_change'] for a, x in r['coverage_by_mapping']['all_other_layouts'].items()} for p, r in reports.items()}), ensure_ascii=False))


if __name__ == '__main__': main()
