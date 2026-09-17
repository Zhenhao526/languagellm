"""Frozen compound-need cases for the partner-direction probe.

The preceding single-axis probe changed an actor's need while preserving the
set of legal partner pairs.  This design keeps the same two-alternative-plan
ecology but selects *compound* need changes that alter two semantic axes and
the legal partner-pair set.  It is a posthoc probe of frozen policies; no
policy-dependent case selection is allowed.
"""
from __future__ import annotations

from collections import Counter
from itertools import product

from research_program.triadic_action_dependency_study import dataset, environment

SEEDS = tuple(range(66701, 66717))
SCHEDULES = ('static', 'rematched')
CONDITIONS = tuple(f'{s}_altpair_factorized_strict_PL_{"live" if live else "silent"}'
                   for s in SCHEDULES for live in (True, False))
PARTS = ('new_layouts',)
CHECKPOINT = 6000
CHUNK_SIZE = 2048
SHAM_ROWS_PER_SENDER = 256
AXES = ('kind_length', 'kind_destination', 'length_destination')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    require(condition in CONDITIONS, 'Unknown condition')
    schedule, suffix = condition.split('_', 1)
    require(schedule in SCHEDULES and
            (suffix.endswith('_live') or suffix.endswith('_silent')),
            'Malformed condition')
    return schedule, suffix.endswith('_live')


def all_needs():
    rows = []
    for need in product(range(24), repeat=3):
        plans = environment.full_success_plans(tuple(need))
        if len(plans) == 2 and len({tuple(p[:2]) for p in plans}) == 2:
            rows.append(tuple(map(int, need)))
    require(len(rows) == 1560, 'Unexpected alternative-partner need count')
    return sorted(rows)


def _semantic_axes(before, after):
    """Return changed semantic acceptance dimensions for two need ids."""
    br, bd = divmod(int(before), 3)
    ar, ad = divmod(int(after), 3)
    before_materials = {environment.MATERIALS[m] for m in environment.RESOURCE_ACCEPTANCE[br]}
    after_materials = {environment.MATERIALS[m] for m in environment.RESOURCE_ACCEPTANCE[ar]}
    result = []
    if {m[0] for m in before_materials} != {m[0] for m in after_materials}:
        result.append('kind')
    if {m[1] for m in before_materials} != {m[1] for m in after_materials}:
        result.append('length')
    if set(environment.DESTINATION_ACCEPTANCE[bd]) != set(environment.DESTINATION_ACCEPTANCE[ad]):
        result.append('destination')
    return tuple(result)


def compound_neighbors(needs):
    """Enumerate every supported one-sender change affecting exactly two axes.

    The returned rows are ``(receiver_need, sender, category, donor_need)``.
    Both endpoint triples remain in the two-alternative-partner support and
    their legal pair sets differ.  The enumeration is balanced by sender and
    pair-set transition, which is checked below before freezing.
    """
    support = set(tuple(map(int, row)) for row in needs)
    result = []
    category_counts = Counter()
    sender_counts = Counter()
    pair_transition_counts = Counter()
    for row in sorted(support):
        receiver_pairs = {tuple(p[:2]) for p in environment.full_success_plans(row)}
        for sender in range(3):
            for donor_value in range(24):
                if donor_value == row[sender]:
                    continue
                donor = list(row); donor[sender] = donor_value; donor = tuple(donor)
                if donor not in support:
                    continue
                donor_pairs = {tuple(p[:2]) for p in environment.full_success_plans(donor)}
                if donor_pairs == receiver_pairs:
                    continue
                changed = _semantic_axes(row[sender], donor_value)
                if len(changed) != 2:
                    continue
                category = '_'.join(changed)
                require(category in AXES, 'Unexpected compound axis category')
                result.append((row, sender, category, donor))
                category_counts[category] += 1
                sender_counts[category, sender] += 1
                pair_transition_counts[category, tuple(sorted(receiver_pairs)),
                                        tuple(sorted(donor_pairs))] += 1
    require(len(result) == 3456, 'Unexpected compound-neighbor count')
    require(dict(category_counts) == {
        'kind_length': 1536, 'kind_destination': 960,
        'length_destination': 960,
    }, 'Compound categories are unbalanced')
    require(all(sender_counts[category, sender] == category_counts[category] // 3
                for category in AXES for sender in range(3)),
            'Compound cases are not sender-balanced')
    require(all(value in (160, 256) for value in pair_transition_counts.values()),
            'Unexpected pair transition multiplicity')
    return result


def source_partition():
    source = dataset.make_prepared()['partitions']['new_layouts']
    needs = all_needs()
    return dict(partition='new_layouts', needs=[list(r) for r in needs],
                layouts=source['layouts'], private_sites=source['private_sites'],
                world_count=len(needs) * len(source['layouts']) * len(source['private_sites']))


def make_cases(spec):
    needs = [tuple(r) for r in spec['needs']]
    lookup = {row: i for i, row in enumerate(needs)}
    neighbors = compound_neighbors(needs)
    nl, no = len(spec['layouts']), len(spec['private_sites'])
    receiver = []; donor = []; sender = []; axis = []; background = []
    for row, actor, category, donor_row in neighbors:
        ni = lookup[row]; di = lookup[donor_row]
        for li in range(nl):
            for oi in range(no):
                receiver.append((ni * nl + li) * no + oi)
                donor.append((di * nl + li) * no + oi)
                sender.append(actor); axis.append(category); background.append(li * no + oi)
    result = dict(
        schema='factorized_altpartner_compound_direction_probe_cases_v1',
        partition=spec['partition'], needs=[list(r) for r in needs],
        layouts=spec['layouts'], private_sites=spec['private_sites'],
        world_count=int(spec['world_count']), background_count=nl * no,
        receiver_state_indices=receiver, donor_state_indices=donor,
        sender=sender, axis=axis, background_indices=background,
        case_count=len(receiver), case_order='need,sender,compound_axis,layout,owner',
        donor_rule='same layout and private-site owner; one supported two-axis need neighbor',
        selection='all supported two-axis neighbors with changed legal partner pair set; no policy/outcome filtering',
        category_counts={name: axis.count(name) for name in AXES},
    )
    require(result['case_count'] == 3456 * nl * no, 'Unexpected compound case count')
    require(all(result['category_counts'][name] == expected * nl * no
                for name, expected in {'kind_length': 1536, 'kind_destination': 960,
                                       'length_destination': 960}.items()),
            'Compound case category counts changed')
    return result


def make_prepared():
    spec = source_partition(); cases = make_cases(spec)
    return dict(
        schema='triadic_factorized_neutral_altpartner_compound_direction_probe_v1',
        seeds=list(SEEDS), schedules=list(SCHEDULES), conditions=list(CONDITIONS),
        parts=list(PARTS), checkpoint=CHECKPOINT, chunk_size=CHUNK_SIZE,
        sham_rows_per_sender=SHAM_ROWS_PER_SENDER, partition=spec, cases=cases,
        compound_axes=list(AXES),
        scientific_question='Does replacing one sender W1 with a compound neighboring need move receivers toward a different legal partner pair?',
        primary='rematching-by-communication interaction on intervention-induced donor-only minus receiver-only partner-edge transfer',
        secondary='full-plan transfer, partner-pair transfer, physical execution, Q, conditional Q, exact sham replay',
        posthoc_relative_to='triadic_factorized_neutral_altpartner_study/results/altpair_001',
        posthoc=True,
        claim_boundary='A compound directional response is a causal task communication probe, not evidence for words, compositionality, intergenerational transmission or human language origin.',
    )


if __name__ == '__main__':
    import json
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))
