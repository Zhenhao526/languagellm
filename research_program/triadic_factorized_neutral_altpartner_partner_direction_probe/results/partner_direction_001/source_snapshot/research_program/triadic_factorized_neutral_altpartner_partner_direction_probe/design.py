"""Frozen composite-need cases whose legal partner set changes."""
from __future__ import annotations

from collections import defaultdict
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
    require(schedule in SCHEDULES and (suffix.endswith('_live') or suffix.endswith('_silent')),
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


def _semantic_axes(old, new):
    old_resource, old_destination = divmod(int(old), 3)
    new_resource, new_destination = divmod(int(new), 3)
    old_materials = set(environment.RESOURCE_ACCEPTANCE[old_resource])
    new_materials = set(environment.RESOURCE_ACCEPTANCE[new_resource])
    axes = []
    if {m // 2 for m in old_materials} != {m // 2 for m in new_materials}:
        axes.append('kind')
    if {m % 2 for m in old_materials} != {m % 2 for m in new_materials}:
        axes.append('length')
    if set(environment.DESTINATION_ACCEPTANCE[old_destination]) != set(
            environment.DESTINATION_ACCEPTANCE[new_destination]):
        axes.append('destination')
    return tuple(axes)


def _pair_set(needs):
    return frozenset(tuple(plan[:2]) for plan in environment.full_success_plans(tuple(needs)))


def neighbors(needs):
    """Return all two-axis replacements that change the legal pair set.

    The return value is grouped by (receiver need, sender), with potentially
    several donor endpoints per composite axis.  No outcome or policy is used
    to select cases.
    """
    support = set(needs)
    result = defaultdict(list)
    counts = defaultdict(int)
    for row in needs:
        before_pairs = _pair_set(row)
        for actor in range(3):
            old = row[actor]
            for new in range(24):
                if new == old:
                    continue
                axes = _semantic_axes(old, new)
                if len(axes) != 2:
                    continue
                donor = list(row)
                donor[actor] = new
                donor = tuple(donor)
                if donor not in support or _pair_set(donor) == before_pairs:
                    continue
                label = '_'.join(axes)
                require(label in AXES, 'Unexpected composite axis')
                result[(row, actor)].append((label, donor))
                counts[label] += 1
    for key in result:
        result[key].sort(key=lambda item: (AXES.index(item[0]), item[1]))
    require(sum(counts.values()) == 3456 and dict(counts) == {
        'kind_length': 1536, 'kind_destination': 960, 'length_destination': 960},
            f'Unexpected composite-neighbor support: {dict(counts)}')
    return dict(result)


def source_partition():
    source = dataset.make_prepared()['partitions']['new_layouts']
    needs = all_needs()
    return dict(partition='new_layouts', needs=[list(r) for r in needs], layouts=source['layouts'],
                private_sites=source['private_sites'],
                world_count=len(needs) * len(source['layouts']) * len(source['private_sites']))


def make_cases(spec):
    needs = [tuple(r) for r in spec['needs']]
    table = neighbors(needs)
    lookup = {r: i for i, r in enumerate(needs)}
    nl, no = len(spec['layouts']), len(spec['private_sites'])
    receiver, donor, sender, axis, background = [], [], [], [], []
    for ni, row in enumerate(needs):
        for actor in range(3):
            for name, donor_row in table.get((row, actor), []):
                for li in range(nl):
                    for oi in range(no):
                        receiver.append((ni * nl + li) * no + oi)
                        donor.append((lookup[donor_row] * nl + li) * no + oi)
                        sender.append(actor)
                        axis.append(name)
                        background.append(li * no + oi)
    result = dict(schema='factorized_altpartner_partner_direction_probe_cases_v1',
                  partition=spec['partition'], needs=[list(r) for r in needs], layouts=spec['layouts'],
                  private_sites=spec['private_sites'], world_count=int(spec['world_count']),
                  background_count=nl * no, receiver_state_indices=receiver,
                  donor_state_indices=donor, sender=sender, axis=axis,
                  background_indices=background, case_count=len(receiver),
                  case_order='need,sender,composite_axis,donor,layout,owner',
                  donor_rule='same layout and private-site owner; one same-person two-axis endpoint that changes the legal partner set',
                  selection='all available two-axis endpoint changes; no policy or outcome filtering')
    require(result['case_count'] == 3456 * nl * no, 'Unexpected composite directional case count')
    require(set(result['axis']) == set(AXES), 'Missing composite axis stratum')
    return result


def make_prepared():
    spec = source_partition()
    cases = make_cases(spec)
    return dict(schema='factorized_neutral_altpartner_partner_direction_probe_v1', seeds=list(SEEDS),
                schedules=list(SCHEDULES), conditions=list(CONDITIONS), parts=list(PARTS),
                checkpoint=CHECKPOINT, chunk_size=CHUNK_SIZE, sham_rows_per_sender=SHAM_ROWS_PER_SENDER,
                axes=list(AXES), partition=spec, cases=cases,
                scientific_question='Does replacing one sender W1 with a same-context composite need that changes the legal partner set move receivers toward the donor partner relation?',
                primary='rematching-by-communication interaction on intervention-induced donor-only minus receiver-only partner-edge transfer',
                secondary='full-plan transfer, physical execution, team Q, conditional Q, exact sham replay',
                posthoc_relative_to='triadic_factorized_neutral_altpartner_study/results/altpair_001',
                claim_boundary='A partner-directed response is a causal task communication probe, not evidence for words, compositionality, intergenerational transmission or human language origin.')


if __name__ == '__main__':
    import json
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))
