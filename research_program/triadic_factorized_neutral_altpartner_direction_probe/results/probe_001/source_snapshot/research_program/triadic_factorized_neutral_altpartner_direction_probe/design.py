"""Frozen cases for the factorized alternative-partner W1 probe."""
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
PAIR_TUPLES = ((0, 1), (0, 2), (1, 2))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    require(condition in CONDITIONS, 'Unknown condition')
    schedule, suffix = condition.split('_', 1)
    require(schedule in SCHEDULES and (suffix.endswith('_live') or suffix.endswith('_silent')), 'Malformed condition')
    return schedule, suffix.endswith('_live')


def all_needs():
    rows = []
    for need in product(range(24), repeat=3):
        plans = environment.full_success_plans(tuple(need))
        if len(plans) == 2 and len({tuple(p[:2]) for p in plans}) == 2:
            rows.append(tuple(map(int, need)))
    require(len(rows) == 1560, 'Unexpected alternative-partner need count')
    return sorted(rows)


def neighbors(needs):
    support = set(needs); table = {}
    counts = defaultdict(int)
    for row in needs:
        for actor in range(3):
            for axis, value in dataset.flips(row[actor]):
                donor = list(row); donor[actor] = value; donor = tuple(donor)
                if donor in support:
                    table[(row, actor, axis)] = donor; counts[axis] += 1
    require(sum(counts.values()) == 2112 and dict(counts) == {'kind': 576, 'length': 576, 'destination': 960},
            'Unexpected same-context neighbor support')
    return table


def source_partition():
    source = dataset.make_prepared()['partitions']['new_layouts']
    needs = all_needs()
    return dict(partition='new_layouts', needs=[list(r) for r in needs], layouts=source['layouts'],
                private_sites=source['private_sites'],
                world_count=len(needs) * len(source['layouts']) * len(source['private_sites']))


def make_cases(spec):
    needs = [tuple(r) for r in spec['needs']]; table = neighbors(needs); lookup = {r: i for i, r in enumerate(needs)}
    nl, no = len(spec['layouts']), len(spec['private_sites'])
    receiver = []; donor = []; sender = []; axis = []; background = []
    for ni, row in enumerate(needs):
        for actor in range(3):
            for name in ('kind', 'length', 'destination'):
                if (row, actor, name) not in table:
                    continue
                donor_row = table[row, actor, name]
                for li in range(nl):
                    for oi in range(no):
                        receiver.append((ni * nl + li) * no + oi)
                        donor.append((lookup[donor_row] * nl + li) * no + oi)
                        sender.append(actor); axis.append(name); background.append(li * no + oi)
    result = dict(schema='factorized_altpartner_direction_probe_cases_v1', partition=spec['partition'],
                  needs=[list(r) for r in needs], layouts=spec['layouts'], private_sites=spec['private_sites'],
                  world_count=int(spec['world_count']), background_count=nl * no,
                  receiver_state_indices=receiver, donor_state_indices=donor, sender=sender,
                  axis=axis, background_indices=background, case_count=len(receiver),
                  case_order='need,sender,axis,layout,owner',
                  donor_rule='same layout and private-site owner; one same-context kind/length/destination neighbor',
                  selection='all available same-context endpoint changes; no policy or outcome filtering')
    require(result['case_count'] == 2112 * nl * no, 'Unexpected directional case count')
    return result


def make_prepared():
    spec = source_partition(); cases = make_cases(spec)
    return dict(schema='factorized_neutral_altpartner_direction_probe_v1', seeds=list(SEEDS),
                schedules=list(SCHEDULES), conditions=list(CONDITIONS), parts=list(PARTS),
                checkpoint=CHECKPOINT, chunk_size=CHUNK_SIZE, sham_rows_per_sender=SHAM_ROWS_PER_SENDER,
                partition=spec, cases=cases,
                scientific_question='Does replacing one sender W1 with a same-context neighboring need move receivers toward donor-only partner/plan alternatives?',
                primary='rematching-by-communication interaction on intervention-induced donor-only minus receiver-only event-mass transfer',
                secondary='partner-only transfer, full-plan transfer, physical execution, target-pair selection, exact sham replay',
                posthoc_relative_to='triadic_factorized_neutral_altpartner_study/results/altpair_001',
                claim_boundary='A directional response is a causal task communication probe, not evidence for words, compositionality, intergenerational transmission or human language origin.')


if __name__ == '__main__':
    import json
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))
