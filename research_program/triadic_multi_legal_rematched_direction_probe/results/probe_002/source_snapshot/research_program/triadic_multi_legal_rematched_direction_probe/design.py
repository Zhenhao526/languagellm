"""Frozen case construction for the rematching directional message probe."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from research_program.triadic_action_dependency_study import dataset, environment
from research_program.triadic_multi_legal_coordination_study import design as multi

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(66501, 66517))
SCHEDULES = ('static', 'rematched')
CONDITIONS = tuple(
    f'{schedule}_partial_multi_strict_PL_{"live" if live else "silent"}'
    for schedule in SCHEDULES for live in (True, False)
)
PARTS = ('new_layouts',)
CHECKPOINT = 6000
STATE_ORDER = 'need-major, then layout, then owner'
CHUNK_SIZE = 1024
SHAM_ROWS_PER_SENDER = 256


def require(ok, message):
    if not ok:
        raise ValueError(message)


def probe_needs():
    """Use all balanced multi-plan needs with a deterministic same-context neighbor."""
    base = [tuple(map(int, row)) for row in multi.multi_needs()]
    valid = []
    for row in base:
        if all(any(other[actor] != row[actor] and
                   all(other[i] == row[i] for i in range(3) if i != actor)
                   for other in base) for actor in range(3)):
            valid.append(row)
    valid = sorted(valid)
    require(len(valid) == 1404, 'Unexpected directional probe need count')
    require([sum(multi.pair_index(row) == p for row in valid) for p in range(3)] == [468, 468, 468],
            'Probe need pair strata are unbalanced')
    require(all(len({row[a] for row in valid}) == 24 for a in range(3)),
            'Probe need marginal coverage is incomplete')
    by_context = defaultdict(list)
    for row in valid:
        for actor in range(3):
            by_context[(actor, tuple(row[i] for i in range(3) if i != actor))].append(row[actor])
    neighbors = {}
    for row in valid:
        for actor in range(3):
            values = sorted(by_context[(actor, tuple(row[i] for i in range(3) if i != actor))])
            require(len(values) > 1, 'Every probe need must have a same-context neighbor')
            pos = values.index(row[actor]); replacement = values[(pos + 1) % len(values)]
            donor = list(row); donor[actor] = replacement; donor = tuple(donor)
            require(donor in set(valid) and donor[actor] != row[actor], 'Invalid cyclic neighbor')
            neighbors[(row, actor)] = donor
    return valid, neighbors


def source_partitions():
    prepared = dataset.make_prepared()
    source = prepared['partitions']['new_layouts']
    needs, _ = probe_needs()
    return dict(partition='new_layouts', needs=[list(row) for row in needs], layouts=source['layouts'],
                private_sites=source['private_sites'], world_count=len(needs) * len(source['layouts']) * len(source['private_sites']))


def make_cases(spec):
    needs, neighbors = probe_needs()
    lookup = {row: i for i, row in enumerate(needs)}
    layouts = [tuple(map(int, row)) for row in spec['layouts']]
    owners = [tuple(map(int, row)) for row in spec['private_sites']]
    b = len(layouts) * len(owners)
    receiver = []; donor = []; sender = []; need_index = []; donor_need_index = []; background = []
    for ni, row in enumerate(needs):
        for li in range(len(layouts)):
            for oi in range(len(owners)):
                bg = li * len(owners) + oi
                for actor in range(3):
                    donor_row = neighbors[(row, actor)]
                    receiver.append((ni * len(layouts) + li) * len(owners) + oi)
                    donor.append((lookup[donor_row] * len(layouts) + li) * len(owners) + oi)
                    sender.append(actor); need_index.append(ni); donor_need_index.append(lookup[donor_row]); background.append(bg)
    result = dict(schema='multi_legal_directional_W1_probe_cases_v1', partition=spec['partition'],
        needs=[list(row) for row in needs], donor_needs=[list(neighbors[(row, actor)]) for row in needs for actor in range(3)],
        layouts=[list(row) for row in layouts], private_sites=[list(row) for row in owners],
        world_count=int(spec['world_count']), background_count=b, state_order=STATE_ORDER,
        receiver_state_indices=receiver, donor_state_indices=donor, sender=sender,
        need_indices=need_index, donor_need_indices=donor_need_index, background_indices=background,
        case_count=len(receiver), case_order='need,layout,owner,sender',
        donor_rule='same layout and private-site owner; cyclic next need value within the same other-agent context',
        sham_rule=f'first {SHAM_ROWS_PER_SENDER} cases per sender in canonical case order',
        selection='all valid multi-plan needs; no policy, message or outcome filtering')
    require(result['case_count'] == 1404 * b * 3, 'Unexpected directional probe case count')
    return result


def make_prepared():
    spec = source_partitions()
    cases = make_cases(spec)
    return dict(schema='triadic_multi_legal_rematched_direction_probe_v1', seeds=list(SEEDS), conditions=list(CONDITIONS),
        parts=list(PARTS), checkpoint=CHECKPOINT, chunk_size=CHUNK_SIZE,
        sham_rows_per_sender=SHAM_ROWS_PER_SENDER, partition=spec, cases=cases,
        scientific_question='Does replacing one sender W1 with a same-context neighboring need move recipients toward the donor legal plan, and does random need rematching change that effect?',
        primary='mean signed donor-plan-only minus receiver-plan-only probability transfer',
        secondary='signed partner transfer, natural/intervened physical execution and Q, sham exact replay',
        posthoc_relative_to='triadic_multi_legal_rematched_confirmation_study/results/confirm_001',
        claim_boundary='Directional response is an exploratory causal communication probe, not evidence for words, compositionality, or language origin.')


if __name__ == '__main__':
    import json
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))
