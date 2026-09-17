"""Frozen case construction for the rematched-policy directional probe."""
from __future__ import annotations

from research_program.triadic_action_dependency_study import dataset
from research_program.triadic_multi_legal_direction_probe import design as base

SEEDS = tuple(range(66501, 66517))
SCHEDULES = ('static', 'rematched')
CONDITIONS = tuple(
    f'{schedule}_partial_multi_strict_PL_{"live" if live else "silent"}'
    for schedule in SCHEDULES for live in (True, False)
)
PARTS = ('new_layouts',)
CHECKPOINT = 6000
STATE_ORDER = 'need-major, then layout, then owner'
CHUNK_SIZE = base.CHUNK_SIZE
SHAM_ROWS_PER_SENDER = base.SHAM_ROWS_PER_SENDER


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    require(condition in CONDITIONS, 'Unknown condition')
    schedule, suffix = condition.split('_', 1)
    require(schedule in SCHEDULES and suffix in ('partial_multi_strict_PL_live', 'partial_multi_strict_PL_silent'),
            'Malformed condition')
    return schedule, suffix.endswith('_live')


def probe_needs():
    return base.probe_needs()


def source_partitions():
    source = dataset.make_prepared()['partitions']['new_layouts']
    needs, _ = probe_needs()
    spec = dict(partition='new_layouts', needs=[list(row) for row in needs], layouts=source['layouts'],
                private_sites=source['private_sites'], world_count=len(needs) * len(source['layouts']) * len(source['private_sites']))
    require(spec['world_count'] == 50544, 'Unexpected probe world count')
    return spec


def make_cases(spec):
    cases = base.make_cases(spec)
    require(cases['case_count'] == 151632, 'Unexpected directional case count')
    return cases


def make_prepared():
    spec = source_partitions(); cases = make_cases(spec)
    return dict(
        schema='triadic_multi_legal_rematched_direction_probe_v1', seeds=list(SEEDS), schedules=list(SCHEDULES),
        conditions=list(CONDITIONS), parts=list(PARTS), checkpoint=CHECKPOINT, chunk_size=CHUNK_SIZE,
        sham_rows_per_sender=SHAM_ROWS_PER_SENDER, partition=spec, cases=cases,
        scientific_question='Does a trained rematched policy respond directionally when one sender W1 is replaced by a same-context neighboring need?',
        primary='rematching-by-communication interaction on signed donor-only minus receiver-only legal-plan probability transfer',
        secondary='signed partner transfer, natural/intervened physical execution and Q, conditional Q and sham exact replay',
        posthoc_relative_to='triadic_multi_legal_rematched_confirmation_study/results/confirm_001',
        case_selection='All 1,404 needs with a same-context neighbor for every actor; no policy, message or outcome filtering.',
        claim_boundary='This is a posthoc directional mechanism probe, not evidence for words, compositionality, intergenerational transmission or language origin.',
    )


if __name__ == '__main__':
    import json
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))
