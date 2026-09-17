"""Frozen design for an independent full-information capacity pilot."""
from __future__ import annotations

from research_program.triadic_factorized_neutral_altpartner_study import design as source_design

SEEDS = tuple(range(66721, 66729))
SCHEDULES = ('static', 'rematched')
LIVES = (False, True)
CONDITIONS = tuple(
    f'{schedule}_altpair_factorized_strict_PL_{"live" if live else "silent"}'
    for schedule in SCHEDULES for live in LIVES
)
PARTS = ('train', 'new_layouts')
UPDATES = (0, 100, 500, 1500, 3000, 6000)
INFORMATION = 'FI'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    require(condition in CONDITIONS, 'Unknown condition')
    schedule, suffix = condition.split('_', 1)
    require(schedule in SCHEDULES and suffix in
            ('altpair_factorized_strict_PL_live', 'altpair_factorized_strict_PL_silent'),
            'Malformed condition')
    return schedule, suffix.endswith('_live')


def make_prepared():
    source = source_design.make_prepared()
    partitions = {part: source['partitions'][part] for part in PARTS}
    runs = [dict(seed=seed, schedule=schedule, live=live,
                 condition=f'{schedule}_altpair_factorized_strict_PL_{"live" if live else "silent"}',
                 information=INFORMATION)
            for seed in SEEDS for schedule in SCHEDULES for live in LIVES]
    return dict(
        schema='triadic_factorized_neutral_altpartner_fi_capacity_pilot_v1',
        information=INFORMATION, seeds=list(SEEDS), schedules=list(SCHEDULES),
        lives=['live', 'silent'], conditions=list(CONDITIONS), parts=list(PARTS),
        checkpoints=list(UPDATES), partitions=partitions, runs=runs,
        independent_seed_count=len(SEEDS), source_schema=source['schema'],
        need_count=source['need_count'], legal_plan_multiplicity=2,
        legal_pair_multiplicity=2, train_layout_count=source['train_layout_count'],
        new_layout_count=source['new_layout_count'], owner_count=source['owner_count'],
        scientific_question='Does giving every actor the other actors’ needs recover partner/content choice that is absent under PL?',
        primary='FI capacity diagnostic: rematching-by-communication interaction on target-pair legal rate conditional on physical execution',
        secondary='team Q, physical execution, engagement, proposal legality and target-pair choice',
        observation='FI own need plus all three need descriptions, public layout and private-site owners; FI flag set',
        action_factorization='intent neutral/engage plus 16-way site-destination-partner proposal',
        pairing='same initial parameters, world uniforms, batch indices and message uniforms within each seed; rematched live/silent share permutation assignments',
        claim_boundary='This is an information-capacity control, not evidence for words, compositionality, intergenerational transmission or human language origin.',
        pilot=True, no_early_stopping=True,
    )


if __name__ == '__main__':
    import json
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))
