"""Frozen aligned/placebo cases for the receiver-role crossover endpoints."""
from __future__ import annotations

from collections import defaultdict

from research_program.triadic_factorized_neutral_altpartner_direction_probe import design as source_design
from research_program.triadic_receiver_role_crossover_study import design as role_design

SEEDS = tuple(role_design.SEEDS)
ROLES = tuple(role_design.ROLES)
SCHEDULES = tuple(role_design.SCHEDULES)
LIVES = (True, False)
CONDITIONS = tuple(role_design.CONDITIONS)
PARTS = ("new_layouts",)
CHECKPOINT = 6000
CHUNK_SIZE = 2048
SHAM_ROWS_PER_SENDER = 256
AXES = ("kind", "length", "destination")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_spec():
    """Use the same held-out layouts and factorized PL cases as altpair_001."""
    return source_design.source_partition()


def make_cases(spec):
    base = source_design.make_cases(spec)
    receiver = base["receiver_state_indices"]
    sender = base["sender"]
    axis = base["axis"]
    background = base["background_indices"]
    groups = defaultdict(list)
    for index, (who, name, bg) in enumerate(zip(sender, axis, background)):
        groups[(int(who), str(name), int(bg))].append(index)
    require(len(groups) == 3 * len(AXES) * 36, "Unexpected placebo strata")
    require(all(len(indices) >= 2 for indices in groups.values()), "Placebo stratum has one case")
    mapping = list(range(len(receiver)))
    for indices in groups.values():
        for pos, index in enumerate(indices):
            mapping[index] = indices[(pos + 1) % len(indices)]
    require(all(mapping[index] != index for index in range(len(mapping))), "Placebo identity")
    require(len(mapping) == 76032, "Unexpected case count")
    return dict(
        schema="receiver_role_crossover_semantic_transfer_cases_v1",
        partition=spec["partition"], needs=spec["needs"], layouts=spec["layouts"],
        private_sites=spec["private_sites"], world_count=int(spec["world_count"]),
        background_count=36, receiver_state_indices=receiver,
        donor_state_indices=base["donor_state_indices"], sender=sender, axis=axis,
        background_indices=background, placebo_donor_case_indices=mapping,
        case_count=len(receiver), case_order=base["case_order"],
        aligned_rule="same-background neighboring need packet from the changed sender",
        placebo_rule="cyclic donor packet within sender×axis×background; donor need sets remain the aligned reference",
    )


def make_prepared():
    spec = source_spec()
    cases = make_cases(spec)
    return dict(
        schema="receiver_role_crossover_semantic_transfer_prepared_v1",
        seeds=list(SEEDS), roles=list(ROLES), schedules=list(SCHEDULES), lives=["live", "silent"],
        conditions=list(CONDITIONS), parts=list(PARTS), checkpoint=CHECKPOINT,
        chunk_size=CHUNK_SIZE, sham_rows_per_sender=SHAM_ROWS_PER_SENDER,
        partition=spec, cases=cases,
        policy_blocks=len(SEEDS) * len(CONDITIONS),
        source_experiment="triadic_receiver_role_crossover_study/results/role_crossover_003",
        source_result="role-crossover final checkpoints; one policy row per seed×role×schedule×channel",
        scientific_question="Does a source-demand aligned packet move each replaced role toward donor-only plans more than a same-stratum placebo packet?",
        primary="receiver-role contrast on aligned-minus-placebo signed plan transfer",
        secondary="partner transfer, physical execution, Q, conditional Q, action-change rate and exact sham replay",
        posthoc_relative_to="triadic_receiver_role_crossover_study/results/role_crossover_003",
        claim_boundary="Aligned-minus-placebo is a frozen task content-transfer diagnostic, not evidence for words, compositionality, intergenerational transmission or human language origin.",
    )


if __name__ == "__main__":
    import json
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))
