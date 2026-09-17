"""Frozen slot and token-recoding variants for the role-crossover endpoints.

The primary role contrast is C versus A.  B remains in the inherited full
packet probe, while this focused follow-up spends the intervention budget on
the two roles needed to test the pre-existing C-minus-A contrast.
"""
from __future__ import annotations

from collections import defaultdict

from research_program.triadic_factorized_neutral_altpartner_direction_probe import design as source_design
from research_program.triadic_receiver_role_crossover_confirmatory_study import design as role_design

SEEDS = tuple(role_design.SEEDS)
ROLES = ("A", "C")
SCHEDULES = tuple(role_design.SCHEDULES)
LIVES = (True, False)
CONDITIONS = tuple(
    condition for condition in role_design.CONDITIONS
    if condition.startswith("replace_A_") or condition.startswith("replace_C_")
)
PARTS = ("new_layouts",)
CHECKPOINT = 6000
CHUNK_SIZE = 2048
SHAM_ROWS_PER_SENDER = 256
AXES = ("kind", "length", "destination")
ALPHABET_SIZE = 8

# The four slot masks test local access. The two fixed bijections test
# dependence on absolute token identity while preserving packet length.
VARIANTS = (
    {"name": "full", "slots": [0, 1, 2, 3], "recode": "identity"},
    {"name": "slot_0", "slots": [0], "recode": "identity"},
    {"name": "slot_1", "slots": [1], "recode": "identity"},
    {"name": "slot_2", "slots": [2], "recode": "identity"},
    {"name": "slot_3", "slots": [3], "recode": "identity"},
    {"name": "recode_add1", "slots": [0, 1, 2, 3], "recode": "add1"},
    {"name": "recode_xor4", "slots": [0, 1, 2, 3], "recode": "xor4"},
)
VARIANT_NAMES = tuple(item["name"] for item in VARIANTS)
VARIANT_BY_NAME = {item["name"]: item for item in VARIANTS}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_spec():
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
        schema="receiver_role_crossover_slot_recode_cases_v1",
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
        schema="receiver_role_crossover_slot_recode_prepared_v1",
        seeds=list(SEEDS), roles=list(ROLES), schedules=list(SCHEDULES), lives=["live", "silent"],
        conditions=list(CONDITIONS), parts=list(PARTS), checkpoint=CHECKPOINT,
        chunk_size=CHUNK_SIZE, sham_rows_per_sender=SHAM_ROWS_PER_SENDER,
        axes=list(AXES), variants=list(VARIANTS), partition=spec, cases=cases,
        policy_blocks=len(SEEDS) * len(CONDITIONS),
        source_experiment="triadic_receiver_role_crossover_confirmatory_study/results/role_crossover_confirm_001",
        source_result="role-crossover final checkpoints; A/C policy rows one per seed×role×schedule×channel",
        scientific_question="Does the C-minus-A source-content response survive slot-selective replacement and fixed token bijections?",
        primary="C-minus-A contrast on aligned-minus-placebo signed plan transfer by packet variant",
        secondary="physical execution, Q, conditional Q, partner transfer, action change and closed-channel silent alias",
        intervention="Replace selected first-window token positions in the receiver's selected sender packet, then optionally apply a fixed token bijection; recompute W2 and actions causally.",
        placebo="Same cyclic sender×axis×background donor mapping as the role semantic-transfer probe; variants share case pairing.",
        controls="silent is a closed-channel alias; full is the whole-packet reference; slot_i isolates one position; recode_add1 and recode_xor4 are fixed alphabet bijections.",
        no_training_updates=True,
        claim_boundary="Variant-selective aligned-minus-placebo transfer is a frozen-policy task diagnostic, not lexical meaning, compositional syntax or language-origin evidence.",
    )


if __name__ == "__main__":
    import json
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))
