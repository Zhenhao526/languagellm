"""Frozen design for matched new-receiver replacement across A, B and C."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from research_program.triadic_factorized_neutral_altpartner_study import design as source_design

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_ROOT = ROOT / "research_program/triadic_factorized_neutral_altpartner_study/results/altpair_001"
SOURCE_PACKAGE = ROOT / "research_program/triadic_factorized_neutral_altpartner_study"

# Independent source-seed block reserved for confirmation of the role effect.
SEEDS = tuple(range(66709, 66717))
ROLES = ("A", "B", "C")
ROLE_INDEX = {role: index for index, role in enumerate(ROLES)}
SCHEDULES = ("static", "rematched")
LIVES = (True, False)
CONDITIONS = tuple(
    f"replace_{role}_new_receiver_PL_{schedule}_{'live' if live else 'silent'}"
    for role in ROLES for schedule in SCHEDULES for live in LIVES
)
CHECKPOINTS = (0, 100, 500, 1500, 3000, 6000)
UPDATES = 6000
BATCH_SIZE = 256
MONITOR_LAYOUT_COUNT = 2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def role_index(role):
    require(role in ROLE_INDEX, "Unknown role")
    return ROLE_INDEX[role]


def parse_condition(condition):
    require(condition in CONDITIONS, "Unknown condition")
    left, suffix = condition.split("_new_receiver_PL_", 1)
    require(left.startswith("replace_"), "Malformed role")
    role = left[len("replace_"):]
    schedule = suffix.rsplit("_", 1)[0]
    suffix = suffix.rsplit("_", 1)[1]
    require(role in ROLES and schedule in SCHEDULES and suffix in ("live", "silent"), "Malformed condition")
    return role, schedule, suffix == "live"


def name(seed, condition):
    require(seed in SEEDS and condition in CONDITIONS, "Unknown run")
    return f"seed_{seed}_{condition}"


def source_condition(schedule):
    require(schedule in SCHEDULES, "Unknown source schedule")
    return f"{schedule}_altpair_factorized_strict_PL_live"


def monitor_spec(source_prepared):
    train = source_prepared["partitions"]["train"]
    layouts = train["layouts"][:MONITOR_LAYOUT_COUNT]
    spec = deepcopy(train)
    spec.update(partition="monitor_train_two_layouts", layouts=layouts,
                world_count=len(spec["needs"]) * len(layouts) * len(spec["private_sites"]),
                role="all_altpair_needs_two_fixed_training_layouts",
                weighting="uniform needs × two fixed layouts × owners")
    require(spec["world_count"] == 18720, "Unexpected monitor world count")
    return spec


def prepared(source_prepared):
    monitor = monitor_spec(source_prepared); train = source_prepared["partitions"]["train"]; final = source_prepared["partitions"]["new_layouts"]
    runs = [dict(seed=seed, role=role, schedule=schedule, channel="live" if live else "silent", condition=f"replace_{role}_new_receiver_PL_{schedule}_{'live' if live else 'silent'}", directory=name(seed, f"replace_{role}_new_receiver_PL_{schedule}_{'live' if live else 'silent'}")) for seed in SEEDS for role in ROLES for schedule in SCHEDULES for live in LIVES]
    return dict(schema="triadic_receiver_role_crossover_v1", seeds=list(SEEDS), roles=list(ROLES), role_indices=ROLE_INDEX, schedules=list(SCHEDULES), lives=["live", "silent"], conditions=list(CONDITIONS), checkpoints=list(CHECKPOINTS), runs=runs, source_experiment="triadic_factorized_neutral_altpartner_study/results/altpair_001", source_schema=source_prepared.get("schema"), source_prepared_sha256=None, source_seed_count=len(SEEDS), source_agent_count=3, replaced_roles=list(ROLES), frozen_role_rule="For each arm, the two non-replaced roles remain bitwise equal to the same source checkpoint.", fresh_role_rule="The same fresh three-module initialization is inserted into A, B or C within each seed/schedule; live and silent share it.", source_train_partition=dict(world_count=train["world_count"], needs=len(train["needs"]), layouts=len(train["layouts"]), owners=len(train["private_sites"])), monitor_partition=dict(partition=monitor["partition"], world_count=monitor["world_count"], needs=len(monitor["needs"]), layouts=len(monitor["layouts"]), owners=len(monitor["private_sites"])), final_partition=dict(partition=final["partition"], world_count=final["world_count"], needs=len(final["needs"]), layouts=len(final["layouts"]), owners=len(final["private_sites"])), source_policy="PL factorized neutral/engage source checkpoint; local NumPy tanh MLP", new_agent_policy="fresh local NumPy tanh MLP inserted at the replaced role", adaptation="Only the selected role's sender1, sender2 and action modules receive optimizer updates; the other six modules are frozen", channels="live routes all packets; silent routes only each sender's own packets", training="same source task, PL observations, strict mutual settlement, 6000 updates, batch 256", evaluation="fixed two-layout monitor at six checkpoints; all six heldout layouts at final", primary="live-minus-silent new-layout Q-rate and monitor Q AUC by replaced role", secondary=["conditional Q-rate", "physical execution", "target-pair legality", "proposal legality", "engagement", "neutral"], scientific_question="Does the identity of the replaced receiver role change how a fresh learner uses the existing packet channel?", claim_boundary="This is a matched role-adaptation and task-protocol diagnostic, not evidence for words, compositionality or human language origin.", no_external_model=True, no_llm=True, no_vision_model=True, monitor_spec=monitor, final_spec=deepcopy(final), train_spec=deepcopy(train), source_checkpoint_conditions={schedule: source_condition(schedule) for schedule in SCHEDULES}, update_count=UPDATES, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    import json
    print(json.dumps(dict(seeds=SEEDS, conditions=CONDITIONS), ensure_ascii=False))
