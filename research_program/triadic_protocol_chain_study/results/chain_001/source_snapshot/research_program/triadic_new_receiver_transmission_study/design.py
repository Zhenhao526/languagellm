"""Pre-registered design for a frozen-pair new-receiver transmission test.

The source population is the completed PL factorized neutral/engage study.
Two source agents and their learned packet protocol are frozen.  The third
agent is replaced by a freshly initialized policy and is adapted on the same
task.  The only experimental manipulation is whether packets are routed
between agents (``live``) or only back to their sender (``silent``).
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from research_program.triadic_factorized_neutral_altpartner_study import design as source_design

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_ROOT = ROOT / "research_program" / "triadic_factorized_neutral_altpartner_study" / "results" / "altpair_001"
SOURCE_PACKAGE = ROOT / "research_program" / "triadic_factorized_neutral_altpartner_study"

# A bounded, independent pilot block selected before execution.  The source
# study has 16 seeds; this block is deliberately the first eight and is not
# selected by outcome.
SEEDS = tuple(range(66701, 66709))
SCHEDULES = ("static", "rematched")
LIVES = (True, False)
CONDITIONS = tuple(
    f"{schedule}_new_receiver_PL_{'live' if live else 'silent'}"
    for schedule in SCHEDULES for live in LIVES
)
CHECKPOINTS = (0, 100, 500, 1500, 3000, 6000)
UPDATES = 6000
BATCH_SIZE = 256
MONITOR_LAYOUT_COUNT = 2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    require(condition in CONDITIONS, "Unknown condition")
    schedule, suffix = condition.split("_new_receiver_PL_", 1)
    require(schedule in SCHEDULES and suffix in ("live", "silent"), "Malformed condition")
    return schedule, suffix == "live"


def name(seed, condition):
    require(seed in SEEDS and condition in CONDITIONS, "Unknown run")
    return f"seed_{seed}_{condition}"


def source_condition(schedule):
    require(schedule in SCHEDULES, "Unknown source schedule")
    return f"{schedule}_altpair_factorized_strict_PL_live"


def monitor_spec(source_prepared):
    """All source needs on two fixed training layouts and all six owners."""
    train = source_prepared["partitions"]["train"]
    layouts = train["layouts"][:MONITOR_LAYOUT_COUNT]
    spec = deepcopy(train)
    spec.update(
        partition="monitor_train_two_layouts",
        layouts=layouts,
        world_count=len(spec["needs"]) * len(layouts) * len(spec["private_sites"]),
        role="all_altpair_needs_two_fixed_training_layouts",
        weighting="uniform needs × two fixed layouts × owners",
    )
    require(spec["world_count"] == 18720, "Unexpected monitor world count")
    return spec


def prepared(source_prepared):
    """Return the frozen experiment input, without initializing or training."""
    monitor = monitor_spec(source_prepared)
    train = source_prepared["partitions"]["train"]
    final = source_prepared["partitions"]["new_layouts"]
    runs = [
        dict(seed=seed, schedule=schedule, channel="live" if live else "silent",
             condition=f"{schedule}_new_receiver_PL_{'live' if live else 'silent'}",
             directory=name(seed, f"{schedule}_new_receiver_PL_{'live' if live else 'silent'}"))
        for seed in SEEDS for schedule in SCHEDULES for live in LIVES
    ]
    return dict(
        schema="triadic_new_receiver_transmission_v1",
        seeds=list(SEEDS), schedules=list(SCHEDULES), lives=["live", "silent"],
        conditions=list(CONDITIONS), checkpoints=list(CHECKPOINTS), runs=runs,
        source_experiment="triadic_factorized_neutral_altpartner_study/results/altpair_001",
        source_schema=source_prepared.get("schema"),
        source_prepared_sha256=None,
        source_seed_count=len(SEEDS), source_agent_count=3,
        frozen_agents=["A", "B"], replaced_agent="C",
        source_train_partition=dict(world_count=train["world_count"], needs=len(train["needs"]),
                                    layouts=len(train["layouts"]), owners=len(train["private_sites"])),
        monitor_partition=dict(partition=monitor["partition"], world_count=monitor["world_count"],
                               needs=len(monitor["needs"]), layouts=len(monitor["layouts"]),
                               owners=len(monitor["private_sites"])),
        final_partition=dict(partition=final["partition"], world_count=final["world_count"],
                             needs=len(final["needs"]), layouts=len(final["layouts"]),
                             owners=len(final["private_sites"])),
        source_policy="PL factorized neutral/engage source checkpoint; A/B/C source networks are local NumPy tanh MLPs",
        new_agent_policy="fresh local NumPy tanh MLP with sender1 54→64→64→32, sender2 153→64→64→32, action 252→64→64→18",
        adaptation="Only C sender1, sender2 and action modules receive optimizer updates; A/B parameters are frozen",
        channels="live routes all agents' 4-token windows; silent routes only each sender's own window",
        training="same source task, PL observations, strict mutual settlement, 6000 updates, batch 256",
        evaluation="fixed two-layout monitor at six checkpoints; all six heldout layouts at final",
        primary="live-minus-silent new-layout Q-rate after adaptation and centered time-AUC of monitor Q-rate",
        secondary=["conditional Q-rate", "physical execution", "target-pair legality", "proposal legality", "engagement", "neutral"],
        claim_boundary="Transmission of a task protocol is tested. This is not evidence for words, compositionality, or human language origin.",
        no_external_model=True, no_llm=True, no_vision_model=True,
        monitor_spec=monitor, final_spec=deepcopy(final),
        train_spec=deepcopy(train),
        source_checkpoint_conditions={schedule: source_condition(schedule) for schedule in SCHEDULES},
        update_count=UPDATES, batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    import json
    # This entry point is informational; execution requires the frozen source.
    print(json.dumps(dict(seeds=SEEDS, conditions=CONDITIONS), ensure_ascii=False))
