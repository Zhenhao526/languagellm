"""Frozen design for module-selective adaptation of a new receiver."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from research_program.triadic_new_receiver_transmission_study import design as transmission_design

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRANSMISSION_ROOT = ROOT / "research_program" / "triadic_new_receiver_transmission_study" / "results" / "transmission_002"
FULL_RESULT = TRANSMISSION_ROOT / "execution" / "results.json"
FULL_AUDIT = ROOT / "research_program" / "triadic_new_receiver_transmission_study" / "results" / "audit_transmission_002" / "verification.json"

SEEDS = tuple(range(66701, 66709))
SCHEDULES = ("static", "rematched")
ARMS = ("action_only", "sender_only")
LIVES = (True, False)
CONDITIONS = tuple(
    f"{schedule}_new_receiver_{arm}_PL_{'live' if live else 'silent'}"
    for schedule in SCHEDULES for arm in ARMS for live in LIVES
)
CHECKPOINTS = (0, 100, 500, 1500, 3000, 6000)
UPDATES = 6000
BATCH_SIZE = 256


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    require(condition in CONDITIONS, "Unknown condition")
    schedule, rest = condition.split("_new_receiver_", 1)
    arm, suffix = rest.split("_PL_", 1)
    require(schedule in SCHEDULES and arm in ARMS and suffix in ("live", "silent"), "Malformed condition")
    return schedule, arm, suffix == "live"


def name(seed, condition):
    require(seed in SEEDS and condition in CONDITIONS, "Unknown run")
    return f"seed_{seed}_{condition}"


def allowed_modules(arm):
    require(arm in ARMS, "Unknown arm")
    return {"action_only": ["C_action"], "sender_only": ["C_sender1", "C_sender2"]}[arm]


def prepared(full_prepared):
    """Use exactly the already frozen transmission task partitions."""
    monitor = deepcopy(full_prepared["monitor_spec"])
    train = deepcopy(full_prepared["train_spec"])
    final = deepcopy(full_prepared["final_spec"])
    runs = [
        dict(seed=seed, schedule=schedule, arm=arm, channel="live" if live else "silent",
             condition=f"{schedule}_new_receiver_{arm}_PL_{'live' if live else 'silent'}",
             directory=name(seed, f"{schedule}_new_receiver_{arm}_PL_{'live' if live else 'silent'}"))
        for seed in SEEDS for schedule in SCHEDULES for arm in ARMS for live in LIVES
    ]
    return dict(
        schema="triadic_new_receiver_module_ablation_v1",
        seeds=list(SEEDS), schedules=list(SCHEDULES), arms=list(ARMS), lives=["live", "silent"],
        conditions=list(CONDITIONS), checkpoints=list(CHECKPOINTS), runs=runs,
        source_experiment="triadic_factorized_neutral_altpartner_study/results/altpair_001",
        full_reference_experiment="triadic_new_receiver_transmission_study/results/transmission_002",
        full_reference_result=str(FULL_RESULT.resolve().relative_to(ROOT)),
        full_reference_audit=str(FULL_AUDIT.resolve().relative_to(ROOT)),
        source_seed_count=len(SEEDS), frozen_agents=["A", "B"], replaced_agent="C",
        source_policy="PL factorized neutral/engage source checkpoint; local NumPy tanh MLP",
        fresh_agent_policy="same fresh C initialization as the full-C transmission reference",
        arm_modules={arm: dict(trainable=allowed_modules(arm), frozen=["A/*", "B/*", "C/* except trainable"]) for arm in ARMS},
        adaptation="A/B are frozen; each arm updates only the listed C modules using zero gradients for all other modules",
        channel="live routes all agents' packets; silent routes only each sender's own packets",
        pairing="same source checkpoint, fresh C initialization, world uniforms, batch indices, packet uniforms and rematching assignments across arms and channels",
        evaluation="same two-layout monitor and six-layout heldout final evaluation as the full-C reference",
        primary="heldout live-minus-silent Q-rate and centered monitor Q time-AUC by adaptation arm",
        secondary=["conditional Q", "physical execution", "target-pair legality", "proposal legality", "engagement", "neutral"],
        claim_boundary="Module-level transmission mechanism is tested. This is not evidence for words, compositionality, or human language origin.",
        no_external_model=True, no_llm=True, no_vision_model=True,
        train_spec=train, monitor_spec=monitor, final_spec=final,
        update_count=UPDATES, batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    import json
    print(json.dumps(dict(seeds=SEEDS, conditions=CONDITIONS, arms=ARMS), ensure_ascii=False))
