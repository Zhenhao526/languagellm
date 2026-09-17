"""Frozen design for a two-generation learner-replacement chain."""
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
GENERATIONS = (2, 3)
LIVES = (True, False)
CHECKPOINTS = (0, 100, 500, 1500, 3000, 6000)
UPDATES = 6000
BATCH_SIZE = 256
REPLACED_AGENT = {2: 0, 3: 1}  # generation 2 replaces A; generation 3 replaces B
ARMS = ("replace_A", "replace_B")
CONDITIONS = tuple(
    f"generation{generation}_{arm}_{schedule}_PL_{'live' if live else 'silent'}"
    for generation, arm in ((2, "replace_A"), (3, "replace_B"))
    for schedule in SCHEDULES for live in LIVES
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    require(condition in CONDITIONS, "Unknown condition")
    pieces = condition.split("_")
    require(len(pieces) == 6 and pieces[4] == "PL", "Malformed condition")
    generation = int(pieces[0].replace("generation", ""))
    arm = pieces[1] + "_" + pieces[2]
    schedule = pieces[3]
    channel = pieces[5]
    require(generation in GENERATIONS and arm in ARMS and schedule in SCHEDULES and channel in ("live", "silent"), "Malformed condition")
    require((generation == 2 and arm == "replace_A") or (generation == 3 and arm == "replace_B"), "Generation/role mismatch")
    return generation, arm, schedule, channel == "live"


def name(seed, condition):
    require(seed in SEEDS and condition in CONDITIONS, "Unknown run")
    return f"seed_{seed}_{condition}"


def replaced_agent(generation):
    require(generation in GENERATIONS, "Unknown generation")
    return REPLACED_AGENT[generation]


def prepared(full_prepared):
    monitor = deepcopy(full_prepared["monitor_spec"])
    train = deepcopy(full_prepared["train_spec"])
    final = deepcopy(full_prepared["final_spec"])
    runs = [
        dict(seed=seed, generation=generation, arm=arm, schedule=schedule, channel="live" if live else "silent",
             condition=f"generation{generation}_{arm}_{schedule}_PL_{'live' if live else 'silent'}",
             directory=name(seed, f"generation{generation}_{arm}_{schedule}_PL_{'live' if live else 'silent'}"))
        for generation, arm in ((2, "replace_A"), (3, "replace_B"))
        for seed in SEEDS for schedule in SCHEDULES for live in LIVES
    ]
    return dict(
        schema="triadic_protocol_chain_v1", seeds=list(SEEDS), schedules=list(SCHEDULES),
        generations=list(GENERATIONS), replacement_roles={"generation2": "A", "generation3": "B"},
        lives=["live", "silent"], conditions=list(CONDITIONS), checkpoints=list(CHECKPOINTS), runs=runs,
        source_experiment="triadic_factorized_neutral_altpartner_study/results/altpair_001",
        generation1_reference="triadic_new_receiver_transmission_study/results/transmission_002",
        full_reference_result=str(FULL_RESULT.resolve().relative_to(ROOT)),
        full_reference_audit=str(FULL_AUDIT.resolve().relative_to(ROOT)),
        source_policy="PL factorized neutral/engage local NumPy tanh MLP",
        chain="generation 1 is the audited live full-C receiver; generation 2 replaces A; generation 3 replaces B using generation-2 live endpoint",
        fresh_agent_policy="same architecture as the replaced source agent, fresh deterministic initialization per seed/generation/schedule",
        adaptation="Only the replaced agent's three modules are updated; other two agents and their modules are frozen",
        channel="live routes all packets; silent routes only each sender's own packets",
        pairing="same parent endpoint, new-agent initialization, worlds, batches, token uniforms and rematching assignments within each generation/schedule",
        evaluation="same two-layout monitor and six-layout heldout final evaluation as the full-C reference",
        primary="live-minus-silent Q by generation and parent-to-child live Q retention",
        secondary=["conditional Q", "physical execution", "target-pair legality", "proposal legality", "engagement", "neutral"],
        claim_boundary="This tests task-protocol transmission across a short learner-replacement chain. It is not evidence for words, compositionality, or human language origin.",
        no_external_model=True, no_llm=True, no_vision_model=True,
        train_spec=train, monitor_spec=monitor, final_spec=final,
        update_count=UPDATES, batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    import json
    print(json.dumps(dict(seeds=SEEDS, conditions=CONDITIONS), ensure_ascii=False))
