"""Frozen design for joint-combination holdout in new-receiver transmission."""
from __future__ import annotations

import hashlib
import json
from itertools import permutations
from pathlib import Path

from research_program.triadic_action_dependency_study import environment
from research_program.triadic_factorized_neutral_altpartner_study import design as source_design

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRANSMISSION_ROOT = ROOT / "research_program" / "triadic_new_receiver_transmission_study" / "results" / "transmission_002"
CHAIN_ROOT = ROOT / "research_program" / "triadic_protocol_chain_study" / "results" / "chain_001"
FULL_RESULT = TRANSMISSION_ROOT / "execution" / "results.json"
FULL_AUDIT = ROOT / "research_program" / "triadic_new_receiver_transmission_study" / "results" / "audit_transmission_002" / "verification.json"
CHAIN_RESULT = CHAIN_ROOT / "execution" / "results.json"
CHAIN_AUDIT = ROOT / "research_program" / "triadic_protocol_chain_study" / "results" / "audit_chain_001" / "verification.json"

SEEDS = tuple(range(66701, 66709))
SCHEDULES = ("static", "rematched")
LIVES = (True, False)
CHECKPOINTS = (0, 100, 500, 1500, 3000, 6000)
UPDATES = 6000
BATCH_SIZE = 256
HELDOUT_ORBIT_COUNT = 6
ARM = "seen_joint_only"
CONDITIONS = tuple(
    f"{schedule}_new_receiver_compositional_{ARM}_PL_{'live' if live else 'silent'}"
    for schedule in SCHEDULES for live in LIVES
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    require(condition in CONDITIONS, "Unknown condition")
    schedule, rest = condition.split("_new_receiver_compositional_", 1)
    arm, suffix = rest.split("_PL_", 1)
    require(schedule in SCHEDULES and arm == ARM and suffix in ("live", "silent"), "Malformed condition")
    return schedule, arm, suffix == "live"


def name(seed, condition):
    require(seed in SEEDS and condition in CONDITIONS, "Unknown run")
    return f"seed_{seed}_{condition}"


def _transform_need(need, kind_flip, length_flip, swap_axes, destination_flip):
    resource, destination = divmod(int(need), 3)
    materials = []
    for material in environment.RESOURCE_ACCEPTANCE[resource]:
        kind, length = (material // 2) ^ kind_flip, (material % 2) ^ length_flip
        if swap_axes:
            kind, length = length, kind
        materials.append(kind * 2 + length)
    resource = environment.RESOURCE_ACCEPTANCE.index(tuple(sorted(materials)))
    if destination_flip and destination != 2:
        destination = 1 - destination
    return resource * 3 + destination


def _orbit(need):
    values = set()
    for kind_flip in range(2):
        for length_flip in range(2):
            for swap_axes in range(2):
                for destination_flip in range(2):
                    transformed = tuple(_transform_need(value, kind_flip, length_flip, swap_axes, destination_flip)
                                       for value in need)
                    for order in permutations(range(3)):
                        values.add(tuple(transformed[i] for i in order))
    return values


def semantic_orbits(needs=None):
    if needs is None:
        needs = source_design.altpair_needs()
    support = set(tuple(map(int, row)) for row in needs)
    remaining = set(support)
    rows = []
    while remaining:
        canonical = min(remaining)
        orbit = _orbit(canonical)
        require(orbit <= support, "Orbit outside support")
        require(orbit <= remaining, "Semantic orbit overlaps an earlier orbit")
        rows.append((canonical, orbit))
        remaining.difference_update(orbit)
    rows.sort(key=lambda pair: (hashlib.sha256(("altpair_orbit_v1|" + json.dumps(list(pair[0]), separators=(",", ":"))).encode()).hexdigest(), pair[0]))
    return rows


def split_needs():
    rows = semantic_orbits()
    require(len(rows) == 23, "Unexpected altpair semantic orbit count")
    held_orbits = rows[:HELDOUT_ORBIT_COUNT]
    heldout = set().union(*(orbit for _, orbit in held_orbits))
    all_needs = set(tuple(map(int, row)) for row in source_design.altpair_needs())
    train = all_needs - heldout
    require(len(heldout) == 312 and len(train) == 1248, "Unexpected joint holdout sizes")
    require(all(len({row[a] for row in heldout}) == 24 for a in range(3)), "Heldout marginal need coverage incomplete")
    require(all(len({row[a] for row in train}) == 24 for a in range(3)), "Training marginal need coverage incomplete")
    return dict(
        train=sorted([list(row) for row in train]), heldout=sorted([list(row) for row in heldout]),
        all=sorted([list(row) for row in all_needs]),
        orbit_count=len(rows), heldout_orbit_count=len(held_orbits),
        train_orbit_count=len(rows) - len(held_orbits),
        heldout_orbit_canonicals=[list(canonical) for canonical, _ in held_orbits],
        serialization="altpair_orbit_v1|compact JSON canonical need; SHA256 rank; first six complete orbits held out",
    )


def _spec(partition, needs, layouts, owners, role):
    needs = [list(map(int, row)) for row in sorted(tuple(map(int, row)) for row in needs)]
    layouts = [list(map(int, row)) for row in layouts]
    owners = [list(map(int, row)) for row in owners]
    return dict(partition=partition, role=role, needs=needs, layouts=layouts, private_sites=owners,
                world_count=len(needs) * len(layouts) * len(owners),
                state_order="need-major, then layout, then owner", weighting="uniform needs × layouts × owners")


def prepared(full_prepared):
    split = split_needs()
    train_source = full_prepared["train_spec"]
    final_source = full_prepared["final_spec"]
    train = _spec("seen_joint_train", split["train"], train_source["layouts"], train_source["private_sites"],
                  "1248_seen_semantic_orbits_on_source_training_layouts")
    monitor = _spec("seen_joint_monitor", split["train"], train_source["layouts"][:2], train_source["private_sites"],
                    "seen_joint_orbits_two_fixed_training_layouts")
    heldout = _spec("heldout_joint", split["heldout"], final_source["layouts"], final_source["private_sites"],
                    "312_unseen_semantic_orbits_on_heldout_layouts")
    seen_new = _spec("seen_joint_new_layout", split["train"], final_source["layouts"], final_source["private_sites"],
                     "seen_joint_orbits_on_heldout_layouts")
    runs = [
        dict(seed=seed, schedule=schedule, arm=ARM, channel="live" if live else "silent",
             condition=f"{schedule}_new_receiver_compositional_{ARM}_PL_{'live' if live else 'silent'}",
             directory=name(seed, f"{schedule}_new_receiver_compositional_{ARM}_PL_{'live' if live else 'silent'}"))
        for seed in SEEDS for schedule in SCHEDULES for live in LIVES
    ]
    return dict(
        schema="triadic_new_receiver_compositional_holdout_v1", seeds=list(SEEDS), schedules=list(SCHEDULES),
        lives=["live", "silent"], conditions=list(CONDITIONS), checkpoints=list(CHECKPOINTS), runs=runs,
        split=split, train_spec=train, monitor_spec=monitor, heldout_spec=heldout, seen_new_spec=seen_new,
        source_experiment="triadic_factorized_neutral_altpartner_study/results/altpair_001",
        generation1_reference="triadic_new_receiver_transmission_study/results/transmission_002",
        all_needs_control="triadic_protocol_chain_study/results/chain_001 generation2_replace_A",
        source_policy="PL factorized neutral/engage local NumPy tanh MLP",
        replacement="A is fresh; B/C are copied from audited generation-1 live full-C endpoint and frozen",
        adaptation="Only fresh A sender1, sender2 and action modules are updated on 1248 seen joint need orbits",
        channel="live routes all agents' packets; silent routes only each sender's own packets",
        pairing="same generation-1 parent, fresh A initialization, world/message/rematch streams as all-needs generation-2 control",
        evaluation="seen-orbit two-layout monitor and unseen-orbit heldout-layout final; seen-orbit new-layout control",
        primary="heldout joint-combination live-minus-silent Q and compositional transfer gap",
        secondary=["seen-new-layout Q", "heldout conditional Q", "physical execution", "target-pair legality", "proposal legality"],
        claim_boundary="Joint-combination behavioral transfer is not evidence for words, compositionality, intergenerational transmission, or human language origin.",
        no_external_model=True, no_llm=True, no_vision_model=True,
        update_count=UPDATES, batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    print(json.dumps(dict(seeds=SEEDS, conditions=CONDITIONS, split=split_needs()), ensure_ascii=False))
