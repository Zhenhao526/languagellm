"""Frozen design for the public/private symbol-codebook pilot.

The task and learning core are inherited from the audited reciprocal PL
formation task.  The only training intervention is the stable map used at the
wire boundary before a message is routed to the other agents.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import numpy as np

from research_program.triadic_rule_formation_study import runner as formation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

SEEDS = (68101, 68102, 68103, 68104, 68105, 68106, 68107, 68108)
STEPS = (0, 100, 500, 1500, 3000, 6000)
PARTS = ("train", "new_needs", "new_layouts", "new_needs_and_layouts")
TARGET = "new_needs_and_layouts"
RULE = "reciprocal"
BATCH_SIZE = 256
UPDATES = 6000
ALPHABET_SIZE = 8

# The global map is the fixed permutation used in the earlier recoding control.
# Private maps are distinct, fixed, and published in the frozen plan.  They are
# deliberately simple bijections so the intervention is not a learned cipher.
GLOBAL_MAP = (3, 7, 1, 6, 0, 4, 2, 5)
PRIVATE_MAPS = (
    (6, 7, 1, 0, 4, 5, 3, 2),
    (4, 1, 3, 2, 6, 5, 7, 0),
    (4, 3, 2, 6, 1, 5, 7, 0),
)
MAP_NAMES = ("identity", "public", "private")
CONDITIONS = ("silent", "identity_live", "public_live", "private_live")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


def map_for_condition(condition: str):
    require(condition in CONDITIONS, f"Unknown condition: {condition}")
    if condition == "silent" or condition == "identity_live":
        return "identity", (tuple(range(ALPHABET_SIZE)),) * 3
    if condition == "public_live":
        return "public", (GLOBAL_MAP,) * 3
    return "private", PRIVATE_MAPS


def parse_condition(condition: str):
    name, maps = map_for_condition(condition)
    return name, condition != "silent", maps


def source_static():
    value = formation.prepared()
    return value["partitions"], value["need_response_cases"]


def make_prepared():
    partitions, response_cases = source_static()
    runs = [
        dict(seed=seed, condition=condition, directory=f"seed_{seed}_{condition}")
        for seed in SEEDS for condition in CONDITIONS
    ]
    evaluation_worlds = len(SEEDS) * len(CONDITIONS) * (
        len(STEPS) * partitions[TARGET]["world_count"]
        + sum(partitions[p]["world_count"] for p in PARTS if p != TARGET)
    )
    training_worlds = len(SEEDS) * len(CONDITIONS) * UPDATES * BATCH_SIZE
    return dict(
        schema="triadic_public_codebook_training_v1",
        seeds=list(SEEDS), conditions=list(CONDITIONS), rule=RULE,
        checkpoints=list(STEPS), parts=list(PARTS), target=TARGET,
        batch_size=BATCH_SIZE, updates=UPDATES, alphabet_size=ALPHABET_SIZE,
        maps=dict(identity=list(range(ALPHABET_SIZE)), public=list(GLOBAL_MAP),
                  private=[list(row) for row in PRIVATE_MAPS]),
        map_names=list(MAP_NAMES), runs=runs, partitions=partitions,
        need_response_cases=response_cases,
        evaluation_storage=dict(
            schema="compact_replay_v1",
            fields=[
                "state_indices", "messages", "wire_messages", "action_indices",
                "conditional_exact_expected_reward",
                "conditional_exact_full_success_probability",
                "conditional_exact_execution_probability",
                "conditional_full_posterior_mass",
            ],
            omitted=["states", "action_probabilities", "settlement_detail"],
            rationale="Complete partition arrays are reconstructed from the frozen design; compact files retain all sampled communication, greedy actions, and exact conditional probability vectors needed for replay.",
        ),
        budget=dict(
            runs=len(runs), training_updates=training_worlds // BATCH_SIZE,
            training_world_samples=training_worlds,
            training_forward_module_samples=training_worlds * 2 * 9,
            checkpoints=len(runs) * len(STEPS),
            target_trajectory_files=len(runs) * len(STEPS),
            final_files=len(runs) * len(PARTS),
            evaluation_worlds=evaluation_worlds,
            evaluation_forward_module_samples=evaluation_worlds * 9,
            total_forward_module_samples=training_worlds * 2 * 9 + evaluation_worlds * 9,
        ),
        scientific_question=(
            "Does a stable public symbol relabeling support the same cooperative "
            "content response as an identity wire, and does it outperform "
            "sender-private relabelings when messages are transferred across senders?"
        ),
        claim_boundary=(
            "A public/private codebook contrast is a convention-alignment task "
            "diagnostic; it is not lexical meaning, compositional syntax or human "
            "language-origin evidence."
        ),
    )


def sources():
    paths = [
        HERE / name for name in ("__init__.py", "design.py", "wire.py", "runner.py", "audit.py", "plan.md")
    ]
    paths += [
        Path(formation.__file__), Path(formation.core.__file__),
        Path(formation.kernel.__file__), Path(formation.environment.__file__),
        Path(formation.dataset.__file__), Path(formation.cases.__file__),
        formation.ORIGINAL / "prepared.json", formation.ORIGINAL / "plan.json", formation.ORIGINAL / "freeze.json",
        formation.PHYSICS / "plan.json", formation.PHYSICS / "freeze.json",
    ]
    require(all(path.is_file() for path in paths), "Missing frozen source")
    return {str(path.resolve().relative_to(ROOT)): sha(path) for path in paths}
