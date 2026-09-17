"""Frozen design for the sender-identity ambiguity follow-up.

The experiment keeps the audited discrete triadic task and changes one wire
factor: whether the receiver can rely on a stable sender slot.  Public and
sender-private token bijections are crossed with stable versus randomly
permuted sender slots.  Silent arms retain their own local message stream but
receive no cross-agent packets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from research_program.triadic_message_study import runner as message_core

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

SEEDS = (70101, 70102, 70103, 70104)
PARTITIONS = ("train", "new_needs", "new_layouts", "new_needs_and_layouts")
TARGET = "new_needs_and_layouts"
STEPS = (0, 100, 500, 1200)
UPDATES = 1200
BATCH_SIZE = 256
ALPHABET_SIZE = 8

CONDITIONS = (
    "stable_public_live",
    "stable_private_live",
    "masked_public_live",
    "masked_private_live",
    "stable_silent",
    "masked_silent",
)

GLOBAL_MAP = (3, 7, 1, 6, 0, 4, 2, 5)
PRIVATE_MAPS = (
    (6, 7, 1, 0, 4, 5, 3, 2),
    (4, 1, 3, 2, 6, 5, 7, 0),
    (4, 3, 2, 6, 1, 5, 7, 0),
)
IDENTITY_MAPS = (tuple(range(ALPHABET_SIZE)),) * 3
IDENTITY_PERMUTATION = (0, 1, 2)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf8")).hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


def array_sha(value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def parse_condition(condition: str) -> dict:
    require(condition in CONDITIONS, f"Unknown condition: {condition}")
    bits = condition.split("_")
    masked = bits[0] == "masked"
    live = bits[-1] == "live"
    if not live:
        return dict(condition=condition, route_mode="stable" if not masked else "masked",
                    communication=False, map_name="identity", maps=IDENTITY_MAPS)
    public = bits[1] == "public"
    return dict(condition=condition, route_mode="masked" if masked else "stable",
                communication=True, map_name="public" if public else "private",
                maps=(GLOBAL_MAP,) * 3 if public else PRIVATE_MAPS)


def make_prepared() -> dict:
    base = message_core.make_prepared()
    require(tuple(base["partitions"]) == PARTITIONS, "Unexpected frozen partition names")
    return dict(
        schema="triadic_identity_mask_codebook_v1",
        base_prepared_sha256=json_hash(base),
        base_partitions_sha256=json_hash(base["partitions"]),
        partitions=base["partitions"],
        actions=base["actions"],
        runs=[dict(seed=seed, condition=condition, directory=f"seed_{seed}_{condition}")
              for seed in SEEDS for condition in CONDITIONS],
        seeds=list(SEEDS), conditions=list(CONDITIONS), partition_names=list(PARTITIONS),
        target=TARGET, checkpoints=list(STEPS), updates=UPDATES, batch_size=BATCH_SIZE,
        alphabet_size=ALPHABET_SIZE,
        maps=dict(identity=list(IDENTITY_MAPS[0]), public=list(GLOBAL_MAP),
                  private=[list(row) for row in PRIVATE_MAPS]),
        route_permutation_sampling=(
            "For masked live arms, each world and window samples a uniform permutation "
            "of sender packets. The same permutation is reused by the two paired "
            "message trajectories; all conditions consume the same permutation stream."
        ),
        silent_definition="Silent arms retain self-only local packets and suppress cross-agent packets.",
        scientific_question=(
            "Does a shared public codebook preserve task-relevant coordination when "
            "sender identity is randomized, relative to sender-private codebooks?"
        ),
        primary_measure=(
            "[(masked_public_live - masked_silent) - "
            "(masked_private_live - masked_silent)] at the held-out joint partition, "
            "with stable-route interaction reported separately."
        ),
        claim_boundary=(
            "This is a training-time convention-alignment and identity-ambiguity test. "
            "It is not lexical meaning, compositional syntax, cultural transmission, "
            "or evidence about human language origins."
        ),
        budget=dict(
            runs=len(SEEDS) * len(CONDITIONS),
            training_updates=len(SEEDS) * len(CONDITIONS) * UPDATES,
            training_world_samples=len(SEEDS) * len(CONDITIONS) * UPDATES * BATCH_SIZE,
            checkpoints=len(SEEDS) * len(CONDITIONS) * len(STEPS),
            target_evaluation_files=(
                len(SEEDS) * len(STEPS) *
                sum(2 if parse_condition(condition)["communication"] else 1 for condition in CONDITIONS)
            ),
        ),
    )


def source_hashes() -> dict[str, str]:
    paths = [
        HERE / name for name in ("__init__.py", "design.py", "wire.py", "runner.py", "plan.md")
    ]
    paths += [Path(message_core.__file__), Path(message_core.base.__file__),
              Path(message_core.coordination.__file__), Path(message_core.base.env.__file__)]
    require(all(path.is_file() for path in paths), "Missing source needed for freeze")
    return {str(path.resolve().relative_to(ROOT)): sha(path) for path in paths}
