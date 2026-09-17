"""Generate a formal zero-shot identity-holdout endpoint probe.

Two sealed v0.34 populations are reused: fixed-A residents and rotating A/B
residents.  A same-private-type newcomer is freshly reset and replaces only
one role in the one team where a held identity occupies that role.  This file
does endpoint model inference and writes raw NumPy tables; the accompanying
analysis and audit scripts do not import the training stack.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
V034 = PROJECT / "redesign_v0.34"
V034_ROTATION = V034 / "results" / "rotation_001"
V034_FIXED = V034 / "results" / "fixed_001"
SOURCE = PROJECT / "redesign_v0.28" / "results" / "formation_001"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
CONDITIONS = ("fixed_A", "rotating_AB")
SCHEDULES = ("A", "B", "C")
ROLES = ("receiver", "food_sender", "water_sender")
PRIVATE_TYPES = (0, 1, 0, 1)
TEAMS = {
    "A": ((0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)),
    "B": ((0, 2, 3), (1, 3, 0), (2, 0, 1), (3, 1, 2)),
    "C": ((0, 3, 1), (1, 0, 2), (2, 1, 3), (3, 2, 0)),
}
ROLE_INDEX = {"receiver": 0, "food_sender": 1, "water_sender": 2}


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def source_hashes():
    files = [ROOT / "newcomer_identity.py", ROOT / "identity_design.json"]
    files += [V034 / name for name in ("run_support.py", "support.py", "views.py", "dual.py")]
    files += [PROJECT / name for name in ("redesign_v0.21/social_model.py", "redesign_v0.28/world.py", "redesign_v0.20/temporal_model.py", "redesign_v0.20/temporal_world.py", "redesign_v0.8/camp.py")]
    return {str(path.resolve()): sha(path) for path in files}


def input_hashes():
    files = [
        V034_ROTATION / "training_complete.json",
        V034_FIXED / "training_complete.json",
        V034_ROTATION / "test_worlds.npz",
        V034_FIXED / "test_worlds.npz",
        PROJECT / "redesign_v0.28/data/feature_cache.pt",
        PROJECT / "redesign_v0.28/data/selection.json",
    ]
    for seed in SEEDS:
        files += [SOURCE / f"prepared_{seed}.pt"]
        for part in PARTITIONS:
            for private_type in (0, 1):
                files += [SOURCE / "private" / f"s{seed}_p{part}_d{private_type}_all" / "final.pt"]
            for condition_out in (V034_ROTATION, V034_FIXED):
                files.append(condition_out / "social" / f"s{seed}_p{part}_dual_complementary" / "final.pt")
    return {str(path.resolve()): sha(path) for path in files}


def load_residents(seed: int, part: int, prepared, condition: str):
    out = V034_ROTATION if condition == "rotating_AB" else V034_FIXED
    folder = out / "social" / f"s{seed}_p{part}_dual_complementary"
    residents, _ = _RUN_SUPPORT.restore_population(seed, part, prepared, SOURCE)
    states = torch.load(folder / "final.pt", weights_only=True)
    if not isinstance(states, list) or len(states) != 4:
        raise ValueError("resident endpoint must contain four state dictionaries")
    for agent, state in zip(residents, states):
        agent.load_state_dict(state)
        agent.requires_grad_(False)
        agent.eval()
    return residents, folder


def load_newcomers(seed: int, part: int, prepared):
    templates = _RUN_SUPPORT.remake_agents(seed, prepared, 7, 2, "identity")
    newcomers = []
    reset_records = []
    for private_type in (0, 1):
        agent = copy.deepcopy(templates[private_type])
        blob = torch.load(SOURCE / "private" / f"s{seed}_p{part}_d{private_type}_all" / "final.pt", weights_only=True)
        agent.load_state_dict(blob["agent"])
        reset_seed = _SUPPORT.seed_value(seed, part, 4 + private_type, 99)
        reset_records.append(_COMMUNICATION.reset_communication(agent, reset_seed))
        agent.requires_grad_(False)
        agent.eval()
        newcomers.append(agent)
    return newcomers, reset_records


def model_tables(seed: int, part: int, agents, worlds, bank):
    food_logs, water_logs, receivers = [], [], []
    for agent in agents:
        with torch.no_grad():
            projected = agent.project(bank.features).detach()
        h_food = _RUN_SUPPORT._encode_chunks(agent, projected, worlds, "food_only")
        h_water = _RUN_SUPPORT._encode_chunks(agent, projected, worlds, "water_only")
        food = _DUAL.first_token_log_probs(agent, torch.from_numpy(h_food)).detach().numpy().astype(np.float32)
        water = _DUAL.first_token_log_probs(agent, torch.from_numpy(h_water)).detach().numpy().astype(np.float32)
        receiver = _COMMUNICATION.receiver_logits(agent).detach().numpy().astype(np.float32)
        food_logs.append(food); water_logs.append(water); receivers.append(receiver)
    return {
        "food_log_probs": np.stack(food_logs),
        "water_log_probs": np.stack(water_logs),
        "food_tokens": np.argmax(np.stack(food_logs), axis=-1).astype(np.int64),
        "water_tokens": np.argmax(np.stack(water_logs), axis=-1).astype(np.int64),
        "receiver_logits": np.stack(receivers),
    }


def episode_raw(worlds, tables, receiver_model, food_model, water_model):
    food_log = tables["food_log_probs"][food_model]
    water_log = tables["water_log_probs"][water_model]
    raw = dict(worlds)
    raw["sender_log_probs_food"] = food_log
    raw["sender_log_probs_water"] = water_log
    raw["sender_log_probs"] = (food_log[:, :, None] + water_log[:, None, :]).reshape(len(food_log), 49).astype(np.float32)
    raw["tokens"] = np.stack((tables["food_tokens"][food_model], tables["water_tokens"][water_model]), axis=1)
    raw["receiver_logits"] = tables["receiver_logits"][receiver_model]
    return raw


def source_modules():
    # Import order mirrors run_support.py.  It is deliberately kept in the
    # endpoint generator only; NumPy analyses below never import these files.
    sys.path.insert(0, str(PROJECT / "redesign_v0.8"))
    sys.path.insert(0, str(PROJECT / "redesign_v0.21"))
    sys.path.insert(0, str(PROJECT / "redesign_v0.28"))
    sys.path.insert(0, str(V034))
    import run_support
    import support
    import views
    import dual
    import social_model as communication
    import world
    return run_support, support, views, dual, communication, world


def run(out: Path):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    inputs = input_hashes()
    design = read(ROOT / "identity_design.json")
    write(out / "invocation.json", {
        "formal": True,
        "version": "v0.35",
        "seeds": list(SEEDS),
        "partitions": list(PARTITIONS),
        "conditions": list(CONDITIONS),
        "schedules": list(SCHEDULES),
        "roles": list(ROLES),
        "source_hashes": hashes,
        "input_hashes": inputs,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "threads": 1,
        "new_dino_inferences": 0,
        "new_private_fits": 0,
        "population_agents": 4,
        "newcomer_private_types": [0, 1],
        "endpoint_rows_expected": 864,
        "study_scope": design["research_question"],
    })
    for path in hashes:
        target = out / "frozen_sources" / Path(path).relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    worlds = npz(V034_ROTATION / "test_worlds.npz")
    for key in ("map_id", "photo_ids", "positions", "shown"):
        if key not in worlds:
            raise ValueError("test world table incomplete")
    np.savez_compressed(out / "test_worlds.npz", **worlds)
    bank = _WORLD.NonSquareImageBank(PROJECT / "redesign_v0.28/data/feature_cache.pt", PROJECT / "redesign_v0.28/data/selection.json")
    episodes = []
    started = time.monotonic()
    for seed, part in itertools.product(SEEDS, PARTITIONS):
        prepared = torch.load(SOURCE / f"prepared_{seed}.pt", weights_only=True)
        newcomers, reset_records = load_newcomers(seed, part, prepared)
        for condition in CONDITIONS:
            residents, resident_folder = load_residents(seed, part, prepared, condition)
            agents = residents + newcomers
            tables = model_tables(seed, part, agents, worlds, bank)
            condition_folder = out / "social" / f"s{seed}_p{part}_{condition}"
            condition_folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(condition_folder / "model_tables.npz", **tables)
            config = {
                "seed": seed,
                "partition": part,
                "condition": condition,
                "resident_source": str(resident_folder.resolve()),
                "resident_ids": [0, 1, 2, 3],
                "newcomer_model_ids": {"type0": 4, "type1": 5},
                "newcomer_social_training": False,
                "newcomer_reset": reset_records,
                "schedules": {key: [list(team) for team in value] for key, value in TEAMS.items()},
                "roles": list(ROLES),
                "worlds": 180,
            }
            write(condition_folder / "config.json", config)
            for schedule in SCHEDULES:
                for role in ROLES:
                    role_index = ROLE_INDEX[role]
                    for held_identity in range(4):
                        slots = [slot for slot, team in enumerate(TEAMS[schedule]) if team[role_index] == held_identity]
                        if len(slots) != 1:
                            raise ValueError("each identity must occupy one slot per role")
                        slot = slots[0]
                        receiver, food_sender, water_sender = TEAMS[schedule][slot]
                        replacement = 4 + PRIVATE_TYPES[held_identity]
                        receiver_model = replacement if role == "receiver" else receiver
                        food_model = replacement if role == "food_sender" else food_sender
                        water_model = replacement if role == "water_sender" else water_sender
                        raw = episode_raw(worlds, tables, receiver_model, food_model, water_model)
                        filename = f"episode_{schedule}_{role}_id{held_identity}.npz"
                        path = condition_folder / filename
                        np.savez_compressed(path, **raw)
                        baseline_out = V034_ROTATION if condition == "rotating_AB" else V034_FIXED
                        baseline_prefix = "protocol" if schedule == "A" else f"transfer_{schedule}"
                        baseline = baseline_out / "social" / f"s{seed}_p{part}_dual_complementary" / f"{baseline_prefix}_2400_team{slot}.npz"
                        episodes.append({
                            "path": str(path.relative_to(out)),
                            "seed": seed,
                            "partition": part,
                            "condition": condition,
                            "schedule": schedule,
                            "role": role,
                            "held_identity": held_identity,
                            "private_type": PRIVATE_TYPES[held_identity],
                            "slot": slot,
                            "receiver_model": receiver_model,
                            "food_model": food_model,
                            "water_model": water_model,
                            "baseline_path": str(baseline.resolve()),
                        })
            print(json.dumps({"phase": "identity_endpoint", "condition": condition, "seed": seed, "partition": part, "episodes": 36, "seconds": round(time.monotonic() - started, 2)}), flush=True)
    if len(episodes) != 864:
        raise AssertionError(f"episode count {len(episodes)} != 864")
    write(out / "episodes.json", {"status": "complete", "rows": episodes, "count": len(episodes)})
    write(out / "training_complete.json", {
        "status": "complete",
        "formal": True,
        "probe": "zero_shot_identity_holdout",
        "social_runs": 0,
        "endpoint_rows": len(episodes),
        "new_dino_inferences": 0,
        "new_private_fits": 0,
        "seconds": time.monotonic() - started,
        "source_hashes": hashes,
        "input_hashes": inputs,
        "files": {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()},
    })
    return {"status": "complete", "endpoint_rows": len(episodes), "seconds": time.monotonic() - started, "source_sha256": sha(ROOT / "newcomer_identity.py")}


_RUN_SUPPORT, _SUPPORT, _VIEWS, _DUAL, _COMMUNICATION, _WORLD = source_modules()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    result = run(args.out.resolve())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
