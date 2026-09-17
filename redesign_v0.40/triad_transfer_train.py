"""Factorial cultural transmission of the v0.39 three-resource protocol."""
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
V039 = PROJECT / "redesign_v0.39"
V039_OUT = V039 / "results" / "triad_001"
SOURCE = PROJECT / "redesign_v0.28" / "results" / "formation_001"
FEATURE_CACHE = PROJECT / "redesign_v0.28" / "data" / "feature_cache.pt"
SELECTION = PROJECT / "redesign_v0.28" / "data" / "selection.json"

SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
FORMATION_CONDITIONS = ("origin_fixed_A", "origin_rotating_AB", "origin_random_ABC")
TRANSMISSION_CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
SCHEDULES = ("A", "B", "C")
PRIVATE_TYPES = (0, 1, 0, 1)
REPLACEMENT_ORDER = (0, 1, 2, 3)
GENERATIONS = 4
UPDATES_PER_GENERATION = 300
BASELINE = 0.1
ENTROPY_WEIGHT = 0.02

# The role topology is inherited exactly from v0.39. Each tuple is
# (receiver, resource-0 sender, resource-1 sender, resource-2 sender).
TEAMS = {
    "A": ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)),
    "B": ((0, 2, 1, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0)),
    "C": ((0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0)),
}


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def source_hashes():
    files = [ROOT / "triad_transfer_train.py", ROOT / "triad_transfer_design.json"]
    files += [V039 / name for name in ("triad_train.py", "triad_design.json")]
    files += [PROJECT / name for name in (
        "redesign_v0.8/camp.py",
        "redesign_v0.20/temporal_model.py",
        "redesign_v0.20/temporal_world.py",
        "redesign_v0.28/world.py",
        "redesign_v0.21/social_model.py",
    )]
    return {str(path.resolve()): sha(path) for path in files}


def input_hashes():
    files = [
        V039_OUT / "training_complete.json",
        V039_OUT / "test_worlds.npz",
        *[V039_OUT / f"train_worlds_p{part}.npz" for part in PARTITIONS],
        FEATURE_CACHE,
        SELECTION,
    ]
    # The population constructor reads the prepared model and the frozen
    # private perceptual endpoints; bind those inputs as well as every v0.39
    # formation endpoint used by this experiment.
    for seed in SEEDS:
        files.append(SOURCE / f"prepared_{seed}.pt")
        for part in PARTITIONS:
            for private_type in (0, 1):
                files.append(SOURCE / "private" / f"s{seed}_p{part}_d{private_type}_all" / "final.pt")
            for formation in FORMATION_CONDITIONS:
                name = formation.removeprefix("origin_")
                files.append(V039_OUT / "social" / f"s{seed}_p{part}_{name}" / "final.pt")
    return {str(path.resolve()): sha(path) for path in files}


def load_modules():
    # Import the sealed v0.39 implementation as a library. Its main() is not
    # executed; this gives us the exact triad architecture and fixture.
    sys.path.insert(0, str(V039))
    import triad_train as triad
    return triad


T = load_modules()


def load_initial_population(seed, part, prepared, formation_condition):
    agents = T.load_population(seed, prepared, part)
    name = formation_condition.removeprefix("origin_")
    endpoint = V039_OUT / "social" / f"s{seed}_p{part}_{name}" / "final.pt"
    states = torch.load(endpoint, weights_only=True)
    if len(states) != len(agents):
        raise ValueError(f"endpoint population size mismatch: {endpoint}")
    for agent, state in zip(agents, states):
        agent.load_state_dict(state)
        agent.requires_grad_(False)
        agent.eval()
    return agents, endpoint


def fresh_newcomer(seed, part, prepared, held, generation):
    # load_population creates the same private visual frontend and a fresh
    # three-resource communication head. reset_communication makes the
    # initialization explicit and reproducible for the replacement event.
    agent = T.load_population(seed, prepared, part)[held]
    reset_seed = int(
        np.random.SeedSequence([39040, int(seed), int(part), int(held), int(generation), 901])
        .generate_state(1, dtype=np.uint64)[0]
        >> np.uint64(1)
    )
    reset_record = T.reset_communication(agent, reset_seed)
    agent.requires_grad_(False)
    for group in T.trainable_groups(agent).values():
        for parameter in group:
            parameter.requires_grad_(True)
    agent.train()
    return agent, reset_record


def schedule_for(condition, seed, part, generation, step):
    # Global parity makes the rotating condition independent of replacement
    # boundaries. The random schedule has a fixed, auditable namespace.
    global_step = (generation - 1) * UPDATES_PER_GENERATION + step
    if condition == "fixed_A":
        return "A", global_step
    if condition == "rotating_AB":
        return ("A" if global_step % 2 == 0 else "B"), global_step
    rng = np.random.default_rng(np.random.SeedSequence([39040, int(seed), int(part), int(global_step), 77]))
    return str(rng.choice(np.asarray(SCHEDULES))), global_step


def train_one_step(agents, views_train, train_worlds, seed, part, generation, step, held, condition):
    schedule, global_step = schedule_for(condition, seed, part, generation, step)
    selected_losses = []
    fixture_slots = {}
    trace_slots = {}
    for slot, team in enumerate(TEAMS[schedule]):
        receiver, sender0, sender1, sender2 = team
        fixture_data = T.fixture(seed, part, slot, global_step, train_worlds)
        idx = fixture_data["indices"]
        sender_losses, receiver_loss, trace = T.tri_loss(
            [agents[sender0], agents[sender1], agents[sender2]],
            agents[receiver],
            [
                views_train[sender0][0]["train"][idx],
                views_train[sender1][1]["train"][idx],
                views_train[sender2][2]["train"][idx],
            ],
            fixture_data["uniforms"],
            train_worlds["positions"][idx],
        )
        if sender0 == held:
            selected_losses.append(sender_losses[0])
        if sender1 == held:
            selected_losses.append(sender_losses[1])
        if sender2 == held:
            selected_losses.append(sender_losses[2])
        if receiver == held:
            selected_losses.append(receiver_loss)
        fixture_slots[slot] = fixture_data
        trace_slots[slot] = trace
    if not selected_losses:
        raise ValueError(f"held identity {held} did not receive a role loss")
    return sum(selected_losses) / len(selected_losses), fixture_slots, trace_slots, schedule, global_step


def save_trace(folder, step, slot, fixture_data, trace, schedule, global_step, team):
    payload = {
        "schedule_id": np.asarray([schedule]),
        "global_step": np.asarray([global_step], dtype=np.int64),
        "slot": np.asarray([slot], dtype=np.int64),
        "team": np.asarray(team, dtype=np.int64),
        "world__indices": fixture_data["indices"],
        "world__uniforms": fixture_data["uniforms"],
    }
    payload.update({f"trace__{key}": value for key, value in trace.items()})
    np.savez_compressed(folder / f"train_{step + 1:04d}_slot{slot}.npz", **payload)


def run(
    out: Path,
    seeds=SEEDS,
    parts=PARTITIONS,
    formation_conditions=FORMATION_CONDITIONS,
    transmission_conditions=TRANSMISSION_CONDITIONS,
    generations=GENERATIONS,
    updates=UPDATES_PER_GENERATION,
):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    hashes, inputs = source_hashes(), input_hashes()
    formal = (
        list(seeds) == list(SEEDS)
        and list(parts) == list(PARTITIONS)
        and list(formation_conditions) == list(FORMATION_CONDITIONS)
        and list(transmission_conditions) == list(TRANSMISSION_CONDITIONS)
        and generations == GENERATIONS
        and updates == UPDATES_PER_GENERATION
    )
    checkpoints = list(range(generations + 1))
    write(
        out / "invocation.json",
        {
            "formal": formal,
            "version": "v0.40-triad-transfer",
            "seeds": list(seeds),
            "partitions": list(parts),
            "formation_conditions": list(formation_conditions),
            "transmission_conditions": list(transmission_conditions),
            "schedules": list(SCHEDULES),
            "generations": generations,
            "updates_per_generation": updates,
            "replacement_order": list(REPLACEMENT_ORDER[:generations]),
            "resources": 3,
            "vocabulary": 7,
            "schedule_seed_namespace": 39040,
            "source_hashes": hashes,
            "input_hashes": inputs,
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "threads": 1,
            "new_private_fits": 0,
            "new_dino_inferences": 0,
            "study_scope": "factorial transmission of a grounded three-resource protocol",
        },
    )
    for path in hashes:
        target = out / "frozen_sources" / Path(path).relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    worlds_test = npz(V039_OUT / "test_worlds.npz")
    np.savez_compressed(out / "test_worlds.npz", **worlds_test)
    worlds_train = {}
    for part in parts:
        worlds_train[part] = npz(V039_OUT / f"train_worlds_p{part}.npz")
        np.savez_compressed(out / f"train_worlds_p{part}.npz", **worlds_train[part])

    bank = T.TriBank()
    rows = []
    started = time.monotonic()
    total = 0
    for seed, part, formation_condition, transmission_condition in itertools.product(
        seeds, parts, formation_conditions, transmission_conditions
    ):
        prepared = torch.load(SOURCE / f"prepared_{seed}.pt", weights_only=True)
        agents, endpoint = load_initial_population(seed, part, prepared, formation_condition)
        # Communication resets do not alter the private visual frontends, so
        # these views remain valid for every replacement generation.
        views = T.load_views(agents, bank, {"train": worlds_train[part], "test": worlds_test})
        chain = out / "chains" / f"s{seed}_p{part}_{formation_condition}__{transmission_condition}"
        chain.mkdir(parents=True, exist_ok=True)
        config = {
            "seed": seed,
            "partition": part,
            "formation_condition": formation_condition,
            "transmission_condition": transmission_condition,
            "generations": generations,
            "updates_per_generation": updates,
            "replacement_order": list(REPLACEMENT_ORDER[:generations]),
            "initial_residents": "v0.39 teacher-free triad endpoint",
            "formation_source": str(endpoint.resolve()),
            "training_schedule": {
                "fixed_A": "A_every_update",
                "rotating_AB": "A_even_B_odd_global_update",
                "random_ABC": "deterministic_random_A_B_C_namespace_39040",
            }[transmission_condition],
            "evaluation_schedules": list(SCHEDULES),
            "newcomer_training": "communication_modules_only",
            "private_types": list(PRIVATE_TYPES),
            "resources": 3,
            "message_length": 3,
            "vocabulary": 7,
        }
        write(chain / "config.json", config)
        torch.save([copy.deepcopy(agent.state_dict()) for agent in agents], chain / "generation_000_before.pt")

        generation_rows = []
        g0 = chain / "generation_00"
        g0.mkdir(exist_ok=True)
        g0_scores = {schedule: T.evaluate(agents, views, worlds_test, part, schedule, g0, 0) for schedule in SCHEDULES}
        generation_rows.append({"generation": 0, "replaced_identity": None, "scores": g0_scores, "schedule_counts": {schedule: 0 for schedule in SCHEDULES}})
        write(chain / "curve.json", generation_rows)

        for generation in range(1, generations + 1):
            held = REPLACEMENT_ORDER[generation - 1]
            newcomer, reset_record = fresh_newcomer(seed, part, prepared, held, generation)
            current = list(agents)
            current[held] = newcomer
            params = [param for group in T.trainable_groups(newcomer).values() for param in group]
            optimizer = torch.optim.Adam(params, lr=0.0007)
            folder = chain / f"generation_{generation:02d}"
            folder.mkdir(exist_ok=True)
            write(
                folder / "config.json",
                {
                    "generation": generation,
                    "replaced_identity": held,
                    "private_type": PRIVATE_TYPES[held],
                    "updates": updates,
                    "reset": reset_record,
                    "formation_condition": formation_condition,
                    "transmission_condition": transmission_condition,
                    "schedule_seed_namespace": 39040,
                },
            )
            schedule_counts = {schedule: 0 for schedule in SCHEDULES}
            trace_names = []
            with (folder / "training.jsonl").open("w") as log:
                for step in range(updates):
                    loss, fixture_slots, trace_slots, schedule, global_step = train_one_step(
                        current, views, worlds_train[part], seed, part, generation, step, held, transmission_condition
                    )
                    schedule_counts[schedule] += 1
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norms = {
                        role: float(torch.nn.utils.clip_grad_norm_(group, 2.0))
                        for role, group in T.trainable_groups(newcomer).items()
                    }
                    if not all(np.isfinite(value) for value in norms.values()):
                        raise ValueError("non-finite newcomer gradient")
                    optimizer.step()
                    if step in (0, updates - 1):
                        for slot, team in enumerate(TEAMS[schedule]):
                            name = f"train_{step + 1:04d}_slot{slot}.npz"
                            save_trace(folder, step, slot, fixture_slots[slot], trace_slots[slot], schedule, global_step, team)
                            trace_names.append(name)
                    log.write(
                        json.dumps(
                            {
                                "update": step + 1,
                                "global_step": global_step,
                                "schedule": schedule,
                                "loss": float(loss.detach()),
                                "norms": norms,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            newcomer.eval()
            scores = {schedule: T.evaluate(current, views, worlds_test, part, schedule, folder, generation) for schedule in SCHEDULES}
            torch.save([copy.deepcopy(agent.state_dict()) for agent in current], folder / "generation_after.pt")
            generation_rows.append(
                {
                    "generation": generation,
                    "replaced_identity": held,
                    "scores": scores,
                    "schedule_counts": schedule_counts,
                    "trace_files": sorted(trace_names),
                }
            )
            for agent in current:
                agent.requires_grad_(False)
                agent.eval()
            agents = current
            write(chain / "curve.json", generation_rows)
            print(
                json.dumps(
                    {
                        "phase": "triad_transfer",
                        "chain": chain.name,
                        "generation": generation,
                        "runs": total + 1,
                        "seconds": round(time.monotonic() - started, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        rows.append(
            {
                "seed": seed,
                "partition": part,
                "formation_condition": formation_condition,
                "transmission_condition": transmission_condition,
                "generations": generation_rows,
            }
        )
        total += 1

    write(out / "runs.json", {"status": "complete", "rows": rows, "count": total})
    write(
        out / "training_complete.json",
        {
            "status": "complete",
            "formal": formal,
            "probe": "triad_transfer",
            "runs": total,
            "generations_per_run": generations,
            "updates_per_generation": updates,
            "replacement_events": total * generations,
            "messages": total * generations * updates * 4 * 240 * 3,
            "token_instances": total * generations * updates * 4 * 240 * 3,
            "actions": total * generations * updates * 4 * 240 * 3,
            "trace_files_expected": total * generations * 2 * 4,
            "new_dino_inferences": 0,
            "new_private_fits": 0,
            "seconds": time.monotonic() - started,
            "source_hashes": hashes,
            "input_hashes": inputs,
            "files": {
                str(path.relative_to(out)): sha(path)
                for path in sorted(out.rglob("*"))
                if path.is_file()
            },
        },
    )
    return {
        "status": "complete",
        "formal": formal,
        "runs": total,
        "replacement_events": total * generations,
        "seconds": time.monotonic() - started,
        "source_sha256": sha(ROOT / "triad_transfer_train.py"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument("--updates", type=int, default=UPDATES_PER_GENERATION)
    args = parser.parse_args()
    if args.dev:
        result = run(
            args.out.resolve(),
            seeds=(SEEDS[0],),
            parts=(1,),
            formation_conditions=(FORMATION_CONDITIONS[0],),
            transmission_conditions=(TRANSMISSION_CONDITIONS[0],),
            generations=min(args.generations, 2),
            updates=min(args.updates, 30),
        )
    else:
        result = run(args.out.resolve(), generations=args.generations, updates=args.updates)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
