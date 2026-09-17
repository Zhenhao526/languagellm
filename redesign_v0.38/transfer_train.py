"""Factorial transfer of formed protocols through newcomer replacement."""
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
V037 = PROJECT / "redesign_v0.37"
SOURCE = PROJECT / "redesign_v0.28" / "results" / "formation_001"
FIXED = V034 / "results" / "fixed_001"
ORIGIN = V037 / "results" / "origin_001"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
FORMATION_CONDITIONS = ("origin_fixed_A", "origin_rotating_AB", "origin_random_ABC")
TRANSMISSION_CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
SCHEDULES = ("A", "B", "C")
PRIVATE_TYPES = (0, 1, 0, 1)
REPLACEMENT_ORDER = (0, 1, 2, 3)
GENERATIONS = 4
UPDATES_PER_GENERATION = 300
TEAMS = {
    "A": ((0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)),
    "B": ((0, 2, 3), (1, 3, 0), (2, 0, 1), (3, 1, 2)),
    "C": ((0, 3, 1), (1, 0, 2), (2, 1, 3), (3, 2, 0)),
}


def read(path: Path): return json.loads(path.read_text())
def write(path: Path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def npz(path: Path):
    with np.load(path, allow_pickle=False) as z: return {key: z[key] for key in z.files}


def source_modules():
    sys.path.insert(0, str(PROJECT / "redesign_v0.8")); sys.path.insert(0, str(PROJECT / "redesign_v0.21")); sys.path.insert(0, str(PROJECT / "redesign_v0.28")); sys.path.insert(0, str(V034)); sys.path.insert(0, str(PROJECT / "redesign_v0.35"))
    import newcomer_adaptation as adaptation
    return adaptation


_ADAPT = source_modules()


def source_hashes():
    files = [ROOT / "transfer_train.py", ROOT / "transfer_design.json", PROJECT / "redesign_v0.35" / "newcomer_adaptation.py"]
    files += [V037 / name for name in ("origin_train.py", "origin_design.json")]
    files += [V034 / name for name in ("run_support.py", "support.py", "views.py", "dual.py", "metrics.py", "code_relabelings.npy")]
    files += [PROJECT / name for name in ("redesign_v0.21/social_model.py", "redesign_v0.28/world.py", "redesign_v0.20/temporal_model.py", "redesign_v0.20/temporal_world.py", "redesign_v0.8/camp.py")]
    return {str(path.resolve()): sha(path) for path in files}


def input_hashes():
    files = [FIXED / "training_complete.json", FIXED / "train_worlds.npz", FIXED / "test_worlds.npz", ORIGIN / "training_complete.json", ORIGIN / "test_worlds.npz", PROJECT / "redesign_v0.28/data/feature_cache.pt", PROJECT / "redesign_v0.28/data/selection.json"]
    for seed in SEEDS:
        files.append(SOURCE / f"prepared_{seed}.pt")
        for part in PARTITIONS:
            for private_type in (0, 1): files.append(SOURCE / "private" / f"s{seed}_p{part}_d{private_type}_all" / "final.pt")
            for formation in FORMATION_CONDITIONS:
                origin_name = formation.removeprefix("origin_")
                files.append(ORIGIN / "social" / f"s{seed}_p{part}_{origin_name}" / "final.pt")
            for view in ("food_only", "water_only"):
                for private_type in (0, 1):
                    for split in ("train", "test"): files.append(FIXED / "cache" / f"s{seed}_p{part}" / f"{split}_{view}_d{private_type}.npy")
    return {str(path.resolve()): sha(path) for path in files}


def load_initial_population(seed, part, prepared, formation_condition):
    """Restore one v0.37 teacher-free endpoint into fresh model objects."""
    residents, _ = _ADAPT._RUN_SUPPORT.restore_population(seed, part, prepared, SOURCE)
    origin_name = formation_condition.removeprefix("origin_")
    folder = ORIGIN / "social" / f"s{seed}_p{part}_{origin_name}"
    states = torch.load(folder / "final.pt", weights_only=True)
    if len(states) != len(residents):
        raise ValueError(f"endpoint population size mismatch: {folder}")
    for agent, state in zip(residents, states):
        agent.load_state_dict(state)
        agent.requires_grad_(False)
        agent.eval()
    return residents, folder


def fresh_newcomer(seed, part, prepared, held, generation):
    templates = _ADAPT._RUN_SUPPORT.remake_agents(seed, prepared, 7, 2, "identity")
    private_type = PRIVATE_TYPES[held]
    agent = copy.deepcopy(templates[private_type])
    blob = torch.load(SOURCE / "private" / f"s{seed}_p{part}_d{private_type}_all" / "final.pt", weights_only=True)
    agent.load_state_dict(blob["agent"])
    reset_seed = _ADAPT._SUPPORT.seed_value(seed, part, 4 + held, 1000 + generation)
    reset_record = _ADAPT._COMMUNICATION.reset_communication(agent, reset_seed)
    agent.requires_grad_(False)
    for params in _ADAPT._COMMUNICATION.trainable_groups(agent).values():
        for parameter in params: parameter.requires_grad_(True)
    agent.train()
    return agent, reset_record


def schedule_for(condition, seed, part, generation, step):
    global_step = (generation - 1) * UPDATES_PER_GENERATION + step
    if condition == "fixed_A": return "A", global_step
    if condition == "rotating_AB": return ("A" if global_step % 2 == 0 else "B"), global_step
    rng = np.random.default_rng(np.random.SeedSequence([34038, int(seed), int(part), int(generation), int(step), 77]))
    return str(rng.choice(np.asarray(SCHEDULES))), global_step


def train_one_step(agents, views_train, train_worlds, seed, part, generation, step, held, transmission_condition):
    schedule, global_step = schedule_for(transmission_condition, seed, part, generation, step)
    selected_losses = []; fixture_worlds = {}; traces = {}
    for slot, team in enumerate(TEAMS[schedule]):
        receiver, sender_food, sender_water = team
        fixture = _ADAPT._SUPPORT.fixture(seed, part, slot, global_step, "dual_complementary", train_worlds)
        idx = fixture["indices"]
        food_loss, water_loss, receiver_loss, trace = _ADAPT._DUAL.dual_loss(
            agents[sender_food], agents[sender_water], agents[receiver],
            views_train[sender_food]["food_only"][idx], views_train[sender_water]["water_only"][idx],
            fixture["uniforms"], train_worlds["positions"][idx], 0.02)
        if sender_food == held: selected_losses.append(food_loss)
        if sender_water == held: selected_losses.append(water_loss)
        if receiver == held: selected_losses.append(receiver_loss)
        fixture_worlds[f"slot{slot}__indices"] = fixture["indices"]
        fixture_worlds[f"slot{slot}__uniforms"] = fixture["uniforms"]
        traces[f"slot{slot}__r{receiver}_sf{sender_food}_sw{sender_water}"] = trace
    if not selected_losses: raise ValueError("held identity did not receive a role loss")
    return sum(selected_losses) / len(selected_losses), fixture_worlds, traces, schedule, global_step


def save_trace(folder, name, fixture_worlds, traces, schedule, global_step):
    payload = {"schedule_id": np.asarray([schedule]), "global_step": np.asarray([global_step], dtype=np.int64)}
    payload.update({f"world__{key}": value for key, value in fixture_worlds.items()})
    for prefix, trace in traces.items(): payload.update({f"trace__{prefix}__{key}": value for key, value in trace.items()})
    np.savez_compressed(folder / name, **payload)


def run(out: Path, seeds=SEEDS, parts=PARTITIONS, formation_conditions=FORMATION_CONDITIONS, transmission_conditions=TRANSMISSION_CONDITIONS, generations=GENERATIONS, updates=UPDATES_PER_GENERATION):
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    hashes, inputs = source_hashes(), input_hashes()
    formal = list(seeds) == list(SEEDS) and list(parts) == list(PARTITIONS) and list(formation_conditions) == list(FORMATION_CONDITIONS) and list(transmission_conditions) == list(TRANSMISSION_CONDITIONS) and generations == GENERATIONS and updates == UPDATES_PER_GENERATION
    write(out / "invocation.json", {"formal": formal, "version": "v0.38-origin-transfer", "seeds": list(seeds), "partitions": list(parts), "formation_conditions": list(formation_conditions), "transmission_conditions": list(transmission_conditions), "schedules": list(SCHEDULES), "generations": generations, "updates_per_generation": updates, "replacement_order": list(REPLACEMENT_ORDER[:generations]), "source_hashes": hashes, "input_hashes": inputs, "torch_version": str(torch.__version__), "numpy_version": np.__version__, "platform": platform.platform(), "threads": 1, "new_dino_inferences": 0, "new_private_fits": 0, "study_scope": "factorial cultural transmission from teacher-free formed endpoints"})
    for path in hashes:
        target = out / "frozen_sources" / Path(path).relative_to(PROJECT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    tables = {split: npz(FIXED / f"{split}_worlds.npz") for split in ("train", "test")}
    for split, table in tables.items(): np.savez_compressed(out / f"{split}_worlds.npz", **table)
    run_rows = []; started = time.monotonic(); total = 0
    for seed, part, formation_condition, transmission_condition in itertools.product(seeds, parts, formation_conditions, transmission_conditions):
        prepared = torch.load(SOURCE / f"prepared_{seed}.pt", weights_only=True)
        agents, origin_folder = load_initial_population(seed, part, prepared, formation_condition)
        views_train = _ADAPT.load_views(FIXED, seed, part, "train"); views_test = _ADAPT.load_views(FIXED, seed, part, "test")
        chain = out / "chains" / f"s{seed}_p{part}_{formation_condition}__{transmission_condition}"; chain.mkdir(parents=True, exist_ok=True)
        config = {"seed": seed, "partition": part, "formation_condition": formation_condition, "transmission_condition": transmission_condition, "generations": generations, "updates_per_generation": updates, "replacement_order": list(REPLACEMENT_ORDER[:generations]), "initial_residents": "v0.37 teacher-free endpoint", "formation_source": str(origin_folder.resolve()), "training_schedule": {"fixed_A": "A_every_update", "rotating_AB": "A_even_B_odd_global_update", "random_ABC": "deterministic_random_A_B_C"}[transmission_condition], "evaluation_schedules": list(SCHEDULES), "newcomer_training": "communication_modules_only", "private_types": list(PRIVATE_TYPES)}
        write(chain / "config.json", config)
        torch.save([copy.deepcopy(agent.state_dict()) for agent in agents], chain / "generation_000_before.pt")
        generation_rows = []
        g0 = chain / "generation_00"; g0.mkdir(exist_ok=True)
        g0_scores = {schedule: _ADAPT.evaluate(agents, views_test, tables["test"], part, schedule, g0, 0) for schedule in SCHEDULES}
        generation_rows.append({"generation": 0, "replaced_identity": None, "scores": g0_scores, "schedule_counts": {schedule: 0 for schedule in SCHEDULES}})
        for generation in range(1, generations + 1):
            held = REPLACEMENT_ORDER[generation - 1]
            newcomer, reset_record = fresh_newcomer(seed, part, prepared, held, generation)
            current = list(agents); current[held] = newcomer
            params = [param for group in _ADAPT._COMMUNICATION.trainable_groups(newcomer).values() for param in group]
            optimizer = torch.optim.Adam(params, lr=0.0007)
            folder = chain / f"generation_{generation:02d}"; folder.mkdir(exist_ok=True)
            write(folder / "config.json", {"generation": generation, "replaced_identity": held, "private_type": PRIVATE_TYPES[held], "updates": updates, "reset": reset_record, "formation_condition": formation_condition, "transmission_condition": transmission_condition})
            schedule_counts = {schedule: 0 for schedule in SCHEDULES}; trace_names = []
            with (folder / "training.jsonl").open("w") as log:
                for step in range(updates):
                    loss, fixture_worlds, traces, schedule, global_step = train_one_step(current, views_train, tables["train"], seed, part, generation, step, held, transmission_condition)
                    schedule_counts[schedule] += 1
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    norms = {role: float(torch.nn.utils.clip_grad_norm_(params, 2.0)) for role, group in _ADAPT._COMMUNICATION.trainable_groups(newcomer).items() for params in [group]}
                    if not all(np.isfinite(value) for value in norms.values()): raise ValueError("non-finite newcomer gradient")
                    optimizer.step()
                    if step in (0, updates - 1):
                        name = f"train_{step + 1:04d}.npz"; save_trace(folder, name, fixture_worlds, traces, schedule, global_step); trace_names.append(name)
                    log.write(json.dumps({"update": step + 1, "global_step": global_step, "schedule": schedule, "loss": float(loss.detach()), "norms": norms}, ensure_ascii=False) + "\n")
            newcomer.eval(); scores = {schedule: _ADAPT.evaluate(current, views_test, tables["test"], part, schedule, folder, generation) for schedule in SCHEDULES}; newcomer.train(False)
            torch.save([copy.deepcopy(agent.state_dict()) for agent in current], folder / "generation_after.pt")
            generation_rows.append({"generation": generation, "replaced_identity": held, "scores": scores, "schedule_counts": schedule_counts, "trace_files": trace_names})
            for agent in current: agent.requires_grad_(False); agent.eval()
            agents = current
            write(chain / "curve.json", generation_rows)
            print(json.dumps({"phase": "origin_transfer", "chain": chain.name, "generation": generation, "runs": total + 1, "seconds": round(time.monotonic() - started, 2)}), flush=True)
        run_rows.append({"seed": seed, "partition": part, "formation_condition": formation_condition, "transmission_condition": transmission_condition, "generations": generation_rows})
        total += 1
    write(out / "runs.json", {"status": "complete", "rows": run_rows, "count": total})
    write(out / "training_complete.json", {"status": "complete", "formal": formal, "probe": "origin_transfer", "runs": total, "generations_per_run": generations, "updates_per_generation": updates, "replacement_events": total * generations, "messages": total * generations * updates * 8 * 240, "actions": total * generations * updates * 8 * 240, "new_dino_inferences": 0, "new_private_fits": 0, "seconds": time.monotonic() - started, "source_hashes": hashes, "input_hashes": inputs, "files": {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}})
    return {"status": "complete", "formal": formal, "runs": total, "replacement_events": total * generations, "seconds": time.monotonic() - started, "source_sha256": sha(ROOT / "transfer_train.py")}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--dev", action="store_true"); parser.add_argument("--generations", type=int, default=GENERATIONS); parser.add_argument("--updates", type=int, default=UPDATES_PER_GENERATION); args = parser.parse_args()
    if args.dev:
        result = run(args.out.resolve(), seeds=(SEEDS[0],), parts=(1,), formation_conditions=(FORMATION_CONDITIONS[0],), transmission_conditions=(TRANSMISSION_CONDITIONS[0],), generations=min(args.generations, 2), updates=min(args.updates, 30))
    else: result = run(args.out.resolve(), generations=args.generations, updates=args.updates)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
