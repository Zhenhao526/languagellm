"""Teacher-free joint protocol formation from random communication modules."""
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
SOURCE = PROJECT / "redesign_v0.28" / "results" / "formation_001"
FIXED = V034 / "results" / "fixed_001"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
SCHEDULES = ("A", "B", "C")
PRIVATE_TYPES = (0, 1, 0, 1)
CHECKPOINTS = (0, 100, 600, 1200)
UPDATES = 1200
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
    files = [ROOT / "origin_train.py", ROOT / "origin_design.json", PROJECT / "redesign_v0.35" / "newcomer_adaptation.py"]
    files += [V034 / name for name in ("run_support.py", "support.py", "views.py", "dual.py", "metrics.py", "code_relabelings.npy")]
    files += [PROJECT / name for name in ("redesign_v0.21/social_model.py", "redesign_v0.28/world.py", "redesign_v0.20/temporal_model.py", "redesign_v0.20/temporal_world.py", "redesign_v0.8/camp.py")]
    return {str(path.resolve()): sha(path) for path in files}


def input_hashes():
    files = [FIXED / "training_complete.json", FIXED / "train_worlds.npz", FIXED / "test_worlds.npz", PROJECT / "redesign_v0.28/data/feature_cache.pt", PROJECT / "redesign_v0.28/data/selection.json"]
    for seed in SEEDS:
        files.append(SOURCE / f"prepared_{seed}.pt")
        for part in PARTITIONS:
            for private_type in (0, 1): files.append(SOURCE / "private" / f"s{seed}_p{part}_d{private_type}_all" / "final.pt")
            for view in ("food_only", "water_only"):
                for split in ("train", "test"):
                    for private_type in (0, 1): files.append(FIXED / "cache" / f"s{seed}_p{part}" / f"{split}_{view}_d{private_type}.npy")
    return {str(path.resolve()): sha(path) for path in files}


def random_population(seed, part, prepared):
    templates = _ADAPT._RUN_SUPPORT.remake_agents(seed, prepared, 7, 2, "identity")
    agents = []
    resets = []
    for identity, private_type in enumerate(PRIVATE_TYPES):
        agent = copy.deepcopy(templates[private_type]); blob = torch.load(SOURCE / "private" / f"s{seed}_p{part}_d{private_type}_all" / "final.pt", weights_only=True); agent.load_state_dict(blob["agent"])
        reset_seed = _ADAPT._SUPPORT.seed_value(seed, part, identity, 901)
        resets.append(_ADAPT._COMMUNICATION.reset_communication(agent, reset_seed)); agents.append(agent)
        agent.train()
    return agents, resets


def schedule_for(condition, seed, part, step):
    if condition == "fixed_A": return "A"
    if condition == "rotating_AB": return "A" if step % 2 == 0 else "B"
    rng = np.random.default_rng(np.random.SeedSequence([34037, int(seed), int(part), int(step), 77])); return str(rng.choice(np.asarray(SCHEDULES)))


def train_step(agents, views_train, worlds, seed, part, step, condition):
    schedule = schedule_for(condition, seed, part, step); losses = [[] for _ in agents]; fixture_worlds = {}; traces = {}
    for slot, team in enumerate(TEAMS[schedule]):
        receiver, sender_food, sender_water = team; fixture = _ADAPT._SUPPORT.fixture(seed, part, slot, step, "dual_complementary", worlds); idx = fixture["indices"]
        food_loss, water_loss, receiver_loss, trace = _ADAPT._DUAL.dual_loss(agents[sender_food], agents[sender_water], agents[receiver], views_train[sender_food]["food_only"][idx], views_train[sender_water]["water_only"][idx], fixture["uniforms"], worlds["positions"][idx], .02)
        losses[sender_food].append(food_loss); losses[sender_water].append(water_loss); losses[receiver].append(receiver_loss); fixture_worlds[f"slot{slot}__indices"] = fixture["indices"]; fixture_worlds[f"slot{slot}__uniforms"] = fixture["uniforms"]; traces[f"slot{slot}__r{receiver}_sf{sender_food}_sw{sender_water}"] = trace
    logs = []
    for agent, role_losses in zip(agents, losses):
        if not role_losses: raise ValueError("every agent needs a role loss")
        loss = sum(role_losses) / len(role_losses); logs.append(loss)
    return logs, fixture_worlds, traces, schedule


def save_trace(folder, name, fixture_worlds, traces, schedule, step):
    payload = {"schedule_id": np.asarray([schedule]), "global_step": np.asarray([step], dtype=np.int64)}; payload.update({f"world__{key}": value for key, value in fixture_worlds.items()})
    for prefix, trace in traces.items(): payload.update({f"trace__{prefix}__{key}": value for key, value in trace.items()})
    np.savez_compressed(folder / name, **payload)


def run(out: Path, seeds=SEEDS, parts=PARTITIONS, conditions=CONDITIONS, updates=UPDATES):
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=False); hashes, inputs = source_hashes(), input_hashes(); checkpoints = sorted(set(t for t in CHECKPOINTS if t <= updates) | {updates}); formal = list(seeds) == list(SEEDS) and list(parts) == list(PARTITIONS) and list(conditions) == list(CONDITIONS) and updates == UPDATES
    write(out / "invocation.json", {"formal": formal, "version": "v0.37-origin", "seeds": list(seeds), "partitions": list(parts), "conditions": list(conditions), "schedules": list(SCHEDULES), "updates": updates, "checkpoints": checkpoints, "source_hashes": hashes, "input_hashes": inputs, "torch_version": str(torch.__version__), "numpy_version": np.__version__, "platform": platform.platform(), "threads": 1, "new_dino_inferences": 0, "new_private_fits": 0, "communication_initialization": "fresh_random_reset_all_agents", "study_scope": "teacher-free joint grounded protocol formation"})
    for path in hashes:
        target = out / "frozen_sources" / Path(path).relative_to(PROJECT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    tables = {split: npz(FIXED / f"{split}_worlds.npz") for split in ("train", "test")}
    for split, table in tables.items(): np.savez_compressed(out / f"{split}_worlds.npz", **table)
    rows = []; started = time.monotonic(); total = 0
    bank = None
    for seed, part, condition in itertools.product(seeds, parts, conditions):
        prepared = torch.load(SOURCE / f"prepared_{seed}.pt", weights_only=True); agents, resets = random_population(seed, part, prepared); views_train = _ADAPT.load_views(FIXED, seed, part, "train"); views_test = _ADAPT.load_views(FIXED, seed, part, "test"); folder = out / "social" / f"s{seed}_p{part}_{condition}"; folder.mkdir(parents=True, exist_ok=True)
        write(folder / "config.json", {"seed": seed, "partition": part, "condition": condition, "updates": updates, "checkpoints": checkpoints, "training_schedule": {"fixed_A": "A_every_update", "rotating_AB": "A_even_B_odd", "random_ABC": "deterministic_random_A_B_C"}[condition], "evaluation_schedules": list(SCHEDULES), "private_types": list(PRIVATE_TYPES), "communication_initialization": "fresh_random_reset_all_agents", "resets": resets})
        params = [[p for group in _ADAPT._COMMUNICATION.trainable_groups(agent).values() for p in group] for agent in agents]; optimizers = [torch.optim.Adam(p, lr=.0007) for p in params]; curve = []
        for step in range(updates + 1):
            if step in checkpoints:
                for agent in agents: agent.eval()
                scores = {schedule: _ADAPT.evaluate(agents, views_test, tables["test"], part, schedule, folder, step) for schedule in SCHEDULES}
                for agent in agents: agent.train()
                curve.append({"update": step, "scores": scores}); write(folder / "curve.json", curve)
                if step == updates: torch.save([copy.deepcopy(agent.state_dict()) for agent in agents], folder / "final.pt")
            if step == updates: break
            losses, fixture_worlds, traces, schedule = train_step(agents, views_train, tables["train"], seed, part, step, condition)
            for optimizer, agent_loss in zip(optimizers, losses): optimizer.zero_grad(set_to_none=True); agent_loss.backward()
            norms = []
            for agent, p in zip(agents, params): norms.append({role: float(torch.nn.utils.clip_grad_norm_(group, 2.0)) for role, group in _ADAPT._COMMUNICATION.trainable_groups(agent).items()})
            if not all(np.isfinite(value) for norm in norms for value in norm.values()): raise ValueError("non-finite gradient")
            for optimizer in optimizers: optimizer.step()
            if step in (0, updates - 1): save_trace(folder, f"train_{step + 1:04d}.npz", fixture_worlds, traces, schedule, step)
            with (folder / "training.jsonl").open("a") as log: log.write(json.dumps({"update": step + 1, "schedule": schedule, "norms": norms}, ensure_ascii=False) + "\n")
        rows.append({"seed": seed, "partition": part, "condition": condition, "curve": curve}); total += 1; print(json.dumps({"phase": "origin", "run": folder.name, "runs": total, "seconds": round(time.monotonic() - started, 2)}), flush=True)
    write(out / "runs.json", {"status": "complete", "rows": rows, "count": total}); write(out / "training_complete.json", {"status": "complete", "formal": formal, "probe": "teacher_free_origin", "runs": total, "updates_per_run": updates, "messages": total * updates * 8 * 240, "actions": total * updates * 8 * 240, "new_dino_inferences": 0, "new_private_fits": 0, "seconds": time.monotonic() - started, "source_hashes": hashes, "input_hashes": inputs, "files": {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}})
    return {"status": "complete", "formal": formal, "runs": total, "seconds": time.monotonic() - started, "source_sha256": sha(ROOT / "origin_train.py")}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--dev", action="store_true"); parser.add_argument("--updates", type=int, default=UPDATES); args = parser.parse_args()
    result = run(args.out.resolve(), seeds=(SEEDS[0],), parts=(1,), conditions=("fixed_A",), updates=min(args.updates, 60)) if args.dev else run(args.out.resolve(), updates=args.updates); print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
