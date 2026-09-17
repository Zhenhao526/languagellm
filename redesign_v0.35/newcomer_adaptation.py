"""Train only a newcomer communication head inside frozen v0.34 societies."""
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
ROTATION = V034 / "results" / "rotation_001"
FIXED = V034 / "results" / "fixed_001"
SOURCE = PROJECT / "redesign_v0.28" / "results" / "formation_001"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
CONDITIONS = ("fixed_A", "rotating_AB")
SCHEDULES = ("A", "B", "C")
HELD_IDENTITIES = (0, 1)
CHECKPOINTS = (0, 100, 300, 600)
PRIVATE_TYPES = (0, 1, 0, 1)
TEAMS = {
    "A": ((0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)),
    "B": ((0, 2, 3), (1, 3, 0), (2, 0, 1), (3, 1, 2)),
    "C": ((0, 3, 1), (1, 0, 2), (2, 1, 3), (3, 2, 0)),
}


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
    files = [ROOT / "newcomer_adaptation.py", ROOT / "adaptation_design.json"]
    files += [V034 / name for name in ("run_support.py", "support.py", "views.py", "dual.py", "metrics.py", "code_relabelings.npy")]
    files += [PROJECT / name for name in ("redesign_v0.21/social_model.py", "redesign_v0.28/world.py", "redesign_v0.20/temporal_model.py", "redesign_v0.20/temporal_world.py", "redesign_v0.8/camp.py")]
    return {str(path.resolve()): sha(path) for path in files}


def input_hashes():
    files = [
        ROTATION / "training_complete.json",
        FIXED / "training_complete.json",
        ROTATION / "test_worlds.npz",
        ROTATION / "train_worlds.npz",
        FIXED / "test_worlds.npz",
        FIXED / "train_worlds.npz",
        PROJECT / "redesign_v0.28/data/feature_cache.pt",
        PROJECT / "redesign_v0.28/data/selection.json",
    ]
    for seed in SEEDS:
        files.append(SOURCE / f"prepared_{seed}.pt")
        for part in PARTITIONS:
            for private_type in (0, 1):
                files.append(SOURCE / "private" / f"s{seed}_p{part}_d{private_type}_all" / "final.pt")
            for base in (ROTATION, FIXED):
                folder = base / "social" / f"s{seed}_p{part}_dual_complementary"
                files.append(folder / "final.pt")
                for split in ("train", "test"):
                    for view in ("food_only", "water_only"):
                        for private_type in (0, 1):
                            files.append(base / "cache" / f"s{seed}_p{part}" / f"{split}_{view}_d{private_type}.npy")
    return {str(path.resolve()): sha(path) for path in files}


def load_residents(seed, part, prepared, condition):
    base = ROTATION if condition == "rotating_AB" else FIXED
    folder = base / "social" / f"s{seed}_p{part}_dual_complementary"
    residents, _ = _RUN_SUPPORT.restore_population(seed, part, prepared, SOURCE)
    states = torch.load(folder / "final.pt", weights_only=True)
    for agent, state in zip(residents, states):
        agent.load_state_dict(state)
        agent.requires_grad_(False)
        agent.eval()
    return residents, folder, base


def load_newcomer(seed, part, prepared, held_identity):
    templates = _RUN_SUPPORT.remake_agents(seed, prepared, 7, 2, "identity")
    private_type = PRIVATE_TYPES[held_identity]
    agent = copy.deepcopy(templates[private_type])
    blob = torch.load(SOURCE / "private" / f"s{seed}_p{part}_d{private_type}_all" / "final.pt", weights_only=True)
    agent.load_state_dict(blob["agent"])
    reset_seed = _SUPPORT.seed_value(seed, part, 4 + held_identity, 101)
    reset_record = _COMMUNICATION.reset_communication(agent, reset_seed)
    agent.requires_grad_(False)
    for params in _COMMUNICATION.trainable_groups(agent).values():
        for parameter in params:
            parameter.requires_grad_(True)
    agent.train()
    return agent, reset_record


def load_views(base, seed, part, split):
    folder = base / "cache" / f"s{seed}_p{part}"
    result = {}
    for agent in range(4):
        private_type = PRIVATE_TYPES[agent]
        result[agent] = {
            view: torch.from_numpy(np.load(folder / f"{split}_{view}_d{private_type}.npy"))
            for view in ("food_only", "water_only")
        }
    return result


def newcomer_views(agent, worlds, bank):
    with torch.no_grad():
        projected = agent.project(bank.features).detach()
    result = {}
    for view in ("food_only", "water_only"):
        encoded = _RUN_SUPPORT._encode_chunks(agent, projected, worlds, view)
        result[view] = torch.from_numpy(encoded)
    return result


def train_one_step(agents, h_views, train_worlds, seed, part, step, held_identity, condition):
    if condition == "fixed_A":
        teams = TEAMS["A"]
    else:
        teams = TEAMS["A" if step % 2 == 0 else "B"]
    selected_losses = []
    traces = {}
    fixture_worlds = {}
    entropy_weight = 0.02 if step < 2100 else 0.0
    for slot, team in enumerate(teams):
        receiver, sender_food, sender_water = team
        fixture = _SUPPORT.fixture(seed, part, slot, step, "dual_complementary", train_worlds)
        idx = fixture["indices"]
        hf = h_views[sender_food]["food_only"][idx]
        hw = h_views[sender_water]["water_only"][idx]
        sender_food_loss, sender_water_loss, receiver_loss, trace = _DUAL.dual_loss(
            agents[sender_food], agents[sender_water], agents[receiver], hf, hw,
            fixture["uniforms"], train_worlds["positions"][idx], entropy_weight)
        if sender_food == held_identity:
            selected_losses.append(sender_food_loss)
        if sender_water == held_identity:
            selected_losses.append(sender_water_loss)
        if receiver == held_identity:
            selected_losses.append(receiver_loss)
        fixture_worlds[f"slot{slot}__indices"] = fixture["indices"]
        fixture_worlds[f"slot{slot}__uniforms"] = fixture["uniforms"]
        traces[f"slot{slot}__r{receiver}_sf{sender_food}_sw{sender_water}"] = trace
    if not selected_losses:
        raise ValueError("held identity did not receive a role loss")
    return sum(selected_losses) / len(selected_losses), fixture_worlds, traces


def evaluate(agents, h_views, worlds, part, schedule, folder, update):
    # Build one frozen table per model for this checkpoint, then compose every
    # team from those tables.  The newcomer may occur in all three roles.
    food_logs, water_logs, receivers = [], [], []
    for agent in agents:
        food_logs.append(_DUAL.first_token_log_probs(agent, h_views[len(food_logs)]["food_only"]).detach().numpy().astype(np.float32))
        water_logs.append(_DUAL.first_token_log_probs(agent, h_views[len(water_logs)]["water_only"]).detach().numpy().astype(np.float32))
        receivers.append(_COMMUNICATION.receiver_logits(agent).detach().numpy().astype(np.float32))
    scores = {}
    for slot, team in enumerate(TEAMS[schedule]):
        receiver, sender_food, sender_water = team
        lf, lw = food_logs[sender_food], water_logs[sender_water]
        raw = dict(worlds)
        raw["sender_log_probs_food"] = lf
        raw["sender_log_probs_water"] = lw
        raw["sender_log_probs"] = (lf[:, :, None] + lw[:, None, :]).reshape(len(lf), 49).astype(np.float32)
        raw["tokens"] = np.stack((np.argmax(lf, axis=-1), np.argmax(lw, axis=-1)), axis=1).astype(np.int64)
        raw["receiver_logits"] = receivers[receiver]
        path = folder / f"protocol_{schedule}_{update:04d}_team{slot}.npz"
        np.savez_compressed(path, **raw)
        scores[f"team{slot}"] = _METRICS.social(raw, part, "dual_complementary")
    return scores


def run(out: Path, seeds=SEEDS, parts=PARTITIONS, conditions=CONDITIONS, held_ids=HELD_IDENTITIES, updates=600):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    hashes, inputs = source_hashes(), input_hashes()
    checkpoint_list = sorted(set(t for t in CHECKPOINTS if t <= updates) | {updates})
    write(out / "invocation.json", {
        "formal": list(seeds) == list(SEEDS) and list(parts) == list(PARTITIONS) and list(conditions) == list(CONDITIONS) and list(held_ids) == list(HELD_IDENTITIES) and updates == 600,
        "version": "v0.35-adaptation",
        "seeds": list(seeds), "partitions": list(parts), "conditions": list(conditions), "held_identities": list(held_ids),
        "updates": updates, "checkpoints": checkpoint_list, "source_hashes": hashes, "input_hashes": inputs,
        "torch_version": str(torch.__version__), "numpy_version": np.__version__, "platform": platform.platform(), "threads": 1,
        "new_dino_inferences": 0, "new_private_fits": 0, "resident_population": 4, "newcomer_training": "communication_modules_only",
        "study_scope": "online newcomer cultural acquisition under frozen resident endpoint policies",
    })
    for path in hashes:
        target = out / "frozen_sources" / Path(path).relative_to(PROJECT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    rotation_worlds = {split: npz(ROTATION / f"{split}_worlds.npz") for split in ("train", "test")}
    for split, worlds in rotation_worlds.items(): np.savez_compressed(out / f"{split}_worlds.npz", **worlds)
    bank = _WORLD.NonSquareImageBank(PROJECT / "redesign_v0.28/data/feature_cache.pt", PROJECT / "redesign_v0.28/data/selection.json")
    run_rows = []; started = time.monotonic(); total = 0
    for seed, part, condition, held_identity in itertools.product(seeds, parts, conditions, held_ids):
        prepared = torch.load(SOURCE / f"prepared_{seed}.pt", weights_only=True)
        residents, resident_folder, base = load_residents(seed, part, prepared, condition)
        newcomer, reset_record = load_newcomer(seed, part, prepared, held_identity)
        agents = list(residents); agents[held_identity] = newcomer
        resident_views_train = load_views(base, seed, part, "train")
        resident_views_test = load_views(base, seed, part, "test")
        newcomer_train = newcomer_views(newcomer, rotation_worlds["train"], bank)
        newcomer_test = newcomer_views(newcomer, rotation_worlds["test"], bank)
        h_train = dict(resident_views_train); h_train[held_identity] = newcomer_train
        h_test = dict(resident_views_test); h_test[held_identity] = newcomer_test
        folder = out / "social" / f"s{seed}_p{part}_{condition}_id{held_identity}"; folder.mkdir(parents=True, exist_ok=True)
        config = {"seed": seed, "partition": part, "condition": condition, "held_identity": held_identity, "private_type": PRIVATE_TYPES[held_identity], "updates": updates, "checkpoints": checkpoint_list, "resident_source": str(resident_folder.resolve()), "training_schedule": "A_every_update" if condition == "fixed_A" else "A_even_B_odd", "newcomer_training": "communication_modules_only", "reset": reset_record, "evaluation_schedules": list(SCHEDULES)}
        write(folder / "config.json", config)
        params = [param for group in _COMMUNICATION.trainable_groups(newcomer).values() for param in group]
        optimizer = torch.optim.Adam(params, lr=0.0007)
        curve = []
        for step in range(updates + 1):
            if step in checkpoint_list:
                newcomer.eval(); scores = {schedule: evaluate(agents, h_test, rotation_worlds["test"], part, schedule, folder, step) for schedule in SCHEDULES}; newcomer.train()
                curve.append({"update": step, "scores": scores}); write(folder / "curve.json", curve)
                if step == updates: torch.save(newcomer.state_dict(), folder / "final_newcomer.pt")
            if step == updates: break
            loss, fixture_worlds, traces = train_one_step(agents, h_train, rotation_worlds["train"], seed, part, step, held_identity, condition)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norms = {role: float(torch.nn.utils.clip_grad_norm_(params, 2.0)) for role, group in _COMMUNICATION.trainable_groups(newcomer).items() for params in [group]}
            if not all(np.isfinite(value) for value in norms.values()): raise ValueError("non-finite newcomer gradient")
            optimizer.step()
            if step in (0, updates - 1):
                payload = {f"world__{key}": value for key, value in fixture_worlds.items()}
                for prefix, trace in traces.items(): payload.update({f"trace__{prefix}__{key}": value for key, value in trace.items()})
                np.savez_compressed(folder / f"train_{step + 1:04d}.npz", **payload)
            with (folder / "training.jsonl").open("a") as log:
                log.write(json.dumps({"update": step + 1, "loss": float(loss.detach()), "norms": norms}, ensure_ascii=False) + "\n")
        run_rows.append({"seed": seed, "partition": part, "condition": condition, "held_identity": held_identity, "private_type": PRIVATE_TYPES[held_identity], "resident_source": str(resident_folder.resolve()), "curve": curve})
        total += 1
        print(json.dumps({"phase": "newcomer_adaptation", "run": folder.name, "runs": total, "seconds": round(time.monotonic() - started, 2)}), flush=True)
    write(out / "runs.json", {"status": "complete", "rows": run_rows, "count": len(run_rows)})
    write(out / "training_complete.json", {"status": "complete", "formal": list(seeds) == list(SEEDS) and list(parts) == list(PARTITIONS) and list(conditions) == list(CONDITIONS) and list(held_ids) == list(HELD_IDENTITIES) and updates == 600, "probe": "online_newcomer_adaptation", "runs": total, "updates_per_run": updates, "messages": total * updates * 8 * 240, "actions": total * updates * 8 * 240, "new_dino_inferences": 0, "new_private_fits": 0, "seconds": time.monotonic() - started, "source_hashes": hashes, "input_hashes": inputs, "files": {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}})
    return {"status": "complete", "formal": total == len(SEEDS) * len(PARTITIONS) * len(CONDITIONS) * len(HELD_IDENTITIES), "runs": total, "seconds": time.monotonic() - started, "source_sha256": sha(ROOT / "newcomer_adaptation.py")}


def source_modules():
    sys.path.insert(0, str(PROJECT / "redesign_v0.8")); sys.path.insert(0, str(PROJECT / "redesign_v0.21")); sys.path.insert(0, str(PROJECT / "redesign_v0.28")); sys.path.insert(0, str(V034))
    import run_support, support, views, dual, metrics, social_model as communication, world
    return run_support, support, views, dual, metrics, communication, world


_RUN_SUPPORT, _SUPPORT, _VIEWS, _DUAL, _METRICS, _COMMUNICATION, _WORLD = source_modules()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--dev", action="store_true"); parser.add_argument("--updates", type=int, default=600); args = parser.parse_args()
    if args.dev:
        result = run(args.out.resolve(), seeds=(34101,), parts=(1,), conditions=("fixed_A",), held_ids=(0,), updates=min(args.updates, 60))
    else:
        result = run(args.out.resolve(), updates=args.updates)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
