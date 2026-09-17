"""Train reset newcomers with a bounded amount of resident co-adaptation.

The v0.47 batch held the resident population fixed.  This batch keeps the
hardest newcomer intervention (both communication sides reset), extends the
adaptation horizon to 600 updates, and lets residents update communication
parameters on at most every twentieth step.  Sparse resident updates make the
amount of social plasticity an explicit, auditable budget.
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
V043 = PROJECT / "redesign_v0.43"
V039 = PROJECT / "redesign_v0.39"
SOURCE = PROJECT / "redesign_v0.28" / "results" / "formation_001"
FEATURE_CACHE = PROJECT / "redesign_v0.28" / "data" / "feature_cache.pt"
SELECTION = PROJECT / "redesign_v0.28" / "data" / "selection.json"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
ASSIGNMENTS = ("012", "021", "102", "120", "201", "210")
ROLE_MODES = ("static_role", "random_role")
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
CULTURES = tuple(f"{mode}__{condition}" for mode in ROLE_MODES for condition in CONDITIONS)
SCHEDULES = ("A", "B", "C")
ROLE_PERMS = tuple(itertools.permutations(range(3)))
ROLE_KEYS = tuple("".join(map(str, p)) for p in ROLE_PERMS)
UPDATES = 600
CHECKPOINTS = (0, 100, 300, 600)
BATCH = 240
TEST_BATCH = 120
RESOURCES, TOKENS, VOCAB, SITES = 3, 2, 7, 6
ADAPTATION_SCHEDULE = "random_ABC"
NEWCOMER_RESET_MODE = "both"
RESIDENT_ADAPTATION_MODES = (
    "resident_frozen",
    "resident_sender_sparse",
    "resident_receiver_sparse",
    "resident_both_sparse",
)
RESIDENT_UPDATE_INTERVAL = 20
RESIDENT_UPDATE_BUDGET = 30
NEWCOMER_LR = 0.0007
RESIDENT_LR = 0.0002
PRIVATE_TYPES = (0, 1, 0, 1)
SCHEDULE_NAMESPACE, FIXTURE_NAMESPACE, ROLE_NAMESPACE, RESET_NAMESPACE = 48040, 48041, 48042, 48043

COMMUNICATION_MODULES = (
    "send_context", "send_embedding", "send_recur", "send_out",
    "receive_embedding", "actor",
)
RESET_GROUPS = {"both": COMMUNICATION_MODULES}
RESIDENT_GROUPS = {
    "resident_frozen": (),
    "resident_sender_sparse": ("sender",),
    "resident_receiver_sparse": ("receiver",),
    "resident_both_sparse": ("sender", "receiver"),
}


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def train_ids(partition):
    maps = np.asarray(list(itertools.permutations(range(SITES), RESOURCES)), dtype=np.int64)
    panels = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
    rank = {site: i for i, site in enumerate(panels[partition - 1])}
    return np.asarray([i for i, triple in enumerate(maps) if rank[int(triple[0])] < rank[int(triple[1])]], dtype=np.int64)


def score_actions(actions, positions, map_id, partition):
    correct = np.asarray(actions) == np.asarray(positions)
    train = np.isin(map_id, train_ids(partition))
    return {
        split: {
            "J": float(np.mean(np.all(correct[mask], axis=-1))),
            "resource0": float(np.mean(correct[mask, 0])),
            "resource1": float(np.mean(correct[mask, 1])),
            "resource2": float(np.mean(correct[mask, 2])),
        }
        for split, mask in (("train60", train), ("target60", ~train), ("all120", np.ones(len(correct), dtype=bool)))
    }


def schedule_for(condition, seed, part, step):
    if condition == "fixed_A":
        return "A"
    if condition == "rotating_AB":
        return "A" if step % 2 == 0 else "B"
    if condition == "random_ABC":
        return str(np.random.default_rng(np.random.SeedSequence([SCHEDULE_NAMESPACE, seed, part, step, 77])).choice(np.asarray(SCHEDULES)))
    raise ValueError(condition)


def role_for(mode, seed, part, step, slot):
    if mode == "static_role":
        return (0, 1, 2)
    return tuple(int(x) for x in np.random.default_rng(np.random.SeedSequence([ROLE_NAMESPACE, seed, part, step, slot, 77])).permutation(RESOURCES))


def fixture(seed, part, slot, step, worlds):
    indices = np.random.default_rng(np.random.SeedSequence([FIXTURE_NAMESPACE, seed, part, slot, step, 1])).integers(0, len(worlds["map_id"]), size=BATCH, dtype=np.int64)
    uniforms = np.random.default_rng(np.random.SeedSequence([FIXTURE_NAMESPACE, seed, part, slot, step, 2])).random((BATCH, RESOURCES * TOKENS + RESOURCES), dtype=np.float32)
    return {"indices": indices, "uniforms": uniforms}


def reset_seed(seed, part, assignment_index, culture_index):
    return int(np.random.SeedSequence([RESET_NAMESPACE, seed, part, assignment_index, culture_index, 901]).generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def load_base(permutation, seed, prepared, part):
    return permutation.T.load_population(seed, prepared, part)


def load_resident(permutation, base_agents, chain):
    agents = [copy.deepcopy(agent) for agent in base_agents]
    for agent in agents:
        permutation.replace_actor(agent)
    states = torch.load(chain / "final.pt", weights_only=True)
    for agent, state in zip(agents, states):
        agent.load_state_dict(state)
        agent.eval()
    return agents


def reset_communication_parts(agent, seed, reset_mode):
    if reset_mode not in RESET_GROUPS:
        raise ValueError(reset_mode)
    names = RESET_GROUPS[reset_mode]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for name in names:
            for module in getattr(agent, name).modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
    agent.requires_grad_(False)
    for name in names:
        for parameter in getattr(agent, name).parameters():
            parameter.requires_grad_(True)
    for parameter in agent.parameters():
        parameter.grad = None
    return {
        "seed": int(seed),
        "reset_mode": reset_mode,
        "reset_modules": list(names),
        "trainable_parameters": sum(parameter.numel() for name in names for parameter in getattr(agent, name).parameters()),
    }


def make_newcomer(permutation, resident_agent, seed, part, assignment_index, culture_index):
    newcomer = copy.deepcopy(resident_agent)
    reset = reset_seed(seed, part, assignment_index, culture_index)
    reset_record = reset_communication_parts(newcomer, reset, NEWCOMER_RESET_MODE)
    newcomer.train()
    return newcomer, reset_record


def communication_reference(agent):
    return {name: [p.detach().clone() for p in getattr(agent, name).parameters()] for name in COMMUNICATION_MODULES}


def communication_drift(agent, reference):
    numerator = 0.0
    denominator = 0.0
    for name in COMMUNICATION_MODULES:
        for parameter, initial in zip(getattr(agent, name).parameters(), reference[name]):
            delta = parameter.detach() - initial
            numerator += float(torch.sum(delta * delta))
            denominator += float(torch.sum(initial * initial))
    return float(np.sqrt(numerator / max(denominator, 1e-12)))


def evaluate_compact(permutation, agents, views, worlds, partition, folder, update, resident_references):
    logs = {resource: {identity: permutation.sequence_log_probs(agents[identity], views[identity][resource]["test"]) for identity in range(4)} for resource in range(RESOURCES)}
    scores = {}
    for schedule in SCHEDULES:
        scores[schedule] = {}
        for slot, team in enumerate(permutation.T.TEAMS[schedule]):
            receiver = team[0]
            role_tokens, role_actions, role_scores = [], [], {}
            for role_perm in ROLE_PERMS:
                tokens = np.stack([logs[resource][sender][3].cpu().numpy().astype(np.int64) for resource, sender in zip(role_perm, team[1:])], axis=1)
                logits = permutation.receiver_logits_for_tokens(agents[receiver], tokens).cpu().numpy().astype(np.float32)
                actions = logits.argmax(-1).astype(np.int16)
                role_tokens.append(tokens)
                role_actions.append(actions)
                key = "".join(map(str, role_perm))
                role_scores[key] = {"literal": score_actions(actions, worlds["positions"], worlds["map_id"], partition), "equivariant": score_actions(actions, worlds["positions"][:, np.asarray(role_perm)], worlds["map_id"], partition)}
            raw = {"map_id": worlds["map_id"], "photo_ids": worlds["photo_ids"], "positions": worlds["positions"], "shown": worlds["shown"], "role_permutations": np.asarray(ROLE_PERMS, dtype=np.int64), "tokens": np.stack(role_tokens).astype(np.int64), "actions": np.stack(role_actions).astype(np.int16)}
            np.savez_compressed(folder / f"protocol_{schedule}_{update:04d}_team{slot}.npz", **raw)
            scores[schedule][f"team{slot}"] = {"role_permutations": role_scores}
    drifts = [communication_drift(agents[identity], resident_references[identity]) for identity in (1, 2, 3)]
    return scores, {"resident_communication_drift_mean": float(np.mean(drifts)), "resident_communication_drift_max": float(np.max(drifts)), "resident_communication_drift_by_identity": drifts}


def adapt_step(permutation, agents, views_train, worlds, seed, part, step, role_mode, resident_mode):
    schedule = schedule_for(ADAPTATION_SCHEDULE, seed, part, step)
    losses_by_identity = {identity: {"sender": [], "receiver": []} for identity in range(4)}
    fixtures, traces, roles = {}, {}, {}
    for slot, team in enumerate(permutation.T.TEAMS[schedule]):
        role_perm = role_for(role_mode, seed, part, step, slot)
        fixture_data = fixture(seed, part, slot, step, worlds)
        idx = fixture_data["indices"]
        senders = team[1:]
        sender_views = [views_train[senders[i]][role_perm[i]]["train"][idx] for i in range(RESOURCES)]
        target_positions = worlds["positions"][idx][:, np.asarray(role_perm, dtype=np.int64)]
        sender_losses, receiver_loss, trace = permutation.two_token_loss([agents[sender] for sender in senders], agents[team[0]], sender_views, fixture_data["uniforms"], target_positions)
        for resource, sender in enumerate(senders):
            losses_by_identity[sender]["sender"].append(sender_losses[resource])
        losses_by_identity[team[0]]["receiver"].append(receiver_loss)
        fixtures[slot], traces[slot], roles[slot] = fixture_data, trace, role_perm
    newcomer_selected = list(losses_by_identity[0]["sender"]) + list(losses_by_identity[0]["receiver"])
    newcomer_loss = sum(newcomer_selected) / len(newcomer_selected)
    resident_step = step % RESIDENT_UPDATE_INTERVAL == 0 and step // RESIDENT_UPDATE_INTERVAL < RESIDENT_UPDATE_BUDGET
    resident_selected = []
    if resident_step:
        resident_roles = RESIDENT_GROUPS[resident_mode]
        for identity in (1, 2, 3):
            for role in resident_roles:
                resident_selected.extend(losses_by_identity[identity][role])
    # Keep the newcomer gradient scale identical across modes.  Resident
    # losses are averaged separately and added only on their sparse update
    # steps, so the comparison changes social plasticity rather than silently
    # reducing newcomer learning rate.
    loss = newcomer_loss
    if resident_selected:
        loss = loss + sum(resident_selected) / len(resident_selected)
    return loss, fixtures, traces, roles, schedule, resident_step


def set_requires_grad(agents, resident_mode):
    for agent in agents:
        agent.requires_grad_(False)
    for name in COMMUNICATION_MODULES:
        for parameter in getattr(agents[0], name).parameters():
            parameter.requires_grad_(True)
    for identity in (1, 2, 3):
        groups = permutation_groups(agents[identity])
        for role in RESIDENT_GROUPS[resident_mode]:
            for parameter in groups[role]:
                parameter.requires_grad_(True)


def permutation_groups(agent):
    return {
        "sender": [p for name in COMMUNICATION_MODULES[:4] for p in getattr(agent, name).parameters()],
        "receiver": [p for name in COMMUNICATION_MODULES[4:] for p in getattr(agent, name).parameters()],
    }


def make_optimizer(agents, resident_mode, permutation):
    param_groups, named_groups = [], []
    newcomer_groups = permutation.T.trainable_groups(agents[0])
    for role in ("sender", "receiver"):
        params = newcomer_groups[role]
        param_groups.append({"params": params, "lr": NEWCOMER_LR})
        named_groups.append((f"newcomer_{role}", params))
    for identity in (1, 2, 3):
        groups = permutation.T.trainable_groups(agents[identity])
        for role in RESIDENT_GROUPS[resident_mode]:
            params = groups[role]
            param_groups.append({"params": params, "lr": RESIDENT_LR})
            named_groups.append((f"resident{identity}_{role}", params))
    return torch.optim.Adam(param_groups), named_groups


def save_trace(folder, step, slot, fixture_data, trace, schedule, team, role_perm, original_positions):
    payload = {"schedule_id": np.asarray([schedule]), "global_step": np.asarray([step], dtype=np.int64), "slot": np.asarray([slot], dtype=np.int64), "team": np.asarray(team, dtype=np.int64), "role_permutation": np.asarray(role_perm, dtype=np.int64), "world__indices": fixture_data["indices"], "world__uniforms": fixture_data["uniforms"], "world__original_positions": original_positions}
    payload.update({f"trace__{key}": value for key, value in trace.items()})
    np.savez_compressed(folder / f"train_{step + 1:04d}_slot{slot}.npz", **payload)


def run(out: Path, seeds=SEEDS, parts=PARTITIONS, assignments=ASSIGNMENTS, cultures=CULTURES, resident_modes=RESIDENT_ADAPTATION_MODES, updates=UPDATES):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    if str(V043) not in sys.path:
        sys.path.insert(0, str(V043))
    import permutation_train as permutation
    torch.set_num_threads(1)
    seeds, parts, assignments, cultures, resident_modes = tuple(seeds), tuple(parts), tuple(assignments), tuple(cultures), tuple(resident_modes)
    formal = seeds == SEEDS and parts == PARTITIONS and assignments == ASSIGNMENTS and cultures == CULTURES and resident_modes == RESIDENT_ADAPTATION_MODES and updates == UPDATES
    checkpoints = sorted(set(t for t in CHECKPOINTS if t <= updates) | {updates})
    hashes, inputs = source_hashes(), input_hashes()
    write(out / "invocation.json", {"formal": formal, "version": "v0.48-coadaptation-horizon", "seeds": list(seeds), "partitions": list(parts), "assignments": list(assignments), "cultures": list(cultures), "role_modes": list(ROLE_MODES), "conditions": list(CONDITIONS), "adaptation_schedule": ADAPTATION_SCHEDULE, "newcomer_reset_mode": NEWCOMER_RESET_MODE, "resident_adaptation_modes": list(resident_modes), "resident_update_interval": RESIDENT_UPDATE_INTERVAL, "resident_update_budget": RESIDENT_UPDATE_BUDGET, "newcomer_learning_rate": NEWCOMER_LR, "resident_learning_rate": RESIDENT_LR, "updates": updates, "checkpoints": checkpoints, "newcomer_identity": 0, "study_scope": "600-update bounded resident co-adaptation after a full newcomer communication reset", "source_hashes": hashes, "input_hashes": inputs})
    for path in hashes:
        target = out / "frozen_sources" / Path(path).relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    prepared = {seed: torch.load(SOURCE / f"prepared_{seed}.pt", weights_only=True) for seed in seeds}
    bank = permutation.T.TriBank()
    started, total = time.monotonic(), 0
    assignment_index = {assignment: i for i, assignment in enumerate(ASSIGNMENTS)}
    culture_index = {culture: i for i, culture in enumerate(CULTURES)}
    rows = []
    for seed, part, assignment, culture_label, resident_mode in itertools.product(seeds, parts, assignments, cultures, resident_modes):
        mode, condition = culture_label.split("__", 1)
        test_worlds = npz(V043 / "results" / "permutation_001" / f"test_worlds_{assignment}.npz")
        train_worlds = npz(V043 / "results" / "permutation_001" / f"train_worlds_p{part}_{assignment}.npz")
        base_agents = load_base(permutation, seed, prepared[seed], part)
        views = permutation.T.load_views(base_agents, bank, {"train": train_worlds, "test": test_worlds})
        resident_chain = V043 / "results" / "permutation_001" / "social" / f"s{seed}_p{part}_{assignment}_{mode}_{condition}"
        agents = load_resident(permutation, base_agents, resident_chain)
        resident_references = {identity: communication_reference(agents[identity]) for identity in (1, 2, 3)}
        newcomer, reset_record = make_newcomer(permutation, agents[0], seed, part, assignment_index[assignment], culture_index[culture_label])
        agents[0] = newcomer
        set_requires_grad(agents, resident_mode)
        optimizer, named_groups = make_optimizer(agents, resident_mode, permutation)
        chain = out / "social" / f"s{seed}_p{part}_{assignment}_{culture_label}__resident_{resident_mode}"
        chain.mkdir(parents=True, exist_ok=True)
        write(chain / "config.json", {"seed": seed, "partition": part, "assignment": assignment, "resident_culture": culture_label, "resident_role_mode": mode, "resident_condition": condition, "adaptation_schedule": ADAPTATION_SCHEDULE, "adaptation_role_mode": mode, "newcomer_identity": 0, "newcomer_reset_mode": NEWCOMER_RESET_MODE, "resident_adaptation_mode": resident_mode, "resident_update_interval": RESIDENT_UPDATE_INTERVAL, "resident_update_budget": RESIDENT_UPDATE_BUDGET, "newcomer_learning_rate": NEWCOMER_LR, "resident_learning_rate": RESIDENT_LR, "updates": updates, "checkpoints": checkpoints, "communication_initialization": "resident_endpoint_full_newcomer_communication_reset", "reset_record": reset_record, "evaluation_schedules": list(SCHEDULES), "train_map_count": 60, "test_map_count": 120})
        curve = []
        for step in range(updates + 1):
            if step in checkpoints:
                for agent in agents:
                    agent.eval()
                scores, drift = evaluate_compact(permutation, agents, views, test_worlds, part, chain, step, resident_references)
                curve.append({"update": step, "scores": scores, "resident_drift": drift})
                write(chain / "curve.json", curve)
                for agent in agents:
                    agent.train()
                if step == updates:
                    torch.save(agents[0].state_dict(), chain / "newcomer_final.pt")
                    for identity in (1, 2, 3):
                        torch.save(agents[identity].state_dict(), chain / f"resident_{identity}_final.pt")
            if step == updates:
                break
            loss, fixture_slots, trace_slots, role_slots, schedule, resident_step = adapt_step(permutation, agents, views, train_worlds, seed, part, step, mode, resident_mode)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norms = {}
            for name, params in named_groups:
                active = [p for p in params if p.grad is not None]
                norms[name] = float(torch.nn.utils.clip_grad_norm_(active, 2.0)) if active else 0.0
            if not all(np.isfinite(value) for value in norms.values()):
                raise ValueError("non-finite adaptation gradient")
            optimizer.step()
            if step in (0, updates - 1):
                for slot, team in enumerate(permutation.T.TEAMS[schedule]):
                    idx = fixture_slots[slot]["indices"]
                    save_trace(chain, step, slot, fixture_slots[slot], trace_slots[slot], schedule, team, role_slots[slot], train_worlds["positions"][idx])
            with (chain / "training.jsonl").open("a") as log:
                log.write(json.dumps({"update": step + 1, "global_step": step, "schedule": schedule, "role_mode": mode, "resident_adaptation_mode": resident_mode, "resident_update": resident_step, "role_permutations": {str(slot): list(role_slots[slot]) for slot in role_slots}, "loss": float(loss.detach()), "norms": norms}, ensure_ascii=False) + "\n")
        rows.append({"seed": seed, "partition": part, "assignment": assignment, "resident_culture": culture_label, "resident_adaptation_mode": resident_mode, "file": str(chain.relative_to(out))})
        total += 1
        print(json.dumps({"phase": "coadaptation_adaptation", "chain": chain.name, "runs": total, "seconds": round(time.monotonic() - started, 2)}, ensure_ascii=False), flush=True)
    write(out / "runs.json", {"status": "complete", "formal": formal, "rows": rows, "count": total})
    files = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}
    write(out / "training_complete.json", {"status": "complete", "formal": formal, "probe": "coadaptation_horizon", "runs": total, "updates_per_run": updates, "resident_cultures": list(cultures), "resident_adaptation_modes": list(resident_modes), "adaptation_schedule": ADAPTATION_SCHEDULE, "newcomer_reset_mode": NEWCOMER_RESET_MODE, "checkpoints": checkpoints, "trace_files_expected": total * 8, "protocol_files_expected": total * len(checkpoints) * len(SCHEDULES) * 4, "resident_update_interval": RESIDENT_UPDATE_INTERVAL, "resident_update_budget": RESIDENT_UPDATE_BUDGET, "seconds": time.monotonic() - started, "source_hashes": hashes, "input_hashes": inputs, "files": files})
    return {"status": "complete", "formal": formal, "runs": total, "seconds": time.monotonic() - started, "source_sha256": sha(ROOT / "coadaptation_train.py")}


def source_hashes():
    paths = [ROOT / "coadaptation_train.py", ROOT / "coadaptation_design.json", V043 / "permutation_train.py", V043 / "permutation_design.json", V039 / "triad_train.py", V039 / "triad_design.json"]
    paths += [PROJECT / name for name in ("redesign_v0.8/camp.py", "redesign_v0.20/temporal_model.py", "redesign_v0.20/temporal_world.py", "redesign_v0.28/world.py", "redesign_v0.21/social_model.py")]
    return {str(path.resolve()): sha(path) for path in paths}


def input_hashes():
    v043 = V043 / "results" / "permutation_001"
    paths = [FEATURE_CACHE, SELECTION, v043 / "completion_manifest.json", v043 / "training_complete.json", v043 / "permutation_analysis.json", *(v043 / f"test_worlds_{a}.npz" for a in ASSIGNMENTS), *(v043 / f"train_worlds_p{p}_{a}.npz" for p in PARTITIONS for a in ASSIGNMENTS), *(SOURCE / f"prepared_{s}.pt" for s in SEEDS)]
    return {str(path.resolve()): sha(path) for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    if args.dev:
        result = run(args.out.resolve(), seeds=(SEEDS[0],), parts=(PARTITIONS[0],), assignments=(ASSIGNMENTS[0],), cultures=(CULTURES[0],), resident_modes=(RESIDENT_ADAPTATION_MODES[0],), updates=min(args.updates, 5))
    else:
        result = run(args.out.resolve(), updates=args.updates)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
