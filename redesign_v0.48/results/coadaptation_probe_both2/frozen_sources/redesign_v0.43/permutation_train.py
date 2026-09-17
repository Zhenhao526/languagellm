"""Formation with all resource permutations and randomized resource-role order."""
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
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
V039 = PROJECT / "redesign_v0.39"
SOURCE = PROJECT / "redesign_v0.28" / "results" / "formation_001"
FEATURE_CACHE = PROJECT / "redesign_v0.28" / "data" / "feature_cache.pt"
SELECTION = PROJECT / "redesign_v0.28" / "data" / "selection.json"

SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
SCHEDULES = ("A", "B", "C")
ROLE_MODES = ("static_role", "random_role")
ROLE_PERMS = tuple(itertools.permutations(range(3)))
ASSIGNMENTS = {"".join(map(str, permutation)): permutation for permutation in ROLE_PERMS}
PRIVATE_TYPES = (0, 1, 0, 1)
VOCAB = 7
RESOURCES = 3
SITES = 6
WIDTH = 96
HISTORY = 18
MESSAGE_TOKENS = 2
UPDATES = 600
CHECKPOINTS = (0, 100, 300, 600)
BATCH = 240
BASELINE = 0.1
ENTROPY_WEIGHT = 0.02
SCHEDULE_NAMESPACE = 43040
FIXTURE_NAMESPACE = 43041
ROLE_NAMESPACE = 43042

# (receiver, resource-0 sender, resource-1 sender, resource-2 sender)
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


def load_modules():
    sys.path.insert(0, str(V039))
    import triad_train as triad

    return triad


T = load_modules()


def source_hashes():
    files = [ROOT / "permutation_train.py", ROOT / "permutation_design.json"]
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
    v039_out = V039 / "results" / "triad_001"
    files = [
        FEATURE_CACHE,
        SELECTION,
        v039_out / "training_complete.json",
        v039_out / "test_worlds.npz",
        *[v039_out / f"train_worlds_p{part}.npz" for part in PARTITIONS],
        PROJECT / "redesign_v0.41" / "results" / "two_token_001" / "completion_manifest.json",
    ]
    for seed in SEEDS:
        files.append(SOURCE / f"prepared_{seed}.pt")
        for part in PARTITIONS:
            for private_type in (0, 1):
                files.append(SOURCE / "private" / f"s{seed}_p{part}_d{private_type}_all" / "final.pt")
    return {str(path.resolve()): sha(path) for path in files}


def remap_worlds(worlds, permutation):
    result = {key: value.copy() for key, value in worlds.items()}
    result["photo_ids"] = worlds["photo_ids"][:, np.asarray(permutation, dtype=np.int64)]
    return result


def fixture(seed, part, slot, step, worlds):
    rng = np.random.default_rng(np.random.SeedSequence([FIXTURE_NAMESPACE, int(seed), int(part), int(slot), int(step), 1]))
    indices = rng.integers(0, len(worlds["map_id"]), size=BATCH, dtype=np.int64)
    uniforms = np.random.default_rng(
        np.random.SeedSequence([FIXTURE_NAMESPACE, int(seed), int(part), int(slot), int(step), 2])
    ).random((BATCH, RESOURCES * MESSAGE_TOKENS + RESOURCES), dtype=np.float32)
    return {"indices": indices, "uniforms": uniforms}


def role_permutation_for(mode, seed, part, step, slot):
    if mode == "static_role":
        return (0, 1, 2)
    if mode != "random_role":
        raise ValueError(mode)
    rng = np.random.default_rng(np.random.SeedSequence([ROLE_NAMESPACE, int(seed), int(part), int(step), int(slot), 77]))
    return tuple(int(x) for x in rng.permutation(RESOURCES))


def replace_actor(agent):
    input_width = RESOURCES * MESSAGE_TOKENS * 16 + 2 + HISTORY
    agent.actor = nn.Sequential(nn.Linear(input_width, WIDTH), nn.Tanh(), nn.Linear(WIDTH, RESOURCES * SITES))
    return agent


def make_population(seed, prepared, part, reset_namespace):
    agents = [replace_actor(agent) for agent in T.load_population(seed, prepared, part)]
    reset_records = []
    for identity, agent in enumerate(agents):
        reset_seed = int(
            np.random.SeedSequence([reset_namespace, int(seed), int(part), int(identity), 901])
            .generate_state(1, dtype=np.uint64)[0]
            >> np.uint64(1)
        )
        reset_records.append(T.reset_communication(agent, reset_seed))
        agent.train()
    return agents, reset_records


def sequence_start(agent, h):
    return T.sender_start(agent, h)


@torch.no_grad()
def sequence_log_probs(agent, h):
    state0, logits0 = sequence_start(agent, h)
    log0 = F.log_softmax(logits0, -1)
    conditional = []
    for prefix in range(VOCAB):
        token = torch.full((len(h),), prefix, dtype=torch.int64)
        state1 = agent.send_recur(agent.send_embedding(token), state0)
        conditional.append(F.log_softmax(agent.send_out(state1), -1))
    log1 = torch.stack(conditional, dim=1)
    joint = (log0[:, :, None] + log1).reshape(len(h), VOCAB * VOCAB)
    token0 = log0.argmax(-1)
    token1 = log1[torch.arange(len(h)), token0].argmax(-1)
    return log0, log1, joint, torch.stack((token0, token1), dim=1)


def receiver_from_sequence(agent, messages):
    flat = messages.reshape(len(messages), RESOURCES * MESSAGE_TOKENS)
    return T.receiver_from_messages(agent, flat)


@torch.no_grad()
def receiver_logits_for_tokens(agent, tokens):
    return receiver_from_sequence(agent, torch.from_numpy(np.asarray(tokens, dtype=np.int64)))


def score_actions(actions, positions, map_id, partition):
    actions = np.asarray(actions)
    positions = np.asarray(positions)
    correct = actions == positions
    rank = {site: i for i, site in enumerate(T.PANELS[partition - 1])}
    train_ids = np.asarray([i for i, triple in enumerate(T.MAPS3) if rank[int(triple[0])] < rank[int(triple[1])]], dtype=np.int64)
    train = np.isin(map_id, train_ids)
    return {
        split: {
            "J": float(np.mean(np.all(correct[mask], axis=-1))),
            "resource0": float(np.mean(correct[mask, 0])),
            "resource1": float(np.mean(correct[mask, 1])),
            "resource2": float(np.mean(correct[mask, 2])),
        }
        for split, mask in (("train60", train), ("target60", ~train), ("all120", np.ones(len(correct), dtype=bool)))
    }


def evaluate(agents, views, worlds, partition, schedule, folder, update):
    logs = {
        resource: {identity: sequence_log_probs(agents[identity], views[identity][resource]["test"]) for identity in range(4)}
        for resource in range(RESOURCES)
    }
    scores = {}
    for slot, team in enumerate(TEAMS[schedule]):
        receiver = team[0]
        role_tokens, role_logits, role_scores = [], [], {}
        for permutation in ROLE_PERMS:
            tokens = np.stack(
                [logs[resource][sender][3].cpu().numpy().astype(np.int64) for resource, sender in zip(permutation, team[1:])], axis=1
            )
            logits = receiver_logits_for_tokens(agents[receiver], tokens).cpu().numpy().astype(np.float32)
            actions = logits.argmax(-1).astype(np.int16)
            role_tokens.append(tokens)
            role_logits.append(logits)
            key = "".join(map(str, permutation))
            role_scores[key] = {
                "literal": score_actions(actions, worlds["positions"], worlds["map_id"], partition),
                "equivariant": score_actions(actions, worlds["positions"][:, np.asarray(permutation)], worlds["map_id"], partition),
            }
        raw = dict(worlds)
        raw.update({"role_permutations": np.asarray(ROLE_PERMS, dtype=np.int64), "tokens": np.stack(role_tokens, axis=0).astype(np.int64), "receiver_logits": np.stack(role_logits, axis=0).astype(np.float32)})
        np.savez_compressed(folder / f"protocol_{schedule}_{update:04d}_team{slot}.npz", **raw)
        scores[f"team{slot}"] = {"role_permutations": role_scores}
    return scores


def two_token_loss(senders, receiver, hs, uniforms, positions):
    uniform_t = torch.from_numpy(uniforms)
    token0s, token1s, sender_logps, sender_entropies, token_probs = [], [], [], [], []
    for resource, (agent, h) in enumerate(zip(senders, hs)):
        state0, logits0 = sequence_start(agent, h)
        token0, logp0, entropy0, prob0 = T._DRAW(logits0, uniform_t[:, resource : resource + 1])
        state1 = agent.send_recur(agent.send_embedding(token0.detach()), state0)
        logits1 = agent.send_out(state1)
        token1, logp1, entropy1, prob1 = T._DRAW(logits1, uniform_t[:, RESOURCES + resource : RESOURCES + resource + 1])
        token0s.append(token0)
        token1s.append(token1)
        sender_logps.append(torch.stack((logp0, logp1), dim=1))
        sender_entropies.append(torch.stack((entropy0, entropy1), dim=1))
        token_probs.append(torch.stack((prob0, prob1), dim=1))
    messages = torch.stack([torch.stack((token0s[r], token1s[r]), dim=1) for r in range(RESOURCES)], dim=1)
    action_logits = receiver_from_sequence(receiver, messages.detach())
    actions, action_logps, action_entropies, action_probs = [], [], [], []
    for resource in range(RESOURCES):
        action, logp, entropy, prob = T._DRAW(action_logits[:, resource], uniform_t[:, 2 * RESOURCES + resource : 2 * RESOURCES + resource + 1])
        actions.append(action)
        action_logps.append(logp)
        action_entropies.append(entropy)
        action_probs.append(prob)
    action_tensor = torch.stack(actions, dim=1)
    success = (action_tensor.detach().numpy() == positions).astype(np.float32)
    reward = (success.sum(axis=1) / np.float32(6.0) + np.float32(0.5) * success.prod(axis=1)).astype(np.float32)
    advantage = torch.from_numpy(reward - np.float32(BASELINE))
    sender_losses = [-(logp.sum(1) * advantage).mean() - ENTROPY_WEIGHT * entropy.sum(1).mean() for logp, entropy in zip(sender_logps, sender_entropies)]
    receiver_loss = -(sum(action_logps) * advantage).mean() - ENTROPY_WEIGHT * sum(action_entropies).mean()
    losses = sender_losses + [receiver_loss]
    if not all(bool(torch.isfinite(loss)) for loss in losses):
        raise ValueError("non-finite permutation loss")
    to_np = lambda tensor: tensor.detach().numpy().copy()
    trace = {
        "uniforms": uniforms.copy(), "messages": to_np(messages), "token_probabilities": to_np(torch.stack(token_probs, dim=1)),
        "action_logits": to_np(action_logits), "action_probabilities": to_np(torch.stack(action_probs, dim=1)), "actions": to_np(action_tensor),
        "positions": positions.copy(), "success": success, "reward": reward, "advantage": to_np(advantage),
        "sender_logp": to_np(torch.stack(sender_logps, dim=1)), "receiver_logp": to_np(torch.stack(action_logps, dim=1).sum(1)),
        "sender_entropy": to_np(torch.stack(sender_entropies, dim=1)), "receiver_entropy": to_np(torch.stack(action_entropies, dim=1).sum(1)),
        "baseline": BASELINE, "entropy_weight": ENTROPY_WEIGHT,
    }
    return sender_losses, receiver_loss, trace


def train_one_step(agents, views_train, train_worlds, seed, part, step, condition, role_mode):
    schedule = schedule_for(condition, seed, part, step)
    selected_losses = [[] for _ in agents]
    fixture_slots, trace_slots, role_slots = {}, {}, {}
    for slot, team in enumerate(TEAMS[schedule]):
        receiver, sender0, sender1, sender2 = team
        role_perm = role_permutation_for(role_mode, seed, part, step, slot)
        fixture_data = fixture(seed, part, slot, step, train_worlds)
        idx = fixture_data["indices"]
        senders = (sender0, sender1, sender2)
        sender_views = [views_train[senders[i]][role_perm[i]]["train"][idx] for i in range(RESOURCES)]
        target_positions = train_worlds["positions"][idx][:, np.asarray(role_perm, dtype=np.int64)]
        sender_losses, receiver_loss, trace = two_token_loss([agents[sender] for sender in senders], agents[receiver], sender_views, fixture_data["uniforms"], target_positions)
        for resource, sender in enumerate(senders):
            selected_losses[sender].append(sender_losses[resource])
        selected_losses[receiver].append(receiver_loss)
        fixture_slots[slot], trace_slots[slot], role_slots[slot] = fixture_data, trace, role_perm
    losses = [sum(role_losses) / len(role_losses) for role_losses in selected_losses]
    return sum(losses), fixture_slots, trace_slots, role_slots, schedule


def schedule_for(condition, seed, part, step):
    if condition == "fixed_A":
        return "A"
    if condition == "rotating_AB":
        return "A" if step % 2 == 0 else "B"
    if condition == "random_ABC":
        return str(np.random.default_rng(np.random.SeedSequence([SCHEDULE_NAMESPACE, int(seed), int(part), int(step), 77])).choice(np.asarray(SCHEDULES)))
    raise ValueError(condition)


def save_trace(folder, step, slot, fixture_data, trace, schedule, team, role_perm, original_positions):
    payload = {
        "schedule_id": np.asarray([schedule]), "global_step": np.asarray([step], dtype=np.int64), "slot": np.asarray([slot], dtype=np.int64),
        "team": np.asarray(team, dtype=np.int64), "role_permutation": np.asarray(role_perm, dtype=np.int64),
        "world__indices": fixture_data["indices"], "world__uniforms": fixture_data["uniforms"], "world__original_positions": original_positions,
    }
    payload.update({f"trace__{key}": value for key, value in trace.items()})
    np.savez_compressed(folder / f"train_{step + 1:04d}_slot{slot}.npz", **payload)


def run(out: Path, seeds=SEEDS, parts=PARTITIONS, assignments=tuple(ASSIGNMENTS), role_modes=ROLE_MODES, conditions=CONDITIONS, updates=UPDATES):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    seeds, parts, assignments, role_modes, conditions = tuple(seeds), tuple(parts), tuple(assignments), tuple(role_modes), tuple(conditions)
    formal = seeds == SEEDS and parts == PARTITIONS and assignments == tuple(ASSIGNMENTS) and role_modes == ROLE_MODES and conditions == CONDITIONS and updates == UPDATES
    checkpoints = sorted(set(t for t in CHECKPOINTS if t <= updates) | {updates})
    hashes, inputs = source_hashes(), input_hashes()
    write(out / "invocation.json", {
        "formal": formal, "version": "v0.43-permutation-formation", "seeds": list(seeds), "partitions": list(parts),
        "assignments": {name: list(ASSIGNMENTS[name]) for name in assignments}, "role_modes": list(role_modes),
        "role_permutations": [list(p) for p in ROLE_PERMS], "conditions": list(conditions), "schedules": list(SCHEDULES),
        "resources": RESOURCES, "tokens_per_sender": MESSAGE_TOKENS, "message_length": RESOURCES * MESSAGE_TOKENS, "vocabulary": VOCAB,
        "updates": updates, "checkpoints": checkpoints, "batch": BATCH, "schedule_namespace": SCHEDULE_NAMESPACE,
        "fixture_namespace": FIXTURE_NAMESPACE, "role_namespace": ROLE_NAMESPACE, "source_hashes": hashes, "input_hashes": inputs,
        "torch_version": str(torch.__version__), "numpy_version": np.__version__, "platform": platform.platform(), "threads": 1,
        "new_private_fits": 0, "new_dino_inferences": 0, "communication_initialization": "fresh_random_reset_all_communication_modules",
        "study_scope": "formation under all resource permutations and static versus randomized resource-role order",
    })
    for path in hashes:
        target = out / "frozen_sources" / Path(path).relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    bank = T.TriBank()
    worlds_train, worlds_test = {}, {}
    for part in parts:
        base_train = npz(V039 / "results" / "triad_001" / f"train_worlds_p{part}.npz")
        base_test = npz(V039 / "results" / "triad_001" / "test_worlds.npz")
        worlds_train[part], worlds_test[part] = {}, {}
        for assignment in assignments:
            worlds_train[part][assignment] = remap_worlds(base_train, ASSIGNMENTS[assignment])
            worlds_test[part][assignment] = remap_worlds(base_test, ASSIGNMENTS[assignment])
            np.savez_compressed(out / f"train_worlds_p{part}_{assignment}.npz", **worlds_train[part][assignment])
            np.savez_compressed(out / f"test_worlds_{assignment}.npz", **worlds_test[part][assignment])

    prepared_cache = {seed: torch.load(SOURCE / f"prepared_{seed}.pt", weights_only=True) for seed in seeds}
    views_cache = {}
    for seed, part, assignment in itertools.product(seeds, parts, assignments):
        base_agents = T.load_population(seed, prepared_cache[seed], part)
        views_cache[seed, part, assignment] = T.load_views(base_agents, bank, {"train": worlds_train[part][assignment], "test": worlds_test[part][assignment]})

    rows, started, total = [], time.monotonic(), 0
    condition_index = {condition: i for i, condition in enumerate(CONDITIONS)}
    assignment_index = {assignment: i for i, assignment in enumerate(ASSIGNMENTS)}
    role_index = {mode: i for i, mode in enumerate(ROLE_MODES)}
    for seed, part, assignment, role_mode, condition in itertools.product(seeds, parts, assignments, role_modes, conditions):
        reset_namespace = 43043 + assignment_index[assignment] * 1000 + role_index[role_mode] * 100 + condition_index[condition]
        agents, reset_records = make_population(seed, prepared_cache[seed], part, reset_namespace)
        views = views_cache[seed, part, assignment]
        train_table, test_table = worlds_train[part][assignment], worlds_test[part][assignment]
        chain = out / "social" / f"s{seed}_p{part}_{assignment}_{role_mode}_{condition}"
        chain.mkdir(parents=True, exist_ok=True)
        write(chain / "config.json", {
            "seed": seed, "partition": part, "assignment": assignment, "resource_permutation": list(ASSIGNMENTS[assignment]), "role_mode": role_mode,
            "condition": condition, "resources": RESOURCES, "tokens_per_sender": MESSAGE_TOKENS, "message_length": RESOURCES * MESSAGE_TOKENS, "vocabulary": VOCAB,
            "role_permutations": [list(p) for p in ROLE_PERMS], "updates": updates, "checkpoints": checkpoints,
            "training_schedule": {"fixed_A": "A_every_update", "rotating_AB": "A_even_B_odd", "random_ABC": "deterministic_random_A_B_C_namespace_43040"}[condition],
            "role_schedule": "identity_every_team" if role_mode == "static_role" else "deterministic_random_per_team_update_namespace_43042",
            "evaluation_schedules": list(SCHEDULES), "communication_initialization": "fresh_random_reset_all_communication_modules", "reset_records": reset_records,
            "train_map_count": 60, "test_map_count": 120,
        })
        optimizer_params = [parameter for agent in agents for group in T.trainable_groups(agent).values() for parameter in group]
        optimizer = torch.optim.Adam(optimizer_params, lr=0.0007)
        curve = []
        for step in range(updates + 1):
            if step in checkpoints:
                for agent in agents:
                    agent.eval()
                scores = {schedule: evaluate(agents, views, test_table, part, schedule, chain, step) for schedule in SCHEDULES}
                curve.append({"update": step, "scores": scores})
                write(chain / "curve.json", curve)
                for agent in agents:
                    agent.train()
                if step == updates:
                    torch.save([copy.deepcopy(agent.state_dict()) for agent in agents], chain / "final.pt")
            if step == updates:
                break
            loss, fixture_slots, trace_slots, role_slots, schedule = train_one_step(agents, views, train_table, seed, part, step, condition, role_mode)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            groups = {}
            for identity, agent in enumerate(agents):
                for role, group in T.trainable_groups(agent).items():
                    groups[f"agent{identity}_{role}"] = group
            norms = {role: float(torch.nn.utils.clip_grad_norm_(group, 2.0)) for role, group in groups.items()}
            if not all(np.isfinite(value) for value in norms.values()):
                raise ValueError("non-finite permutation gradient")
            optimizer.step()
            if step in (0, updates - 1):
                for slot, team in enumerate(TEAMS[schedule]):
                    idx = fixture_slots[slot]["indices"]
                    save_trace(chain, step, slot, fixture_slots[slot], trace_slots[slot], schedule, team, role_slots[slot], train_table["positions"][idx])
            with (chain / "training.jsonl").open("a") as log:
                log.write(json.dumps({"update": step + 1, "global_step": step, "schedule": schedule, "role_mode": role_mode, "role_permutations": {str(slot): list(role_slots[slot]) for slot in role_slots}, "loss": float(loss.detach()), "norms": norms}, ensure_ascii=False) + "\n")
        rows.append({"seed": seed, "partition": part, "assignment": assignment, "role_mode": role_mode, "condition": condition, "curve": curve})
        total += 1
        print(json.dumps({"phase": "permutation_formation", "chain": chain.name, "runs": total, "seconds": round(time.monotonic() - started, 2)}, ensure_ascii=False), flush=True)

    write(out / "runs.json", {"status": "complete", "rows": rows, "count": total})
    write(out / "training_complete.json", {
        "status": "complete", "formal": formal, "probe": "permutation_formation", "runs": total, "updates_per_run": updates,
        "role_modes": list(role_modes), "assignments": list(assignments), "messages": total * updates * 4 * BATCH,
        "token_instances": total * updates * 4 * BATCH * RESOURCES * MESSAGE_TOKENS, "actions": total * updates * 4 * BATCH * RESOURCES,
        "protocol_tables": total * len(checkpoints) * len(SCHEDULES) * 4, "trace_files_expected": total * 2 * 4, "trace_rows_expected": total * 2 * 4 * BATCH,
        "seconds": time.monotonic() - started, "source_hashes": hashes, "input_hashes": inputs,
        "files": {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()},
    })
    return {"status": "complete", "formal": formal, "runs": total, "seconds": time.monotonic() - started, "source_sha256": sha(ROOT / "permutation_train.py")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--updates", type=int, default=UPDATES)
    args = parser.parse_args()
    if args.dev:
        result = run(args.out.resolve(), seeds=(SEEDS[0],), parts=(PARTITIONS[0],), assignments=("012",), role_modes=(ROLE_MODES[0],), conditions=(CONDITIONS[0],), updates=min(args.updates, 30))
    else:
        result = run(args.out.resolve(), updates=args.updates)
    print(json.dumps(result, ensure_ascii=False))
