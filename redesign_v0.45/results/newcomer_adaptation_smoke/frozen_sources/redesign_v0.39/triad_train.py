"""Teacher-free formation with three resource-specific senders and one receiver."""
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
SOURCE = PROJECT / "redesign_v0.28" / "results" / "formation_001"
PRIVATE_SOURCE = SOURCE / "private"
FEATURE_CACHE = PROJECT / "redesign_v0.28" / "data" / "feature_cache.pt"
SELECTION = PROJECT / "redesign_v0.28" / "data" / "selection.json"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
SCHEDULES = ("A", "B", "C")
PRIVATE_TYPES = (0, 1, 0, 1)
RESOURCE_STRATA = ("apple", "banana", "orange")
VOCAB = 7
RESOURCES = 3
SITES = 6
WIDTH = 96
HISTORY = 18
UPDATES = 1200
CHECKPOINTS = (0, 100, 600, 1200)
BATCH = 240
BASELINE = 0.1
ENTROPY_WEIGHT = 0.02
MAPS3 = np.asarray(list(itertools.permutations(range(SITES), RESOURCES)), dtype=np.int64)
PANELS = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
TEAMS = {
    "A": ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)),
    "B": ((0, 2, 1, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0)),
    "C": ((0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0)),
}


def write(path: Path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def npz(path: Path):
    with np.load(path, allow_pickle=False) as z: return {key: z[key] for key in z.files}


def load_modules():
    sys.path.insert(0, str(PROJECT / "redesign_v0.8")); sys.path.insert(0, str(PROJECT / "redesign_v0.20")); sys.path.insert(0, str(PROJECT / "redesign_v0.28")); sys.path.insert(0, str(PROJECT / "redesign_v0.21"))
    import camp
    import temporal_model
    import world
    from social_model import _draw
    return camp, temporal_model, world, _draw


_CAMP, _TEMPORAL, _WORLD, _DRAW = load_modules()


def source_hashes():
    files = [ROOT / "triad_train.py", ROOT / "triad_design.json"]
    files += [PROJECT / name for name in ("redesign_v0.8/camp.py", "redesign_v0.20/temporal_model.py", "redesign_v0.20/temporal_world.py", "redesign_v0.28/world.py", "redesign_v0.21/social_model.py")]
    return {str(path.resolve()): sha(path) for path in files}


def input_hashes():
    files = [FEATURE_CACHE, SELECTION]
    for seed in SEEDS:
        files.append(SOURCE / f"prepared_{seed}.pt")
        for part in PARTITIONS:
            for private_type in (0, 1): files.append(PRIVATE_SOURCE / f"s{seed}_p{part}_d{private_type}_all" / "final.pt")
    return {str(path.resolve()): sha(path) for path in files}


class TriBank:
    def __init__(self):
        base = _WORLD.NonSquareImageBank(FEATURE_CACHE, SELECTION)
        self.features = base.features
        self.entries = base.entries
        self.pools = {(split, stratum): np.asarray([i for i, row in enumerate(self.entries) if row["split"] == split and row["stratum"] == stratum], dtype=np.int64) for split in ("train", "test") for stratum in RESOURCE_STRATA}
        if any(len(self.pools[split, stratum]) != (2 if split == "train" else 1) for split in ("train", "test") for stratum in RESOURCE_STRATA):
            raise ValueError("three resource strata require 2 train and 1 test image each")


def train_map_ids(partition):
    rank = {site: i for i, site in enumerate(PANELS[partition - 1])}
    return np.asarray([i for i, triple in enumerate(MAPS3) if rank[int(triple[0])] < rank[int(triple[1])]], dtype=np.int64)


def table(bank: TriBank, split: str, map_ids=None):
    if split not in ("train", "test"): raise ValueError(split)
    chosen = np.arange(len(MAPS3), dtype=np.int64) if map_ids is None else np.asarray(map_ids, dtype=np.int64)
    photos = np.asarray(list(itertools.product(*[np.sort(bank.pools[split, stratum]) for stratum in RESOURCE_STRATA])), dtype=np.int64)
    map_id = np.repeat(chosen, len(photos)); photo_ids = np.tile(photos, (len(chosen), 1)); positions = np.repeat(MAPS3[chosen], len(photos), axis=0)
    return {"map_id": map_id, "photo_ids": photo_ids, "positions": positions, "shown": np.zeros(len(map_id), dtype=np.int64)}


def fixture(seed, part, slot, step, worlds):
    rng = np.random.default_rng(np.random.SeedSequence([39039, int(seed), int(part), int(slot), int(step), 1]))
    indices = rng.integers(0, len(worlds["map_id"]), size=BATCH, dtype=np.int64)
    uniforms = np.random.default_rng(np.random.SeedSequence([39039, int(seed), int(part), int(slot), int(step), 2])).random((BATCH, 2 * RESOURCES), dtype=np.float32)
    return {"indices": indices, "uniforms": uniforms}


def make_agent(seed, prepared, private_type):
    templates = _CAMP.remake_agents(seed, prepared, VOCAB, RESOURCES, "identity")
    agent = copy.deepcopy(templates[private_type])
    blob = torch.load(PRIVATE_SOURCE / f"s{seed}_p{1}_d{private_type}_all" / "final.pt", weights_only=True)
    # The private endpoint's perceptual frontend is shared with the two-resource
    # experiment. Communication modules are deliberately excluded and reset below.
    state = blob["agent"]
    keep = {key: value for key, value in state.items() if key == "input_transform" or key.startswith(("project.", "memory.", "slot_phi."))}
    agent.load_state_dict(keep, strict=False)
    agent.actor = nn.Sequential(nn.Linear(RESOURCES * 16 + 2 + HISTORY, WIDTH), nn.Tanh(), nn.Linear(WIDTH, RESOURCES * SITES))
    return agent


def load_population(seed, prepared, part):
    agents = []
    for identity, private_type in enumerate(PRIVATE_TYPES):
        templates = _CAMP.remake_agents(seed, prepared, VOCAB, RESOURCES, "identity")
        agent = copy.deepcopy(templates[private_type])
        blob = torch.load(PRIVATE_SOURCE / f"s{seed}_p{part}_d{private_type}_all" / "final.pt", weights_only=True)
        state = blob["agent"]; keep = {key: value for key, value in state.items() if key == "input_transform" or key.startswith(("project.", "memory.", "slot_phi."))}
        agent.load_state_dict(keep, strict=False)
        agent.actor = nn.Sequential(nn.Linear(RESOURCES * 16 + 2 + HISTORY, WIDTH), nn.Tanh(), nn.Linear(WIDTH, RESOURCES * SITES))
        agents.append(agent)
    return agents


COMMUNICATION_MODULES = ("send_context", "send_embedding", "send_recur", "send_out", "receive_embedding", "actor")


def reset_communication(agent, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for name in COMMUNICATION_MODULES:
            for module in getattr(agent, name).modules():
                if hasattr(module, "reset_parameters"): module.reset_parameters()
    agent.requires_grad_(False)
    for name in COMMUNICATION_MODULES:
        for parameter in getattr(agent, name).parameters(): parameter.requires_grad_(True)
    for parameter in agent.parameters(): parameter.grad = None
    return {"seed": int(seed), "reset_module_order": list(COMMUNICATION_MODULES), "trainable_parameters": sum(parameter.numel() for name in COMMUNICATION_MODULES for parameter in getattr(agent, name).parameters())}


def trainable_groups(agent):
    groups = {"sender": [p for name in ("send_context", "send_embedding", "send_recur", "send_out") for p in getattr(agent, name).parameters()], "receiver": [p for name in ("receive_embedding", "actor") for p in getattr(agent, name).parameters()]}
    ids = [id(p) for group in groups.values() for p in group]
    if len(ids) != len(set(ids)): raise ValueError("communication parameter alias")
    return groups


def encode_resource(agent, projected, worlds, resource):
    n = len(worlds["map_id"]); first_slots = projected.new_zeros(n, SITES, 64); first_exists = projected.new_zeros(n, SITES); rows = torch.arange(n); sites = torch.from_numpy(worlds["positions"][:, resource]); ids = torch.from_numpy(worlds["photo_ids"][:, resource]); first_slots[rows, sites] = projected[ids]; first_exists[rows, sites] = 1.
    first = torch.cat((first_slots.flatten(1), first_exists), dim=1)
    second = first.clone()
    frames = torch.stack((first, second), dim=1); bits = projected.new_zeros(n, 2); bits[:, 0] = 1.
    return _TEMPORAL.observe_sequence(agent, frames, bits, "full").detach()


def load_views(agents, bank, worlds_by_split):
    result = {identity: {resource: {} for resource in range(RESOURCES)} for identity in range(4)}
    for identity, agent in enumerate(agents):
        with torch.no_grad(): projected = agent.project(bank.features).detach()
        for split, worlds in worlds_by_split.items():
            for resource in range(RESOURCES): result[identity][resource][split] = encode_resource(agent, projected, worlds, resource)
    return result


def sender_start(agent, h):
    state = agent.send_context(torch.cat((h, h.new_zeros(len(h), 4)), dim=-1))
    return state, agent.send_out(state)


def receiver_from_messages(agent, messages):
    embedded = agent.receive_embedding(messages.detach()).flatten(1); context = torch.cat((embedded, embedded.new_zeros(len(messages), 2 + HISTORY)), dim=-1)
    return agent.actor(context).reshape(len(messages), RESOURCES, SITES)


def tri_loss(senders, receiver, hs, uniforms, positions):
    states, logits = zip(*(sender_start(agent, h) for agent, h in zip(senders, hs))); uniform_t = torch.from_numpy(uniforms)
    sampled_tokens = []; sender_logps = []; sender_entropies = []
    for resource in range(RESOURCES):
        token, logp, entropy, _ = _DRAW(logits[resource], uniform_t[:, resource:resource + 1]); sampled_tokens.append(token); sender_logps.append(logp); sender_entropies.append(entropy)
    messages = torch.stack(sampled_tokens, dim=1).detach(); action_logits = receiver_from_messages(receiver, messages)
    actions = []; action_logps = []; action_entropies = []
    for resource in range(RESOURCES):
        action, logp, entropy, _ = _DRAW(action_logits[:, resource], uniform_t[:, RESOURCES + resource:RESOURCES + resource + 1]); actions.append(action); action_logps.append(logp); action_entropies.append(entropy)
    action_tensor = torch.stack(actions, dim=1); success = (action_tensor.detach().numpy() == positions).astype(np.float32); reward = (success.sum(axis=1) / 6.0 + .5 * success.prod(axis=1)).astype(np.float32); advantage = torch.from_numpy(reward - np.float32(BASELINE))
    sender_losses = [-(logp * advantage).mean() - ENTROPY_WEIGHT * entropy.mean() for logp, entropy in zip(sender_logps, sender_entropies)]
    receiver_loss = -(sum(action_logps) * advantage).mean() - ENTROPY_WEIGHT * sum(action_entropies).mean()
    losses = sender_losses + [receiver_loss]
    if not all(bool(torch.isfinite(loss)) for loss in losses): raise ValueError("non-finite triad loss")
    to_np = lambda tensor: tensor.detach().numpy().copy()
    trace = {"uniforms": uniforms.copy(), "messages": to_np(messages), "token_probabilities": to_np(torch.stack([F.softmax(x, -1) for x in logits], dim=1)), "action_logits": to_np(action_logits), "action_probabilities": to_np(torch.stack([F.softmax(x, -1) for x in action_logits.unbind(1)], dim=1)), "actions": to_np(action_tensor), "positions": positions.copy(), "success": success, "reward": reward, "advantage": to_np(advantage), "sender_logp": to_np(torch.stack(sender_logps, dim=1)), "receiver_logp": to_np(torch.stack(action_logps, dim=1).sum(1)), "sender_entropy": to_np(torch.stack(sender_entropies, dim=1)), "receiver_entropy": to_np(torch.stack(action_entropies, dim=1).sum(1)), "baseline": BASELINE, "entropy_weight": ENTROPY_WEIGHT}
    return sender_losses, receiver_loss, trace


@torch.no_grad()
def first_token_log_probs(agent, h):
    _, logits = sender_start(agent, h); return F.log_softmax(logits, -1)


@torch.no_grad()
def receiver_logits(agent):
    codes = torch.tensor(list(itertools.product(range(VOCAB), repeat=RESOURCES)), dtype=torch.int64); return receiver_from_messages(agent, codes)


def train_ids(partition): return train_map_ids(partition)


def score(raw, partition):
    decoder = np.argmax(raw["receiver_logits"], axis=-1); codes = raw["tokens"][:, 0] * VOCAB * VOCAB + raw["tokens"][:, 1] * VOCAB + raw["tokens"][:, 2]; actions = decoder[codes]; correct = actions == raw["positions"]; train = np.isin(raw["map_id"], train_ids(partition)); target = ~train
    return {split: {"J": float(np.mean(np.all(correct[mask], axis=-1))), "resource0": float(np.mean(correct[mask, 0])), "resource1": float(np.mean(correct[mask, 1])), "resource2": float(np.mean(correct[mask, 2]))} for split, mask in (("train60", train), ("target60", target), ("all120", np.ones(len(correct), dtype=bool)))}


def evaluate(agents, views, worlds, partition, schedule, folder, update):
    logs = {resource: [first_token_log_probs(agents[identity], views[identity][resource]["test"]).numpy().astype(np.float32) for identity in range(4)] for resource in range(RESOURCES)}; receivers = [receiver_logits(agent).numpy().astype(np.float32) for agent in agents]; scores = {}
    for slot, team in enumerate(TEAMS[schedule]):
        receiver, sender0, sender1, sender2 = team; l0, l1, l2 = (logs[0][sender0], logs[1][sender1], logs[2][sender2]); raw = dict(worlds, sender_log_probs_r0=l0, sender_log_probs_r1=l1, sender_log_probs_r2=l2, sender_log_probs=(l0[:, :, None, None] + l1[:, None, :, None] + l2[:, None, None, :]).reshape(len(l0), VOCAB ** RESOURCES), tokens=np.stack((np.argmax(l0, -1), np.argmax(l1, -1), np.argmax(l2, -1)), axis=1).astype(np.int64), receiver_logits=receivers[receiver]); np.savez_compressed(folder / f"protocol_{schedule}_{update:04d}_team{slot}.npz", **raw); scores[f"team{slot}"] = score(raw, partition)
    return scores


def schedule_for(condition, seed, part, step):
    if condition == "fixed_A": return "A"
    if condition == "rotating_AB": return "A" if step % 2 == 0 else "B"
    return str(np.random.default_rng(np.random.SeedSequence([39039, int(seed), int(part), int(step), 77])).choice(np.asarray(SCHEDULES)))


def save_trace(folder, name, fixture_worlds, trace, schedule, step):
    payload = {"schedule_id": np.asarray([schedule]), "global_step": np.asarray([step], dtype=np.int64), "world__indices": fixture_worlds["indices"], "world__uniforms": fixture_worlds["uniforms"]}; payload.update({f"trace__{key}": value for key, value in trace.items()}); np.savez_compressed(folder / name, **payload)


def run(out: Path, seeds=SEEDS, parts=PARTITIONS, conditions=CONDITIONS, updates=UPDATES):
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1); hashes, inputs = source_hashes(), input_hashes(); checkpoints = sorted(set(t for t in CHECKPOINTS if t <= updates) | {updates}); formal = list(seeds) == list(SEEDS) and list(parts) == list(PARTITIONS) and list(conditions) == list(CONDITIONS) and updates == UPDATES
    write(out / "invocation.json", {"formal": formal, "version": "v0.39-triad-origin", "seeds": list(seeds), "partitions": list(parts), "conditions": list(conditions), "schedules": list(SCHEDULES), "resources": RESOURCES, "vocabulary": VOCAB, "updates": updates, "checkpoints": checkpoints, "batch": BATCH, "source_hashes": hashes, "input_hashes": inputs, "torch_version": str(torch.__version__), "numpy_version": np.__version__, "platform": platform.platform(), "threads": 1, "new_private_fits": 0, "new_dino_inferences": 0, "communication_initialization": "fresh_random_reset_all_agents", "study_scope": "teacher-free three-resource grounded protocol formation"})
    for path in hashes:
        target = out / "frozen_sources" / Path(path).relative_to(PROJECT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    bank = TriBank(); worlds_train = {}; worlds_test = {}; tables_train = {}; tables_test = {}
    for part in parts:
        worlds_train[part] = table(bank, "train", train_map_ids(part)); worlds_test[part] = table(bank, "test"); tables_train[part] = worlds_train[part]; tables_test[part] = worlds_test[part]
    # A single copy of the public test table is sufficient; the map IDs and
    # partition-specific split are stored in every protocol table.
    np.savez_compressed(out / "test_worlds.npz", **worlds_test[parts[0]])
    for part in parts: np.savez_compressed(out / f"train_worlds_p{part}.npz", **worlds_train[part])
    rows = []; started = time.monotonic(); total = 0
    for seed, part, condition in itertools.product(seeds, parts, conditions):
        prepared = torch.load(SOURCE / f"prepared_{seed}.pt", weights_only=True); agents = load_population(seed, prepared, part); resets = [reset_communication(agent, int(np.random.SeedSequence([39039, int(seed), int(part), identity, 901]).generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))) for identity, agent in enumerate(agents)]; views = load_views(agents, bank, {"train": worlds_train[part], "test": worlds_test[part]}); folder = out / "social" / f"s{seed}_p{part}_{condition}"; folder.mkdir(parents=True, exist_ok=True); write(folder / "config.json", {"seed": seed, "partition": part, "condition": condition, "resources": RESOURCES, "private_types": list(PRIVATE_TYPES), "updates": updates, "checkpoints": checkpoints, "training_schedule": {"fixed_A": "A_every_update", "rotating_AB": "A_even_B_odd", "random_ABC": "deterministic_random_A_B_C"}[condition], "evaluation_schedules": list(SCHEDULES), "communication_initialization": "fresh_random_reset_all_agents", "reset_records": resets, "train_map_count": int(len(train_map_ids(part))), "test_map_count": int(len(MAPS3))})
        params = [[p for group in trainable_groups(agent).values() for p in group] for agent in agents]; optimizers = [torch.optim.Adam(p, lr=.0007) for p in params]; curve = []
        for step in range(updates + 1):
            if step in checkpoints:
                for agent in agents: agent.eval()
                scores = {schedule: evaluate(agents, views, worlds_test[part], part, schedule, folder, step) for schedule in SCHEDULES}
                curve.append({"update": step, "scores": scores}); write(folder / "curve.json", curve)
                for agent in agents: agent.train()
                if step == updates: torch.save([copy.deepcopy(agent.state_dict()) for agent in agents], folder / "final.pt")
            if step == updates: break
            schedule = schedule_for(condition, seed, part, step); losses = [[] for _ in agents]; trace_slots = {}; fixture_slots = {}
            for slot, team in enumerate(TEAMS[schedule]):
                receiver, sender0, sender1, sender2 = team; fixture_data = fixture(seed, part, slot, step, worlds_train[part]); idx = fixture_data["indices"]; sender_losses, receiver_loss, trace = tri_loss([agents[sender0], agents[sender1], agents[sender2]], agents[receiver], [views[sender0][0]["train"][idx], views[sender1][1]["train"][idx], views[sender2][2]["train"][idx]], fixture_data["uniforms"], worlds_train[part]["positions"][idx]); losses[sender0].append(sender_losses[0]); losses[sender1].append(sender_losses[1]); losses[sender2].append(sender_losses[2]); losses[receiver].append(receiver_loss); trace_slots[f"slot{slot}"] = trace; fixture_slots[f"slot{slot}__indices"] = idx; fixture_slots[f"slot{slot}__uniforms"] = fixture_data["uniforms"]
            norms = []
            for optimizer, agent, role_losses in zip(optimizers, agents, losses):
                loss = sum(role_losses) / len(role_losses); optimizer.zero_grad(set_to_none=True); loss.backward(); groups = trainable_groups(agent); group_norms = {role: float(torch.nn.utils.clip_grad_norm_(group, 2.0)) for role, group in groups.items()}; norms.append(group_norms); optimizer.step()
            if not all(np.isfinite(value) for norm in norms for value in norm.values()): raise ValueError("non-finite triad gradient")
            if step in (0, updates - 1):
                for slot, trace in trace_slots.items():
                    payload = {"schedule_id": np.asarray([schedule]), "global_step": np.asarray([step], dtype=np.int64), "world__indices": fixture_slots[slot + "__indices"], "world__uniforms": fixture_slots[slot + "__uniforms"]}; payload.update({f"trace__{key}": value for key, value in trace.items()}); np.savez_compressed(folder / f"train_{step + 1:04d}_{slot}.npz", **payload)
            with (folder / "training.jsonl").open("a") as log: log.write(json.dumps({"update": step + 1, "schedule": schedule, "norms": norms}, ensure_ascii=False) + "\n")
        rows.append({"seed": seed, "partition": part, "condition": condition, "curve": curve}); total += 1; print(json.dumps({"phase": "triad_origin", "run": folder.name, "runs": total, "seconds": round(time.monotonic() - started, 2)}), flush=True)
    write(out / "runs.json", {"status": "complete", "rows": rows, "count": total}); write(out / "training_complete.json", {"status": "complete", "formal": formal, "probe": "triad_origin", "runs": total, "updates_per_run": updates, "messages": total * updates * 4 * BATCH, "token_instances": total * updates * 4 * BATCH * RESOURCES, "actions": total * updates * 4 * BATCH * RESOURCES, "seconds": time.monotonic() - started, "source_hashes": hashes, "input_hashes": inputs, "files": {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}})
    return {"status": "complete", "formal": formal, "runs": total, "seconds": time.monotonic() - started, "source_sha256": sha(ROOT / "triad_train.py")}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--dev", action="store_true"); parser.add_argument("--updates", type=int, default=UPDATES); args = parser.parse_args()
    if args.dev: result = run(args.out.resolve(), seeds=(SEEDS[0],), parts=(1,), conditions=(CONDITIONS[0],), updates=min(args.updates, 60))
    else: result = run(args.out.resolve(), updates=args.updates)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
