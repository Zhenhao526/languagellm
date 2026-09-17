"""Endpoint partner-replacement assay for the v0.43 two-token cultures."""
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
REPLACEMENTS = ("none", "slot0", "slot1", "slot2", "all")
TEAMS = {
    "A": ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)),
    "B": ((0, 2, 1, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0)),
    "C": ((0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0)),
}
SITES, RESOURCES, TOKENS, VOCAB, BATCH = 6, 3, 2, 7, 120
METRICS = ("J", "resource0", "resource1", "resource2")
SPLITS = ("target60", "all120")


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def source_hashes():
    paths = [ROOT / "partner_transfer_train.py", ROOT / "partner_transfer_design.json"]
    paths += [V043 / name for name in ("permutation_train.py", "permutation_design.json")]
    paths += [V039 / name for name in ("triad_train.py", "triad_design.json")]
    paths += [PROJECT / name for name in ("redesign_v0.8/camp.py", "redesign_v0.20/temporal_model.py", "redesign_v0.20/temporal_world.py", "redesign_v0.28/world.py", "redesign_v0.21/social_model.py")]
    return {str(path.resolve()): sha(path) for path in paths}


def input_hashes():
    v043_out = V043 / "results" / "permutation_001"
    paths = [FEATURE_CACHE, SELECTION, v043_out / "completion_manifest.json", v043_out / "training_complete.json", v043_out / "permutation_analysis.json"]
    paths += [v043_out / f"test_worlds_{assignment}.npz" for assignment in ASSIGNMENTS]
    paths += [SOURCE / f"prepared_{seed}.pt" for seed in SEEDS]
    return {str(path.resolve()): sha(path) for path in paths}


def train_ids(partition):
    panels = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
    rank = {site: i for i, site in enumerate(panels[partition - 1])}
    maps = np.asarray(list(itertools.permutations(range(SITES), RESOURCES)), dtype=np.int64)
    return np.asarray([i for i, triple in enumerate(maps) if rank[int(triple[0])] < rank[int(triple[1])]], dtype=np.int64)


def metrics(actions, positions, map_id, partition, role_perm):
    correct = np.asarray(actions) == np.asarray(positions)
    train = np.isin(map_id, train_ids(partition))
    result = np.zeros((2, 2, 4), dtype=np.float32)
    for si, mask in enumerate((~train, np.ones(len(correct), dtype=bool))):
        result[si, 0, 0] = np.mean(np.all(correct[mask], axis=-1))
        result[si, 0, 1:] = np.mean(correct[mask], axis=0)
        equiv_positions = positions[:, np.asarray(role_perm, dtype=np.int64)]
        equiv_correct = np.asarray(actions) == equiv_positions
        result[si, 1, 0] = np.mean(np.all(equiv_correct[mask], axis=-1))
        result[si, 1, 1:] = np.mean(equiv_correct[mask], axis=0)
    return result


def load_culture(base_agents, chain):
    agents = [copy.deepcopy(agent) for agent in base_agents]
    for agent in agents:
        # v0.43 uses a 3-resource/two-token receiver actor interface.
        import permutation_train as permutation

        permutation.replace_actor(agent)
    states = torch.load(chain / "final.pt", weights_only=True)
    if len(states) != 4:
        raise ValueError(f"expected four agents in {chain}")
    for agent, state in zip(agents, states):
        agent.load_state_dict(state)
        agent.eval()
    return agents


def token_tables(permutation, agents, views):
    tokens = np.zeros((4, RESOURCES, BATCH, TOKENS), dtype=np.int16)
    for identity, agent in enumerate(agents):
        for resource in range(RESOURCES):
            with torch.no_grad():
                sequence = permutation.sequence_log_probs(agent, views[identity][resource]["test"])[3]
            tokens[identity, resource] = sequence.cpu().numpy().astype(np.int16)
    return tokens


def evaluate_group(permutation, base_agents, views, worlds, chain_paths):
    cultures = []
    culture_tokens = np.zeros((len(CULTURES), 4, RESOURCES, BATCH, TOKENS), dtype=np.int16)
    for ci, label in enumerate(CULTURES):
        mode, condition = label.split("__", 1)
        agents = load_culture(base_agents, chain_paths[mode, condition])
        cultures.append(agents)
        culture_tokens[ci] = token_tables(permutation, agents, views)
    # recipient, donor, replacement, schedule, role permutation, team, world, resource
    actions_all = np.zeros((6, 6, 5, 3, 6, 4, BATCH, RESOURCES), dtype=np.uint8)
    scores = np.zeros((6, 6, 5, 3, 6, 4, 2, 2, 4), dtype=np.float32)
    for recipient in range(6):
        for si, schedule in enumerate(SCHEDULES):
            for slot, team in enumerate(TEAMS[schedule]):
                receiver = cultures[recipient][team[0]]
                batches, specs = [], []
                for donor in range(6):
                    for replacement in range(5):
                        for role_i, role_perm in enumerate(ROLE_PERMS):
                            sender_streams = []
                            for sender_pos, sender_identity in enumerate(team[1:]):
                                donor_selected = replacement == 4 or (replacement > 0 and sender_pos == replacement - 1)
                                culture_index = donor if donor_selected else recipient
                                resource = role_perm[sender_pos]
                                sender_streams.append(culture_tokens[culture_index, sender_identity, resource])
                            batches.append(np.stack(sender_streams, axis=1))
                            specs.append((donor, replacement, role_i, role_perm))
                messages = torch.from_numpy(np.concatenate(batches, axis=0).astype(np.int64))
                with torch.no_grad():
                    logits = permutation.receiver_from_sequence(receiver, messages).cpu().numpy()
                sampled_actions = logits.argmax(axis=-1).astype(np.uint8).reshape(len(specs), BATCH, RESOURCES)
                for combo_i, (donor, replacement, role_i, role_perm) in enumerate(specs):
                    act = sampled_actions[combo_i]
                    actions_all[recipient, donor, replacement, si, role_i, slot] = act
                    scores[recipient, donor, replacement, si, role_i, slot] = metrics(act, worlds["positions"], worlds["map_id"], int(worlds["partition"]), role_perm)
    return culture_tokens, actions_all, scores


def run(out: Path):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if str(V043) not in sys.path:
        sys.path.insert(0, str(V043))
    import permutation_train as permutation

    formal = True
    hashes, inputs = source_hashes(), input_hashes()
    write(out / "invocation.json", {
        "formal": formal,
        "version": "v0.44-partner-transfer-endpoint",
        "visual_groups": len(SEEDS) * len(PARTITIONS) * len(ASSIGNMENTS),
        "seeds": list(SEEDS), "partitions": list(PARTITIONS), "assignments": list(ASSIGNMENTS),
        "cultures": list(CULTURES), "role_modes": list(ROLE_MODES), "conditions": list(CONDITIONS),
        "schedules": list(SCHEDULES), "role_permutations": [list(p) for p in ROLE_PERMS],
        "replacement_modes": list(REPLACEMENTS), "vocabulary": VOCAB, "tokens_per_resource_sender": TOKENS,
        "resource_count": RESOURCES, "site_count": SITES, "test_world_count": BATCH,
        "adaptation_updates": 0, "training_updates": 0, "deterministic_endpoint_inference": True,
        "source_hashes": hashes, "input_hashes": inputs, "torch_version": str(torch.__version__),
        "numpy_version": np.__version__, "platform": platform.platform(), "threads": 1,
    })
    for path in hashes:
        target = out / "frozen_sources" / Path(path).relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    prepared = {seed: torch.load(SOURCE / f"prepared_{seed}.pt", weights_only=True) for seed in SEEDS}
    bank = permutation.T.TriBank()
    rows, started = [], time.monotonic()
    groups = out / "groups"
    groups.mkdir()
    for seed, part, assignment in itertools.product(SEEDS, PARTITIONS, ASSIGNMENTS):
        worlds = npz(V043 / "results" / "permutation_001" / f"test_worlds_{assignment}.npz")
        worlds["partition"] = np.asarray(part, dtype=np.int64)
        base_agents = permutation.T.load_population(seed, prepared[seed], part)
        views = permutation.T.load_views(base_agents, bank, {"test": worlds})
        chain_paths = {(mode, condition): V043 / "results" / "permutation_001" / "social" / f"s{seed}_p{part}_{assignment}_{mode}_{condition}" for mode, condition in itertools.product(ROLE_MODES, CONDITIONS)}
        for chain in chain_paths.values():
            if not (chain / "final.pt").is_file():
                raise FileNotFoundError(chain / "final.pt")
        culture_tokens, actions, scores = evaluate_group(permutation, base_agents, views, worlds, chain_paths)
        target = groups / f"transfer_s{seed}_p{part}_{assignment}.npz"
        np.savez_compressed(target, culture_labels=np.asarray(CULTURES), replacement_modes=np.asarray(REPLACEMENTS), schedules=np.asarray(SCHEDULES), role_permutations=np.asarray(ROLE_PERMS, dtype=np.int64), map_id=worlds["map_id"], positions=worlds["positions"], photo_ids=worlds["photo_ids"], shown=worlds["shown"], tokens=culture_tokens, actions=actions, scores=scores)
        rows.append({"seed": seed, "partition": part, "assignment": assignment, "file": str(target.relative_to(out)), "bytes": target.stat().st_size})
        print(json.dumps({"phase": "partner_transfer_endpoint", "group": f"s{seed}_p{part}_{assignment}", "groups": len(rows), "seconds": round(time.monotonic() - started, 2)}, ensure_ascii=False), flush=True)
    write(out / "runs.json", {"status": "complete", "formal": formal, "rows": rows, "count": len(rows)})
    files = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "transfer_complete.json"}
    write(out / "transfer_complete.json", {"status": "complete", "formal": formal, "probe": "partner_transfer_endpoint", "groups": len(rows), "cultures": len(CULTURES), "replacement_modes": list(REPLACEMENTS), "schedules": list(SCHEDULES), "role_permutations": len(ROLE_PERMS), "team_slots": 4, "worlds_per_group": BATCH, "files": files, "seconds": time.monotonic() - started, "source_hashes": hashes, "input_hashes": inputs})
    return {"status": "complete", "formal": formal, "groups": len(rows), "seconds": time.monotonic() - started, "source_sha256": sha(ROOT / "partner_transfer_train.py")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    result = run(args.out.resolve())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
