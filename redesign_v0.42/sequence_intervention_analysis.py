"""Run endpoint interventions on the formal v0.41 two-token formation batch."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
V041 = PROJECT / "redesign_v0.41"
INPUT = V041 / "results" / "two_token_001"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
ASSIGNMENTS = ("canonical", "cyclic")
SCHEDULES = ("A", "B", "C")
VOCAB = 7
RESOURCES = 3
TOKENS = 2
SITES = 6
MASK_VALUES = tuple(range(VOCAB))
BLOCK_PERMS = tuple(itertools.permutations(range(RESOURCES)))
SITE_PERMS = tuple(tuple((site + shift) % SITES for site in range(SITES)) for shift in range(SITES))

sys.path.insert(0, str(V041))
import two_token_train as TRAIN  # noqa: E402

T = TRAIN.T
TEAMS = TRAIN.TEAMS


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def source_hashes():
    return {
        str(path.resolve()): sha(path)
        for path in (ROOT / "sequence_intervention_analysis.py", ROOT / "sequence_intervention_design.json", V041 / "two_token_train.py", V041 / "two_token_design.json")
    }


def input_hashes():
    invocation = read(INPUT / "invocation.json")
    files = {
        str(Path(path).resolve()): digest
        for path, digest in invocation["input_hashes"].items()
    }
    for path in (INPUT / "invocation.json", INPUT / "training_complete.json", INPUT / "completion_manifest.json"):
        if path.exists():
            files[str(path.resolve())] = sha(path)
    for seed, part, assignment, condition in itertools.product(SEEDS, PARTITIONS, ASSIGNMENTS, CONDITIONS):
        chain = INPUT / "social" / f"s{seed}_p{part}_{assignment}_{condition}"
        files[str((chain / "final.pt").resolve())] = sha(chain / "final.pt")
        for schedule in SCHEDULES:
            for slot in range(4):
                path = chain / f"protocol_{schedule}_1200_team{slot}.npz"
                files[str(path.resolve())] = sha(path)
    return files


def site_world(worlds, permutation):
    result = {key: value.copy() for key, value in worlds.items()}
    result["positions"] = np.asarray(permutation, dtype=np.int64)[worlds["positions"]]
    return result


def train_ids(partition):
    rank = {site: i for i, site in enumerate(T.PANELS[partition - 1])}
    return np.asarray(
        [i for i, triple in enumerate(T.MAPS3) if rank[int(triple[0])] < rank[int(triple[1])]],
        dtype=np.int64,
    )


def split_masks(map_id, partition):
    train = np.isin(map_id, train_ids(partition))
    return {"train60": train, "target60": ~train, "all120": np.ones(len(map_id), dtype=bool)}


def metrics(actions, positions, map_id, partition):
    actions = np.asarray(actions)
    positions = np.asarray(positions)
    if actions.shape != positions.shape:
        raise ValueError(f"action/position shape mismatch: {actions.shape} {positions.shape}")
    correct = actions == positions
    return {
        split: {
            "J": float(np.mean(np.all(correct[mask], axis=1))),
            "resource0": float(np.mean(correct[mask, 0])),
            "resource1": float(np.mean(correct[mask, 1])),
            "resource2": float(np.mean(correct[mask, 2])),
        }
        for split, mask in split_masks(map_id, partition).items()
    }


@torch.no_grad()
def receiver_actions(receiver, tokens):
    flat = torch.from_numpy(np.asarray(tokens, dtype=np.int64).reshape(len(tokens), RESOURCES * TOKENS))
    logits = TRAIN.receiver_from_sequence(receiver, flat)
    return logits.argmax(-1).cpu().numpy().astype(np.int16)


def sender_tokens(agents, views, schedule):
    output = {}
    for slot, team in enumerate(TEAMS[schedule]):
        tokens = []
        for resource, sender in enumerate(team[1:]):
            greedy = TRAIN.sequence_log_probs(agents[sender], views[sender][resource]["test"])[3]
            tokens.append(greedy.cpu().numpy().astype(np.int64))
        output[slot] = np.stack(tokens, axis=1)
    return output


def load_endpoint_agents(seed, part, assignment, condition):
    prepared = torch.load(TRAIN.SOURCE / f"prepared_{seed}.pt", weights_only=True)
    agents, _ = TRAIN.make_population(seed, prepared, part, 42042)
    chain = INPUT / "social" / f"s{seed}_p{part}_{assignment}_{condition}"
    states = torch.load(chain / "final.pt", weights_only=True)
    if len(states) != len(agents):
        raise ValueError("checkpoint agent count mismatch")
    for agent, state in zip(agents, states):
        agent.load_state_dict(state, strict=True)
        agent.eval()
    return agents


def endpoint_worlds(assignment):
    return npz(INPUT / f"test_worlds_{assignment}.npz")


def evaluate_chain(out_chain, seed, part, assignment, condition, bank, worlds):
    agents = load_endpoint_agents(seed, part, assignment, condition)
    base_views = T.load_views(agents, bank, {"test": worlds})
    transformed_views = []
    transformed_worlds = []
    for permutation in SITE_PERMS:
        moved = site_world(worlds, permutation)
        transformed_worlds.append(moved)
        transformed_views.append(T.load_views(agents, bank, {"test": moved}))

    n = len(worlds["map_id"])
    shape = (len(SCHEDULES), 4, n, RESOURCES)
    arrays = {
        "map_id": worlds["map_id"].astype(np.int64),
        "positions": worlds["positions"].astype(np.int64),
        "baseline_actions": np.empty(shape, dtype=np.int16),
        "swap_actions": np.empty(shape, dtype=np.int16),
        "shuffle_t1_actions": np.empty(shape, dtype=np.int16),
        "mask_t0_actions": np.empty((len(MASK_VALUES),) + shape, dtype=np.int16),
        "mask_t1_actions": np.empty((len(MASK_VALUES),) + shape, dtype=np.int16),
        "block_actions": np.empty((len(BLOCK_PERMS),) + shape, dtype=np.int16),
        "site_actions": np.empty((len(SITE_PERMS),) + shape, dtype=np.int16),
    }
    records = []
    baseline_mismatches = 0
    for si, schedule in enumerate(SCHEDULES):
        raw_tokens = {}
        for slot in range(4):
            raw = npz(INPUT / "social" / f"s{seed}_p{part}_{assignment}_{condition}" / f"protocol_{schedule}_1200_team{slot}.npz")
            raw_tokens[slot] = raw["tokens"].astype(np.int64)
            expected = np.argmax(raw["receiver_logits"], axis=-1)[raw["receiver_code_ids"]]
            actions = receiver_actions(agents[TEAMS[schedule][slot][0]], raw_tokens[slot])
            baseline_mismatches += int(np.sum(actions != expected))
            arrays["baseline_actions"][si, slot] = actions
            arrays["swap_actions"][si, slot] = receiver_actions(agents[TEAMS[schedule][slot][0]], raw_tokens[slot][:, :, ::-1])
            shuffled = raw_tokens[slot].copy()
            shuffled[:, :, 1] = np.roll(shuffled[:, :, 1], 1, axis=0)
            arrays["shuffle_t1_actions"][si, slot] = receiver_actions(agents[TEAMS[schedule][slot][0]], shuffled)
            for mi, mask_value in enumerate(MASK_VALUES):
                masked = raw_tokens[slot].copy()
                masked[:, :, 0] = mask_value
                arrays["mask_t0_actions"][mi, si, slot] = receiver_actions(agents[TEAMS[schedule][slot][0]], masked)
                masked = raw_tokens[slot].copy()
                masked[:, :, 1] = mask_value
                arrays["mask_t1_actions"][mi, si, slot] = receiver_actions(agents[TEAMS[schedule][slot][0]], masked)
            for pi, permutation in enumerate(BLOCK_PERMS):
                permuted = raw_tokens[slot][:, permutation, :]
                arrays["block_actions"][pi, si, slot] = receiver_actions(agents[TEAMS[schedule][slot][0]], permuted)
            for pi, views in enumerate(transformed_views):
                moved_tokens = sender_tokens(agents, views, schedule)[slot]
                arrays["site_actions"][pi, si, slot] = receiver_actions(agents[TEAMS[schedule][slot][0]], moved_tokens)

            positions = worlds["positions"]
            base = metrics(arrays["baseline_actions"][si, slot], positions, worlds["map_id"], part)
            record = {
                "schedule": schedule,
                "slot": slot,
                "baseline": base,
                "swap": metrics(arrays["swap_actions"][si, slot], positions, worlds["map_id"], part),
                "shuffle_t1": metrics(arrays["shuffle_t1_actions"][si, slot], positions, worlds["map_id"], part),
                "mask_t0": {},
                "mask_t1": {},
                "block_permutations": {},
                "site_relocations": {},
            }
            for mi, mask_value in enumerate(MASK_VALUES):
                record["mask_t0"][str(mask_value)] = metrics(arrays["mask_t0_actions"][mi, si, slot], positions, worlds["map_id"], part)
                record["mask_t1"][str(mask_value)] = metrics(arrays["mask_t1_actions"][mi, si, slot], positions, worlds["map_id"], part)
            for pi, permutation in enumerate(BLOCK_PERMS):
                literal = metrics(arrays["block_actions"][pi, si, slot], positions, worlds["map_id"], part)
                equiv_positions = positions[:, np.asarray(permutation, dtype=np.int64)]
                equiv = metrics(arrays["block_actions"][pi, si, slot], equiv_positions, worlds["map_id"], part)
                record["block_permutations"]["".join(map(str, permutation))] = {"literal": literal, "equivariant": equiv}
            for pi, permutation in enumerate(SITE_PERMS):
                moved_positions = transformed_worlds[pi]["positions"]
                record["site_relocations"][str(pi)] = metrics(arrays["site_actions"][pi, si, slot], moved_positions, worlds["map_id"], part)
            records.append(record)

    arrays["baseline_mismatches"] = np.asarray([baseline_mismatches], dtype=np.int64)
    np.savez_compressed(out_chain / "interventions.npz", **arrays)
    write(out_chain / "interventions.json", {
        "seed": seed,
        "partition": part,
        "assignment": assignment,
        "condition": condition,
        "endpoint_update": 1200,
        "records": records,
        "baseline_replay_mismatches": baseline_mismatches,
        "block_permutations": [list(p) for p in BLOCK_PERMS],
        "site_permutations": [list(p) for p in SITE_PERMS],
        "mask_values": list(MASK_VALUES),
    })
    return records, arrays


def mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "values": values.tolist()}


def aggregate(records, key, split="target60", metric="J"):
    return mean_sd([row[key][split][metric] for row in records])


def aggregate_mask(records, key, value, split="target60", metric="J"):
    value = str(value)
    return mean_sd([row[key][value][split][metric] for row in records])


def run(out: Path, seeds=SEEDS, parts=PARTITIONS, assignments=ASSIGNMENTS, conditions=CONDITIONS):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    seeds = tuple(seeds)
    parts = tuple(parts)
    assignments = tuple(assignments)
    conditions = tuple(conditions)
    formal = seeds == SEEDS and parts == PARTITIONS and assignments == ASSIGNMENTS and conditions == CONDITIONS
    invocation = {
        "formal": formal,
        "version": "v0.42-sequence-intervention",
        "source_version": "v0.41-two-token-per-sender",
        "seeds": list(seeds),
        "partitions": list(parts),
        "assignments": list(assignments),
        "conditions": list(conditions),
        "schedules": list(SCHEDULES),
        "endpoint_update": 1200,
        "resource_block_permutations": [list(p) for p in BLOCK_PERMS],
        "site_permutations": [list(p) for p in SITE_PERMS],
        "mask_values": list(MASK_VALUES),
        "source_hashes": source_hashes(),
        "input_hashes": input_hashes(),
        "model_calls_added": 0,
        "new_training": False,
        "study_scope": "endpoint counterfactual assay for two-token sequence structure",
    }
    write(out / "invocation.json", invocation)
    bank = T.TriBank()
    all_rows = []
    chain_count = 0
    baseline_mismatch_total = 0
    for seed, part, assignment, condition in itertools.product(seeds, parts, assignments, conditions):
        chain_count += 1
        chain = out / "social" / f"s{seed}_p{part}_{assignment}_{condition}"
        chain.mkdir(parents=True, exist_ok=True)
        records, arrays = evaluate_chain(chain, seed, part, assignment, condition, bank, endpoint_worlds(assignment))
        baseline_mismatch_total += int(arrays["baseline_mismatches"][0])
        for row in records:
            all_rows.append({"seed": seed, "partition": part, "assignment": assignment, "condition": condition, **row})
        print(json.dumps({"phase": "sequence_intervention", "chain": chain.name, "chains": chain_count, "seconds": round(time.monotonic() - started, 2)}, ensure_ascii=False), flush=True)

    # Keep the detailed rows available for an independent reader while adding
    # compact condition-level summaries for the report and figure.
    summary = {
        assignment: {
            condition: {
                "baseline": aggregate([r for r in all_rows if r["assignment"] == assignment and r["condition"] == condition], "baseline"),
                "swap": aggregate([r for r in all_rows if r["assignment"] == assignment and r["condition"] == condition], "swap"),
                "shuffle_t1": aggregate([r for r in all_rows if r["assignment"] == assignment and r["condition"] == condition], "shuffle_t1"),
                "mask_t0": {},
                "mask_t1": {},
                "block_permutations": {},
                "site_relocations": {},
            }
            for condition in conditions
        }
        for assignment in assignments
    }
    # The helper above cannot index a nested block/site dictionary directly;
    # replace those summaries with explicit extraction to keep the schema clear.
    for assignment in assignments:
        for condition in conditions:
            subset = [r for r in all_rows if r["assignment"] == assignment and r["condition"] == condition]
            summary[assignment][condition]["mask_t0"] = {
                str(v): aggregate_mask(subset, "mask_t0", v)
                for v in MASK_VALUES
            }
            summary[assignment][condition]["mask_t1"] = {
                str(v): aggregate_mask(subset, "mask_t1", v)
                for v in MASK_VALUES
            }
            summary[assignment][condition]["block_permutations"] = {
                "".join(map(str, p)): {
                    kind: mean_sd([r["block_permutations"]["".join(map(str, p))][kind]["target60"]["J"] for r in subset])
                    for kind in ("literal", "equivariant")
                }
                for p in BLOCK_PERMS
            }
            summary[assignment][condition]["site_relocations"] = {
                str(i): mean_sd([r["site_relocations"][str(i)]["target60"]["J"] for r in subset])
                for i in range(len(SITE_PERMS))
            }
    record = {
        "status": "complete",
        "formal": formal,
        "probe": "sequence_intervention",
        "chains": chain_count,
        "rows": all_rows,
        "summary": summary,
        "baseline_replay_mismatches": baseline_mismatch_total,
        "checks": {
            "chains": chain_count,
            "rows": len(all_rows),
            "expected_rows": chain_count * len(SCHEDULES) * 4,
        },
        "source_sha256": sha(ROOT / "sequence_intervention_analysis.py"),
        "input_manifest_sha256": sha(INPUT / "completion_manifest.json"),
        "seconds": time.monotonic() - started,
        "limits": [
            "Endpoint intervention only; no additional training.",
            "Cyclic site relocation reuses the six-site frozen visual bank.",
            "Block permutation is a receiver-side counterfactual and needs formation controls.",
        ],
    }
    write(out / "sequence_intervention_analysis.json", record)
    write(out / "rows.json", {"status": "complete", "rows": all_rows, "count": len(all_rows)})
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    if args.dev:
        result = run(args.out.resolve(), seeds=(SEEDS[0],), parts=(PARTITIONS[0],), assignments=(ASSIGNMENTS[0],), conditions=(CONDITIONS[0],))
    else:
        result = run(args.out.resolve())
    print(json.dumps({"status": result["status"], "chains": result["chains"], "rows": len(result["rows"]), "seconds": result["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
