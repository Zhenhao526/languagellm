"""Independent NumPy reanalysis of the v0.47 reset-granularity batch."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / "results" / "reset_granularity_001"
V043 = PROJECT / "redesign_v0.43"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
ASSIGNMENTS = ("012", "021", "102", "120", "201", "210")
ROLE_MODES = ("static_role", "random_role")
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
CULTURES = tuple(f"{mode}__{condition}" for mode in ROLE_MODES for condition in CONDITIONS)
RESET_MODES = ("sender_only", "receiver_only", "both")
SCHEDULES = ("A", "B", "C")
ROLE_PERMS = tuple(itertools.permutations(range(3)))
ROLE_KEYS = tuple("".join(map(str, p)) for p in ROLE_PERMS)
UPDATES = (0, 100, 300)
METRICS = ("J", "resource0", "resource1", "resource2")
SPLITS = ("train60", "target60", "all120")
SITES, RESOURCES, BATCH = 6, 3, 120
SCHEDULE_NAMESPACE = 47040
ROLE_NAMESPACE = 47042
TEAMS = {
    "A": ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)),
    "B": ((0, 2, 1, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0)),
    "C": ((0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0)),
}


def read(path: Path):
    return json.loads(path.read_text())


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


def metric(actions, positions, map_id, partition):
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


def nmi(x, y):
    x, y = np.asarray(x), np.asarray(y)
    mi = 0.0
    for xv in np.unique(x):
        px = np.mean(x == xv)
        for yv in np.unique(y):
            pxy = np.mean((x == xv) & (y == yv))
            py = np.mean(y == yv)
            if pxy > 0:
                mi += pxy * np.log(pxy / (px * py))
    hy = 0.0
    for yv in np.unique(y):
        py = np.mean(y == yv)
        if py > 0:
            hy -= py * np.log(py)
    return float(mi / hy) if hy > 0 else 0.0


def ms(values):
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0, "values": a.tolist()}


def schedule_for(seed, part, step):
    return str(np.random.default_rng(np.random.SeedSequence([SCHEDULE_NAMESPACE, seed, part, step, 77])).choice(np.asarray(SCHEDULES)))


def role_for(mode, seed, part, step, slot):
    if mode == "static_role":
        return (0, 1, 2)
    return tuple(int(x) for x in np.random.default_rng(np.random.SeedSequence([ROLE_NAMESPACE, seed, part, step, slot, 77])).permutation(RESOURCES))


def sequence_metrics(raw_by_schedule):
    pair, token0, token1, nm0, nm1, nmpair = [], [], [], [], [], []
    for schedule in SCHEDULES:
        sender_table = {}
        for slot, team in enumerate(TEAMS[schedule]):
            for resource, identity in enumerate(team[1:]):
                sender_table[resource, identity] = raw_by_schedule[schedule][slot]["tokens"][0, :, resource, :]
        for resource in range(RESOURCES):
            newcomer = sender_table[resource, 0]
            for identity in (1, 2, 3):
                token0.append(float(np.mean(newcomer[:, 0] == sender_table[resource, identity][:, 0])))
                token1.append(float(np.mean(newcomer[:, 1] == sender_table[resource, identity][:, 1])))
                pair.append(float(np.mean(np.all(newcomer == sender_table[resource, identity], axis=1))))
            positions = raw_by_schedule[schedule][next(slot for slot, team in enumerate(TEAMS[schedule]) if 0 in team[1:])]["positions"][:, resource]
            nm0.append(nmi(newcomer[:, 0], positions)); nm1.append(nmi(newcomer[:, 1], positions)); nmpair.append(nmi(newcomer[:, 0] * 7 + newcomer[:, 1], positions))
    return {
        "newcomer_resident_token0_agreement": float(np.mean(token0)),
        "newcomer_resident_token1_agreement": float(np.mean(token1)),
        "newcomer_resident_pair_agreement": float(np.mean(pair)),
        "newcomer_token0_position_nmi": float(np.mean(nm0)),
        "newcomer_token1_position_nmi": float(np.mean(nm1)),
        "newcomer_pair_position_nmi": float(np.mean(nmpair)),
    }


def run(out: Path):
    checks = 0; comparisons = 0; max_error = 0.0; coverage = defaultdict(int)

    def require(value, label):
        nonlocal checks
        checks += 1
        if not bool(value):
            raise AssertionError(label)

    def exact(actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected)
        require(a.shape == b.shape, label + " shape")
        require(np.array_equal(a, b), label)

    def close(actual, expected, label):
        nonlocal comparisons, max_error
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        require(a.shape == b.shape, label + " shape")
        require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        comparisons += int(a.size)
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        max_error = max(max_error, error)
        require(np.allclose(a, b, atol=1e-6, rtol=1e-6), label)

    invocation = read(OUT / "invocation.json")
    complete = read(OUT / "training_complete.json")
    require(invocation.get("formal") is True and invocation.get("version") == "v0.47-reset-granularity", "formal invocation")
    require(invocation.get("adaptation_schedule") == "random_ABC", "fixed adaptation schedule")
    require(tuple(invocation.get("reset_modes", ())) == RESET_MODES, "reset mode coverage")
    require(complete.get("formal") is True and complete.get("status") == "complete" and complete.get("runs") == 1296, "formal completion")
    rows, sequence_rows = [], []
    for seed, part, assignment, culture, reset_mode in itertools.product(SEEDS, PARTITIONS, ASSIGNMENTS, CULTURES, RESET_MODES):
        mode, condition = culture.split("__", 1)
        chain = OUT / "social" / f"s{seed}_p{part}_{assignment}_{culture}__reset_{reset_mode}"
        cfg = read(chain / "config.json")
        require(cfg["seed"] == seed and cfg["partition"] == part and cfg["assignment"] == assignment and cfg["resident_culture"] == culture and cfg["reset_mode"] == reset_mode and cfg["adaptation_schedule"] == "random_ABC", f"{chain.name} config")
        curve = read(chain / "curve.json")
        exact([int(item["update"]) for item in curve], list(UPDATES), f"{chain.name} curve updates")
        test_worlds = npz(V043 / "results" / "permutation_001" / f"test_worlds_{assignment}.npz")
        for item in curve:
            update = int(item["update"]); raw_by_schedule = {}
            for schedule in SCHEDULES:
                raw_by_schedule[schedule] = []
                for slot in range(4):
                    raw = npz(chain / f"protocol_{schedule}_{update:04d}_team{slot}.npz")
                    for key in ("map_id", "photo_ids", "positions", "shown"):
                        exact(raw[key], test_worlds[key], f"{chain.name}/{schedule}/{update}/{slot}/{key}")
                    exact(raw["role_permutations"], np.asarray(ROLE_PERMS, dtype=np.int64), f"{chain.name} roles")
                    require(raw["tokens"].shape == (6, 120, 3, 2) and raw["actions"].shape == (6, 120, 3), f"{chain.name} protocol shape")
                    require(np.isin(raw["tokens"], np.arange(7)).all() and np.isin(raw["actions"], np.arange(6)).all(), f"{chain.name} protocol domain")
                    raw_by_schedule[schedule].append(raw)
                    saved = item["scores"][schedule][f"team{slot}"]["role_permutations"]
                    for role_i, role_key in enumerate(ROLE_KEYS):
                        actions = raw["actions"][role_i]
                        for kind, positions in (("literal", raw["positions"]), ("equivariant", raw["positions"][:, np.asarray(ROLE_PERMS[role_i], dtype=np.int64)])):
                            actual = metric(actions, positions, raw["map_id"], part)
                            for split in SPLITS:
                                for metric_name in METRICS:
                                    close(actual[split][metric_name], saved[role_key][kind][split][metric_name], f"{chain.name} saved metric")
                        rows.append({"seed": seed, "partition": part, "assignment": assignment, "resident_culture": culture, "role_mode": mode, "condition": condition, "reset_mode": reset_mode, "update": update, "schedule": schedule, "slot": slot, "role_permutation": role_key, "literal": metric(actions, raw["positions"], raw["map_id"], part)["target60"], "equivariant": metric(actions, raw["positions"][:, np.asarray(ROLE_PERMS[role_i], dtype=np.int64)], raw["map_id"], part)["target60"]})
                    coverage["protocol_files"] += 1
            sequence_rows.append({"seed": seed, "partition": part, "assignment": assignment, "resident_culture": culture, "role_mode": mode, "condition": condition, "reset_mode": reset_mode, "update": update, "schedule": "all", **sequence_metrics(raw_by_schedule)})
        coverage["chains"] += 1
    require(len(rows) == 1296 * 3 * 3 * 4 * 6, f"row coverage {len(rows)}")
    require(len(sequence_rows) == 1296 * 3, f"sequence coverage {len(sequence_rows)}")
    identity_groups = defaultdict(lambda: defaultdict(list)); spread_groups = defaultdict(list); sequence_groups = defaultdict(lambda: defaultdict(list))
    sequence_metric_names = ("newcomer_resident_token0_agreement", "newcomer_resident_token1_agreement", "newcomer_resident_pair_agreement", "newcomer_token0_position_nmi", "newcomer_token1_position_nmi", "newcomer_pair_position_nmi")
    for row in rows:
        group = (row["resident_culture"], row["reset_mode"], row["update"])
        if row["role_permutation"] == "012":
            identity_groups[group]["equivariant_J"].append(row["equivariant"]["J"])
            identity_groups[group]["literal_J"].append(row["literal"]["J"])
            for name in METRICS[1:]:
                identity_groups[group][f"equivariant_{name}"].append(row["equivariant"][name])
        spread_groups[(row["resident_culture"], row["reset_mode"], row["update"], row["seed"], row["partition"], row["assignment"], row["schedule"], row["slot"])].append(row["equivariant"]["J"])
    for row in sequence_rows:
        group = (row["resident_culture"], row["reset_mode"], row["update"])
        for name in sequence_metric_names:
            sequence_groups[group][name].append(row[name])
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for culture, reset_mode, update in itertools.product(CULTURES, RESET_MODES, UPDATES):
        group = (culture, reset_mode, update); cell = summary[culture][reset_mode][str(update)]
        cell["identity_equivariant_target60_J"] = ms(identity_groups[group]["equivariant_J"])
        cell["identity_literal_target60_J"] = ms(identity_groups[group]["literal_J"])
        for name in METRICS[1:]:
            cell[f"identity_equivariant_target60_{name}"] = ms(identity_groups[group][f"equivariant_{name}"])
        cell["role_spread_target60_J"] = ms([max(values) - min(values) for key, values in spread_groups.items() if key[:3] == group])
        for name in sequence_metric_names:
            cell[name] = ms(sequence_groups[group][name])
    record = {"status": "complete", "formal": True, "probe": "reset_granularity", "runs": 1296, "updates": list(UPDATES), "cultures": list(CULTURES), "reset_modes": list(RESET_MODES), "adaptation_schedule": "random_ABC", "summary": summary, "rows": rows, "sequence_rows": sequence_rows, "checks": checks, "scalar_comparisons": comparisons, "maximum_metric_absolute_difference": max_error, "coverage": dict(coverage), "source_sha256": sha(ROOT / "reset_granularity_analysis.py"), "v043_training_complete_sha256": sha(V043 / "results/permutation_001/training_complete.json"), "limits": ["Only selected communication modules are reset; the visual frontend and unreset modules are inherited from the resident endpoint.", "Residents remain frozen during adaptation.", "Finite six-site two-token grounded protocol; recovery is not open-ended language learning."]}
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "reset_granularity_analysis.py", out / "reset_granularity_analysis_source.py")
    return record


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); started = time.monotonic(); result = run(args.out.resolve()); result["seconds"] = time.monotonic() - started; (args.out.resolve() / "reset_granularity_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"); print(json.dumps({"status": result["status"], "runs": result["runs"], "checks": result["checks"], "scalar_comparisons": result["scalar_comparisons"], "maximum_metric_absolute_difference": result["maximum_metric_absolute_difference"], "seconds": result["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
