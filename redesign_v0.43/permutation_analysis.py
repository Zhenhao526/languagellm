"""Independent NumPy reanalysis of the v0.43 permutation formation batch."""
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
OUT = ROOT / "results" / "permutation_001"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
ROLE_MODES = ("static_role", "random_role")
ASSIGNMENTS = tuple("".join(map(str, p)) for p in itertools.permutations(range(3)))
SCHEDULES = ("A", "B", "C")
ROLE_PERMS = tuple(itertools.permutations(range(3)))
ROLE_KEYS = tuple("".join(map(str, p)) for p in ROLE_PERMS)
SITES = 6
RESOURCES = 3
TOKENS = 2

TEAMS = {
    "A": ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)),
    "B": ((0, 2, 1, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0)),
    "C": ((0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0)),
}
PANELS = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
MAPS3 = np.asarray(list(itertools.permutations(range(SITES), RESOURCES)), dtype=np.int64)
UPDATES = (0, 100, 300, 600)


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


class Checks:
    def __init__(self):
        self.checks = 0
        self.comparisons = 0
        self.max_error = 0.0
        self.coverage = defaultdict(int)

    def require(self, value, label):
        self.checks += 1
        if not bool(value):
            raise AssertionError(label)

    def exact(self, actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape, label + " shape")
        self.require(np.array_equal(a, b), label)

    def close(self, actual, expected, label, atol=1e-6):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.require(a.shape == b.shape, label + " shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        self.comparisons += int(a.size)
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        self.max_error = max(self.max_error, error)
        self.require(np.allclose(a, b, rtol=1e-6, atol=atol), label)


def train_ids(partition):
    rank = {site: i for i, site in enumerate(PANELS[partition - 1])}
    return np.asarray([i for i, triple in enumerate(MAPS3) if rank[int(triple[0])] < rank[int(triple[1])]], dtype=np.int64)


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
        px = float(np.mean(x == xv))
        for yv in np.unique(y):
            pxy = float(np.mean((x == xv) & (y == yv)))
            py = float(np.mean(y == yv))
            if pxy > 0:
                mi += pxy * np.log(pxy / (px * py))
    hy = 0.0
    for yv in np.unique(y):
        py = float(np.mean(y == yv))
        if py > 0:
            hy -= py * np.log(py)
    return float(mi / hy) if hy > 0 else 0.0


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "values": a.tolist()}


def aggregate(rows, key, split="target60", metric_name="J"):
    return mean_sd([row[key][split][metric_name] for row in rows])


def sequence_metrics(raw_by_slot, schedule, role_index=0):
    sender_tokens = {resource: {} for resource in range(RESOURCES)}
    sender_positions = {resource: {} for resource in range(RESOURCES)}
    for slot, team in enumerate(TEAMS[schedule]):
        for resource, sender in enumerate(team[1:]):
            sender_tokens[resource][sender] = raw_by_slot[slot]["tokens"][role_index, :, resource, :]
            sender_positions[resource][sender] = raw_by_slot[slot]["positions"][:, resource]
    t0, t1, pair, lexicon, mi0, mi1, mipair = [], [], [], [], [], [], []
    for resource in range(RESOURCES):
        for a, b in ((0, 2), (1, 3)):
            t0.append(float(np.mean(sender_tokens[resource][a][:, 0] == sender_tokens[resource][b][:, 0])))
            t1.append(float(np.mean(sender_tokens[resource][a][:, 1] == sender_tokens[resource][b][:, 1])))
            pair.append(float(np.mean(np.all(sender_tokens[resource][a] == sender_tokens[resource][b], axis=1))))
        lexicon.extend(float(len(np.unique(sender_tokens[resource][sender], axis=0))) for sender in range(4))
        for sender in range(4):
            mi0.append(nmi(sender_tokens[resource][sender][:, 0], sender_positions[resource][sender]))
            mi1.append(nmi(sender_tokens[resource][sender][:, 1], sender_positions[resource][sender]))
            mipair.append(nmi(sender_tokens[resource][sender][:, 0] * 7 + sender_tokens[resource][sender][:, 1], sender_positions[resource][sender]))
    sequence_joint = []
    for a, b in ((0, 2), (1, 3)):
        sequence_joint.append(float(np.mean(np.all(np.stack([sender_tokens[r][a] == sender_tokens[r][b] for r in range(RESOURCES)], axis=1), axis=(1, 2)))) )
    return {
        "token0_agreement": float(np.mean(t0)),
        "token1_agreement": float(np.mean(t1)),
        "pair_agreement": float(np.mean(pair)),
        "sequence_joint_agreement": float(np.mean(sequence_joint)),
        "pair_lexicon_size": float(np.mean(lexicon)),
        "nmi_token0_position": float(np.mean(mi0)),
        "nmi_token1_position": float(np.mean(mi1)),
        "nmi_pair_position": float(np.mean(mipair)),
    }


def run(out: Path):
    checks = Checks()
    invocation = read(OUT / "invocation.json")
    complete = read(OUT / "training_complete.json")
    checks.require(invocation.get("formal") is True, "formal invocation")
    checks.require(complete.get("formal") is True and complete.get("status") == "complete", "formal completion")
    checks.require(complete.get("probe") == "permutation_formation", "probe")
    checks.require(complete.get("runs") == 432, "runs")
    checks.exact(invocation["assignments"], {name: list(p) for name, p in zip(ASSIGNMENTS, ROLE_PERMS)}, "assignment map")
    rows, sequence_rows = [], []
    for seed, part, assignment, role_mode, condition in itertools.product(SEEDS, PARTITIONS, ASSIGNMENTS, ROLE_MODES, CONDITIONS):
        chain = OUT / "social" / f"s{seed}_p{part}_{assignment}_{role_mode}_{condition}"
        cfg = read(chain / "config.json")
        checks.require(cfg["assignment"] == assignment and cfg["role_mode"] == role_mode and cfg["condition"] == condition, "config identity")
        checks.exact(cfg["role_permutations"], np.asarray(ROLE_PERMS).tolist(), "config role permutations")
        curve = read(chain / "curve.json")
        checks.exact([int(item["update"]) for item in curve], list(UPDATES), "curve updates")
        for item in curve:
            update = int(item["update"])
            raw_by_schedule = {schedule: [] for schedule in SCHEDULES}
            for schedule in SCHEDULES:
                for slot in range(4):
                    raw = npz(chain / f"protocol_{schedule}_{update:04d}_team{slot}.npz")
                    worlds = npz(OUT / f"test_worlds_{assignment}.npz")
                    for key in ("map_id", "photo_ids", "positions", "shown"):
                        checks.exact(raw[key], worlds[key], f"{chain.name}/{schedule}/{update}/{slot}/{key}")
                    checks.exact(raw["role_permutations"], np.asarray(ROLE_PERMS, dtype=np.int64), "role permutation table")
                    checks.require(raw["tokens"].shape == (6, 120, 3, 2), "token shape")
                    checks.require(np.isin(raw["tokens"], np.arange(7)).all(), "token domain")
                    checks.require(raw["receiver_logits"].shape == (6, 120, 3, 6), "receiver logit shape")
                    checks.require(np.isfinite(raw["receiver_logits"]).all(), "receiver logits finite")
                    raw_by_schedule[schedule].append(raw)
                    saved = item["scores"][schedule][f"team{slot}"]["role_permutations"]
                    for pi, key in enumerate(ROLE_KEYS):
                        actions = np.argmax(raw["receiver_logits"][pi], axis=-1)
                        literal = metric(actions, raw["positions"], raw["map_id"], part)
                        equiv_positions = raw["positions"][:, np.asarray(ROLE_PERMS[pi], dtype=np.int64)]
                        equiv = metric(actions, equiv_positions, raw["map_id"], part)
                        checks.require(key in saved, "saved role key")
                        for kind, actual in (("literal", literal), ("equivariant", equiv)):
                            for split in ("train60", "target60", "all120"):
                                for metric_name in ("J", "resource0", "resource1", "resource2"):
                                    checks.close(actual[split][metric_name], saved[key][kind][split][metric_name], "saved role metric")
                        rows.append({"seed": seed, "partition": part, "assignment": assignment, "role_mode": role_mode, "condition": condition, "update": update, "schedule": schedule, "slot": slot, "role_permutation": key, "literal": literal, "equivariant": equiv})
                        checks.coverage["protocol_tables"] += 1
                ag = sequence_metrics(raw_by_schedule[schedule], schedule, role_index=0)
                sequence_rows.append({"seed": seed, "partition": part, "assignment": assignment, "role_mode": role_mode, "condition": condition, "update": update, "schedule": schedule, **ag})

    summary = {}
    for role_mode in ROLE_MODES:
        summary[role_mode] = {}
        for condition in CONDITIONS:
            summary[role_mode][condition] = {}
            for update in UPDATES:
                summary[role_mode][condition][str(update)] = {}
                for key in ROLE_KEYS:
                    sub = [r for r in rows if r["role_mode"] == role_mode and r["condition"] == condition and r["update"] == update and r["role_permutation"] == key]
                    summary[role_mode][condition][str(update)][key] = {kind: aggregate(sub, kind) for kind in ("literal", "equivariant")}

    assignment_summary = {}
    for assignment in ASSIGNMENTS:
        assignment_summary[assignment] = {}
        for role_mode in ROLE_MODES:
            assignment_summary[assignment][role_mode] = {}
            for condition in CONDITIONS:
                assignment_summary[assignment][role_mode][condition] = {}
                for key in ROLE_KEYS:
                    sub = [r for r in rows if r["assignment"] == assignment and r["role_mode"] == role_mode and r["condition"] == condition and r["update"] == 600 and r["role_permutation"] == key]
                    assignment_summary[assignment][role_mode][condition][key] = {kind: aggregate(sub, kind) for kind in ("literal", "equivariant")}

    sequence_summary = {}
    for role_mode in ROLE_MODES:
        sequence_summary[role_mode] = {}
        for condition in CONDITIONS:
            sub = [r for r in sequence_rows if r["role_mode"] == role_mode and r["condition"] == condition and r["update"] == 600]
            sequence_summary[role_mode][condition] = {key: mean_sd([r[key] for r in sub]) for key in ("token0_agreement", "token1_agreement", "pair_agreement", "sequence_joint_agreement", "pair_lexicon_size", "nmi_token0_position", "nmi_token1_position", "nmi_pair_position")}

    role_mode_effects = {}
    for condition in CONDITIONS:
        role_mode_effects[condition] = {}
        for key in ROLE_KEYS:
            static = summary["static_role"][condition]["600"][key]["equivariant"]["mean"]
            random = summary["random_role"][condition]["600"][key]["equivariant"]["mean"]
            role_mode_effects[condition][key] = {"random_minus_static_equivariant_target_J": random - static}

    record = {
        "status": "complete", "formal": True, "probe": "permutation_formation", "runs": 432,
        "updates": list(UPDATES), "assignments": {name: list(p) for name, p in zip(ASSIGNMENTS, ROLE_PERMS)}, "role_modes": list(ROLE_MODES),
        "conditions": list(CONDITIONS), "schedules": list(SCHEDULES), "role_permutations": [list(p) for p in ROLE_PERMS],
        "rows": rows, "sequence_rows": sequence_rows, "summary": summary, "assignment_summary": assignment_summary,
        "sequence_summary": sequence_summary, "role_mode_effects": role_mode_effects,
        "checks": checks.checks, "scalar_comparisons": checks.comparisons, "maximum_metric_absolute_difference": checks.max_error,
        "coverage": dict(checks.coverage), "source_sha256": sha(ROOT / "permutation_analysis.py"),
        "probe_training_complete_sha256": sha(OUT / "training_complete.json"), "limits": [
            "The six resource permutations are formation factors, but resource visual strata are not identical in the frozen bank.",
            "Role metrics are endpoint evaluations; they do not prove open-ended syntax.",
            "All private visual frontends and reward machinery are inherited and frozen/centralized.",
        ],
    }
    write(out / "permutation_analysis.json", record)
    qa = {
        "status": "passed_permutation_recheck", "formal": True, "checks": checks.checks, "scalar_comparisons": checks.comparisons,
        "maximum_metric_absolute_difference": checks.max_error, "coverage": dict(checks.coverage),
        "analysis_source_sha256": sha(ROOT / "permutation_analysis.py"), "analysis_sha256": sha(out / "permutation_analysis.json"),
        "production_modules_imported": False, "model_calls": 0,
    }
    write(out / "permutation_raw_validation.json", qa)
    shutil.copy2(ROOT / "permutation_analysis.py", out / "permutation_analysis_source.py")
    return record, qa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    record, qa = run(args.out.resolve())
    qa["seconds"] = time.monotonic() - started
    write(args.out.resolve() / "permutation_raw_validation.json", qa)
    print(json.dumps({"status": record["status"], "checks": record["checks"], "scalar_comparisons": record["scalar_comparisons"], "maximum_metric_absolute_difference": record["maximum_metric_absolute_difference"], "seconds": qa["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
