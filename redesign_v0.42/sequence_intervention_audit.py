"""Independently audit the v0.42 endpoint intervention artifacts."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path

import numpy as np


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
SITE_PERMS = tuple(tuple((site + shift) % SITES for site in range(SITES)) for shift in range(SITES)
)
TEAMS = {
    "A": ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)),
    "B": ((0, 2, 1, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0)),
    "C": ((0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0)),
}


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
        self.coverage = {"chains": 0, "rows": 0, "mask_tables": 0, "block_tables": 0, "site_tables": 0}

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
    panels = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
    rank = {site: i for i, site in enumerate(panels[partition - 1])}
    maps = np.asarray(list(itertools.permutations(range(SITES), RESOURCES)), dtype=np.int64)
    return np.asarray([i for i, triple in enumerate(maps) if rank[int(triple[0])] < rank[int(triple[1])]], dtype=np.int64)


def metric(actions, positions, map_id, partition):
    actions = np.asarray(actions)
    positions = np.asarray(positions)
    correct = actions == positions
    train = np.isin(map_id, train_ids(partition))
    return {
        split: {
            "J": float(np.mean(np.all(correct[mask], axis=1))),
            "resource0": float(np.mean(correct[mask, 0])),
            "resource1": float(np.mean(correct[mask, 1])),
            "resource2": float(np.mean(correct[mask, 2])),
        }
        for split, mask in (("train60", train), ("target60", ~train), ("all120", np.ones(len(map_id), dtype=bool)))
    }


def compare_metric(actual, expected, checks, label):
    for split in ("train60", "target60", "all120"):
        for key in ("J", "resource0", "resource1", "resource2"):
            checks.close(actual[split][key], expected[split][key], f"{label}/{split}/{key}")


def run(out: Path):
    checks = Checks()
    analysis = read(out / "sequence_intervention_analysis.json")
    invocation = read(out / "invocation.json")
    checks.require(analysis["formal"] is True, "formal analysis")
    checks.require(analysis["chains"] == 72, "chain count")
    checks.require(analysis["baseline_replay_mismatches"] == 0, "saved baseline mismatch")
    checks.require(invocation["formal"] is True, "formal invocation")
    rows = analysis["rows"]
    checks.require(len(rows) == 864, "row count")
    for seed, part, assignment, condition in itertools.product(SEEDS, PARTITIONS, ASSIGNMENTS, CONDITIONS):
        checks.coverage["chains"] += 1
        chain = out / "social" / f"s{seed}_p{part}_{assignment}_{condition}"
        detail = read(chain / "interventions.json")
        arrays = npz(chain / "interventions.npz")
        checks.exact(detail["seed"], seed, "detail seed")
        checks.exact(detail["partition"], part, "detail partition")
        checks.exact(detail["assignment"], assignment, "detail assignment")
        checks.exact(detail["condition"], condition, "detail condition")
        positions = arrays["positions"]
        map_id = arrays["map_id"]
        checks.require(positions.shape == (120, 3), "positions shape")
        checks.require(np.unique(map_id).size == 120, "map ids")
        checks.require(np.isin(arrays["baseline_actions"], np.arange(SITES)).all(), "baseline action domain")
        for key in ("swap_actions", "shuffle_t1_actions", "mask_t0_actions", "mask_t1_actions", "block_actions", "site_actions"):
            checks.require(np.isin(arrays[key], np.arange(SITES)).all(), key + " action domain")
        checks.exact(arrays["baseline_mismatches"], np.asarray([0], dtype=np.int64), "baseline mismatch array")
        for si, schedule in enumerate(SCHEDULES):
            for slot, team in enumerate(TEAMS[schedule]):
                raw = npz(INPUT / "social" / f"s{seed}_p{part}_{assignment}_{condition}" / f"protocol_{schedule}_1200_team{slot}.npz")
                checks.exact(raw["positions"], positions, f"{schedule}/{slot}/positions")
                checks.exact(raw["map_id"], map_id, f"{schedule}/{slot}/map_id")
                expected_actions = np.argmax(raw["receiver_logits"], axis=-1)[raw["receiver_code_ids"]]
                checks.exact(arrays["baseline_actions"][si, slot], expected_actions, f"{schedule}/{slot}/baseline replay")
                base_metric = metric(arrays["baseline_actions"][si, slot], positions, map_id, part)
                row = detail["records"][si * 4 + slot]
                checks.require(row["schedule"] == schedule and row["slot"] == slot, "record order")
                compare_metric(base_metric, row["baseline"], checks, f"{schedule}/{slot}/baseline")
                compare_metric(metric(arrays["swap_actions"][si, slot], positions, map_id, part), row["swap"], checks, f"{schedule}/{slot}/swap")
                compare_metric(metric(arrays["shuffle_t1_actions"][si, slot], positions, map_id, part), row["shuffle_t1"], checks, f"{schedule}/{slot}/shuffle")
                for mi, value in enumerate(MASK_VALUES):
                    compare_metric(metric(arrays["mask_t0_actions"][mi, si, slot], positions, map_id, part), row["mask_t0"][str(value)], checks, f"{schedule}/{slot}/mask0/{value}")
                    compare_metric(metric(arrays["mask_t1_actions"][mi, si, slot], positions, map_id, part), row["mask_t1"][str(value)], checks, f"{schedule}/{slot}/mask1/{value}")
                    checks.coverage["mask_tables"] += 2
                for pi, permutation in enumerate(BLOCK_PERMS):
                    key = "".join(map(str, permutation))
                    literal = metric(arrays["block_actions"][pi, si, slot], positions, map_id, part)
                    equiv_positions = positions[:, np.asarray(permutation, dtype=np.int64)]
                    equiv = metric(arrays["block_actions"][pi, si, slot], equiv_positions, map_id, part)
                    compare_metric(literal, row["block_permutations"][key]["literal"], checks, f"{schedule}/{slot}/block/{key}/literal")
                    compare_metric(equiv, row["block_permutations"][key]["equivariant"], checks, f"{schedule}/{slot}/block/{key}/equivariant")
                    checks.coverage["block_tables"] += 1
                for pi, permutation in enumerate(SITE_PERMS):
                    moved_positions = np.asarray(permutation, dtype=np.int64)[positions]
                    compare_metric(metric(arrays["site_actions"][pi, si, slot], moved_positions, map_id, part), row["site_relocations"][str(pi)], checks, f"{schedule}/{slot}/site/{pi}")
                    checks.coverage["site_tables"] += 1
                checks.coverage["rows"] += 1
    if checks.coverage["rows"] != 864:
        raise AssertionError("row coverage")
    qa = {
        "status": "complete",
        "formal": True,
        "probe": "sequence_intervention",
        "checks": checks.checks,
        "scalar_comparisons": checks.comparisons,
        "maximum_replay_absolute_difference": checks.max_error,
        "coverage": checks.coverage,
        "production_modules_imported": False,
        "model_calls": 0,
        "analysis_sha256": sha(out / "sequence_intervention_analysis.json"),
        "audit_source_sha256": sha(ROOT / "sequence_intervention_audit.py"),
    }
    write(out / "sequence_intervention_audit.json", qa)
    shutil.copy2(ROOT / "sequence_intervention_audit.py", out / "sequence_intervention_audit_source.py")
    return qa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    qa = run(args.out.resolve())
    print(json.dumps(qa, ensure_ascii=False))


if __name__ == "__main__":
    main()
