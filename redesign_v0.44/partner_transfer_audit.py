"""Independent NumPy audit for v0.44 partner-transfer arrays."""
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
OUT = ROOT / "results" / "partner_transfer_001"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
ASSIGNMENTS = ("012", "021", "102", "120", "201", "210")
CULTURES = tuple(f"{mode}__{condition}" for mode in ("static_role", "random_role") for condition in ("fixed_A", "rotating_AB", "random_ABC"))
SCHEDULES = ("A", "B", "C")
ROLE_PERMS = tuple(itertools.permutations(range(3)))
REPLACEMENTS = ("none", "slot0", "slot1", "slot2", "all")
METRICS = 4
SITES = 6
RESOURCES = 3
BATCH = 120


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


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

    def close(self, actual, expected, label, atol=1e-6, rtol=1e-6):
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        self.require(a.shape == b.shape, label + " shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        self.comparisons += int(a.size)
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        self.max_error = max(self.max_error, error)
        self.require(np.allclose(a, b, atol=atol, rtol=rtol), label)


def train_ids(partition):
    panels = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
    maps = np.asarray(list(itertools.permutations(range(SITES), RESOURCES)), dtype=np.int64)
    rank = {site: i for i, site in enumerate(panels[partition - 1])}
    return np.asarray([i for i, triple in enumerate(maps) if rank[int(triple[0])] < rank[int(triple[1])]], dtype=np.int64)


def replay_metrics(actions, positions, map_id, partition, role_perm):
    correct_literal = actions == positions
    correct_equiv = actions == positions[:, np.asarray(role_perm, dtype=np.int64)]
    train = np.isin(map_id, train_ids(partition))
    result = np.zeros((2, 2, METRICS), dtype=np.float32)
    for split, mask in enumerate((~train, np.ones(len(map_id), dtype=bool))):
        for kind, correct in enumerate((correct_literal, correct_equiv)):
            result[split, kind, 0] = np.mean(np.all(correct[mask], axis=-1))
            result[split, kind, 1:] = np.mean(correct[mask], axis=0)
    return result


def run(out: Path):
    checks = Checks()
    invocation = read(OUT / "invocation.json")
    complete = read(OUT / "transfer_complete.json")
    checks.require(invocation.get("formal") is True and invocation.get("version") == "v0.44-partner-transfer-endpoint", "formal invocation")
    checks.require(complete.get("formal") is True and complete.get("status") == "complete" and complete.get("groups") == 72, "formal completion")
    for kind in ("source_hashes", "input_hashes"):
        for path, digest in invocation[kind].items():
            p = Path(path)
            checks.require(p.is_file(), f"{kind} exists {path}")
            checks.require(sha(p) == digest, f"{kind} hash {path}")
            checks.coverage[f"{kind}_hashes"] += 1
    groups = 0
    for seed, part, assignment in itertools.product(SEEDS, PARTITIONS, ASSIGNMENTS):
        path = OUT / "groups" / f"transfer_s{seed}_p{part}_{assignment}.npz"
        d = load(path)
        checks.exact(d["culture_labels"], np.asarray(CULTURES), f"{path.name} cultures")
        checks.exact(d["replacement_modes"], np.asarray(REPLACEMENTS), f"{path.name} replacements")
        checks.exact(d["schedules"], np.asarray(SCHEDULES), f"{path.name} schedules")
        checks.exact(d["role_permutations"], np.asarray(ROLE_PERMS, dtype=np.int64), f"{path.name} roles")
        checks.require(d["map_id"].shape == (BATCH,) and d["positions"].shape == (BATCH, RESOURCES), f"{path.name} worlds")
        checks.require(d["tokens"].shape == (6, 4, RESOURCES, BATCH, 2), f"{path.name} tokens")
        checks.require(d["actions"].shape == (6, 6, 5, 3, 6, 4, BATCH, RESOURCES), f"{path.name} actions")
        checks.require(d["scores"].shape == (6, 6, 5, 3, 6, 4, 2, 2, METRICS), f"{path.name} scores")
        checks.require(np.isin(d["tokens"], np.arange(7)).all() and np.isin(d["actions"], np.arange(SITES)).all(), f"{path.name} domains")
        checks.require(np.isfinite(d["scores"]).all(), f"{path.name} finite scores")
        for recipient, donor, replacement, schedule, role, slot in itertools.product(range(6), range(6), range(5), range(3), range(6), range(4)):
            action = d["actions"][recipient, donor, replacement, schedule, role, slot]
            expected = replay_metrics(action, d["positions"], d["map_id"], part, ROLE_PERMS[role])
            checks.close(d["scores"][recipient, donor, replacement, schedule, role, slot], expected, f"{path.name} score replay")
            if replacement == 0 and donor > 0:
                checks.exact(action, d["actions"][recipient, 0, replacement, schedule, role, slot], f"{path.name} none action invariance")
                checks.close(d["scores"][recipient, donor, replacement, schedule, role, slot], d["scores"][recipient, 0, replacement, schedule, role, slot], f"{path.name} none score invariance")
            checks.coverage["score_cells"] += 1
        # A direct token-domain and no-NaN check on each pair also guards the
        # cultural-comparison inputs used by the analysis.
        checks.require(np.isin(d["tokens"], np.arange(7)).all(), f"{path.name} token replay domain")
        checks.coverage["groups"] += 1
        groups += 1
    checks.require(groups == 72, "group coverage")
    checks.require(checks.coverage["score_cells"] == 72 * 6 * 6 * 5 * 3 * 6 * 4, "score coverage")
    result = {
        "status": "complete", "formal": True, "probe": "partner_transfer_endpoint", "groups": groups,
        "checks": checks.checks, "scalar_comparisons": checks.comparisons, "maximum_replay_absolute_difference": checks.max_error,
        "coverage": dict(checks.coverage), "source_sha256": sha(ROOT / "partner_transfer_audit.py"),
        "production_modules_imported": False, "model_calls": 0,
        "limits": ["This audit replays stored action arrays and endpoint metrics; it does not independently run the neural models.", "Native v0.43 provenance is checked in partner_transfer_analysis.json."],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "partner_transfer_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    shutil.copy2(ROOT / "partner_transfer_audit.py", out / "partner_transfer_audit_source.py")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    result = run(args.out.resolve())
    result["seconds"] = time.monotonic() - started
    (args.out.resolve() / "partner_transfer_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "groups": result["groups"], "checks": result["checks"], "scalar_comparisons": result["scalar_comparisons"], "maximum_replay_absolute_difference": result["maximum_replay_absolute_difference"], "seconds": result["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
