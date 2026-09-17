"""Independent NumPy reanalysis of the v0.44 partner-transfer endpoint assay."""
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
V043 = PROJECT / "redesign_v0.43"
OUT = ROOT / "results" / "partner_transfer_001"
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
METRICS = ("J", "resource0", "resource1", "resource2")
SPLITS = ("target60", "all120")


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def ms(values):
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0}


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
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.require(a.shape == b.shape, label + " shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        self.comparisons += int(a.size)
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        self.max_error = max(self.max_error, err)
        self.require(np.allclose(a, b, atol=atol, rtol=rtol), label)


def add(bucket, key, value):
    bucket[key].append(float(value))


def category(recipient, donor):
    rm, rc = CULTURES[recipient].split("__", 1)
    dm, dc = CULTURES[donor].split("__", 1)
    if recipient == donor:
        return "same_culture"
    if rm == dm:
        return "same_role_mode_different_topology"
    if rc == dc:
        return "different_role_mode_same_topology"
    return "different_role_mode_and_topology"


def run(out: Path):
    checks = Checks()
    invocation = read(OUT / "invocation.json")
    complete = read(OUT / "transfer_complete.json")
    v043_analysis = read(V043 / "results" / "permutation_001" / "permutation_analysis.json")
    checks.require(invocation.get("formal") is True and invocation.get("version") == "v0.44-partner-transfer-endpoint", "formal invocation")
    checks.require(complete.get("formal") is True and complete.get("status") == "complete" and complete.get("groups") == 72, "formal transfer completion")
    checks.require(v043_analysis.get("status") == "complete" and v043_analysis.get("runs") == 432, "v0.43 endpoint source")
    checks.exact(invocation["cultures"], list(CULTURES), "culture order")
    checks.exact(invocation["replacement_modes"], list(REPLACEMENTS), "replacement order")
    # One native endpoint row per v0.43 culture/schedule/role/team is used as
    # an exact provenance control for the transfer inference.
    endpoint_rows = {}
    for row in v043_analysis["rows"]:
        if row["update"] == 600:
            key = (row["seed"], row["partition"], row["assignment"], row["role_mode"], row["condition"], row["schedule"], row["role_permutation"], row["slot"])
            endpoint_rows[key] = row
    checks.require(len(endpoint_rows) == 432 * 3 * 6 * 4, "v0.43 endpoint row coverage")
    aggregate = defaultdict(list)
    drop_aggregate = defaultdict(list)
    token_agreement = defaultdict(list)
    native_values = defaultdict(list)
    group_count = 0
    for seed, part, assignment in itertools.product(SEEDS, PARTITIONS, ASSIGNMENTS):
        path = OUT / "groups" / f"transfer_s{seed}_p{part}_{assignment}.npz"
        d = npz(path)
        checks.exact(d["culture_labels"], np.asarray(CULTURES), f"{path.name} culture labels")
        checks.exact(d["replacement_modes"], np.asarray(REPLACEMENTS), f"{path.name} replacement labels")
        checks.exact(d["schedules"], np.asarray(SCHEDULES), f"{path.name} schedules")
        checks.exact(d["role_permutations"], np.asarray(ROLE_PERMS, dtype=np.int64), f"{path.name} role permutations")
        checks.require(d["map_id"].shape == (120,) and d["positions"].shape == (120, 3), f"{path.name} world shapes")
        checks.require(d["tokens"].shape == (6, 4, 3, 120, 2), f"{path.name} token shape")
        checks.require(d["actions"].shape == (6, 6, 5, 3, 6, 4, 120, 3), f"{path.name} action shape")
        checks.require(d["scores"].shape == (6, 6, 5, 3, 6, 4, 2, 2, 4), f"{path.name} score shape")
        checks.require(np.isin(d["tokens"], np.arange(7)).all() and np.isin(d["actions"], np.arange(6)).all(), f"{path.name} domains")
        checks.require(np.isfinite(d["scores"]).all(), f"{path.name} finite scores")
        for recipient, donor, replacement, schedule, role, slot in itertools.product(range(6), range(6), range(5), range(3), range(6), range(4)):
            # None replacement does not use a donor; all donor replicas must be
            # byte-identical and the donor=recipient cell is the native result.
            if replacement == 0 and donor > 0:
                checks.exact(d["actions"][recipient, donor, replacement, schedule, role, slot], d["actions"][recipient, 0, replacement, schedule, role, slot], f"{path.name} none action donor invariance")
                checks.close(d["scores"][recipient, donor, replacement, schedule, role, slot], d["scores"][recipient, 0, replacement, schedule, role, slot], f"{path.name} none score donor invariance")
            role_key = ROLE_KEYS[role]
            for kind_i, kind in enumerate(("literal", "equivariant")):
                for split_i, split in enumerate(SPLITS):
                    for metric_i, metric_name in enumerate(METRICS):
                        if replacement == 0 and donor == 0:
                            endpoint_key = (seed, part, assignment, *CULTURES[recipient].split("__", 1), SCHEDULES[schedule], role_key, slot)
                            expected = endpoint_rows[endpoint_key][kind][split][metric_name]
                            checks.close(d["scores"][recipient, donor, replacement, schedule, role, slot, split_i, kind_i, metric_i], expected, f"{path.name} native v0.43 provenance")
                        value = d["scores"][recipient, donor, replacement, schedule, role, slot, split_i, kind_i, metric_i]
                        aggregate[(recipient, donor, replacement, schedule, role, kind, split, metric_name)].append(float(value))
            if replacement > 0:
                native = d["scores"][recipient, recipient, 0, schedule, role, slot, 0, 1, 0]
                transfer = d["scores"][recipient, donor, replacement, schedule, role, slot, 0, 1, 0]
                drop_aggregate[(recipient, donor, replacement)].append(float(native - transfer))
        # Sender-form agreement is computed over identity, resource and test
        # worlds for each pair of cultures in this visual group.
        for recipient, donor in itertools.product(range(6), range(6)):
            for identity, resource in itertools.product(range(4), range(3)):
                a = d["tokens"][recipient, identity, resource]
                b = d["tokens"][donor, identity, resource]
                for token in range(2):
                    add(token_agreement, (recipient, donor, f"token{token}"), np.mean(a[:, token] == b[:, token]))
                add(token_agreement, (recipient, donor, "pair"), np.mean(np.all(a == b, axis=-1)))
        group_count += 1
        checks.coverage["groups"] += 1
    # Collapse the detailed transfer cells to readable factors.
    summary = {}
    for recipient in range(6):
        summary[CULTURES[recipient]] = {}
        for donor in range(6):
            summary[CULTURES[recipient]][CULTURES[donor]] = {}
            for replacement in range(5):
                summary[CULTURES[recipient]][CULTURES[donor]][REPLACEMENTS[replacement]] = {}
                for schedule in range(3):
                    summary[CULTURES[recipient]][CULTURES[donor]][REPLACEMENTS[replacement]][SCHEDULES[schedule]] = {}
                    for role in range(6):
                        cell = summary[CULTURES[recipient]][CULTURES[donor]][REPLACEMENTS[replacement]][SCHEDULES[schedule]][ROLE_KEYS[role]] = {}
                        for kind in ("literal", "equivariant"):
                            for split in SPLITS:
                                for metric in METRICS:
                                    cell[f"{kind}_{split}_{metric}"] = ms(aggregate[(recipient, donor, replacement, schedule, role, kind, split, metric)])
    categories = {}
    for target_category in ("same_culture", "same_role_mode_different_topology", "different_role_mode_same_topology", "different_role_mode_and_topology"):
        categories[target_category] = {}
        for recipient in range(6):
            categories[target_category][CULTURES[recipient]] = {}
            for replacement in range(5):
                values = []
                literal = []
                for (r, donor, repl, schedule, role, kind, split, metric), arr in aggregate.items():
                    if r == recipient and repl == replacement and kind == "equivariant" and split == "target60" and metric == "J" and category(recipient, donor) == target_category:
                        values.extend(arr)
                    if r == recipient and repl == replacement and kind == "literal" and split == "target60" and metric == "J" and category(recipient, donor) == target_category:
                        literal.extend(arr)
                categories[target_category][CULTURES[recipient]][REPLACEMENTS[replacement]] = {"equivariant_target_J": ms(values), "literal_target_J": ms(literal)}
    drops = {f"{CULTURES[r]}__{CULTURES[d]}__{REPLACEMENTS[repl]}": ms(values) for (r, d, repl), values in drop_aggregate.items()}
    token_summary = {f"{CULTURES[r]}__{CULTURES[d]}": {key: ms(values) for (rr, dd, key), values in token_agreement.items() if rr == r and dd == d} for r in range(6) for d in range(6)}
    record = {
        "status": "complete", "formal": True, "probe": "partner_transfer_endpoint", "groups": group_count,
        "cultures": list(CULTURES), "replacement_modes": list(REPLACEMENTS), "schedules": list(SCHEDULES), "role_permutations": list(ROLE_KEYS),
        "summary": summary, "category_summary": categories, "replacement_drop": drops, "token_summary": token_summary,
        "native_replay_mismatches": 0, "checks": checks.checks, "scalar_comparisons": checks.comparisons,
        "maximum_metric_absolute_difference": checks.max_error, "coverage": dict(checks.coverage),
        "source_sha256": sha(ROOT / "partner_transfer_analysis.py"), "v043_analysis_sha256": sha(V043 / "results" / "permutation_001" / "permutation_analysis.json"),
        "limits": ["Endpoint compatibility only; no post-replacement adaptation was trained.", "Donor and recipient share seed, partition and resource-column assignment within each visual group, isolating communication compatibility from perceptual mismatch.", "The finite two-token, six-site task remains a grounded protocol assay rather than open-ended language."],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "partner_transfer_analysis.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    shutil.copy2(ROOT / "partner_transfer_analysis.py", out / "partner_transfer_analysis_source.py")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    result = run(args.out.resolve())
    result["seconds"] = time.monotonic() - started
    (args.out.resolve() / "partner_transfer_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "groups": result["groups"], "checks": result["checks"], "scalar_comparisons": result["scalar_comparisons"], "maximum_metric_absolute_difference": result["maximum_metric_absolute_difference"], "seconds": result["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
