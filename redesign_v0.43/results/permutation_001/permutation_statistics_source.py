"""Descriptive paired statistics for v0.43 (NumPy only, no model imports)."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
MODES = ("static_role", "random_role")
ROLES = ("012", "021", "102", "120", "201", "210")
BOOTSTRAPS = 10000
SEED = 43045


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_sd(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def ci(x, rng):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [0.0, 0.0]
    indices = rng.integers(0, len(x), size=(BOOTSTRAPS, len(x)))
    means = x[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summary(x, rng):
    m, s = mean_sd(x)
    return {"n": int(len(x)), "mean": m, "sd": s, "ci95_bootstrap": ci(x, rng), "values": np.asarray(x, dtype=float).tolist()}


def paired(rows, condition, role="012"):
    key = lambda r: (r["seed"], r["partition"], r["assignment"], r["schedule"], r["slot"])
    static = {key(r): r for r in rows if r["condition"] == condition and r["role_mode"] == "static_role" and r["update"] == 600 and r["role_permutation"] == role}
    random = {key(r): r for r in rows if r["condition"] == condition and r["role_mode"] == "random_role" and r["update"] == 600 and r["role_permutation"] == role}
    keys = sorted(set(static) & set(random))
    return np.asarray([random[k]["equivariant"]["target60"]["J"] - static[k]["equivariant"]["target60"]["J"] for k in keys], dtype=float)


def role_spread(rows, condition, mode):
    groups = defaultdict(dict)
    for r in rows:
        if r["condition"] == condition and r["role_mode"] == mode and r["update"] == 600:
            key = (r["seed"], r["partition"], r["assignment"], r["schedule"], r["slot"])
            groups[key][r["role_permutation"]] = r["equivariant"]["target60"]["J"]
    return np.asarray([max(v.values()) - min(v.values()) for v in groups.values()], dtype=float)


def endpoint_values(rows, condition, mode, role, kind):
    return np.asarray([r[kind]["target60"]["J"] for r in rows if r["condition"] == condition and r["role_mode"] == mode and r["role_permutation"] == role and r["update"] == 600], dtype=float)


def run(out):
    data = read(out / "permutation_analysis.json")
    rows = data["rows"]
    rng = np.random.default_rng(SEED)
    record = {"status": "complete", "formal": True, "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": SEED, "identity_role_mode_differences": {}, "role_spread": {}, "nonidentity_literal_equivariant": {}, "assignment_identity_endpoint": {}}
    for condition in CONDITIONS:
        delta = paired(rows, condition, "012")
        record["identity_role_mode_differences"][condition] = {"random_minus_static_equivariant_target_J": summary(delta, rng), "positive_fraction": float(np.mean(delta > 0))}
        record["role_spread"][condition] = {}
        for mode in MODES:
            values = role_spread(rows, condition, mode)
            record["role_spread"][condition][mode] = summary(values, rng)
        spread_delta = role_spread(rows, condition, "random_role") - role_spread(rows, condition, "static_role")
        record["role_spread"][condition]["random_minus_static"] = summary(spread_delta, rng)
        record["nonidentity_literal_equivariant"][condition] = {}
        for mode in MODES:
            lit = np.concatenate([endpoint_values(rows, condition, mode, role, "literal") for role in ROLES[1:]])
            eq = np.concatenate([endpoint_values(rows, condition, mode, role, "equivariant") for role in ROLES[1:]])
            record["nonidentity_literal_equivariant"][condition][mode] = {"literal": summary(lit, rng), "equivariant": summary(eq, rng), "equivariant_minus_literal": summary(eq - lit, rng)}
        record["assignment_identity_endpoint"][condition] = {}
        for mode in MODES:
            means = {}
            for assignment in ROLES:
                values = [r["equivariant"]["target60"]["J"] for r in rows if r["condition"] == condition and r["role_mode"] == mode and r["assignment"] == assignment and r["role_permutation"] == "012" and r["update"] == 600]
                means[assignment] = mean_sd(values)[0]
            record["assignment_identity_endpoint"][condition][mode] = {"means": means, "min": float(min(means.values())), "max": float(max(means.values())), "range": float(max(means.values()) - min(means.values()))}
    record["source_sha256"] = sha(ROOT / "permutation_statistics.py")
    record["analysis_sha256"] = sha(out / "permutation_analysis.json")
    (out / "permutation_statistics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (out / "permutation_statistics_source.py").write_text((ROOT / "permutation_statistics.py").read_text())
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = run(args.out.resolve())
    print(json.dumps({"status": record["status"], "bootstrap_replicates": record["bootstrap_replicates"], "conditions": list(CONDITIONS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
