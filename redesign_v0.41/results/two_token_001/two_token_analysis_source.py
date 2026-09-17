"""Independent NumPy reanalysis of the v0.41 two-token formation matrix."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
ASSIGNMENTS = ("canonical", "cyclic")
SCHEDULES = ("A", "B", "C")
UPDATES = (0, 100, 600, 1200)
RESOURCES = 3
TOKENS = 2
VOCAB = 7
SITES = 6
MAPS3 = tuple(itertools.permutations(range(SITES), RESOURCES))
PANELS = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
TEAMS = {
    "A": ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)),
    "B": ((0, 2, 1, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0)),
    "C": ((0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0)),
}


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def train_ids(partition):
    rank = {site: i for i, site in enumerate(PANELS[partition - 1])}
    return np.asarray([i for i, triple in enumerate(MAPS3) if rank[int(triple[0])] < rank[int(triple[1])]], dtype=np.int64)


def score(raw, partition):
    decoder = np.argmax(raw["receiver_logits"], axis=-1)
    actions = decoder[raw["receiver_code_ids"]]
    correct = actions == raw["positions"]
    train = np.isin(raw["map_id"], train_ids(partition))
    return {
        split: {
            "J": float(np.mean(np.all(correct[mask], axis=-1))),
            "resource0": float(np.mean(correct[mask, 0])),
            "resource1": float(np.mean(correct[mask, 1])),
            "resource2": float(np.mean(correct[mask, 2])),
        }
        for split, mask in (("train60", train), ("target60", ~train), ("all120", np.ones(len(correct), dtype=bool)))
    }


def agreement_and_information(raw_by_slot, schedule):
    sender_tokens = {resource: {} for resource in range(RESOURCES)}
    sender_positions = {resource: {} for resource in range(RESOURCES)}
    for slot, team in enumerate(TEAMS[schedule]):
        for resource, sender in enumerate(team[1:]):
            sender_tokens[resource][sender] = raw_by_slot[slot]["tokens"][:, resource, :]
            sender_positions[resource][sender] = raw_by_slot[slot]["positions"][:, resource]

    token0_values = []
    token1_values = []
    pair_values = []
    pair_lexicon = []
    for resource in range(RESOURCES):
        token0_values.extend(float(np.mean(sender_tokens[resource][a][:, 0] == sender_tokens[resource][b][:, 0])) for a, b in ((0, 2), (1, 3)))
        token1_values.extend(float(np.mean(sender_tokens[resource][a][:, 1] == sender_tokens[resource][b][:, 1])) for a, b in ((0, 2), (1, 3)))
        pair_values.extend(float(np.mean(np.all(sender_tokens[resource][a] == sender_tokens[resource][b], axis=1))) for a, b in ((0, 2), (1, 3)))
        pair_lexicon.extend(float(len(np.unique(sender_tokens[resource][sender], axis=0))) for sender in range(4))

    sequence_joint = []
    for a, b in ((0, 2), (1, 3)):
        sequence_joint.append(
            float(
                np.mean(
                    np.all(
                        np.stack(
                            [sender_tokens[resource][a] == sender_tokens[resource][b] for resource in range(RESOURCES)],
                            axis=1,
                        ),
                        axis=(1, 2),
                    )
                )
            )
        )

    def normalized_mi(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
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

    mi0 = []
    mi1 = []
    mipair = []
    token_dependency = []
    prefix_sensitivity = []
    for resource in range(RESOURCES):
        for sender in range(4):
            tokens = sender_tokens[resource][sender]
            positions = sender_positions[resource][sender]
            mi0.append(normalized_mi(tokens[:, 0], positions))
            mi1.append(normalized_mi(tokens[:, 1], positions))
            mipair.append(normalized_mi(tokens[:, 0] * VOCAB + tokens[:, 1], positions))
            token_dependency.append(normalized_mi(tokens[:, 0], tokens[:, 1]))
            # The saved conditional table gives a direct sequence diagnostic:
            # if p(token1 | token0) were prefix independent, this value would
            # be zero. It is computed from every possible prefix, not only the
            # greedy prefix used by the behavioral score.
            source_slot = next(slot for slot, team in enumerate(TEAMS[schedule]) if team[1 + resource] == sender)
            t1 = np.exp(raw_by_slot[source_slot][f"sender_log_probs_t1_r{resource}"])
            prefix_mean = t1.mean(axis=1, keepdims=True)
            prefix_sensitivity.append(float(0.5 * np.abs(t1 - prefix_mean).sum(axis=-1).mean()))
    return {
        "token0_agreement": float(np.mean(token0_values)),
        "token1_agreement": float(np.mean(token1_values)),
        "pair_agreement": float(np.mean(pair_values)),
        "sequence_joint_agreement": float(np.mean(sequence_joint)),
        "pair_lexicon_size": float(np.mean(pair_lexicon)),
        "nmi_token0_position": float(np.mean(mi0)),
        "nmi_token1_position": float(np.mean(mi1)),
        "nmi_pair_position": float(np.mean(mipair)),
        "token_dependency_mi": float(np.mean(token_dependency)),
        "prefix_sensitivity_tv": float(np.mean(prefix_sensitivity)),
    }


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "values": a.tolist()}


def auc(values):
    x = np.arange(len(values), dtype=float)
    return float(np.trapezoid(np.asarray(values, dtype=float), x) / x[-1]) if len(values) > 1 else float(values[0])


class Checks:
    def __init__(self):
        self.count = 0
        self.comparisons = 0
        self.max_error = 0.0
        self.coverage = Counter()

    def require(self, value, label):
        self.count += 1
        if not bool(value):
            raise AssertionError(label)

    def exact(self, actual, expected, label):
        self.require(np.array_equal(actual, expected), label)

    def close(self, actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape, label + " shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        self.comparisons += int(a.size)
        error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
        self.max_error = max(self.max_error, error)
        self.require(np.allclose(a, b, rtol=1e-6, atol=1e-6), label + " numerical")

    def add(self, key, value=1):
        self.coverage[key] += value


def check_raw(raw, worlds, checks, label):
    for key in ("map_id", "photo_ids", "positions", "shown"):
        checks.exact(raw[key], worlds[key], label + "/" + key)
    n = len(worlds["map_id"])
    checks.require(raw["tokens"].shape == (n, RESOURCES, TOKENS) and raw["tokens"].dtype.kind in "iu", label + " tokens")
    checks.require(((raw["tokens"] >= 0) & (raw["tokens"] < VOCAB)).all(), label + " token domain")
    for resource in range(RESOURCES):
        t0 = raw[f"sender_log_probs_t0_r{resource}"]
        t1 = raw[f"sender_log_probs_t1_r{resource}"]
        pair = raw[f"sender_log_probs_r{resource}"]
        checks.require(t0.shape == (n, VOCAB) and np.isfinite(t0).all(), label + " token0 table")
        checks.require(t1.shape == (n, VOCAB, VOCAB) and np.isfinite(t1).all(), label + " token1 table")
        checks.require(pair.shape == (n, VOCAB * VOCAB) and np.isfinite(pair).all(), label + " pair table")
        checks.require(np.allclose(np.exp(t0).sum(-1), 1, atol=1e-6), label + " token0 normalization")
        checks.require(np.allclose(np.exp(t1).sum(-1), 1, atol=1e-6), label + " conditional token1 normalization")
        checks.require(np.allclose(np.exp(pair).sum(-1), 1, atol=1e-6), label + " pair normalization")
        expected_pair = (t0[:, :, None] + t1).reshape(n, VOCAB * VOCAB)
        checks.require(np.allclose(pair, expected_pair, atol=1e-6, rtol=1e-6), label + " autoregressive factorization")
    codes, inverse = np.unique(raw["tokens"].reshape(n, RESOURCES * TOKENS), axis=0, return_inverse=True)
    checks.exact(raw["receiver_codes"], codes, label + " receiver codes")
    checks.exact(raw["receiver_code_ids"], inverse, label + " receiver code ids")
    checks.require(raw["receiver_logits"].shape == (len(codes), RESOURCES, SITES), label + " receiver shape")
    checks.require(np.isfinite(raw["receiver_logits"]).all(), label + " receiver finite")
    checks.add("protocol_worlds", n)


def run(out: Path):
    checks = Checks()
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    runs = read(out / "runs.json")
    checks.exact(invocation["formal"], True, "formal invocation")
    checks.exact(invocation["seeds"], list(SEEDS), "seeds")
    checks.exact(invocation["partitions"], list(PARTITIONS), "partitions")
    checks.exact(invocation["conditions"], list(CONDITIONS), "conditions")
    checks.exact(tuple(invocation["assignments"].keys()), ASSIGNMENTS, "assignment order")
    checks.exact(invocation["updates"], 1200, "updates")
    checks.exact(invocation["checkpoints"], list(UPDATES), "checkpoints")
    checks.exact(complete["status"], "complete", "completion")
    checks.exact(complete["formal"], True, "formal completion")
    checks.exact(complete["probe"], "two_token_formation", "probe")
    checks.exact(complete["runs"], 72, "runs")
    checks.exact(runs["count"], 72, "run manifest")
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " identity")
        for path, digest in invocation[key].items():
            checks.require(Path(path).is_file(), key + " bound path")
            checks.exact(sha(Path(path)), digest, key + " bound hash")

    worlds = {assignment: npz(out / f"test_worlds_{assignment}.npz") for assignment in ASSIGNMENTS}
    for assignment in ASSIGNMENTS:
        checks.exact(len(worlds[assignment]["map_id"]), 120, assignment + " test worlds")
    rows = []
    for seed, part, assignment, condition in itertools.product(SEEDS, PARTITIONS, ASSIGNMENTS, CONDITIONS):
        chain = out / "social" / f"s{seed}_p{part}_{assignment}_{condition}"
        cfg = read(chain / "config.json")
        checks.exact(cfg["assignment"], assignment, "config assignment")
        checks.exact(cfg["condition"], condition, "config condition")
        checks.exact(cfg["resource_permutation"], invocation["assignments"][assignment], "resource permutation")
        curve = read(chain / "curve.json")
        checks.exact([int(item["update"]) for item in curve], list(UPDATES), "curve updates")
        previous = {}
        base = {}
        for item in curve:
            update = int(item["update"])
            for schedule in SCHEDULES:
                raws = []
                metrics = []
                for slot in range(4):
                    path = chain / f"protocol_{schedule}_{update:04d}_team{slot}.npz"
                    rel = str(path.relative_to(out))
                    checks.require(rel in complete["files"], "protocol completion hash")
                    raw = npz(path)
                    check_raw(raw, worlds[assignment], checks, f"{assignment}/{condition}/{seed}/{part}/{schedule}/{update}/{slot}")
                    metric = score(raw, part)
                    saved = item["scores"][schedule][f"team{slot}"]
                    for split in ("train60", "target60", "all120"):
                        for key in ("J", "resource0", "resource1", "resource2"):
                            checks.close(metric[split][key], saved[split][key], "saved metric")
                    raws.append(raw)
                    metrics.append(metric)
                    checks.add("protocol_tables")
                ag = agreement_and_information(raws, schedule)
                tokens = np.stack([raw["tokens"] for raw in raws], axis=0)
                if schedule not in previous:
                    previous[schedule] = tokens
                    base[schedule] = tokens
                    change_prev = 0.0
                else:
                    change_prev = float(np.mean(np.any(tokens != previous[schedule], axis=(2, 3))))
                    previous[schedule] = tokens
                change_base = float(np.mean(np.any(tokens != base[schedule], axis=(2, 3))))
                pooled = {
                    split: {
                        key: float(np.mean([m[split][key] for m in metrics]))
                        for key in ("J", "resource0", "resource1", "resource2")
                    }
                    for split in ("train60", "target60", "all120")
                }
                rows.append(
                    {
                        "seed": seed,
                        "partition": part,
                        "assignment": assignment,
                        "condition": condition,
                        "update": update,
                        "schedule": schedule,
                        "train_J": pooled["train60"]["J"],
                        "target_J": pooled["target60"]["J"],
                        "all_J": pooled["all120"]["J"],
                        "resource0": pooled["target60"]["resource0"],
                        "resource1": pooled["target60"]["resource1"],
                        "resource2": pooled["target60"]["resource2"],
                        "token0_agreement": ag["token0_agreement"],
                        "token1_agreement": ag["token1_agreement"],
                        "pair_agreement": ag["pair_agreement"],
                        "sequence_joint_agreement": ag["sequence_joint_agreement"],
                        "pair_lexicon_size": ag["pair_lexicon_size"],
                        "nmi_token0_position": ag["nmi_token0_position"],
                        "nmi_token1_position": ag["nmi_token1_position"],
                        "nmi_pair_position": ag["nmi_pair_position"],
                        "token_dependency_mi": ag["token_dependency_mi"],
                        "prefix_sensitivity_tv": ag["prefix_sensitivity_tv"],
                        "token_change_prev": change_prev,
                        "token_change_base": change_base,
                    }
                )

    metric_keys = (
        "train_J", "target_J", "all_J", "resource0", "resource1", "resource2",
        "token0_agreement", "token1_agreement", "pair_agreement", "sequence_joint_agreement",
        "pair_lexicon_size", "nmi_token0_position", "nmi_token1_position", "nmi_pair_position",
        "token_dependency_mi", "prefix_sensitivity_tv",
        "token_change_prev", "token_change_base",
    )
    summary = {
        assignment: {
            condition: {
                str(update): {
                    schedule: {
                        key: mean_sd([
                            row[key]
                            for row in rows
                            if row["assignment"] == assignment
                            and row["condition"] == condition
                            and row["update"] == update
                            and row["schedule"] == schedule
                        ])
                        for key in metric_keys
                    }
                    for schedule in SCHEDULES
                }
                for update in UPDATES
            }
            for condition in CONDITIONS
        }
        for assignment in ASSIGNMENTS
    }

    formation_auc = {}
    for assignment in ASSIGNMENTS:
        formation_auc[assignment] = {}
        for condition in CONDITIONS:
            formation_auc[assignment][condition] = {}
            for schedule in SCHEDULES:
                curves = []
                for seed, part in itertools.product(SEEDS, PARTITIONS):
                    values = [
                        next(
                            row["target_J"]
                            for row in rows
                            if row["seed"] == seed and row["partition"] == part and row["assignment"] == assignment and row["condition"] == condition and row["schedule"] == schedule and row["update"] == update
                        )
                        for update in UPDATES
                    ]
                    curves.append(auc(values))
                formation_auc[assignment][condition][schedule] = mean_sd(curves)

    condition_effects = {}
    for assignment in ASSIGNMENTS:
        condition_effects[assignment] = {}
        for condition in CONDITIONS:
            condition_effects[assignment][condition] = {}
            for update in UPDATES:
                values = [row["target_J"] for row in rows if row["assignment"] == assignment and row["condition"] == condition and row["update"] == update]
                condition_effects[assignment][condition][str(update)] = mean_sd(values)

    assignment_effects = {}
    for condition in CONDITIONS:
        assignment_effects[condition] = {}
        for schedule in SCHEDULES:
            cyclic = summary["cyclic"][condition]["1200"][schedule]["target_J"]["mean"]
            canonical = summary["canonical"][condition]["1200"][schedule]["target_J"]["mean"]
            assignment_effects[condition][schedule] = {"cyclic_minus_canonical_target_J": cyclic - canonical}

    record = {
        "status": "complete",
        "formal": True,
        "probe": "two_token_formation",
        "seeds": list(SEEDS),
        "partitions": list(PARTITIONS),
        "conditions": list(CONDITIONS),
        "assignments": {"canonical": [0, 1, 2], "cyclic": [1, 2, 0]},
        "schedules": list(SCHEDULES),
        "updates": list(UPDATES),
        "resources": RESOURCES,
        "tokens_per_sender": TOKENS,
        "vocabulary": VOCAB,
        "rows": rows,
        "summary": summary,
        "condition_effects": condition_effects,
        "assignment_effects": assignment_effects,
        "formation_auc": formation_auc,
        "checks": checks.count,
        "scalar_comparisons": checks.comparisons,
        "maximum_metric_absolute_difference": checks.max_error,
        "source_sha256": sha(ROOT / "two_token_analysis.py"),
        "probe_training_complete_sha256": sha(out / "training_complete.json"),
        "coverage": dict(checks.coverage),
        "limits": [
            "The private visual frontend is inherited and frozen.",
            "A six-token tuple is a bounded sequential communication probe, not open-ended language.",
            "The receiver code table materializes only six-token codes observed on the 120-map evaluation table.",
            "Shared reward and joint policy-gradient updates are centralized.",
        ],
    }
    write(out / "two_token_analysis.json", record)
    qa = {
        "passed": True,
        "status": "passed_two_token_recheck",
        "formal": True,
        "checks": checks.count,
        "scalar_comparisons": checks.comparisons,
        "maximum_metric_absolute_difference": checks.max_error,
        "coverage": dict(checks.coverage),
        "analysis_source_sha256": sha(ROOT / "two_token_analysis.py"),
        "analysis_sha256": sha(out / "two_token_analysis.json"),
        "production_modules_imported": False,
        "model_calls": 0,
    }
    write(out / "two_token_raw_validation.json", qa)
    shutil.copy2(ROOT / "two_token_analysis.py", out / "two_token_analysis_source.py")
    return record, qa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        record, qa = run(args.out.resolve())
        qa["seconds"] = time.monotonic() - started
        write(args.out.resolve() / "two_token_raw_validation.json", qa)
        print(json.dumps({key: record[key] for key in ("status", "checks", "scalar_comparisons", "maximum_metric_absolute_difference")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns()
        write(args.out.resolve() / f"two_token_analysis_failure_{stamp}.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "two_token_analysis.py")})
        raise


if __name__ == "__main__":
    main()
