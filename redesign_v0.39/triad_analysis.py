"""Independent NumPy reanalysis of the v0.39 three-resource formation runs."""
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
SCHEDULES = ("A", "B", "C")
CHECKPOINTS = (0, 100, 600, 1200)
RESOURCES = 3
VOCAB = 7
SITES = 6
MAPS3 = tuple(itertools.permutations(range(SITES), RESOURCES))
PANELS = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
TEAMS = {"A": ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)), "B": ((0, 2, 1, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0)), "C": ((0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0))}


def read(path: Path): return json.loads(path.read_text())
def write(path: Path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def npz(path: Path):
    with np.load(path, allow_pickle=False) as z: return {key: z[key] for key in z.files}


def train_ids(partition):
    rank = {site: i for i, site in enumerate(PANELS[partition - 1])}
    return np.asarray([i for i, triple in enumerate(MAPS3) if rank[int(triple[0])] < rank[int(triple[1])]], dtype=np.int64)


def score(raw, partition):
    decoder = np.argmax(raw["receiver_logits"], axis=-1); code = raw["tokens"][:, 0] * VOCAB * VOCAB + raw["tokens"][:, 1] * VOCAB + raw["tokens"][:, 2]; actions = decoder[code]; correct = actions == raw["positions"]; train = np.isin(raw["map_id"], train_ids(partition)); target = ~train
    return {split: {"J": float(np.mean(np.all(correct[mask], axis=-1))), "resource0": float(np.mean(correct[mask, 0])), "resource1": float(np.mean(correct[mask, 1])), "resource2": float(np.mean(correct[mask, 2]))} for split, mask in (("train60", train), ("target60", target), ("all120", np.ones(len(correct), dtype=bool)))}


def agreement(raw_by_slot, schedule):
    sender_tokens = {resource: {} for resource in range(RESOURCES)}
    for slot, team in enumerate(TEAMS[schedule]):
        for resource, sender in enumerate(team[1:]): sender_tokens[resource][sender] = raw_by_slot[slot]["tokens"][:, resource]
    pair_values = []
    per_resource = []
    for resource in range(RESOURCES):
        values = [float(np.mean(sender_tokens[resource][a] == sender_tokens[resource][b])) for a, b in ((0, 2), (1, 3))]; per_resource.append(float(np.mean(values)))
    for a, b in ((0, 2), (1, 3)):
        pair_values.append(float(np.mean(np.all(np.stack([sender_tokens[r][a] == sender_tokens[r][b] for r in range(RESOURCES)], axis=1), axis=1))))
    return {"resource0": per_resource[0], "resource1": per_resource[1], "resource2": per_resource[2], "joint": float(np.mean(pair_values))}


def mean_sd(values):
    a = np.asarray(values, dtype=float); return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "values": a.tolist()}


def auc(values): return float(np.trapezoid(np.asarray(values, dtype=float), np.asarray(CHECKPOINTS, dtype=float)) / CHECKPOINTS[-1])


class Checks:
    def __init__(self): self.count = 0; self.comparisons = 0; self.max_error = 0.0; self.coverage = Counter()
    def require(self, value, label):
        self.count += 1
        if not bool(value): raise AssertionError(label)
    def exact(self, actual, expected, label): self.require(np.array_equal(actual, expected), label)
    def close(self, actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected); self.require(a.shape == b.shape, label + " shape"); self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite"); self.comparisons += int(a.size); error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0; self.max_error = max(self.max_error, error); self.require(np.allclose(a, b, rtol=1e-6, atol=1e-6), label + " numerical")
    def add(self, key, value=1): self.coverage[key] += value


def check_raw(raw, worlds, checks, label):
    for key in ("map_id", "photo_ids", "positions", "shown"): checks.exact(raw[key], worlds[key], label + "/" + key)
    n = len(worlds["map_id"]); checks.require(raw["tokens"].shape == (n, RESOURCES) and raw["tokens"].dtype.kind in "iu", label + " tokens"); checks.require(((raw["tokens"] >= 0) & (raw["tokens"] < VOCAB)).all(), label + " token domain")
    for resource in range(RESOURCES):
        key = f"sender_log_probs_r{resource}"; checks.require(raw[key].shape == (n, VOCAB) and np.isfinite(raw[key]).all(), label + " component table"); checks.require(np.allclose(np.exp(raw[key]).sum(-1), 1, atol=1e-6), label + " component normalization")
    checks.require(raw["sender_log_probs"].shape == (n, VOCAB ** RESOURCES) and np.isfinite(raw["sender_log_probs"]).all(), label + " joint table"); checks.require(np.allclose(np.exp(raw["sender_log_probs"]).sum(-1), 1, atol=1e-6), label + " joint normalization")
    expected = (raw["sender_log_probs_r0"][:, :, None, None] + raw["sender_log_probs_r1"][:, None, :, None] + raw["sender_log_probs_r2"][:, None, None, :]).reshape(n, VOCAB ** RESOURCES)
    checks.require(np.allclose(raw["sender_log_probs"], expected, atol=1e-6, rtol=1e-6), label + " factorization"); checks.require(raw["receiver_logits"].shape == (VOCAB ** RESOURCES, RESOURCES, SITES) and np.isfinite(raw["receiver_logits"]).all(), label + " receiver"); checks.add("protocol_worlds", n)


def run(out: Path):
    checks = Checks(); invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json"); runs = read(out / "runs.json")
    checks.exact(invocation["formal"], True, "formal invocation"); checks.exact(invocation["seeds"], list(SEEDS), "seeds"); checks.exact(invocation["partitions"], list(PARTITIONS), "partitions"); checks.exact(invocation["conditions"], list(CONDITIONS), "conditions"); checks.exact(invocation["updates"], 1200, "updates"); checks.exact(invocation["checkpoints"], list(CHECKPOINTS), "checkpoints"); checks.exact(complete["status"], "complete", "completion"); checks.exact(complete["formal"], True, "formal completion"); checks.exact(complete["probe"], "triad_origin", "probe"); checks.exact(complete["runs"], 36, "runs"); checks.exact(runs["count"], 36, "run manifest")
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " identity")
        for path, digest in invocation[key].items(): checks.require(Path(path).is_file(), key + " bound path"); checks.exact(sha(Path(path)), digest, key + " bound hash")
    worlds = npz(out / "test_worlds.npz"); checks.exact(len(worlds["map_id"]), 120, "test worlds"); rows = []
    for part in PARTITIONS:
        train_worlds = npz(out / f"train_worlds_p{part}.npz"); checks.exact(len(train_worlds["map_id"]), 480, f"train worlds p{part}")
    for seed, part, condition in itertools.product(SEEDS, PARTITIONS, CONDITIONS):
        folder = out / "social" / f"s{seed}_p{part}_{condition}"; cfg = read(folder / "config.json"); checks.exact(cfg["condition"], condition, "config condition"); curve = read(folder / "curve.json"); checks.exact([x["update"] for x in curve], list(CHECKPOINTS), "curve updates")
        previous = {}; base = {}
        for item in curve:
            update = int(item["update"])
            for schedule in SCHEDULES:
                raws = []; metrics = []
                for slot in range(4):
                    path = folder / f"protocol_{schedule}_{update:04d}_team{slot}.npz"; rel = str(path.relative_to(out)); checks.require(rel in complete["files"], "protocol completion hash"); raw = npz(path); check_raw(raw, worlds, checks, f"{condition}/{seed}/{part}/{schedule}/{update}/{slot}"); metric = score(raw, part); saved = item["scores"][schedule][f"team{slot}"]
                    for split in ("train60", "target60", "all120"):
                        for key in ("J", "resource0", "resource1", "resource2"): checks.close(metric[split][key], saved[split][key], "saved metric")
                    raws.append(raw); metrics.append(metric); checks.add("protocol_tables")
                tokens = np.stack([raw["tokens"] for raw in raws], axis=0); ag = agreement(raws, schedule)
                if schedule not in previous: previous[schedule] = tokens; base[schedule] = tokens; change_prev = 0.0
                else: change_prev = float(np.mean(np.any(tokens != previous[schedule], axis=-1))); previous[schedule] = tokens
                change_base = float(np.mean(np.any(tokens != base[schedule], axis=-1))); pooled = {split: {key: float(np.mean([m[split][key] for m in metrics])) for key in ("J", "resource0", "resource1", "resource2")} for split in ("train60", "target60", "all120")}
                rows.append({"seed": seed, "partition": part, "condition": condition, "update": update, "schedule": schedule, "train_J": pooled["train60"]["J"], "target_J": pooled["target60"]["J"], "all_J": pooled["all120"]["J"], "resource0": pooled["target60"]["resource0"], "resource1": pooled["target60"]["resource1"], "resource2": pooled["target60"]["resource2"], "agreement_resource0": ag["resource0"], "agreement_resource1": ag["resource1"], "agreement_resource2": ag["resource2"], "agreement_joint": ag["joint"], "token_change_prev": change_prev, "token_change_base": change_base})
    summary = {condition: {schedule: {str(update): {key: mean_sd([row[key] for row in rows if row["condition"] == condition and row["schedule"] == schedule and row["update"] == update]) for key in ("train_J", "target_J", "all_J", "resource0", "resource1", "resource2", "agreement_resource0", "agreement_resource1", "agreement_resource2", "agreement_joint", "token_change_prev", "token_change_base")} for update in CHECKPOINTS} for schedule in SCHEDULES} for condition in CONDITIONS}
    formation_auc = {}
    for condition in CONDITIONS:
        formation_auc[condition] = {}
        for schedule in SCHEDULES:
            curves = []
            for seed, part in itertools.product(SEEDS, PARTITIONS):
                curve = [next(row for row in rows if row["seed"] == seed and row["partition"] == part and row["condition"] == condition and row["schedule"] == schedule and row["update"] == update)["target_J"] for update in CHECKPOINTS]
                curves.append(auc(curve))
            formation_auc[condition][schedule] = mean_sd(curves)
    record = {"status": "complete", "formal": True, "probe": "triad_origin", "seeds": list(SEEDS), "partitions": list(PARTITIONS), "conditions": list(CONDITIONS), "schedules": list(SCHEDULES), "resources": RESOURCES, "vocabulary": VOCAB, "checkpoints": list(CHECKPOINTS), "rows": rows, "summary": summary, "formation_auc": formation_auc, "checks": checks.count, "scalar_comparisons": checks.comparisons, "maximum_metric_absolute_difference": checks.max_error, "source_sha256": sha(ROOT / "triad_analysis.py"), "probe_training_complete_sha256": sha(out / "training_complete.json"), "coverage": dict(checks.coverage), "limits": ["The visual frontend is inherited from two-resource perceptual training and remains frozen.", "The three-token tuple is a controlled composition probe, not open-ended language.", "The reward and joint gradient update are centralized."]}; write(out / "triad_analysis.json", record); qa = {"passed": True, "status": "passed_triad_origin_recheck", "formal": True, "checks": checks.count, "scalar_comparisons": checks.comparisons, "maximum_metric_absolute_difference": checks.max_error, "coverage": dict(checks.coverage), "analysis_source_sha256": sha(ROOT / "triad_analysis.py"), "analysis_sha256": sha(out / "triad_analysis.json"), "production_modules_imported": False, "model_calls": 0}; write(out / "triad_raw_validation.json", qa); shutil.copy2(ROOT / "triad_analysis.py", out / "triad_analysis_source.py"); return record, qa


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); started = time.monotonic()
    try:
        record, qa = run(args.out.resolve()); qa["seconds"] = time.monotonic() - started; write(args.out.resolve() / "triad_raw_validation.json", qa); print(json.dumps({k: record[k] for k in ("status", "checks", "scalar_comparisons", "maximum_metric_absolute_difference")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns(); write(args.out.resolve() / f"triad_analysis_failure_{stamp}.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "triad_analysis.py")}); raise


if __name__ == "__main__": main()
