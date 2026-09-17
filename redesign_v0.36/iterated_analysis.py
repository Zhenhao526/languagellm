"""Independent NumPy reanalysis of iterated replacement endpoints."""
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
GENERATIONS = tuple(range(9))
REPLACEMENT_ORDER = (0, 1, 2, 3, 0, 1, 2, 3)
MAPS = tuple((food, water) for food in range(6) for water in range(6) if food != water)
PANELS = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
TARGET_PAIRS = ((0, 1), (0, 3), (1, 2), (1, 4), (2, 0), (2, 5), (3, 2), (3, 4), (4, 0), (4, 5), (5, 1), (5, 3))
TRAIN_PAIRS = ((0, 2), (0, 5), (1, 0), (1, 3), (2, 1), (2, 4), (3, 1), (3, 5), (4, 2), (4, 3), (5, 0), (5, 4))


def read(path: Path): return json.loads(path.read_text())
def write(path: Path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def npz(path: Path):
    with np.load(path, allow_pickle=False) as z: return {key: z[key] for key in z.files}


def ids(partition, pairs):
    order = PANELS[partition - 1]; index = {pair: i for i, pair in enumerate(MAPS)}
    return np.asarray(sorted(index[(order[i], order[j])] for i, j in pairs), dtype=np.int64)


def score(raw, partition):
    decoder = np.argmax(raw["receiver_logits"], axis=-1)
    action = decoder[7 * raw["tokens"][:, 0] + raw["tokens"][:, 1]]
    correct = action == raw["positions"]
    train = np.isin(raw["map_id"], ids(partition, TRAIN_PAIRS)); target = np.isin(raw["map_id"], ids(partition, TARGET_PAIRS)); held = np.isin(raw["map_id"], np.setdiff1d(np.arange(30, dtype=np.int64), ids(partition, TRAIN_PAIRS)))
    return {split: {"J": float(np.mean(np.all(correct[mask], axis=-1))), "food": float(np.mean(correct[mask, 0])), "water": float(np.mean(correct[mask, 1]))} for split, mask in (("train12", train), ("target12", target), ("held18", held))}


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "values": values.tolist()}


class Checks:
    def __init__(self): self.count = 0; self.comparisons = 0; self.max_error = 0.0; self.coverage = Counter()
    def require(self, value, label):
        self.count += 1
        if not bool(value): raise AssertionError(label)
    def exact(self, actual, expected, label): self.require(np.array_equal(actual, expected), label)
    def close(self, actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected); self.require(a.shape == b.shape, label + " shape"); self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        self.comparisons += int(a.size); error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0; self.max_error = max(self.max_error, error); self.require(np.allclose(a, b, rtol=1e-6, atol=1e-6), label + " numerical")
    def add(self, key, value=1): self.coverage[key] += value


def check_raw(raw, worlds, checks, label):
    for key in ("map_id", "photo_ids", "positions", "shown"): checks.exact(raw[key], worlds[key], label + "/" + key)
    n = len(worlds["map_id"]); checks.require(raw["tokens"].shape == (n, 2) and raw["tokens"].dtype.kind in "iu", label + " tokens"); checks.require(((raw["tokens"] >= 0) & (raw["tokens"] < 7)).all(), label + " token domain")
    for key in ("sender_log_probs_food", "sender_log_probs_water"):
        checks.require(raw[key].shape == (n, 7) and np.isfinite(raw[key]).all(), label + " component table"); checks.require(np.allclose(np.exp(raw[key]).sum(-1), 1, atol=1e-6), label + " component normalization")
    checks.require(raw["sender_log_probs"].shape == (n, 49) and np.isfinite(raw["sender_log_probs"]).all(), label + " joint table"); checks.require(np.allclose(np.exp(raw["sender_log_probs"]).sum(-1), 1, atol=1e-6), label + " joint normalization")
    checks.require(np.allclose(raw["sender_log_probs"], (raw["sender_log_probs_food"][:, :, None] + raw["sender_log_probs_water"][:, None, :]).reshape(n, 49), atol=1e-6, rtol=1e-6), label + " factorization")
    checks.require(raw["receiver_logits"].shape == (49, 2, 6) and np.isfinite(raw["receiver_logits"]).all(), label + " receiver"); checks.add("protocol_worlds", n)


def summarize(rows):
    return {"n": len(rows), "target_J": mean_sd([row["target_J"] for row in rows]), "held_J": mean_sd([row["held_J"] for row in rows]), "train_J": mean_sd([row["train_J"] for row in rows]), "token_change_prev": mean_sd([row["token_change_prev"] for row in rows if row["generation"] > 0]) if any(row["generation"] > 0 for row in rows) else None, "token_change_base": mean_sd([row["token_change_base"] for row in rows])}


def run(out: Path):
    checks = Checks(); invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json"); runs = read(out / "runs.json")
    checks.exact(invocation["formal"], True, "formal invocation"); checks.exact(invocation["seeds"], list(SEEDS), "seeds"); checks.exact(invocation["partitions"], list(PARTITIONS), "partitions"); checks.exact(invocation["conditions"], list(CONDITIONS), "conditions"); checks.exact(invocation["generations"], 8, "generations"); checks.exact(invocation["updates_per_generation"], 300, "updates")
    checks.exact(complete["status"], "complete", "completion"); checks.exact(complete["formal"], True, "formal completion"); checks.exact(complete["probe"], "iterated_replacement", "probe"); checks.exact(complete["runs"], 36, "runs"); checks.exact(runs["count"], 36, "run manifest")
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " identity")
        for path, digest in invocation[key].items():
            p = Path(path); checks.require(p.is_file(), key + " bound path"); checks.exact(sha(p), digest, key + " bound hash")
    worlds = npz(out / "test_worlds.npz"); train_worlds = npz(out / "train_worlds.npz"); checks.exact(len(worlds["map_id"]), 180, "test worlds"); checks.exact(len(train_worlds["map_id"]), 720, "train worlds")
    rows = []; chain_rows = []
    for seed, part, condition in itertools.product(SEEDS, PARTITIONS, CONDITIONS):
        chain = out / "chains" / f"s{seed}_p{part}_{condition}"; cfg = read(chain / "config.json"); checks.exact(cfg["condition"], condition, "chain condition"); checks.exact(cfg["replacement_order"], list(REPLACEMENT_ORDER), "replacement order")
        curve = read(chain / "curve.json"); checks.exact([item["generation"] for item in curve], list(GENERATIONS), "curve generations")
        chain_entry = {"seed": seed, "partition": part, "condition": condition, "generations": []}
        base_tokens = {}
        previous_tokens = {}
        for generation in GENERATIONS:
            gen_folder = chain / f"generation_{generation:02d}"; item = curve[generation]; generation_row = {"generation": generation, "replaced_identity": item["replaced_identity"], "schedules": {}}
            for schedule in SCHEDULES:
                team_scores = []; team_tokens = []
                for slot in range(4):
                    path = gen_folder / f"protocol_{schedule}_{generation:04d}_team{slot}.npz"; rel = str(path.relative_to(out)); checks.require(rel in complete["files"], "protocol completion hash")
                    raw = npz(path); check_raw(raw, worlds, checks, f"{condition}/{seed}/{part}/{generation}/{schedule}/{slot}"); metric = score(raw, part); saved = item["scores"][schedule][f"team{slot}"]
                    for split in ("train12", "target12", "held18"):
                        for key in ("J", "food", "water"): checks.close(metric[split][key], saved[split]["pooled"][key], "saved metric")
                    team_scores.append(metric); team_tokens.append(raw["tokens"]); checks.add("protocol_tables")
                tokens = np.stack(team_tokens, axis=0)
                if generation == 0: base_tokens[schedule] = tokens; previous_tokens[schedule] = tokens; change_prev = 0.0
                else: change_prev = float(np.mean(np.any(tokens != previous_tokens[schedule], axis=-1))); previous_tokens[schedule] = tokens
                change_base = float(np.mean(np.any(tokens != base_tokens[schedule], axis=-1)))
                pooled = {split: {key: float(np.mean([metric[split][key] for metric in team_scores])) for key in ("J", "food", "water")} for split in ("train12", "target12", "held18")}
                row = {"seed": seed, "partition": part, "condition": condition, "generation": generation, "replaced_identity": item["replaced_identity"], "schedule": schedule, "train_J": pooled["train12"]["J"], "target_J": pooled["target12"]["J"], "held_J": pooled["held18"]["J"], "token_change_prev": change_prev, "token_change_base": change_base}
                rows.append(row); generation_row["schedules"][schedule] = {"train_J": pooled["train12"]["J"], "target_J": pooled["target12"]["J"], "held_J": pooled["held18"]["J"], "token_change_prev": change_prev, "token_change_base": change_base}
            chain_entry["generations"].append(generation_row)
            checks.add("generations")
        chain_rows.append(chain_entry); checks.add("chains")

    summary = {condition: {str(generation): {schedule: summarize([row for row in rows if row["condition"] == condition and row["generation"] == generation and row["schedule"] == schedule]) for schedule in SCHEDULES} for generation in GENERATIONS} for condition in CONDITIONS}
    final = {condition: {schedule: summary[condition]["8"][schedule] for schedule in SCHEDULES} for condition in CONDITIONS}
    schedule_counts = {condition: [row["generations"] for row in chain_rows if row["condition"] == condition] for condition in CONDITIONS}
    record = {"status": "complete", "formal": True, "probe": "iterated_replacement", "seeds": list(SEEDS), "partitions": list(PARTITIONS), "conditions": list(CONDITIONS), "schedules": list(SCHEDULES), "generations": list(GENERATIONS), "updates_per_generation": 300, "rows": rows, "chain_rows": chain_rows, "summary": summary, "final": final, "checks": checks.count, "scalar_comparisons": checks.comparisons, "maximum_metric_absolute_difference": checks.max_error, "source_sha256": sha(ROOT / "iterated_analysis.py"), "probe_training_complete_sha256": sha(out / "training_complete.json"), "coverage": dict(checks.coverage), "limits": ["The common generation-0 population is a v0.34 fixed-A endpoint; schedule effects begin at replacement generation 1.", "Only newcomer communication modules are optimized and private visual encoders are frozen.", "Token change is an endpoint surface-drift diagnostic, not a semantic distance or a human-language measure."]}
    write(out / "iterated_analysis.json", record)
    qa = {"passed": True, "status": "passed_iterated_replacement_recheck", "formal": True, "checks": checks.count, "scalar_comparisons": checks.comparisons, "maximum_metric_absolute_difference": checks.max_error, "coverage": dict(checks.coverage), "analysis_source_sha256": sha(ROOT / "iterated_analysis.py"), "analysis_sha256": sha(out / "iterated_analysis.json"), "production_modules_imported": False, "model_calls": 0}
    write(out / "iterated_raw_validation.json", qa); shutil.copy2(ROOT / "iterated_analysis.py", out / "iterated_analysis_source.py")
    return record, qa


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); started = time.monotonic()
    try:
        record, qa = run(args.out.resolve()); qa["seconds"] = time.monotonic() - started; write(args.out.resolve() / "iterated_raw_validation.json", qa); print(json.dumps({key: record[key] for key in ("status", "checks", "scalar_comparisons", "maximum_metric_absolute_difference")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns(); write(args.out.resolve() / f"iterated_analysis_failure_{stamp}.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "iterated_analysis.py")}); raise


if __name__ == "__main__": main()
