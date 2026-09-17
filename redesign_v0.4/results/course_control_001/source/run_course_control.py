"""Direct-full-task control for the completed curriculum_001 baseline.

The original training function is reused unchanged. Only the first 600 updates'
scene distribution changes: full task from the first update, instead of the
eight-case curriculum. Both schedules retain 900 updates of action exploration,
then 300 updates of pure resource reward, with no signalling auxiliary loss.

Run --validate-only to check the matched design and saved artifacts without
training. The normal entry point runs exactly three fixed-budget controls.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import run_curriculum as original
from run_pilot import ImageBank, write_json


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "results/curriculum_001"
SOURCE_NAMES = ("run_curriculum.py", "curriculum_eval.py", "run_pilot.py",
                "agents.py", "resource_env.py")
DATA_NAMES = ("manifest.json", "features.npz", "encoder_report.json")
CONFIG_CHANGES = {
    "conditions": ["baseline"],
    "course_updates": 0,
    "transition_updates": 900,
    "signalling_coefficient": 0.0,
}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_config(reference=REFERENCE):
    """Derive all unaltered parameters from the saved completed experiment."""
    config = copy.deepcopy(read_json(Path(reference) / "config.json"))
    config.update(copy.deepcopy(CONFIG_CHANGES))
    return config


def same_states(left, right):
    """Compare tensor values, not PyTorch archive names or serialization bytes."""
    return len(left) == len(right) and all(
        a.keys() == b.keys() and all(torch.equal(a[name], b[name]) for name in a)
        for a, b in zip(left, right))


def validate_design(config, reference_config):
    allowed = set(CONFIG_CHANGES)
    assert config.keys() == reference_config.keys(), "configuration schema changed"
    for key, value in reference_config.items():
        if key not in allowed:
            assert config[key] == value, f"unintended configuration change: {key}"
    for key, value in CONFIG_CHANGES.items():
        assert config[key] == value, f"required configuration change missing: {key}"
    assert config["seeds"] == [101, 202, 303]
    assert config["pure_reward_updates"] == 300
    assert sum(config[key] for key in ("course_updates", "transition_updates", "pure_reward_updates")) == 1200
    for update in range(1, 1201):
        control = original.schedule(update, "baseline", config)
        curriculum = original.schedule(update, "baseline", reference_config)
        assert control["task"] == "full"
        assert control["aux_weight"] == curriculum["aux_weight"] == 0
        assert control["action_entropy_weight"] == curriculum["action_entropy_weight"]
        if update > 600:
            assert control["task"] == curriculum["task"]
    return {"total_updates_per_run": 1200,
            "joint_samples_per_run": 1200 * config["batch_size"],
            "all_control_updates_use_full_task": True,
            "exploration_schedule_matches_every_update": True,
            "both_auxiliary_weights_zero_every_update": True,
            "only_task_difference_is_first_600_updates": True,
            "config_differences": {key: {"reference": reference_config[key], "control": config[key]}
                                   for key in CONFIG_CHANGES},
            "stage_label_note": "The reused scheduler calls the first 900 full-task updates transition; no scene transition occurs in this control."}


def validate_reference(config, reference=REFERENCE):
    """Abort if any reused source, data, prepared policy, or reference changes."""
    reference = Path(reference)
    validation = validate_design(config, read_json(reference / "config.json"))
    assert read_json(reference / "completed.json")["status"] == "completed"
    source_hashes = read_json(reference / "source_hashes.json")
    for name in SOURCE_NAMES:
        assert sha256(ROOT / name) == source_hashes[name], f"source differs from reference: {name}"
        assert sha256(reference / "source" / name) == source_hashes[name], f"reference source damaged: {name}"
    data_hashes = {}
    for name in DATA_NAMES:
        digest = sha256(ROOT / "data" / name)
        assert digest == sha256(reference / ("data_" + name)), f"data differs from reference: {name}"
        data_hashes[name] = digest
    prepared_hashes = read_json(reference / "initial_checkpoint_hashes.json")
    baseline_hashes = {}
    for seed in config["seeds"]:
        prepared_path = ROOT / f"results/pilot_001/prepared_s{seed}.pt"
        assert sha256(prepared_path) == prepared_hashes[str(seed)]
        prepared = torch.load(prepared_path, map_location="cpu", weights_only=True)
        initial_path = reference / f"baseline_s{seed}/initial.pt"
        baseline = torch.load(initial_path, map_location="cpu", weights_only=True)
        assert same_states(prepared, baseline), f"reference initial parameters differ: seed {seed}"
        baseline_hashes[str(seed)] = {"initial.pt": sha256(initial_path),
                                      "result.json": sha256(initial_path.parent / "result.json")}
    validation.update({"status": "validated_without_training", "reference_directory": str(reference),
                       "source_hashes_match_reference": source_hashes,
                       "data_hashes_match_reference": data_hashes,
                       "prepared_checkpoint_hashes": prepared_hashes,
                       "prepared_tensors_equal_original_baseline_initial_tensors": True,
                       "reference_baseline_hashes": baseline_hashes})
    return validation


def compare_completed(out, config, reference=REFERENCE):
    """Audit paired endpoints and summarize the saved direct/full comparison."""
    out, reference = Path(out), Path(reference)
    runs = []
    for seed in config["seeds"]:
        direct_dir, course_dir = out / f"baseline_s{seed}", reference / f"baseline_s{seed}"
        direct, course = read_json(direct_dir / "result.json"), read_json(course_dir / "result.json")
        direct_initial = torch.load(direct_dir / "initial.pt", map_location="cpu", weights_only=True)
        course_initial = torch.load(course_dir / "initial.pt", map_location="cpu", weights_only=True)
        assert same_states(direct_initial, course_initial), f"initialization mismatch for seed {seed}"
        assert direct["updates"] == course["updates"] == 1200
        assert direct["joint_training_steps"] == course["joint_training_steps"]
        dm = [json.loads(line) for line in (direct_dir / "training_metrics.jsonl").read_text().splitlines()]
        cm = [json.loads(line) for line in (course_dir / "training_metrics.jsonl").read_text().splitlines()]
        assert len(dm) == len(cm) == 1200
        for a, b in zip(dm, cm):
            assert a["update"] == b["update"]
            assert a["aux_weight"] == b["aux_weight"] == 0
            assert a["action_entropy_weight"] == b["action_entropy_weight"]
            assert a["task"] == "full"
        row = {"seed": seed, "initial_tensors_identical": True, "budget_identical": True,
               "exploration_and_auxiliary_schedules_identical": True,
               "training_world_hash_equal_updates": sum(a["world_input_sha256"] == b["world_input_sha256"] for a, b in zip(dm, cm)),
               "training_world_hash_equality_is_not_required": True,
               "tasks": {}}
        for task in ("curriculum", "full"):
            modes = {}
            for mode in ("normal", "shuffle", "blank", "stochastic"):
                d, c = direct[task][mode], course[task][mode]
                assert d["external_cases_sha256"] == c["external_cases_sha256"]
                assert d["cases"] == c["cases"] == config["evaluation_n"]
                modes[mode] = {"direct_full_success": d["balanced_gathering"],
                               "curriculum_success": c["balanced_gathering"],
                               "direct_minus_curriculum": d["balanced_gathering"] - c["balanced_gathering"],
                               "external_cases_sha256": d["external_cases_sha256"],
                               "by_direction": [
                                   {"sender": dd["restricted_sender"], "receiver": dd["mixed_receiver"],
                                    "n": dd["n"], "direct_full_success": dd["success"], "curriculum_success": cd["success"]}
                                   for dd, cd in zip(d["by_direction"], c["by_direction"])]}
                if mode != "stochastic":
                    path_name = f"final_{task}_{mode}_trace.jsonl"
                    direct_rows = (direct_dir / path_name).read_text().splitlines()
                    course_rows = (course_dir / path_name).read_text().splitlines()
                    assert len(direct_rows) == len(course_rows) == config["evaluation_n"]
                    for raw_a, raw_b in zip(direct_rows, course_rows):
                        a, b = json.loads(raw_a), json.loads(raw_b)
                        for key in ("case", "remaining", "inventory", "kinds", "image_ids"):
                            assert a[key] == b[key], f"external evaluation case mismatch: {seed}, {task}, {mode}, {key}"
                    modes[mode]["external_trace_cases_identical_row_by_row"] = True
            row["tasks"][task] = modes
        row["intervention_external_cases_identical"] = direct["intervention"]["external_cases_sha256"] == course["intervention"]["external_cases_sha256"]
        assert row["intervention_external_cases_identical"]
        runs.append(row)
    return {"status": "paired_control_verified", "reference_directory": str(reference),
            "control_directory": str(out), "runs": runs,
            "interpretation": "Effect of replacing the first 600 curriculum updates with full-task training under the same prepared policies, action exploration, total budget, and evaluation cases. Does not isolate visual pretraining or action exploration.",
            "training_case_note": "Training distributions intentionally differ for the first 600 updates. Equal training world hashes are not required; external evaluation cases and initial tensors must match.",
            "statistics_note": "Three paired seeds; report each pair and their mean, not a population confidence interval."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="course_control_001")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name:
        parser.error("--name must be one directory name")
    torch.set_num_threads(4)
    config = build_config()
    validation = validate_reference(config)
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return
    out = ROOT / "results" / args.name
    out.mkdir(exist_ok=False)
    write_json(out / "config.json", config)
    write_json(out / "preflight_validation.json", validation)
    source = out / "source"
    source.mkdir()
    for name in SOURCE_NAMES + (Path(__file__).name,):
        shutil.copyfile(ROOT / name, source / name)
    write_json(out / "source_hashes.json", {p.name: sha256(p) for p in source.iterdir()})
    for name in DATA_NAMES:
        shutil.copyfile(ROOT / "data" / name, out / ("data_" + name))
    write_json(out / "initial_checkpoint_hashes.json", validation["prepared_checkpoint_hashes"])
    bank = ImageBank()
    old_scaling = np.load(REFERENCE / "feature_scaling.npz")
    assert np.array_equal(bank.center, old_scaling["center"])
    assert bank.scale == float(old_scaling["scale"])
    np.savez_compressed(out / "feature_scaling.npz", center=bank.center, scale=bank.scale)
    results, start = [], time.monotonic()
    for seed in config["seeds"]:
        result = original.train(seed, "baseline", bank, out / f"baseline_s{seed}", config)
        results.append(result)
        write_json(out / "results.json", results)
    write_json(out / "course_comparison.json", compare_completed(out, config))
    write_json(out / "completed.json", {"status": "completed", "runs": len(results),
                                         "seconds": time.monotonic() - start,
                                         "paired_reference_control_verified": True})


if __name__ == "__main__":
    main()
