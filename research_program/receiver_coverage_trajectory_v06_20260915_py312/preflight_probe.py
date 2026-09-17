"""Reproduce the pre-execution review's NumPy checks without importing Torch.

This executable artifact was first written after trajectory execution completed.
It reconstructs checks previously run interactively before execution; its output
records the actual rerun time, and labels completed-run checks separately.
It does not establish a new preregistration or load/deserialize model weights.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
from itertools import permutations
import json
from pathlib import Path
import platform
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np


PACKAGE = Path(__file__).resolve().parent
WORK = PACKAGE.parents[1]


def require(value, message):
    if not value:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def run():
    require("torch" not in sys.modules and "camp" not in sys.modules,
            "Use a clean process without model modules")
    source_path = WORK / "research_program/receiver_coverage_trajectory.py"
    plan_before_import = read(PACKAGE / "plan.json")
    require(digest(source_path) == plan_before_import["source_files_sha256"][str(source_path)],
            "Source is not the frozen trajectory implementation")
    spec = importlib.util.spec_from_file_location("reviewed_trajectory", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.verify_plan(PACKAGE)
    require(len(plan["runs"]) == 56, "Run count differs")
    unique_files = {f for r in plan["runs"] for f in r["checkpoint_files"].values()}
    receiver_ids = {(r["seed"], r["condition"], u, a)
                    for r in plan["runs"] for u in module.CHECKPOINTS for a in (0, 1)}
    require(len(unique_files) == 448 and len(receiver_ids) == 896, "Grid is incomplete")

    # Each of the four classifications is tested for map 0 = (food 0, water 1).
    train, heldout = list(range(24)), list(range(24, 30))
    expected_oracles = (1., .5, .5, 0.)
    categories = {}
    for name, expected in zip(module.CATEGORIES, expected_oracles):
        table = np.empty((49, 2, 1), np.int64)
        table[:, 0], table[:, 1] = 2, 3
        if name == module.CATEGORIES[0]:
            table[0, :, 0] = [0, 1]
        elif name == module.CATEGORIES[1]:
            table[0, :, 0], table[1, :, 0] = [0, 3], [2, 1]
        elif name == module.CATEGORIES[2]:
            table[0, :, 0] = [0, 3]
        metrics = module.coverage_metrics(table, train, heldout)
        row = metrics["maps"][0]
        require(row["coverage_classes"][name]["rate"] == 1, name)
        require(row["full_code_uniform_goal_ceiling"]["rate"] == expected,
                "Incorrect uniform-private-goal oracle")
        for item in metrics["maps"]:
            require(sum(v["numerator"] for v in item["coverage_classes"].values()) == 1,
                    "Coverage classes do not form a partition")
        require(metrics["subsets"]["train"]["full_code_uniform_goal_ceiling"]["denominator"] == 48,
                "Train oracle denominator should be 24 maps x 2 goals")
        require(metrics["subsets"]["heldout"]["full_code_uniform_goal_ceiling"]["denominator"] == 12,
                "Heldout oracle denominator should be 6 maps x 2 goals")
        categories[name] = expected

    # A genuine static-logit/final-gather counterexample, not an arbitrary table.
    # Code 0: food has a {0,2} tie; water uniquely selects 1.
    # Code 1: food uniquely selects 0; water has a {1,2} tie.
    # Other codes uniquely choose 2 for both needs. Each menu uses first argmax.
    logits = np.zeros((49, 2, 6))
    logits[:, :, 2] = 1
    logits[0, 0, 0] = 1
    logits[0, 1, :] = 0
    logits[0, 1, 1] = 1
    logits[1, 0, :] = 0
    logits[1, 0, 0] = 1
    logits[1, 1, 1] = 1
    menus = np.asarray(list(permutations(range(6))))
    scores = logits[:, :, menus]
    physical = np.take_along_axis(np.broadcast_to(menus, scores.shape),
                                  scores.argmax(-1)[..., None], -1)[..., 0]
    tie_row = module.coverage_metrics(physical, train, heldout)["maps"][0]
    correct = physical == np.asarray([0, 1])[None, :, None]
    fixed_code = float(correct.sum(1).mean(1).max() / 2)
    require(np.isclose(tie_row["full_code_uniform_goal_ceiling"]["rate"], 5 / 6),
            "Per-menu oracle changed")
    require(np.isclose(fixed_code, 3 / 4), "Menu-blind single-code oracle changed")
    require(tie_row["coverage_classes"]["same_code_both_correct"]["numerator"] == 480,
            "Joint coverage incorrectly collapsed across menus")
    require(tie_row["coverage_classes"]["both_marginals_but_no_joint_code"]["numerator"] == 240,
            "Marginal-only cases lost")
    blocked_row = module.coverage_metrics(physical, train, heldout, blocked=True)["maps"][0]
    require(np.isclose(blocked_row["channel_admissible_uniform_goal_ceiling"]["rate"], 3 / 4),
            "Blocked condition must use code 0 only")

    # Reconstruct the old 112 endpoint tables from JSON. This is not new model
    # inference and does not independently prove checkpoint/final tensor equality.
    endpoint_source = read(plan["source_protocol"])
    old_anchor_checks = 0
    for source_run in endpoint_source["runs"]:
        for direction in source_run["directions"]:
            table = direction["receiver_decoder_table"]
            if direction["menu_audit"]["all_menu_permutations_physically_equivalent"]:
                full = np.repeat(np.asarray([r["actions_by_goal"] for r in table],
                                            dtype=np.int64)[:, :, None], 720, axis=2)
            else:
                full = np.asarray([r["physical_actions_by_goal_and_menu"] for r in table],
                                  dtype=np.int64)
            require(full.shape == (49, 2, 720), "Old physical table shape differs")
            for code, row in enumerate(table):
                expected_counts = [np.bincount(full[code, goal], minlength=6).tolist()
                                   for goal in (0, 1)]
                require(row["menu_action_counts_by_goal"] == expected_counts,
                        "Old menu action frequency counts differ")
            lookup = {"receiver_decoder_table": table,
                      "physical_action_sha256": hashlib.sha256(full.tobytes()).hexdigest(),
                      "physical_menu_invariant": bool(np.all(full == full[:, :, :1]))}
            module.endpoint_check(lookup, direction)
            old_anchor_checks += 1
    require(old_anchor_checks == 112, "Missing old endpoint tables")

    # Test complete summary grouping without using any learned policy actions.
    constant = np.zeros((49, 2, 1), np.int64)
    constant[:, 1] = 1
    synthetic_records = []
    for source_run in plan["runs"]:
        coverage = module.coverage_metrics(constant, source_run["train_map_ids"],
                                           source_run["heldout_map_ids"],
                                           blocked=source_run["plan"]["blocked"])
        for update in module.CHECKPOINTS:
            for receiver in (0, 1):
                synthetic_records.append({"condition": source_run["condition"],
                                          "seed": source_run["seed"], "update": update,
                                          "receiver": receiver, "coverage": coverage})
    summaries = module.trajectory_summaries(synthetic_records)
    require(len(synthetic_records) == 896 and len(summaries["by_seed"]) == 320
            and len(summaries["by_family"]) == 80, "Summary grid differs")
    for row in summaries["by_seed"]:
        full_control = row["family"].startswith("full_")
        require(row["receiver_direction_count"] == (2 if full_control else 6),
                "Directions/splits weighted incorrectly")
        require(not (full_control and row["subset"] == "heldout"),
                "Full-map control must not have artificial heldout splits")
    for row in summaries["by_family"]:
        require(row["training_seeds"] == list(module.SEEDS)
                and all(len(values) == 4 for values in row["seed_values"].values()),
                "Independent seed values not retained")

    # Post-execution supplement: only read the completed run and confirm its
    # internal records. Keep this separate from reconstructed pre-execution tests.
    status = read(PACKAGE / "execution/status.json")
    result = read(PACKAGE / "execution/results.json")
    records_path = PACKAGE / "execution/records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    require(status["status"] == result["status"] == "completed", "Run is not completed")
    require(digest(records_path) == result["records_sha256"], "Completed records hash differs")
    actual_ids = {(r["seed"], r["condition"], r["update"], r["receiver"]) for r in records}
    require(len(records) == len(actual_ids) == 896 and actual_ids == receiver_ids,
            "Completed records are not the complete frozen grid")
    anchors = [r for r in records if r["endpoint_anchor"] is not None]
    require(len(anchors) == 112 and all(r["update"] == 2400
            and r["endpoint_anchor"]["status"] == "exact_match"
            and r["endpoint_anchor"]["checkpoint_final_state_dict_exact"] for r in anchors),
            "Completed endpoint checks differ")
    actual_ties = sum(r["lookup"]["complete_menu_enumeration_triggered"] for r in records)
    require(actual_ties == result["tie_receiver_checkpoint_count"] == 0,
            "Expected reported zero-tie run differs; do not silently discard ties")
    require(all(r["lookup"]["physical_menu_invariant"]
                and r["lookup"]["unique_argmax_equivalence_proof_used"]
                and r["lookup"]["minimum_top_two_margin"] > 0 for r in records),
            "Completed no-tie evidence is inconsistent")

    legacy_dir = PACKAGE.with_name("receiver_coverage_trajectory_v06_20260915")
    legacy = read(legacy_dir / "plan.json")
    require(legacy["source_files_sha256"] == plan["source_files_sha256"],
            "Re-preparation changed sources rather than only runtime/package metadata")
    require(legacy["runs"] == plan["runs"], "Re-preparation changed run selection")
    require(not (legacy_dir / "execution").exists(), "Legacy preparation unexpectedly executed")
    tree = ast.parse((WORK / "redesign_v0.6/camp.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CampAgent")
    receive = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "receive")
    require("torch" not in sys.modules and "camp" not in sys.modules,
            "This read-only probe imported a model module")
    return {
        "status": "passed",
        "written_and_rerun_after_execution": True,
        "timing_note": "The initial interactive review/probes preceded execute. This script and JSON were first saved after execute completed; their actual rerun time is recorded and is not a preregistration timestamp.",
        "rerun_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": digest(source_path), "script_sha256": digest(__file__),
        "plan_sha256": digest(PACKAGE / "plan.json"),
        "reconstructed_pre_execution_checks": {
            "frozen_sources_and_snapshots_verified": len(plan["source_files_sha256"]),
            "policy_files": 448, "receiver_checkpoints": 896,
            "classification_uniform_goal_oracles": categories,
            "train_oracle_denominator": 48, "heldout_oracle_denominator": 12,
            "menu_gather_counterexample": {
                "same_code_joint": tie_row["coverage_classes"]["same_code_both_correct"],
                "both_marginals_no_joint": tie_row["coverage_classes"]["both_marginals_but_no_joint_code"],
                "mean_menu_max_code": tie_row["full_code_uniform_goal_ceiling"],
                "max_code_mean_menu": fixed_code,
                "blocked_zero_code": blocked_row["channel_admissible_uniform_goal_ceiling"]},
            "old_endpoint_tables_reconstructed_and_hashed": old_anchor_checks,
            "synthetic_summary_by_seed_rows": 320, "synthetic_summary_by_family_rows": 80,
            "directions_per_split_family_seed": 6, "directions_per_full_control_seed": 2,
            "four_independent_seeds_retained": True},
        "ast_runtime_incident": {
            "legacy_python": legacy["prepare_runtime"]["python"],
            "replacement_python": plan["prepare_runtime"]["python"],
            "legacy_ast_sha256": legacy["menu_rule"]["receive_ast_sha256"],
            "replacement_ast_sha256": plan["menu_rule"]["receive_ast_sha256"],
            "current_functiondef_fields": list(receive._fields),
            "same_source_hashes_and_run_selection": True,
            "legacy_package_unexecuted": True,
            "repair": "Re-prepare a separate package with the execution Python; retain original package and all SHA/AST validation."},
        "post_execution_read_only_checks": {
            "status": result["status"], "receiver_records": len(records),
            "endpoint_exact_match_records": len(anchors), "ties": actual_ties,
            "receiver_forward_calls_reported_by_execution": result["receiver_forward_calls"],
            "evaluated_rows_reported_by_execution": result["evaluated_code_goal_menu_rows"],
            "records_sha256": digest(records_path),
            "minimum_recorded_argmax_margin": min(r["lookup"]["minimum_top_two_margin"] for r in records),
            "note": "Reads execution records; does not independently reload tensors or rerun receive."},
        "torch_imported": False, "camp_imported": False,
        "weights_deserialized": 0, "policy_forward_calls": 0,
        "limitations": ["The ties counterexample limits metric interpretation; actual execution recorded no ties.",
                        "NumPy/JSON checks are not an independent weight-level reproduction.",
                        "No new scientific outcome selection, model inference or training is performed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PACKAGE / "preflight_probe.json")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; choose a new --out path to preserve the original audit")
    result = run()
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "out": str(args.out),
                      "policy_forward_calls": 0, "torch_imported": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
