"""Independent audit for a fixed-role signaling execution."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import design, runner


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def finite(value, name):
    if isinstance(value, (int, float)):
        require(math.isfinite(float(value)), f"nonfinite {name}")
    elif isinstance(value, list):
        for item in value:
            finite(item, name)


def verify_snapshot(prepared):
    prepared = Path(prepared).resolve()
    plan = json.loads((prepared / "plan.json").read_text(encoding="utf8"))
    freeze = json.loads((prepared / "freeze.json").read_text(encoding="utf8"))
    require(runner.sha(prepared / "plan.json") == freeze["plan_sha256"], "plan hash mismatch")
    require(runner.sha(prepared / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "prepared hash mismatch")
    for relative, digest in plan["sources"].items():
        path = prepared / "source_snapshot" / relative
        require(path.is_file() and runner.sha(path) == digest, f"source snapshot mismatch: {relative}")
    return plan


def summarize(execution):
    rows = json.loads((Path(execution) / "results.json").read_text(encoding="utf8"))["results"]
    cells = []
    for row in rows:
        for split in ("training_support", "heldout"):
            natural = row["final"][split]["natural"]
            closed = row["final"][split]["closed"]
            permuted = row["final"][split]["permuted"]
            cells.append(dict(seed=row["seed"], split=split, memory=row["memory"], scarcity=row["scarcity"],
                              information=row["information"], channel=row["channel"], task=row["task"],
                              natural=natural["team_return_mean"], closed=closed["team_return_mean"],
                              permuted=permuted["team_return_mean"],
                              natural_minus_closed=natural["team_return_mean"] - closed["team_return_mean"],
                              natural_minus_permuted=natural["team_return_mean"] - permuted["team_return_mean"],
                              message_entropy=natural["message_entropy"], message_type_mi=natural["message_type_mi"]))
    live = [cell for cell in cells if cell["split"] == "heldout" and cell["channel"] == "live"]
    def maybe_mean(group, key):
        return None if not group else float(np.mean([cell[key] for cell in group]))
    pi_switching = [cell for cell in live if cell["information"] == "PI" and cell["task"] == "switching"]
    return dict(
        schema="fixed_role_signaling_analysis_v1",
        run_count=len(rows), seeds=sorted({row["seed"] for row in rows}), cells=cells,
        heldout_live=dict(count=len(live), natural_minus_closed_mean=maybe_mean(live, "natural_minus_closed"),
                          natural_minus_permuted_mean=maybe_mean(live, "natural_minus_permuted"),
                          message_type_mi_mean=maybe_mean(live, "message_type_mi")),
        heldout_PI_switching=dict(count=len(pi_switching), natural_minus_closed_mean=maybe_mean(pi_switching, "natural_minus_closed"),
                                  natural_minus_permuted_mean=maybe_mean(pi_switching, "natural_minus_permuted"),
                                  message_type_mi_mean=maybe_mean(pi_switching, "message_type_mi")),
    )


def audit(prepared, execution):
    prepared = Path(prepared).resolve(); execution = Path(execution).resolve()
    plan = verify_snapshot(prepared)
    rows = json.loads((execution / "results.json").read_text(encoding="utf8"))["results"]
    progress = json.loads((execution / "progress.json").read_text(encoding="utf8"))
    seeds, conditions = list(progress["seeds"]), list(progress["conditions"])
    require([(row["seed"], row["condition"]) for row in rows] == [(s, c) for s in seeds for c in conditions], "grid mismatch")
    training_rows = 0; final_blocks = 0; max_oracle_error = 0.0
    for row in rows:
        directory = execution / f"seed_{row['seed']}_{row['condition']}"
        require(directory.is_dir(), f"missing run: {directory}")
        require(json.loads((directory / "result.json").read_text(encoding="utf8")) == row, f"result copy mismatch: {directory.name}")
        lines = (directory / "training.jsonl").read_text(encoding="utf8").splitlines()
        require(len(lines) == row["updates"] and runner.sha(directory / "training.jsonl") == row["training_log_sha256"], f"training log mismatch: {directory.name}")
        for update, line in enumerate(lines, 1):
            record = json.loads(line); require(record["update"] == update, f"update mismatch: {directory.name}")
            for key in ("return_mean", "return_sd", "gradient_norm", "gradient_clip_scale"): finite(record[key], f"{directory.name}:{key}")
        training_rows += len(lines)
        for split in ("training_support", "heldout"):
            block = row["final"][split]
            for mode in ("natural", "closed", "permuted"):
                result = block[mode]; final_blocks += 1
                require(result["episodes"] == 4096 and sum(result["action_histogram"]) == 4096 * design.HORIZON * 2,
                        f"count mismatch: {directory.name}:{split}:{mode}")
                require(sum(result["message_histogram"]) == 4096 * len(design.MESSAGE_ROUNDS), f"message count mismatch: {directory.name}")
                for key, value in result.items():
                    if key not in ("split", "message_mode", "action_histogram", "message_histogram"):
                        finite(value, f"{directory.name}:{key}")
                max_oracle_error = max(max_oracle_error, abs(block["natural"]["oracle_team_return_mean"] - result["oracle_team_return_mean"]))
            if row["channel"] == "silent":
                for mode in ("closed", "permuted"):
                    require(block[mode] == dict(block["natural"], message_mode=mode, reused_natural=True), f"silent alias mismatch: {directory.name}")
    return dict(schema="fixed_role_signaling_audit_v1", status="passed", plan_sha256=runner.sha(prepared / "plan.json"),
                prepared_sha256=runner.sha(prepared / "prepared.json"), result_sha256=runner.sha(execution / "results.json"),
                run_count=len(rows), seed_count=len(seeds), training_log_rows=training_rows, final_evaluation_blocks=final_blocks,
                max_oracle_consistency_error=max_oracle_error, nonfinite_diagnostics=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepared", required=True); parser.add_argument("--execution", required=True); parser.add_argument("--summary", default=None)
    args = parser.parse_args(); answer = audit(args.prepared, args.execution); answer["analysis"] = summarize(args.execution)
    if args.summary: Path(args.summary).write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(answer, ensure_ascii=False, sort_keys=True))
