"""Audit and summarize a frozen private-demand execution."""
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
    """Verify the frozen preparation against its own source snapshot.

    The working tree may contain a later, intentionally different optimizer;
    an old execution must be audited against the source snapshot that created
    it, rather than silently reinterpreted with that later code.
    """
    prepared = Path(prepared).resolve()
    plan = json.loads((prepared / "plan.json").read_text(encoding="utf8"))
    frozen = json.loads((prepared / "freeze.json").read_text(encoding="utf8"))
    require(runner.sha(prepared / "plan.json") == frozen["plan_sha256"], "plan hash mismatch")
    require(runner.sha(prepared / "prepared.json") == frozen["prepared_sha256"] == plan["prepared_sha256"], "prepared hash mismatch")
    for relative, digest in plan["sources"].items():
        path = prepared / "source_snapshot" / relative
        require(path.is_file() and runner.sha(path) == digest, f"source snapshot mismatch: {relative}")
    return plan, json.loads((prepared / "prepared.json").read_text(encoding="utf8"))


def audit(prepared, execution):
    prepared = Path(prepared).resolve()
    execution = Path(execution).resolve()
    plan, frozen = verify_snapshot(prepared)
    payload = json.loads((execution / "results.json").read_text(encoding="utf8"))
    rows = payload["results"]
    progress_path = execution / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf8")) if progress_path.is_file() else None
    seeds = list(progress["seeds"]) if progress is not None else sorted({row["seed"] for row in rows})
    active_conditions = list(progress["conditions"]) if progress is not None else [
        condition for condition in design.CONDITIONS
        if any(row["condition"] == condition for row in rows)
    ]
    require(len(rows) == len(seeds) * len(active_conditions), "incomplete condition grid")
    require([(row["seed"], row["condition"]) for row in rows] == [
        (seed, condition) for seed in seeds for condition in active_conditions
    ], "noncanonical grid order")
    training_rows = 0
    final_blocks = 0
    max_oracle_spread = 0.0
    for row in rows:
        directory = execution / f"seed_{row['seed']}_{row['condition']}"
        require(directory.is_dir(), f"missing run: {directory}")
        saved = json.loads((directory / "result.json").read_text(encoding="utf8"))
        require(saved == row, f"result copy mismatch: {directory.name}")
        log = directory / "training.jsonl"
        lines = log.read_text(encoding="utf8").splitlines()
        require(len(lines) == row["updates"], f"training length mismatch: {directory.name}")
        require(runner.sha(log) == row["training_log_sha256"], f"training hash mismatch: {directory.name}")
        for expected, line in enumerate(lines, 1):
            record = json.loads(line)
            require(record["update"] == expected, f"update mismatch: {directory.name}:{expected}")
            for key in ("return_mean", "return_sd", "gradient_norm", "gradient_clip_scale"):
                finite(record[key], f"{directory.name}:{key}")
        training_rows += len(lines)
        for split in ("training_support", "heldout"):
            block = row["final"][split]
            for mode in ("natural", "closed", "permuted"):
                result = block[mode]
                final_blocks += 1
                for key, value in result.items():
                    if key not in ("split", "message_mode", "action_histogram", "message_histogram"):
                        finite(value, f"{directory.name}:{split}:{mode}:{key}")
                require(result["episodes"] == 4096, f"evaluation size mismatch: {directory.name}")
                require(sum(result["action_histogram"]) == 4096 * design.HORIZON * 2,
                        f"action histogram mismatch: {directory.name}")
                require(sum(result["message_histogram"]) == 4096 * len(design.MESSAGE_ROUNDS) * 2,
                        f"message histogram mismatch: {directory.name}")
                max_oracle_spread = max(max_oracle_spread, abs(block["natural"]["oracle_team_return_mean"] - result["oracle_team_return_mean"]))
            if row["channel"] == "silent":
                for mode in ("closed", "permuted"):
                    require(block[mode] == dict(block["natural"], message_mode=mode, reused_natural=True),
                            f"silent control is not a natural alias: {directory.name}")
            else:
                require(block["closed"]["message_mode"] == "closed", "closed label missing")
                require(block["permuted"]["message_mode"] == "permuted", "permuted label missing")
    return dict(
        schema="partner_demand_audit_v1",
        status="passed",
        prepared_sha256=runner.sha(prepared / "prepared.json"),
        plan_sha256=runner.sha(prepared / "plan.json"),
        result_sha256=runner.sha(execution / "results.json"),
        run_count=len(rows),
        seed_count=len(seeds),
        training_log_rows=training_rows,
        final_evaluation_blocks=final_blocks,
        max_oracle_consistency_error=max_oracle_spread,
        nonfinite_diagnostics=0,
    )


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
    def mean(values):
        return float(np.mean([cell[values] for cell in cells if cell["channel"] == "live"]))
    def maybe_mean(group, key):
        return None if not group else float(np.mean([cell[key] for cell in group]))
    heldout = [cell for cell in cells if cell["split"] == "heldout" and cell["channel"] == "live"]
    pi_switching = [cell for cell in heldout if cell["information"] == "PI" and cell["task"] == "switching"]
    pi_scarce_switching = [cell for cell in pi_switching if cell["scarcity"] == "scarce"]
    return dict(
        schema="partner_demand_analysis_v1",
        run_count=len(rows),
        seeds=sorted({row["seed"] for row in rows}),
        cells=cells,
        heldout_live=dict(
            count=len(heldout),
            natural_minus_closed_mean=maybe_mean(heldout, "natural_minus_closed"),
            natural_minus_permuted_mean=maybe_mean(heldout, "natural_minus_permuted"),
            message_type_mi_mean=maybe_mean(heldout, "message_type_mi"),
        ),
        heldout_PI_switching=dict(
            count=len(pi_switching),
            natural_minus_closed_mean=maybe_mean(pi_switching, "natural_minus_closed"),
            natural_minus_permuted_mean=maybe_mean(pi_switching, "natural_minus_permuted"),
            message_type_mi_mean=maybe_mean(pi_switching, "message_type_mi"),
        ),
        heldout_PI_scarce_switching=dict(
            count=len(pi_scarce_switching),
            natural_minus_closed_mean=maybe_mean(pi_scarce_switching, "natural_minus_closed"),
            natural_minus_permuted_mean=maybe_mean(pi_scarce_switching, "natural_minus_permuted"),
            message_type_mi_mean=maybe_mean(pi_scarce_switching, "message_type_mi"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()
    result = audit(args.prepared, args.execution)
    result["analysis"] = summarize(args.execution)
    if args.summary:
        Path(args.summary).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
