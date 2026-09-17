"""Paired JSON summary for the module-selective new-receiver study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import design
from . import runner

UPDATES = np.asarray(design.CHECKPOINTS, dtype=np.float64)
T7_975 = 2.364624251
METRICS = (
    "q_rate", "conditional_q_rate", "physical_execution_rate",
    "target_pair_legal_rate", "proposal_legal_rate", "engagement_rate", "neutral_rate",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (8,) and np.isfinite(x).all(), "Expected eight seed values")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(8)
    half = T7_975 * se
    return dict(n=8, mean=mean, sample_sd=sd, standard_error=se, df=7,
                ci95_lower=mean - half, ci95_upper=mean + half, t_critical=T7_975,
                positive_count=int((x > 0).sum()), zero_count=int((x == 0).sum()),
                negative_count=int((x < 0).sum()))


def centered_auc(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (6,), "Expected six checkpoints")
    return float(np.trapezoid(x - x[0], UPDATES) / UPDATES[-1])


def trajectory(run, key):
    rows = sorted(run["trajectory"], key=lambda row: row["update"])
    require([int(row["update"]) for row in rows] == list(map(int, UPDATES)), "Checkpoint mismatch")
    return np.asarray([row["target_trajectory"][key] for row in rows], dtype=np.float64)


def endpoint(run, key):
    return float(run["final"]["new_layouts"][key])


def load_full_reference():
    data = json.loads(design.FULL_RESULT.read_text())
    require(len(data["runs"]) == 32, "Full reference must contain 32 runs")
    return {(r["seed"], r["schedule"], bool(r["live"])): r for r in data["runs"]}


def summarize(runs):
    require(len(runs) == 64, "Expected 64 selective runs")
    selective = {(r["seed"], r["schedule"], r["arm"], bool(r["live"])): r for r in runs}
    require(len(selective) == 64, "Duplicate selective run")
    full = load_full_reference()
    seeds = list(range(66701, 66709))
    schedule_rows = {}
    overall_auc = {arm: {key: [] for key in METRICS} for arm in ("full",) + design.ARMS}
    overall_end = {arm: {key: [] for key in METRICS} for arm in ("full",) + design.ARMS}

    for schedule in design.SCHEDULES:
        schedule_rows[schedule] = {}
        for arm in ("full",) + design.ARMS:
            schedule_rows[schedule][arm] = {}
            for key in METRICS:
                auc_values = []
                end_values = []
                live_values = []
                silent_values = []
                for seed in seeds:
                    if arm == "full":
                        live = full[seed, schedule, True]
                        silent = full[seed, schedule, False]
                    else:
                        live = selective[seed, schedule, arm, True]
                        silent = selective[seed, schedule, arm, False]
                    diff = trajectory(live, key) - trajectory(silent, key)
                    auc_values.append(centered_auc(diff))
                    end_values.append(endpoint(live, key) - endpoint(silent, key))
                    live_values.append(endpoint(live, key))
                    silent_values.append(endpoint(silent, key))
                schedule_rows[schedule][arm][key] = dict(
                    centered_AUC=stats(auc_values), endpoint=stats(end_values),
                    centered_AUC_values=auc_values, endpoint_values=end_values,
                    live_endpoint_mean=float(np.mean(live_values)),
                    silent_endpoint_mean=float(np.mean(silent_values)),
                )

    # Average the two schedule strata within each source seed.
    for arm in ("full",) + design.ARMS:
        for key in METRICS:
            for seed_index, seed in enumerate(seeds):
                overall_auc[arm][key].append(float(np.mean([
                    schedule_rows[schedule][arm][key]["centered_AUC_values"][seed_index]
                    for schedule in design.SCHEDULES
                ])))
                overall_end[arm][key].append(float(np.mean([
                    schedule_rows[schedule][arm][key]["endpoint_values"][seed_index]
                    for schedule in design.SCHEDULES
                ])))

    full_minus_selective = {}
    for arm in design.ARMS:
        full_minus_selective[arm] = {}
        for key in METRICS:
            full_minus_selective[arm][key] = dict(
                centered_AUC=stats([a - b for a, b in zip(overall_auc["full"][key], overall_auc[arm][key])]),
                endpoint=stats([a - b for a, b in zip(overall_end["full"][key], overall_end[arm][key])]),
            )

    inherited = {}
    for arm in ("full",) + design.ARMS:
        inherited[arm] = {}
        for channel in (True, False):
            values = []
            source_values = []
            for seed in seeds:
                cells = []
                source_cells = []
                for schedule in design.SCHEDULES:
                    run = full[seed, schedule, channel] if arm == "full" else selective[seed, schedule, arm, channel]
                    cells.append(endpoint(run, "q_rate"))
                    source_cells.append(float(run["inherited_source"]["new_layouts"]["q_rate"]))
                values.append(float(np.mean(cells)))
                source_values.append(float(np.mean(source_cells)))
            inherited[arm]["live" if channel else "silent"] = dict(
                adapted_final=values, source_inherited=source_values,
                adapted_minus_inherited=stats((np.asarray(values) - np.asarray(source_values)).tolist()),
            )

    seed_rows = []
    for i, seed in enumerate(seeds):
        seed_rows.append(dict(
            seed=seed,
            full_q_centered_AUC=float(overall_auc["full"]["q_rate"][i]),
            action_only_q_centered_AUC=float(overall_auc["action_only"]["q_rate"][i]),
            sender_only_q_centered_AUC=float(overall_auc["sender_only"]["q_rate"][i]),
            full_q_endpoint=float(overall_end["full"]["q_rate"][i]),
            action_only_q_endpoint=float(overall_end["action_only"]["q_rate"][i]),
            sender_only_q_endpoint=float(overall_end["sender_only"]["q_rate"][i]),
        ))

    return dict(
        name="new_receiver_module_selective_transmission",
        overall={arm: {key: dict(centered_AUC=stats(overall_auc[arm][key]), endpoint=stats(overall_end[arm][key]))
                       for key in METRICS} for arm in ("full",) + design.ARMS},
        by_schedule=schedule_rows,
        full_minus_selective=full_minus_selective,
        inherited_source_control=inherited,
        by_seed=seed_rows,
        independent_seeds=8,
        paired_unit="source seed; static and rematched schedule gains averaged within seed",
        inference="Student-t intervals across eight independent source initializations; schedules are paired strata.",
        claim_boundary="This is a module-level task-protocol transmission ablation. It does not establish words, compositionality, or human language origin.",
        no_external_model=True, no_llm=True, no_vision_model=True,
    )


def main(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=False, exist_ok=False)
    data = json.loads((source / "execution" / "results.json").read_text())
    summary = summarize(data["runs"])
    path = output / "results.json"
    path.write_text(json.dumps(dict(status="completed_after_json_only_aggregation", source=str(source),
                                     summary=summary, runs=data["runs"], no_model_calls=True),
                               ensure_ascii=False, indent=2) + "\n")
    receipt = dict(status="passed", input_runs=len(data["runs"]), model_forwards=0,
                   output_results_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.source, args.output)
