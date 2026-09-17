"""Paired JSON-only summary for the two-generation protocol chain."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import design

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
    require(x.ndim == 1 and len(x) == len(design.SEEDS) and np.isfinite(x).all(), "Expected eight seed values")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(len(x))
    half = T7_975 * se
    return dict(n=len(x), mean=mean, sample_sd=sd, standard_error=se, df=len(x) - 1,
                ci95_lower=mean - half, ci95_upper=mean + half, t_critical=T7_975,
                positive_count=int((x > 0).sum()), zero_count=int((x == 0).sum()),
                negative_count=int((x < 0).sum()))


def centered_auc(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (len(UPDATES),), "Expected six checkpoints")
    return float(np.trapezoid(x - x[0], UPDATES) / UPDATES[-1])


def trajectory(run, key):
    rows = sorted(run["trajectory"], key=lambda row: row["update"])
    require([int(row["update"]) for row in rows] == list(map(int, UPDATES)), "Checkpoint mismatch")
    return np.asarray([row["target_trajectory"][key] for row in rows], dtype=np.float64)


def endpoint(run, key):
    return float(run["final"]["new_layouts"][key])


def inherited_endpoint(run, key):
    return float(run["inherited_parent"]["new_layouts"][key])


def summarize(runs):
    expected = len(design.SEEDS) * len(design.GENERATIONS) * len(design.SCHEDULES) * 2
    require(len(runs) == expected, "Expected 64 chain runs")
    by = {(r["seed"], r["generation"], r["schedule"], bool(r["live"])): r for r in runs}
    require(len(by) == expected, "Duplicate chain run identity")
    seeds = list(design.SEEDS)

    by_generation_schedule = {}
    overall = {}
    checkpoint_rows = {}
    for generation in design.GENERATIONS:
        gkey = str(generation)
        by_generation_schedule[gkey] = {}
        checkpoint_rows[gkey] = []
        overall[gkey] = {}
        for schedule in design.SCHEDULES:
            by_generation_schedule[gkey][schedule] = {}
            for key in METRICS:
                auc_values, endpoint_values, live_values, silent_values = [], [], [], []
                for seed in seeds:
                    live = by[seed, generation, schedule, True]
                    silent = by[seed, generation, schedule, False]
                    diff = trajectory(live, key) - trajectory(silent, key)
                    auc_values.append(centered_auc(diff))
                    endpoint_values.append(endpoint(live, key) - endpoint(silent, key))
                    live_values.append(endpoint(live, key))
                    silent_values.append(endpoint(silent, key))
                by_generation_schedule[gkey][schedule][key] = dict(
                    centered_AUC=stats(auc_values), endpoint=stats(endpoint_values),
                    centered_AUC_values=auc_values, endpoint_values=endpoint_values,
                    live_endpoint_mean=float(np.mean(live_values)), silent_endpoint_mean=float(np.mean(silent_values)),
                    unit="probability; multiply by 100 for percentage points",
                )
        for key in METRICS:
            auc_by_seed = [float(np.mean([by_generation_schedule[gkey][s][key]["centered_AUC_values"][i] for s in design.SCHEDULES])) for i in range(len(seeds))]
            end_by_seed = [float(np.mean([by_generation_schedule[gkey][s][key]["endpoint_values"][i] for s in design.SCHEDULES])) for i in range(len(seeds))]
            overall[gkey][key] = dict(centered_AUC=stats(auc_by_seed), endpoint=stats(end_by_seed),
                                      centered_AUC_values=auc_by_seed, endpoint_values=end_by_seed)
        for idx, update in enumerate(UPDATES.astype(int)):
            values = []
            for seed in seeds:
                values.append(float(np.mean([
                    trajectory(by[seed, generation, schedule, True], "q_rate")[idx] -
                    trajectory(by[seed, generation, schedule, False], "q_rate")[idx]
                    for schedule in design.SCHEDULES
                ])))
            checkpoint_rows[gkey].append(dict(update=int(update), q_rate_live_minus_silent=stats(values), values=values))

    # Parent-to-child live retention is evaluated against the saved parent
    # policy under the same packet-routing mode.  Generation 3's parent is
    # always the generation-2 live endpoint by construction.
    retention = {}
    absolute = {}
    for generation in design.GENERATIONS:
        gkey = str(generation)
        retention[gkey] = {}
        absolute[gkey] = {}
        for schedule in design.SCHEDULES:
            retention[gkey][schedule] = {}
            absolute[gkey][schedule] = {}
            for channel in (True, False):
                label = "live" if channel else "silent"
                child = [endpoint(by[seed, generation, schedule, channel], "q_rate") for seed in seeds]
                parent = [inherited_endpoint(by[seed, generation, schedule, channel], "q_rate") for seed in seeds]
                retention[gkey][schedule][label] = dict(
                    child=child, parent=parent, child_minus_parent=stats((np.asarray(child) - np.asarray(parent)).tolist()),
                )
                absolute[gkey][schedule][label] = stats(child)

    overall_retention = {}
    for generation in design.GENERATIONS:
        gkey = str(generation)
        overall_retention[gkey] = {}
        for channel in (True, False):
            label = "live" if channel else "silent"
            values = []
            for seed in seeds:
                values.append(float(np.mean([
                    endpoint(by[seed, generation, schedule, channel], "q_rate") -
                    inherited_endpoint(by[seed, generation, schedule, channel], "q_rate")
                    for schedule in design.SCHEDULES
                ])))
            overall_retention[gkey][label] = stats(values)

    seed_rows = []
    for i, seed in enumerate(seeds):
        seed_rows.append(dict(
            seed=seed,
            generation2_q_endpoint=float(overall["2"]["q_rate"]["endpoint_values"][i]),
            generation2_q_centered_AUC=float(overall["2"]["q_rate"]["centered_AUC_values"][i]),
            generation3_q_endpoint=float(overall["3"]["q_rate"]["endpoint_values"][i]),
            generation3_q_centered_AUC=float(overall["3"]["q_rate"]["centered_AUC_values"][i]),
            generation2_live_retention=float(np.mean([endpoint(by[seed, 2, s, True], "q_rate") - inherited_endpoint(by[seed, 2, s, True], "q_rate") for s in design.SCHEDULES])),
            generation3_live_retention=float(np.mean([endpoint(by[seed, 3, s, True], "q_rate") - inherited_endpoint(by[seed, 3, s, True], "q_rate") for s in design.SCHEDULES])),
        ))

    return dict(
        name="two_generation_protocol_chain",
        primary={"by_generation": overall, "by_generation_schedule": by_generation_schedule,
                 "by_checkpoint": checkpoint_rows,
                 "interpretation":"Positive routing values mean the replaced learner benefits more from routed packets than from silent self-only packets."},
        retention=retention, overall_retention=overall_retention, absolute_final_q=absolute,
        by_seed=seed_rows, independent_seeds=len(seeds), paired_unit="source seed; static and rematched schedules averaged within seed",
        inference="Student-t intervals across eight independent source initializations; schedules are paired strata, not additional independent societies.",
        chain_definition="generation 1 audited live endpoint -> generation 2 replaces A -> generation 3 replaces B from generation-2 live endpoint",
        claim_boundary="This is a short-horizon task-protocol transmission experiment. It does not establish words, compositionality, conventionality, or human language origin.",
        no_external_model=True, no_llm=True, no_vision_model=True,
    )


def main(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=False, exist_ok=False)
    data = json.loads((source / "execution" / "results.json").read_text())
    summary = summarize(data["runs"])
    path = output / "results.json"
    path.write_text(json.dumps(dict(status="completed_after_json_only_aggregation", source=str(source), summary=summary,
                                     runs=data["runs"], no_model_calls=True), ensure_ascii=False, indent=2) + "\n")
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
