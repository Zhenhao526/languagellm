"""Paired summary for new-receiver transmission."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

UPDATES = np.asarray([0, 100, 500, 1500, 3000, 6000], dtype=np.float64)
# Two-sided 95% Student-t critical value for df=7.
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
    require(x.ndim == 1 and len(x) == 8 and np.isfinite(x).all(), "Expected eight paired seed values")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(len(x))
    half = T7_975 * se
    return dict(n=8, mean=mean, sample_sd=sd, standard_error=se, df=7,
                ci95_lower=mean - half, ci95_upper=mean + half, t_critical=T7_975,
                positive_count=int((x > 0).sum()), zero_count=int((x == 0).sum()),
                negative_count=int((x < 0).sum()))


def centered_auc(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (6,), "Six checkpoints required")
    return float(np.trapezoid(x - x[0], UPDATES) / UPDATES[-1])


def metric_trajectory(run, key):
    rows = sorted(run["trajectory"], key=lambda row: row["update"])
    require([row["update"] for row in rows] == list(map(int, UPDATES)), "Checkpoint order mismatch")
    return np.asarray([row["target_trajectory"][key] for row in rows], dtype=np.float64)


def final_metric(run, key):
    return float(run["final"]["new_layouts"][key])


def summarize(runs):
    require(len(runs) == 32, "Expected 32 transmission runs")
    by = {(r["seed"], r["schedule"], bool(r["live"])): r for r in runs}
    seeds = sorted({r["seed"] for r in runs})
    require(seeds == list(range(66701, 66709)), "Unexpected seed block")
    require(len(by) == 32, "Duplicate run identity")

    schedule_rows = {}
    overall = {key: [] for key in METRICS}
    overall_end = {key: [] for key in METRICS}
    seed_rows = []
    checkpoint_rows = []
    for schedule in ("static", "rematched"):
        schedule_rows[schedule] = {}
        for key in METRICS:
            auc_values = []
            end_values = []
            for seed in seeds:
                live = by[seed, schedule, True]
                silent = by[seed, schedule, False]
                diff = metric_trajectory(live, key) - metric_trajectory(silent, key)
                endpoint = final_metric(live, key) - final_metric(silent, key)
                auc_values.append(centered_auc(diff))
                end_values.append(endpoint)
                if schedule == "static":
                    overall[key].append(None)
            schedule_rows[schedule][key] = dict(
                centered_AUC=stats(auc_values), endpoint=stats(end_values),
                centered_AUC_values=auc_values, endpoint_values=end_values,
                unit="probability; multiply by 100 for percentage points",
            )
            if schedule == "static":
                # Preserve per-seed values for the schedule average below.
                pass

    for key in METRICS:
        auc_by_seed = []
        end_by_seed = []
        for seed in seeds:
            auc_pair = [schedule_rows[s][key]["centered_AUC_values"][seeds.index(seed)] for s in ("static", "rematched")]
            end_pair = [schedule_rows[s][key]["endpoint_values"][seeds.index(seed)] for s in ("static", "rematched")]
            auc_by_seed.append(float(np.mean(auc_pair)))
            end_by_seed.append(float(np.mean(end_pair)))
        overall[key] = auc_by_seed
        overall_end[key] = end_by_seed

    # Per-seed primary values make the paired unit explicit.
    for i, seed in enumerate(seeds):
        seed_rows.append(dict(
            seed=seed,
            q_centered_AUC=float(overall["q_rate"][i]),
            q_endpoint=float(overall_end["q_rate"][i]),
            static_q_centered_AUC=float(schedule_rows["static"]["q_rate"]["centered_AUC_values"][i]),
            rematched_q_centered_AUC=float(schedule_rows["rematched"]["q_rate"]["centered_AUC_values"][i]),
            static_q_endpoint=float(schedule_rows["static"]["q_rate"]["endpoint_values"][i]),
            rematched_q_endpoint=float(schedule_rows["rematched"]["q_rate"]["endpoint_values"][i]),
        ))
    for checkpoint_index, update in enumerate(UPDATES.astype(int)):
        q_values = []
        for seed in seeds:
            pairs = []
            for schedule in ("static", "rematched"):
                live = metric_trajectory(by[seed, schedule, True], "q_rate")
                silent = metric_trajectory(by[seed, schedule, False], "q_rate")
                pairs.append(float(live[checkpoint_index] - silent[checkpoint_index]))
            q_values.append(float(np.mean(pairs)))
        checkpoint_rows.append(dict(update=int(update), q_rate_live_minus_silent=stats(q_values), values=q_values))

    inherited = {}
    for schedule in ("static", "rematched"):
        inherited[schedule] = {}
        for channel in (True, False):
            vals = [final_metric(by[seed, schedule, channel], "q_rate") for seed in seeds]
            source_vals = [float(by[seed, schedule, channel]["inherited_source"]["new_layouts"]["q_rate"]) for seed in seeds]
            delta = np.asarray(vals) - np.asarray(source_vals)
            inherited[schedule]["live" if channel else "silent"] = dict(
                adapted_final=vals, source_inherited=source_vals,
                adapted_minus_inherited=stats(delta.tolist()),
            )

    output = dict(
        name="new_receiver_live_minus_silent_transmission",
        primary=dict(
            q_rate_centered_AUC=stats(overall["q_rate"]), q_rate_endpoint=stats(overall_end["q_rate"]),
            by_seed=seed_rows, by_checkpoint=checkpoint_rows,
            interpretation="Positive values mean the freshly initialized C benefits more from routed packets than from silent self-only packets.",
        ),
        by_schedule=schedule_rows,
        inherited_source_control=inherited,
        independent_seeds=8, paired_unit="seed; static and rematched schedule gains averaged within seed for the overall estimate",
        inference="Student-t intervals across eight independent source initializations; schedules are paired strata, not additional independent societies.",
        claim_boundary="The test concerns transmission of a task-coordination protocol into a new receiver. It does not establish lexical symbols, compositionality, or human-language origin.",
        no_external_model=True, no_llm=True, no_vision_model=True,
    )
    return output


def main(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=False, exist_ok=False)
    data = json.loads((source / "execution" / "results.json").read_text())
    summary = summarize(data["runs"])
    path = output / "results.json"
    path.write_text(json.dumps(dict(status="completed_after_json_only_aggregation", source=str(source), summary=summary, runs=data["runs"], no_model_calls=True), ensure_ascii=False, indent=2) + "\n")
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
