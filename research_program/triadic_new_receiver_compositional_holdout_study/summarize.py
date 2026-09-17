"""JSON-only aggregation for the joint-combination holdout experiment."""
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
ARMS = ("seen_joint_only", "all_needs_control")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 1 and len(x) == len(design.SEEDS) and np.isfinite(x).all(),
            "Expected eight finite seed values")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(len(x))
    half = T7_975 * se
    return dict(
        n=len(x), mean=mean, sample_sd=sd, standard_error=se, df=len(x) - 1,
        ci95_lower=mean - half, ci95_upper=mean + half, t_critical=T7_975,
        positive_count=int((x > 0).sum()), zero_count=int((x == 0).sum()),
        negative_count=int((x < 0).sum()), values=[float(v) for v in x],
    )


def centered_auc(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (len(UPDATES),), "Expected six checkpoints")
    return float(np.trapezoid(x - x[0], UPDATES) / UPDATES[-1])


def trajectory(run, partition, key):
    rows = sorted(run["trajectory"], key=lambda row: int(row["update"]))
    require([int(row["update"]) for row in rows] == list(map(int, UPDATES)), "Checkpoint mismatch")
    return np.asarray([row["target_trajectory"][partition][key] for row in rows], dtype=np.float64)


def final(run, partition, key):
    return float(run["final"][partition][key])


def endpoint_pair(live, silent, partition, key):
    return final(live, partition, key) - final(silent, partition, key)


def make_pair_summary(by, arm, schedule, partition):
    rows = {}
    for key in METRICS:
        auc_values, endpoint_values = [], []
        live_values, silent_values = [], []
        for seed in design.SEEDS:
            live = by[arm][seed, schedule, True]
            silent = by[arm][seed, schedule, False]
            live_trajectory = trajectory(live, partition, key)
            silent_trajectory = trajectory(silent, partition, key)
            auc_values.append(centered_auc(live_trajectory - silent_trajectory))
            endpoint_values.append(endpoint_pair(live, silent, partition, key))
            live_values.append(final(live, partition, key))
            silent_values.append(final(silent, partition, key))
        rows[key] = dict(
            centered_AUC=stats(auc_values), endpoint=stats(endpoint_values),
            live_endpoint=stats(live_values), silent_endpoint=stats(silent_values),
            unit="probability; multiply by 100 for percentage points",
        )
    return rows


def make_schedule_average(by, arm, partition):
    rows = {}
    for key in METRICS:
        auc_by_seed, endpoint_by_seed = [], []
        for seed in design.SEEDS:
            aucs, endpoints = [], []
            for schedule in design.SCHEDULES:
                live = by[arm][seed, schedule, True]
                silent = by[arm][seed, schedule, False]
                aucs.append(centered_auc(trajectory(live, partition, key) - trajectory(silent, partition, key)))
                endpoints.append(endpoint_pair(live, silent, partition, key))
            auc_by_seed.append(float(np.mean(aucs)))
            endpoint_by_seed.append(float(np.mean(endpoints)))
        rows[key] = dict(
            centered_AUC=stats(auc_by_seed), endpoint=stats(endpoint_by_seed),
            unit="probability; multiply by 100 for percentage points; schedule average within seed",
        )
    return rows


def absolute_summary(by, arm, schedule, partition):
    output = {}
    for channel in (True, False):
        label = "live" if channel else "silent"
        output[label] = {}
        for key in METRICS:
            output[label][key] = stats([final(by[arm][seed, schedule, channel], partition, key) for seed in design.SEEDS])
    return output


def transfer_summary(by, arm, schedule):
    """Heldout joint combinations minus seen combinations on the same new layouts."""
    endpoint_values = []
    heldout_values, seen_new_values = [], []
    for seed in design.SEEDS:
        run = by[arm][seed, schedule, True]
        heldout = final(run, "heldout", "q_rate")
        seen_new = final(run, "seen_new", "q_rate")
        endpoint_values.append(heldout - seen_new)
        heldout_values.append(heldout)
        seen_new_values.append(seen_new)
    return dict(
        heldout_minus_seen_new=stats(endpoint_values),
        heldout_live=stats(heldout_values), seen_new_live=stats(seen_new_values),
        interpretation="Negative values mean the unseen joint combinations score below seen combinations on the same heldout layouts.",
        unit="probability; multiply by 100 for percentage points",
    )


def overall_transfer(by, arm):
    endpoint_values, heldout_values, seen_new_values = [], [], []
    for seed in design.SEEDS:
        heldout, seen_new = [], []
        for schedule in design.SCHEDULES:
            run = by[arm][seed, schedule, True]
            heldout.append(final(run, "heldout", "q_rate"))
            seen_new.append(final(run, "seen_new", "q_rate"))
        endpoint_values.append(float(np.mean(np.asarray(heldout) - np.asarray(seen_new))))
        heldout_values.append(float(np.mean(heldout)))
        seen_new_values.append(float(np.mean(seen_new)))
    return dict(
        heldout_minus_seen_new=stats(endpoint_values), heldout_live=stats(heldout_values),
        seen_new_live=stats(seen_new_values),
        unit="probability; multiply by 100 for percentage points; schedule average within seed",
    )


def checkpoint_rows(by, arm, partition="heldout", key="q_rate"):
    rows = []
    for update_index, update in enumerate(UPDATES.astype(int)):
        values = []
        for seed in design.SEEDS:
            gains = []
            for schedule in design.SCHEDULES:
                live = trajectory(by[arm][seed, schedule, True], partition, key)[update_index]
                silent = trajectory(by[arm][seed, schedule, False], partition, key)[update_index]
                gains.append(float(live - silent))
            values.append(float(np.mean(gains)))
        rows.append(dict(update=int(update), live_minus_silent=stats(values)))
    return rows


def contrast_summary(by, schedule, partition="heldout", key="q_rate"):
    """Paired contrast: seen-only routing gain minus all-needs-control routing gain."""
    endpoint, auc = [], []
    for seed in design.SEEDS:
        seen_live = by["seen_joint_only"][seed, schedule, True]
        seen_silent = by["seen_joint_only"][seed, schedule, False]
        control_live = by["all_needs_control"][seed, schedule, True]
        control_silent = by["all_needs_control"][seed, schedule, False]
        endpoint.append(endpoint_pair(seen_live, seen_silent, partition, key) - endpoint_pair(control_live, control_silent, partition, key))
        auc.append(centered_auc(trajectory(seen_live, partition, key) - trajectory(seen_silent, partition, key)) -
                   centered_auc(trajectory(control_live, partition, key) - trajectory(control_silent, partition, key)))
    return dict(endpoint=stats(endpoint), centered_AUC=stats(auc),
                interpretation="Positive values mean the seen-joint-only learner has a larger routed-packet advantage than the all-needs control.",
                unit="probability; multiply by 100 for percentage points")


def summarize(runs, controls):
    expected = len(design.SEEDS) * len(design.SCHEDULES) * 2
    require(len(runs) == expected and len(controls) == expected, "Expected 32 runs in each arm")
    by = {
        "seen_joint_only": {(r["seed"], r["schedule"], bool(r["live"])): r for r in runs},
        "all_needs_control": {(r["seed"], r["schedule"], bool(r["live"])): r for r in controls},
    }
    for arm in ARMS:
        require(len(by[arm]) == expected, "Duplicate run identity in " + arm)
        require(set(by[arm]) == {(seed, schedule, bool(live)) for seed in design.SEEDS for schedule in design.SCHEDULES for live in design.LIVES},
                "Incomplete run grid in " + arm)

    by_arm_schedule = {arm: {} for arm in ARMS}
    by_arm = {arm: {} for arm in ARMS}
    absolute_heldout = {arm: {} for arm in ARMS}
    absolute_seen_new = {arm: {} for arm in ARMS}
    heldout_checkpoint = {arm: checkpoint_rows(by, arm) for arm in ARMS}
    for arm in ARMS:
        for schedule in design.SCHEDULES:
            by_arm_schedule[arm][schedule] = make_pair_summary(by, arm, schedule, "heldout")
            absolute_heldout[arm][schedule] = absolute_summary(by, arm, schedule, "heldout")
            absolute_seen_new[arm][schedule] = absolute_summary(by, arm, schedule, "seen_new")
        by_arm[arm] = make_schedule_average(by, arm, "heldout")

    transfer = {
        arm: {schedule: transfer_summary(by, arm, schedule) for schedule in design.SCHEDULES}
        for arm in ARMS
    }
    transfer["overall"] = {arm: overall_transfer(by, arm) for arm in ARMS}
    arm_contrast = {schedule: contrast_summary(by, schedule) for schedule in design.SCHEDULES}

    seed_rows = []
    for seed in design.SEEDS:
        row = {"seed": seed}
        for arm in ARMS:
            for schedule in design.SCHEDULES:
                live = by[arm][seed, schedule, True]
                silent = by[arm][seed, schedule, False]
                row[f"{arm}_{schedule}_heldout_q_endpoint"] = endpoint_pair(live, silent, "heldout", "q_rate")
                row[f"{arm}_{schedule}_heldout_q_centered_AUC"] = centered_auc(trajectory(live, "heldout", "q_rate") - trajectory(silent, "heldout", "q_rate"))
                row[f"{arm}_{schedule}_heldout_q_live"] = final(live, "heldout", "q_rate")
                row[f"{arm}_{schedule}_seen_new_q_live"] = final(live, "seen_new", "q_rate")
                row[f"{arm}_{schedule}_heldout_minus_seen_new"] = final(live, "heldout", "q_rate") - final(live, "seen_new", "q_rate")
        seed_rows.append(row)

    return dict(
        name="new_receiver_joint_combination_holdout",
        primary=dict(
            heldout_joint_live_minus_silent={"by_arm_schedule": by_arm_schedule, "overall_by_arm": by_arm,
                                             "by_checkpoint": heldout_checkpoint},
            interpretation="Positive values mean routed packets improve task success relative to self-only packets on unseen joint combinations.",
        ),
        absolute_final=dict(heldout=absolute_heldout, seen_new=absolute_seen_new),
        compositional_transfer=transfer,
        arm_contrast=arm_contrast,
        by_seed=seed_rows,
        independent_seeds=len(design.SEEDS),
        paired_unit="source seed; static and rematched schedules averaged within seed for overall estimates",
        inference="Student-t intervals across eight independent source initializations; schedules are paired strata and the all-needs control is paired by seed/stream.",
        design="1248 seen joint combinations for adaptation; six complete semantic orbits (312 combinations) held out; every individual need value remains present in both partitions.",
        claim_boundary="The result tests behavioral transfer of a task-coordination protocol across unseen joint combinations. It does not establish words, compositionality, conventionality, intergenerational transmission, or human language origin.",
        no_external_model=True, no_llm=True, no_vision_model=True,
    )


def main(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=False, exist_ok=False)
    data = json.loads((source / "execution" / "results.json").read_text())
    summary = summarize(data["runs"], data["all_needs_control_runs"])
    path = output / "results.json"
    path.write_text(json.dumps(dict(status="completed_after_json_only_aggregation", source=str(source), summary=summary,
                                     runs=data["runs"], all_needs_control_runs=data["all_needs_control_runs"],
                                     no_model_calls=True), ensure_ascii=False, indent=2) + "\n")
    receipt = dict(status="passed", input_seen_runs=len(data["runs"]), input_control_runs=len(data["all_needs_control_runs"]),
                   model_forwards=0, output_results_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.source, args.output)
