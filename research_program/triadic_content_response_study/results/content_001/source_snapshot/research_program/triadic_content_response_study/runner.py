"""Run the frozen-policy four-choice content probe without training."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import time
from pathlib import Path

import numpy as np

from research_program.triadic_message_study import runner as core
from research_program.triadic_rule_formation_study import runner as formation
from . import dataset, intervene, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "research_program/triadic_rule_formation_study/results/formation_001"
SEEDS = metrics.SEEDS
CONDITIONS = metrics.CONDITIONS
STEPS = metrics.STEPS
TARGET = "new_needs_and_layouts"
BATCH = 1024
FORMATION_PLAN_SHA = "b3ae053bca41e5acf60fa9108835cfe5bec85f2c6638cc48b2a7db9e5f00354f"
FORMATION_PREPARED_SHA = "92acde1b3e535990fdb120bb9dc19b1c5c2f6c7177d504182d2f6f687ffbe885"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return dataset.sha(path)


def read(path):
    return dataset.read_json(path)


def write(path, value):
    dataset.write_json(path, value)


def source_path(seed, condition, step, kind):
    folder = SOURCE / "execution" / f"seed_{seed}_{condition}"
    if kind == "trajectory":
        return folder / f"trajectory_{step:04d}_{TARGET}.npz"
    if kind == "checkpoint":
        return folder / f"checkpoint_{step:04d}.npz"
    raise ValueError(kind)


def verify_source():
    require(sha(SOURCE / "plan.json") == FORMATION_PLAN_SHA, "Formation plan hash changed")
    require(sha(SOURCE / "prepared.json") == FORMATION_PREPARED_SHA, "Formation prepared hash changed")
    freeze = read(SOURCE / "freeze.json")
    require(freeze == {"plan_sha256": FORMATION_PLAN_SHA, "prepared_sha256": FORMATION_PREPARED_SHA}, "Formation freeze changed")
    status = read(SOURCE / "execution" / "status.json")
    require(status.get("status") == "completed", "Formation execution is not complete")
    for seed in SEEDS:
        for condition in CONDITIONS:
            for step in STEPS:
                require(source_path(seed, condition, step, "trajectory").exists(), "Missing source trajectory")
            if condition.endswith("_live"):
                for step in STEPS:
                    require(source_path(seed, condition, step, "checkpoint").exists(), "Missing source checkpoint")


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def flatten_from(static_arrays):
    return dataset.flatten_rows(static_arrays)


def natural_arrays(pool, row_map):
    ids = row_map["host_indices"]
    p = pool["action_probabilities"][ids]
    require(p.dtype == np.float64 and p.shape == (len(ids), 3, 17), "Natural action probability schema")
    return p


def one_policy_time(seed, condition, step, static, arrays, features, output, row_map):
    rule, visibility = condition.split("_PL_")
    live = visibility == "live"
    out_dir = Path(output) / f"seed_{seed}_{condition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    natural_path = source_path(seed, condition, step, "trajectory")
    pool = load_npz(natural_path)
    target_states = arrays["target_states"]
    require(np.array_equal(pool["states"], target_states), "Natural target states changed")
    require(np.array_equal(pool["state_indices"], np.arange(len(target_states), dtype=np.int64)), "Natural target indices changed")
    require(pool["messages"].shape == (len(target_states), 2, 3, 4), "Natural messages schema")
    host_ids = row_map["host_indices"]; donor_ids = row_map["donor_indices"]; senders = row_map["senders"]
    host_messages = pool["messages"][host_ids]
    donor_packets = pool["messages"][donor_ids, 0, senders, :]
    if live:
        checkpoint_path = source_path(seed, condition, step, "checkpoint")
        networks = core.load_networks(checkpoint_path)
        trace = intervene.first_window_intervene(networks, features[host_ids], host_messages, senders, donor_packets)
        probabilities = trace["action_probabilities"]
        actions = trace["action_indices"]
        messages = trace["messages"]
        diagonal = row_map["host_endpoint"] == row_map["donor_endpoint"]
        natural_p = pool["action_probabilities"][host_ids]
        diag_error = float(np.max(np.abs(probabilities[diagonal] - natural_p[diagonal])))
        require(diag_error <= 2e-12, "Natural diagonal intervention identity failed")
        data_path = out_dir / f"probe_{step:04d}.npz"
        require(not data_path.exists(), "Probe output already exists")
        payload = dict(action_probabilities=probabilities, action_indices=actions, messages=messages,
                       donor_packets=donor_packets, host_indices=host_ids, donor_indices=donor_ids,
                       group_index=row_map["group_index"], background_index=row_map["background_index"],
                       host_endpoint=row_map["host_endpoint"], donor_endpoint=row_map["donor_endpoint"],
                       senders=senders, listeners=row_map["listeners"],
                       candidate_receiver_actions=row_map["candidate_receiver_actions"],
                       target_receiver_actions=row_map["target_receiver_actions"])
        with data_path.open("xb") as stream:
            np.savez_compressed(stream, **payload)
        path_value = str(data_path)
        data_hash = sha(data_path)
        forward_samples = 6 * len(host_ids)
    else:
        probabilities = natural_arrays(pool, row_map)
        actions = np.argmax(probabilities, axis=-1).astype(np.int16)
        messages = None; path_value = None; data_hash = None; diag_error = 0.0; forward_samples = 0
    m = metrics.probe_metrics_with_layers(probabilities, row_map["group_index"], row_map["background_index"],
                                          row_map["host_endpoint"], row_map["donor_endpoint"], row_map["listeners"],
                                          row_map["candidate_receiver_actions"], arrays["group_layer_index"])
    record = dict(seed=seed, condition=condition, rule=rule, live=live, update=step,
                  source_natural_path=str(natural_path), source_natural_sha256=sha(natural_path),
                  path=path_value, data_sha256=data_hash, alias_of_source=not live,
                  worlds=len(host_ids), groups=48, backgrounds=36, host_donor_cells=16,
                  off_diagonal_rows=int(np.sum(row_map["host_endpoint"] != row_map["donor_endpoint"])),
                  metrics={k: v for k, v in m.items() if k not in ("margins",)},
                  max_diagonal_probability_error=diag_error,
                  logical_module_samples=6 * len(host_ids), neural_forward_samples=forward_samples,
                  intervention="first-window sender outward packet only; W2 and all action heads recomputed for live",
                  silent_recipe="reuse natural host action probabilities; packet replacement is invisible and performs zero forward" if not live else None)
    out_record = Path(output) / f"seed_{seed}_{condition}" / f"record_{step:04d}.json"
    write(out_record, record)
    return record


def worker(payload):
    seed, static, output = payload
    output = Path(output)
    source_prepared = read(formation.ORIGINAL / "prepared.json")
    part = source_prepared["partitions"][TARGET]
    target_states = dataset.pack_states(part)
    features = intervene.observations(target_states)
    static_arrays = dataset.load_cases(static)
    row_map = flatten_from(static_arrays)
    arrays = dict(target_states=target_states, group_layer_index=static_arrays["group_layer_index"])
    records = []
    for condition in CONDITIONS:
        for step in STEPS:
            records.append(one_policy_time(seed, condition, step, static, arrays, features, output, row_map))
    feature_hash = __import__("hashlib").sha256(features.tobytes()).hexdigest()
    write(output / f"seed_{seed}_result.json", dict(seed=seed, records=records,
        feature_array_sha256=feature_hash))
    return dict(seed=seed, records=records, feature_array_sha256=feature_hash)


def budget():
    rows = 48 * 36 * 16
    policy_times = 16 * 4 * 6
    live_policy_times = 16 * 2 * 6
    return dict(policy_trajectories=64, checkpoint_times=384, rows_per_policy_time=rows,
                logical_module_samples=policy_times * rows * 6,
                live_policy_times=live_policy_times, live_probe_rows=live_policy_times * rows,
                silent_alias_rows=live_policy_times * rows, new_forward_module_samples=live_policy_times * rows * 6,
                live_probe_files=live_policy_times, silent_alias_records=live_policy_times,
                training_updates=0)


def prepare(out):
    return dataset.save_static(out)


def verify(out):
    plan, prepared, arrays = dataset.verify(out)
    require(plan["config"]["source_formation_plan_sha256"] == FORMATION_PLAN_SHA, "Candidate source formation hash")
    verify_source()
    return plan, prepared, arrays


def execute(out):
    out = Path(out).resolve()
    plan, prepared, static_arrays = verify(out)
    execution = out / "execution"
    require(not execution.exists(), "Never overwrite or resume a probe execution")
    execution.mkdir()
    started = time.perf_counter()
    write(execution / "started.json", dict(at=time.time(), plan_sha256=sha(out / "plan.json"), source_formation_plan_sha256=FORMATION_PLAN_SHA))
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        groups = pool.map(worker, [(seed, str(out), str(execution)) for seed in SEEDS])
    records = [record for group in groups for record in group["records"]]
    require([(r["seed"], r["condition"], r["update"]) for r in records] ==
            [(s, c, t) for s in SEEDS for c in CONDITIONS for t in STEPS], "Noncanonical probe record grid")
    live = [r for r in records if r["live"]]
    silent = [r for r in records if not r["live"]]
    measured = dict(records=len(records), live_probe_files=sum(r["path"] is not None for r in records),
                    silent_alias_records=sum(r["alias_of_source"] for r in records),
                    logical_module_samples=sum(r["logical_module_samples"] for r in records),
                    new_forward_module_samples=sum(r["neural_forward_samples"] for r in records),
                    worlds=sum(r["worlds"] for r in records), off_diagonal_rows=sum(r["off_diagonal_rows"] for r in records))
    require(measured["logical_module_samples"] == budget()["logical_module_samples"], "Logical budget mismatch")
    require(measured["new_forward_module_samples"] == budget()["new_forward_module_samples"], "Forward budget mismatch")
    # Primary expects one six-point trajectory per policy; assemble it from the
    # flat per-time records without altering any metrics.
    result = dict(status="completed", completed_at=time.time(), elapsed_seconds=time.perf_counter() - started,
                  plan_sha256=sha(out / "plan.json"), budget=budget(), measured_budget=measured,
                  records=records, primary=metrics.primary(_policy_records(records)))
    write(execution / "results.json", result)
    write(execution / "status.json", dict(status="completed", results_sha256=sha(execution / "results.json"), completed_at=time.time()))
    return {k: result[k] for k in ("status", "elapsed_seconds", "measured_budget", "primary")}


def _policy_records(records):
    result = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            rows = [r for r in records if r["seed"] == seed and r["condition"] == condition]
            rows.sort(key=lambda r: r["update"])
            rule, visibility = condition.split("_PL_")
            result.append(dict(seed=seed, condition=condition, rule=rule, live=visibility == "live",
                               trajectory=rows))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "prepare": value = prepare(args.out)
    elif args.command == "verify": value = dict(status="verified", plan_sha256=sha(Path(args.out).resolve() / "plan.json"))
    else: value = execute(args.out)
    print(json.dumps(value, ensure_ascii=False))
