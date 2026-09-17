"""Execute frozen symbol and packet-position recoding controls."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

from research_program.triadic_message_study import runner as core
from research_program.triadic_rule_formation_study import runner as formation
from research_program.triadic_content_response_study import dataset as content
from . import dataset, intervene, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "research_program/triadic_rule_formation_study/results/formation_001"
CONTENT = ROOT / "research_program/triadic_content_response_study/results/content_001"
SEEDS = metrics.SEEDS; CONDITIONS = metrics.CONDITIONS; MODES = metrics.MODES; STEPS = metrics.STEPS; TARGET = "new_needs_and_layouts"
FORMATION_PLAN_SHA = "b3ae053bca41e5acf60fa9108835cfe5bec85f2c6638cc48b2a7db9e5f00354f"
FORMATION_PREPARED_SHA = "92acde1b3e535990fdb120bb9dc19b1c5c2f6c7177d504182d2f6f687ffbe885"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return dataset.sha(path)


def read(path):
    return dataset.read(path)


def write(path, value):
    return dataset.write(path, value)


def source_path(seed, condition, step, kind="trajectory"):
    folder = SOURCE / "execution" / f"seed_{seed}_{condition}"
    return folder / (f"trajectory_{step:04d}_{TARGET}.npz" if kind == "trajectory" else f"checkpoint_{step:04d}.npz")


def verify_source():
    require(sha(SOURCE / "plan.json") == FORMATION_PLAN_SHA and sha(SOURCE / "prepared.json") == FORMATION_PREPARED_SHA, "Formation source changed")
    require(read(SOURCE / "freeze.json") == {"plan_sha256": FORMATION_PLAN_SHA, "prepared_sha256": FORMATION_PREPARED_SHA}, "Formation freeze changed")
    require(read(SOURCE / "execution" / "status.json").get("status") == "completed", "Formation execution incomplete")
    for seed in SEEDS:
        for condition in CONDITIONS:
            for step in STEPS:
                require(source_path(seed, condition, step).exists(), "Missing natural source")
                if condition.endswith("_live"):
                    require(source_path(seed, condition, step, "checkpoint").exists(), "Missing checkpoint")


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def recode_packets(raw, mode):
    raw = np.asarray(raw)
    require(raw.ndim == 2 and raw.shape[1] == 4 and raw.dtype.kind in "iu" and np.all((raw >= 0) & (raw < 8)), "Invalid raw packets")
    order = np.asarray(dataset.POSITION_ORDERS[mode], dtype=np.int8)
    values = raw[:, order]
    if mode == "global_symbol_permutation":
        values = np.asarray(dataset.SYMBOL_PERMUTATION, dtype=np.int8)[values]
    return values.astype(np.int8, copy=False)


def one_time(seed, condition, mode, step, static_maps, features, output, row):
    rule, visibility = condition.split("_PL_"); live = visibility == "live"
    source = source_path(seed, condition, step); pool = load_npz(source)
    host = row["host_indices"]; donor = static_maps[f"{mode}_donor_indices"]; senders = row["senders"]
    natural = pool["messages"][host]; raw_packets = pool["messages"][donor, 0, senders, :]; packets = recode_packets(raw_packets, mode)
    if live:
        networks = core.load_networks(source_path(seed, condition, step, "checkpoint"))
        trace = intervene.intervene_first(networks, features[host], natural, senders, packets)
        probabilities = trace["action_probabilities"]; actions = trace["action_indices"]; messages = trace["messages"]
        outdir = Path(output) / f"seed_{seed}_{condition}"; outdir.mkdir(parents=True, exist_ok=True)
        datapath = outdir / f"{mode}_{step:04d}.npz"
        payload = dict(action_probabilities=probabilities, action_indices=actions, messages=messages,
                       raw_donor_packets=raw_packets, donor_packets=packets, host_indices=host,
                       donor_indices=donor, source_group_index=static_maps[f"{mode}_source_group_index"],
                       source_packet_endpoint=static_maps[f"{mode}_source_packet_endpoint"], group_index=row["group_index"],
                       background_index=row["background_index"], host_endpoint=row["host_endpoint"], donor_endpoint=row["donor_endpoint"],
                       senders=senders, listeners=row["listeners"], candidate_receiver_actions=row["candidate_receiver_actions"],
                       target_receiver_actions=row["target_receiver_actions"])
        with datapath.open("xb") as stream:
            np.savez_compressed(stream, **payload)
        path_value = str(datapath); data_hash = sha(datapath); forwards = 6 * len(host)
    else:
        probabilities = pool["action_probabilities"][host]; actions = pool["action_indices"][host]
        messages = None; path_value = None; data_hash = None; forwards = 0
    m = metrics.margins(probabilities, row["group_index"], row["background_index"], row["host_endpoint"], row["donor_endpoint"], row["listeners"], row["candidate_receiver_actions"])
    outdir = Path(output) / f"seed_{seed}_{condition}"; outdir.mkdir(parents=True, exist_ok=True)
    rec = dict(seed=seed, condition=condition, rule=rule, live=live, mode=mode, update=step,
               source_natural_path=str(source), source_natural_sha256=sha(source), path=path_value,
               data_sha256=data_hash, alias_of_source=not live, worlds=len(host), groups=48, backgrounds=36,
               host_donor_cells=16, off_diagonal_rows=int(np.sum(row["host_endpoint"] != row["donor_endpoint"])),
               metrics={key: value for key, value in m.items() if key != "margins"}, logical_module_samples=6 * len(host),
               neural_forward_samples=forwards, transformation={"mode": mode, "symbol_permutation": list(dataset.SYMBOL_PERMUTATION),
                                                                  "position_order": list(dataset.POSITION_ORDERS[mode])},
               intervention="recoded donor W1 packet only; live W2 and action recomputed; silent is invisible alias")
    write(outdir / f"{mode}_record_{step:04d}.json", rec)
    return rec


def worker(payload):
    seed, static_path, output = payload; output = Path(output)
    _plan, _prepared, arrays = dataset.source_arrays()
    source_prepared = read(formation.ORIGINAL / "prepared.json"); states = content.pack_states(source_prepared["partitions"][TARGET])
    features = intervene.observations(states); static_maps = dataset.load_maps(static_path); row = content.flatten_rows(arrays)
    records = []
    for mode in MODES:
        for condition in CONDITIONS:
            for step in STEPS:
                records.append(one_time(seed, condition, mode, step, static_maps, features, output, row))
    write(output / f"seed_{seed}_result.json", dict(seed=seed, records=records, feature_array_sha256=__import__("hashlib").sha256(features.tobytes()).hexdigest()))
    return dict(seed=seed, records=records)


def budget():
    rows = 48 * 36 * 16; records = 16 * 4 * 3 * 6; live = 16 * 2 * 3 * 6
    return dict(records=records, live_probe_files=live, silent_alias_records=records - live,
                rows_per_record=rows, logical_module_samples=records * rows * 6,
                new_forward_module_samples=live * rows * 6, training_updates=0)


def baseline_values():
    summary = read(CONTENT / "summary_001" / "summary.json")
    values = {(int(r["seed"]), r["condition"], r["update"]): r for r in summary["records"]}
    require(len(values) == 384, "Complete content baseline")
    return values


def verify(out):
    plan, prepared, maps = dataset.verify(out)
    require(plan["config"]["primary"] == "same_endpoint_live_M_minus_recoded_live_M_by_mode", "Recoding primary config")
    verify_source()
    return plan, prepared, maps


def execute(out):
    out = Path(out).resolve(); plan, prepared, static_maps = verify(out); execution = out / "execution"
    require(not execution.exists(), "Never overwrite recoding execution"); execution.mkdir()
    started = time.perf_counter(); write(execution / "started.json", dict(at=time.time(), plan_sha256=sha(out / "plan.json")))
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        groups = pool.map(worker, [(seed, str(out), str(execution)) for seed in SEEDS])
    records = [record for group in groups for record in group["records"]]
    require(len(records) == budget()["records"], "Recoding record count")
    measured = dict(records=len(records), live_probe_files=sum(r["path"] is not None for r in records),
                    silent_alias_records=sum(r["alias_of_source"] for r in records), rows_per_record=27648,
                    logical_module_samples=sum(r["logical_module_samples"] for r in records),
                    new_forward_module_samples=sum(r["neural_forward_samples"] for r in records), training_updates=0)
    require(measured == budget(), "Recoding budget mismatch")
    primary = metrics.primary(records, baseline_values())
    result = dict(status="completed", completed_at=time.time(), elapsed_seconds=time.perf_counter() - started,
                  plan_sha256=sha(out / "plan.json"), budget=budget(), measured_budget=measured, records=records, primary=primary)
    write(execution / "results.json", result); write(execution / "status.json", dict(status="completed", results_sha256=sha(execution / "results.json"), completed_at=time.time()))
    return {"status": result["status"], "elapsed_seconds": result["elapsed_seconds"], "measured_budget": measured, "primary": primary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "verify", "execute")); parser.add_argument("--out", required=True); args = parser.parse_args()
    value = dataset.prepare(args.out) if args.command == "prepare" else dict(status="verified", plan_sha256=sha(Path(args.out).resolve() / "plan.json")) if args.command == "verify" else execute(args.out)
    print(json.dumps(value, ensure_ascii=False))
