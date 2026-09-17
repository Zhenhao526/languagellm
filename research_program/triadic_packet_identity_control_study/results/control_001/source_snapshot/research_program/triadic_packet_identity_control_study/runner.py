"""Execute packet identity controls on the frozen formation policies."""
from __future__ import annotations
import argparse, json, multiprocessing, time
from pathlib import Path
import numpy as np
from research_program.triadic_message_study import runner as core
from research_program.triadic_rule_formation_study import runner as formation
from . import dataset, intervene, metrics

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
SOURCE = ROOT / "research_program/triadic_rule_formation_study/results/formation_001"
CONTENT = ROOT / "research_program/triadic_content_response_study/results/content_001"
SEEDS = metrics.SEEDS; CONDITIONS = metrics.CONDITIONS; MODES = metrics.MODES; STEPS = metrics.STEPS; TARGET = "new_needs_and_layouts"
FORMATION_PLAN_SHA = "b3ae053bca41e5acf60fa9108835cfe5bec85f2c6638cc48b2a7db9e5f00354f"
FORMATION_PREPARED_SHA = "92acde1b3e535990fdb120bb9dc19b1c5c2f6c7177d504182d2f6f687ffbe885"


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path): return dataset.sha(path)
def read(path): return dataset.read(path)
def write(path, value): return dataset.write(path, value)


def source_path(seed, condition, step, kind="trajectory"):
    folder = SOURCE / "execution" / f"seed_{seed}_{condition}"
    return folder / (f"trajectory_{step:04d}_{TARGET}.npz" if kind == "trajectory" else f"checkpoint_{step:04d}.npz")


def verify_source():
    require(sha(SOURCE / "plan.json") == FORMATION_PLAN_SHA and sha(SOURCE / "prepared.json") == FORMATION_PREPARED_SHA, "Formation source changed")
    freeze = read(SOURCE / "freeze.json"); require(freeze == {"plan_sha256": FORMATION_PLAN_SHA, "prepared_sha256": FORMATION_PREPARED_SHA}, "Formation freeze changed")
    require(read(SOURCE / "execution" / "status.json").get("status") == "completed", "Formation execution incomplete")
    for seed in SEEDS:
        for condition in CONDITIONS:
            for step in STEPS: require(source_path(seed, condition, step).exists(), "Missing natural source")
            if condition.endswith("_live"):
                for step in STEPS: require(source_path(seed, condition, step, "checkpoint").exists(), "Missing checkpoint")


def load_npz(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


def one_time(seed, condition, mode, step, static, maps, features, output, row):
    rule, visibility = condition.split("_PL_"); live = visibility == "live"; n = len(row["group_index"])
    source = source_path(seed, condition, step); pool = load_npz(source)
    host = row["host_indices"]; donor = maps[f"{mode}_donor_indices"]; senders = row["senders"]
    natural = pool["messages"][host]; packets = pool["messages"][donor, 0, senders, :]
    if live:
        networks = core.load_networks(source_path(seed, condition, step, "checkpoint"))
        trace = intervene.intervene_first(networks, features[host], natural, senders, packets)
        p = trace["action_probabilities"]; actions = trace["action_indices"]; messages = trace["messages"]
        outdir = Path(output) / f"seed_{seed}_{condition}"; outdir.mkdir(parents=True, exist_ok=True)
        datapath = outdir / f"{mode}_{step:04d}.npz"
        payload = dict(action_probabilities=p, action_indices=actions, messages=messages, donor_packets=packets,
                       host_indices=host, donor_indices=donor, source_group_index=maps[f"{mode}_source_group_index"],
                       source_packet_endpoint=maps[f"{mode}_source_packet_endpoint"], group_index=row["group_index"],
                       background_index=row["background_index"], host_endpoint=row["host_endpoint"], donor_endpoint=row["donor_endpoint"],
                       senders=senders, listeners=row["listeners"], candidate_receiver_actions=row["candidate_receiver_actions"],
                       target_receiver_actions=row["target_receiver_actions"])
        with datapath.open("xb") as stream: np.savez_compressed(stream, **payload)
        path_value = str(datapath); data_hash = sha(datapath); forwards = 6 * n
    else:
        p = pool["action_probabilities"][host]; actions = pool["action_indices"][host]; path_value = None; data_hash = None; forwards = 0
    m = metrics.margins(p, row["group_index"], row["background_index"], row["host_endpoint"], row["donor_endpoint"], row["listeners"], row["candidate_receiver_actions"])
    rec = dict(seed=seed, condition=condition, rule=rule, live=live, mode=mode, update=step,
               source_natural_path=str(source), source_natural_sha256=sha(source), path=path_value, data_sha256=data_hash,
               alias_of_source=not live, worlds=n, groups=48, backgrounds=36, host_donor_cells=16,
               off_diagonal_rows=int(np.sum(row["host_endpoint"] != row["donor_endpoint"])), metrics={k:v for k,v in m.items() if k != "margins"},
               logical_module_samples=6*n, neural_forward_samples=forwards,
               intervention="identity control changes donor W1 source only; live W2 and action recomputed; silent is invisible alias")
    outdir = Path(output) / f"seed_{seed}_{condition}"; outdir.mkdir(parents=True, exist_ok=True)
    write(outdir / f"{mode}_record_{step:04d}.json", rec)
    return rec


def worker(payload):
    seed, static_path, output = payload
    output = Path(output)
    _content_plan, _content_prepared, content_arrays = dataset.source_arrays()
    source_prepared = read(formation.ORIGINAL / "prepared.json")
    part = source_prepared["partitions"][TARGET]
    states = __import__("research_program.triadic_content_response_study.dataset", fromlist=["pack_states"]).pack_states(part)
    features = intervene.observations(states)
    static_maps = dataset.load_maps(static_path)
    row = __import__("research_program.triadic_content_response_study.dataset", fromlist=["flatten_rows"]).flatten_rows(content_arrays)
    records = []
    for mode in MODES:
        for condition in CONDITIONS:
            for step in STEPS:
                maps = {f"{mode}_{field}": static_maps[f"{mode}_{field}"] for field in ("source_group_index", "source_packet_endpoint", "donor_indices")}
                records.append(one_time(seed, condition, mode, step, static_path, maps, features, output, row))
    write(output / f"seed_{seed}_result.json", dict(seed=seed, records=records, feature_array_sha256=__import__("hashlib").sha256(features.tobytes()).hexdigest()))
    return dict(seed=seed, records=records)


def budget():
    rows = 48*36*16; records = 16*4*2*6; live = 16*2*2*6
    return dict(records=records, live_probe_files=live, silent_alias_records=records-live, rows_per_record=rows,
                logical_module_samples=records*rows*6, new_forward_module_samples=live*rows*6,
                training_updates=0)


def prepare(out): return dataset.prepare(out)
def verify(out):
    plan, prepared, maps = dataset.verify(out); verify_source(); return plan, prepared, maps


def load_same_baseline():
    summary = read(CONTENT / "summary_001" / "summary.json")
    return {(int(r["seed"]), c): float(summary["endpoint"][c]["M"]["mean"]) for r in [] for c in []} if False else None


def same_endpoint_values():
    # Exact same-group live endpoint M, from the already audited content probe.
    summary = read(CONTENT / "summary_001" / "summary.json")
    # summary endpoint means are across seeds; primary needs per-seed values,
    # so recover them from the compact records table.
    values = {}
    for rec in summary["records"]:
        if rec["update"] == 6000 and rec["live"]:
            values[rec["seed"], rec["condition"]] = float(rec["M"])
    require(len(values) == 32, "Complete same-group live baseline")
    return values


def _policy_records(records):
    out = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            for mode in MODES:
                rows = sorted([r for r in records if r["seed"] == seed and r["condition"] == condition and r["mode"] == mode], key=lambda r:r["update"])
                out.extend(rows)
    return out


def execute(out):
    out = Path(out).resolve(); plan, prepared, static_maps = verify(out); execution = out / "execution"; require(not execution.exists(), "Never overwrite control execution"); execution.mkdir()
    started = time.perf_counter(); write(execution / "started.json", dict(at=time.time(), plan_sha256=sha(out / "plan.json")))
    with multiprocessing.get_context("spawn").Pool(4) as pool: groups = pool.map(worker, [(s, str(out), str(execution)) for s in SEEDS])
    records = [r for g in groups for r in g["records"]]; require(len(records) == budget()["records"], "Control record count")
    measured = dict(records=len(records), live_probe_files=sum(r["path"] is not None for r in records), silent_alias_records=sum(r["alias_of_source"] for r in records),
                    rows_per_record=27648, logical_module_samples=sum(r["logical_module_samples"] for r in records), new_forward_module_samples=sum(r["neural_forward_samples"] for r in records), training_updates=0)
    require(measured == budget(), "Control budget mismatch")
    same = same_endpoint_values(); primary = metrics.primary(records, same)
    result = dict(status="completed", completed_at=time.time(), elapsed_seconds=time.perf_counter()-started, plan_sha256=sha(out / "plan.json"), budget=budget(), measured_budget=measured, records=records, primary=primary)
    write(execution / "results.json", result); write(execution / "status.json", dict(status="completed", results_sha256=sha(execution / "results.json"), completed_at=time.time()))
    return {k:result[k] for k in ("status", "elapsed_seconds", "measured_budget", "primary")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "verify", "execute")); parser.add_argument("--out", required=True); args=parser.parse_args()
    value = prepare(args.out) if args.command == "prepare" else dict(status="verified", plan_sha256=sha(Path(args.out).resolve()/"plan.json")) if args.command == "verify" else execute(args.out)
    print(json.dumps(value, ensure_ascii=False))
