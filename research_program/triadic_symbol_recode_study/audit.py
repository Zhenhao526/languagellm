"""Independent replay audit for symbol and position recoding controls."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research_program.triadic_message_study import runner as core
from research_program.triadic_action_dependency_study import dataset as task_dataset
from research_program.triadic_action_dependency_study import environment as env
from research_program.triadic_content_response_study import dataset as content
from . import dataset, metrics

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
SOURCE = ROOT / "research_program/triadic_rule_formation_study/results/formation_001"
CONTENT = ROOT / "research_program/triadic_content_response_study/results/content_001"
TARGET = "new_needs_and_layouts"; SEEDS = metrics.SEEDS; CONDITIONS = metrics.CONDITIONS; MODES = metrics.MODES; STEPS = metrics.STEPS; TOL = 2e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return dataset.sha(path)


def read(path):
    return dataset.read(path)


def route(tokens):
    t = np.asarray(tokens); require(t.ndim == 3 and t.shape[1:] == (3, 4) and t.dtype.kind in "iu" and np.all((t >= 0) & (t < 8)), "Audit route")
    vis = np.ones((3, 3), dtype=np.float64); one = np.eye(8, dtype=np.float64)[t]
    return np.concatenate(((one[:, None] * vis[None, :, :, None, None]).reshape(len(t), 3, 96), np.broadcast_to(vis[None], (len(t), 3, 3))), axis=-1)


def replay(networks, observations, natural, senders, donor):
    n = len(observations); first = natural[:, 0]; r1 = route(first)
    enc = np.eye(8, dtype=np.float64)[donor].reshape(n, 32)
    for viewer in range(3):
        rows = np.flatnonzero(senders != viewer); slots = 32 * senders[rows, None] + np.arange(32)
        r1[rows[:, None], viewer, slots] = enc[rows]
    z = np.concatenate((observations, r1), axis=-1)
    l2 = np.stack([core.base.actor_forward(networks[3 * a + 1], z[:, a])[0].reshape(n, 4, 8) for a in range(3)], axis=1)
    p2, _ = core.base.policy_distribution(l2); second = np.argmax(p2, axis=-1).astype(np.int8); r2 = route(second)
    action_input = np.concatenate((observations, r1, r2), axis=-1)
    logits = np.stack([core.base.actor_forward(networks[3 * a + 2], action_input[:, a])[0] for a in range(3)], axis=1)
    p, _ = core.base.policy_distribution(logits)
    return np.stack((first, second), axis=1), p, np.argmax(p, axis=-1).astype(np.int16), r1, r2


def features(states):
    packed = np.asarray(states)
    worlds = [env.State(tuple(map(int, row[:3])), tuple(map(int, row[3:7])), tuple(map(int, row[7:10]))) for row in packed]
    return task_dataset.encode_observations([{a: env.observe(world, a, information="PL") for a in env.AGENTS} for world in worlds])


def recode(raw, mode):
    raw = np.asarray(raw); order = np.asarray(dataset.POSITION_ORDERS[mode], dtype=np.int8); result = raw[:, order]
    if mode == "global_symbol_permutation":
        result = np.asarray(dataset.SYMBOL_PERMUTATION, dtype=np.int8)[result]
    return result.astype(np.int8, copy=False)


def independent_metrics(p, row):
    p = np.asarray(p, dtype=np.float64); n = len(p); ix = np.arange(n); listeners = row["listeners"]; candidates = row["candidate_receiver_actions"]
    cp = p[ix[:, None], listeners[:, None], candidates]; target = cp[ix, row["donor_endpoint"]]; cp[ix, row["donor_endpoint"]] = -np.inf; margin = target - cp.max(axis=1)
    G = int(row["group_index"].max()) + 1; B = int(row["background_index"].max()) + 1; tensor = margin.reshape(G, B, 4, 4); off = ~np.eye(4, dtype=bool)
    cell = tensor[:, :, off].reshape(G, B, 12).mean(-1); group = cell.mean(-1)
    return dict(M=float(group.mean()), margin_mean=float(margin.mean()), off_diagonal_margin_mean=float(margin[row["host_endpoint"] != row["donor_endpoint"]].mean()), group_means=group.tolist(), off_diagonal_cell_means=cell.tolist())


def source_path(seed, condition, step, kind="trajectory"):
    folder = SOURCE / "execution" / f"seed_{seed}_{condition}"
    return folder / (f"trajectory_{step:04d}_{TARGET}.npz" if kind == "trajectory" else f"checkpoint_{step:04d}.npz")


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def compare(a, b, label, errors):
    a = np.asarray(a); b = np.asarray(b); require(a.shape == b.shape, label + " shape")
    err = float(np.max(np.abs(a - b))) if a.size else 0.0; errors[label] = max(errors.get(label, 0.0), err); require(err <= TOL, label + " tolerance")


def baseline_values():
    summary = read(CONTENT / "summary_001" / "summary.json")
    values = {(int(r["seed"]), r["condition"], r["update"]): r for r in summary["records"]}
    require(len(values) == 384, "Complete content baseline")
    return values


def audit(run, expected_plan):
    run = Path(run).resolve(); plan, prepared, static_maps = dataset.verify(run)
    require(sha(run / "plan.json") == expected_plan, "Recoding plan hash")
    result = read(run / "execution" / "results.json"); status = read(run / "execution" / "status.json")
    require(status["status"] == result["status"] == "completed" and status["results_sha256"] == sha(run / "execution" / "results.json"), "Completed recoding execution")
    states = content.pack_states(read(SOURCE / "prepared.json")["partitions"][TARGET]); x_all = features(states)
    _content_plan, _content_prepared, content_arrays = dataset.source_arrays(); row = content.flatten_rows(content_arrays); errors = {}
    expected_keys = {(s, c, m, t) for s in SEEDS for c in CONDITIONS for m in MODES for t in STEPS}
    actual = {(r["seed"], r["condition"], r["mode"], r["update"]): r for r in result["records"]}
    require(set(actual) == expected_keys and len(actual) == 1152, "Complete recoding grid")
    counts = dict(records=0, live_files=0, silent_aliases=0, live_rows=0, silent_rows=0, independent_module_samples=0)
    for seed in SEEDS:
        for condition in CONDITIONS:
            for mode in MODES:
                for step in STEPS:
                    rec = actual[seed, condition, mode, step]; source = source_path(seed, condition, step)
                    require(rec["source_natural_path"] == str(source) and rec["source_natural_sha256"] == sha(source), "Natural identity")
                    pool = load(source); host = row["host_indices"]; donor = static_maps[f"{mode}_donor_indices"]; senders = row["senders"]
                    natural = pool["messages"][host]; raw = pool["messages"][donor, 0, senders, :]; packets = recode(raw, mode)
                    if rec["live"]:
                        networks = core.load_networks(source_path(seed, condition, step, "checkpoint")); ps = []; ms = []; acts = []
                        for start in range(0, len(host), 1024):
                            sl = slice(start, min(start + 1024, len(host)))
                            m, p, a, r1, r2 = replay(networks, x_all[host[sl]], natural[sl], senders[sl], packets[sl]); ps.append(p); ms.append(m); acts.append(a)
                        p = np.concatenate(ps); m = np.concatenate(ms); a = np.concatenate(acts); saved = load(rec["path"])
                        require(rec["path"] is not None and rec["data_sha256"] == sha(rec["path"]), "Live probe hash")
                        compare(saved["action_probabilities"], p, "probability", errors); require(np.array_equal(saved["action_indices"], a) and np.array_equal(saved["messages"], m), "Saved action/message replay")
                        for field, expected in (("raw_donor_packets", raw), ("donor_packets", packets), ("host_indices", host), ("donor_indices", donor), ("source_group_index", static_maps[f"{mode}_source_group_index"]), ("source_packet_endpoint", static_maps[f"{mode}_source_packet_endpoint"]), ("group_index", row["group_index"]), ("background_index", row["background_index"]), ("host_endpoint", row["host_endpoint"]), ("donor_endpoint", row["donor_endpoint"]), ("senders", senders), ("listeners", row["listeners"]), ("candidate_receiver_actions", row["candidate_receiver_actions"]), ("target_receiver_actions", row["target_receiver_actions"])):
                            require(np.array_equal(saved[field], expected), "Metadata " + field)
                        z = independent_metrics(p, row)
                        for field in ("M", "margin_mean", "off_diagonal_margin_mean", "group_means", "off_diagonal_cell_means"):
                            compare(rec["metrics"][field], z[field], "metric_" + field, errors)
                        counts["live_files"] += 1; counts["live_rows"] += len(host); counts["independent_module_samples"] += 6 * len(host)
                    else:
                        z = independent_metrics(pool["action_probabilities"][host], row)
                        for field in ("M", "margin_mean", "off_diagonal_margin_mean", "group_means", "off_diagonal_cell_means"):
                            compare(rec["metrics"][field], z[field], "silent_metric_" + field, errors)
                        require(rec["path"] is None and rec["alias_of_source"] and rec["neural_forward_samples"] == 0, "Silent alias")
                        counts["silent_aliases"] += 1; counts["silent_rows"] += len(host)
                    counts["records"] += 1
    calculated = metrics.primary(result["records"], baseline_values())
    require(calculated == result["primary"], "Independent primary reconstruction")
    expected_counts = dict(records=1152, live_files=576, silent_aliases=576, live_rows=576 * 27648, silent_rows=576 * 27648, independent_module_samples=95551488)
    require(counts == expected_counts, "Audit scope")
    return dict(status="passed", plan_sha256=expected_plan, scope=counts, max_errors=errors, primary=calculated, audit_source_sha256=sha(__file__))


def freeze(run, expected_plan):
    run = Path(run).resolve(); path = HERE / "audit_freeze_001.json"
    require(not path.exists(), "Never overwrite audit freeze"); require(not (run / "execution").exists(), "Freeze precedes execution"); require(sha(run / "plan.json") == expected_plan, "Plan hash")
    value = dict(status="frozen_before_recoding_execution", at=time.time(), plan_sha256=expected_plan, source_sha256={str(HERE / "audit.py"): sha(HERE / "audit.py")}, policy_output_files_read=0, neural_forward_samples=0, independent_replay_module_samples=95551488)
    dataset.write(path, value); return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True); parser.add_argument("--plan-sha", required=True); parser.add_argument("--out"); parser.add_argument("--freeze", action="store_true"); args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze(args.run, args.plan_sha), ensure_ascii=False))
    else:
        out = Path(args.out).resolve(); require(not out.exists(), "Never overwrite audit"); out.mkdir(parents=True); started = time.perf_counter()
        try:
            value = audit(args.run, args.plan_sha); value["elapsed_seconds"] = time.perf_counter() - started; dataset.write(out / "verification.json", value); print(json.dumps(value, ensure_ascii=False))
        except BaseException as error:
            dataset.write(out / "failure.json", dict(status="failed", error=repr(error))); raise
