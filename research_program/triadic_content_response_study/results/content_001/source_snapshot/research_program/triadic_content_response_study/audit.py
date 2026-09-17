"""Independent replay audit for the four-choice probe.

This module does not import the candidate intervention or candidate metrics.
It duplicates packet routing, W2/action forwarding, and the four-choice margin
calculation, then compares every saved live row and every silent alias.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research_program.triadic_message_study import runner as core
from research_program.triadic_action_dependency_study import dataset as task_dataset
from research_program.triadic_action_dependency_study import environment as env
from . import dataset
from . import metrics as protocol

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "research_program/triadic_rule_formation_study/results/formation_001"
TARGET = "new_needs_and_layouts"
STEPS = protocol.STEPS
SEEDS = protocol.SEEDS
CONDITIONS = protocol.CONDITIONS
TOL = 2e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return dataset.sha(path)


def read(path):
    return dataset.read_json(path)


def route_window(tokens, live=True):
    tokens = np.asarray(tokens)
    require(tokens.ndim == 3 and tokens.shape[1:] == (3, 4) and tokens.dtype.kind in "iu" and np.all((tokens >= 0) & (tokens < 8)), "Audit route token domain")
    visibility = np.ones((3, 3), dtype=np.float64) if live else np.eye(3, dtype=np.float64)
    onehot = np.eye(8, dtype=np.float64)[tokens]
    visible = onehot[:, None, :, :, :] * visibility[None, :, :, None, None]
    return np.concatenate((visible.reshape(len(tokens), 3, 96), np.broadcast_to(visibility[None], (len(tokens), 3, 3))), axis=-1)


def patch_route(first, sender, donor):
    route = route_window(first, True)
    encoded = np.eye(8, dtype=np.float64)[donor].reshape(len(first), 32)
    for viewer in range(3):
        rows = np.flatnonzero(sender != viewer)
        slots = 32 * sender[rows, None] + np.arange(32)
        route[rows[:, None], viewer, slots] = encoded[rows]
    return route


def replay(networks, x, natural_messages, senders, donor_packets):
    n = len(x)
    first = natural_messages[:, 0]
    route1 = patch_route(first, senders, donor_packets)
    second_input = np.concatenate((x, route1), axis=-1)
    logits2 = np.stack([core.base.actor_forward(networks[3 * a + 1], second_input[:, a])[0].reshape(n, 4, 8)
                         for a in range(3)], axis=1)
    p2, _ = core.base.policy_distribution(logits2)
    second = np.argmax(p2, axis=-1).astype(np.int8)
    route2 = route_window(second, True)
    action_input = np.concatenate((x, route1, route2), axis=-1)
    logits = np.stack([core.base.actor_forward(networks[3 * a + 2], action_input[:, a])[0]
                       for a in range(3)], axis=1)
    p, _ = core.base.policy_distribution(logits)
    return np.stack((first, second), axis=1), p, np.argmax(p, axis=-1).astype(np.int16), route1, route2


def features(states):
    packed = np.asarray(states)
    worlds = [env.State(tuple(map(int, row[:3])), tuple(map(int, row[3:7])), tuple(map(int, row[7:10]))) for row in packed]
    return task_dataset.encode_observations([{a: env.observe(s, a, information="PL") for a in env.AGENTS} for s in worlds])


def independent_metrics(p, row, group_layer):
    p = np.asarray(p, dtype=np.float64); n = len(p)
    g = row["group_index"]; b = row["background_index"]; e = row["host_endpoint"]; d = row["donor_endpoint"]
    l = row["listeners"]; c = row["candidate_receiver_actions"]
    ix = np.arange(n); cp = p[ix[:, None], l[:, None], c]
    target = cp[ix, d]; cp[ix, d] = -np.inf; margin = target - cp.max(axis=1)
    G = int(g.max()) + 1; B = int(b.max()) + 1
    tensor = margin.reshape(G, B, 4, 4); off = ~np.eye(4, dtype=bool)
    cell = tensor[:, :, off].reshape(G, B, 12).mean(-1); group = cell.mean(1)
    layer_labels = np.asarray(group_layer, dtype=np.int64)
    layers = np.asarray([group[layer_labels == i].mean() for i in range(12)])
    return dict(M=float(layers.mean()), layer_means=layers.tolist(), group_means=group.tolist(),
                off_diagonal_cell_means=cell.tolist(), off_diagonal_margin_mean=float(margin[e != d].mean()),
                margin_mean=float(margin.mean()), margins=margin.tolist())


def source_path(seed, condition, step, kind):
    folder = SOURCE / "execution" / f"seed_{seed}_{condition}"
    return folder / (f"trajectory_{step:04d}_{TARGET}.npz" if kind == "trajectory" else f"checkpoint_{step:04d}.npz")


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def compare(a, b, label, errors):
    a = np.asarray(a); b = np.asarray(b)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label + " shape/domain")
    err = float(np.max(np.abs(a - b))) if a.size else 0.0
    errors[label] = max(errors.get(label, 0.0), err)
    require(err <= TOL, label + " exceeds tolerance")


def audit(run, expected_plan_sha):
    run = Path(run).resolve(); plan, prepared, static = dataset.verify(run)
    require(sha(run / "plan.json") == expected_plan_sha, "Candidate plan hash")
    execution = run / "execution"; result = read(execution / "results.json"); status = read(execution / "status.json")
    require(status["status"] == result["status"] == "completed", "Completed probe required")
    require(status["results_sha256"] == sha(execution / "results.json"), "Probe results hash")
    require(result["plan_sha256"] == expected_plan_sha, "Probe result plan chain")
    require(result["budget"]["logical_module_samples"] == 63700992 and result["budget"]["new_forward_module_samples"] == 31850496, "Fixed candidate budget")
    states = dataset.pack_states(read(SOURCE / "prepared.json")["partitions"][TARGET])
    x_all = features(states)
    row = dataset.flatten_rows(static)
    expected_records = {(s, c, t) for s in SEEDS for c in CONDITIONS for t in STEPS}
    actual_records = {(r["seed"], r["condition"], r["update"]): r for r in result["records"]}
    require(set(actual_records) == expected_records and len(actual_records) == 384, "Complete record grid")
    errors = {}; counts = dict(records=0, live_files=0, silent_aliases=0, live_rows=0, silent_rows=0,
                               independent_module_samples=0, diagonal_rows=0)
    for key in [(s, c, t) for s in SEEDS for c in CONDITIONS for t in STEPS]:
        seed, condition, step = key; rec = actual_records[key]; source = source_path(seed, condition, step, "trajectory")
        require(rec["source_natural_path"] == str(source) and rec["source_natural_sha256"] == sha(source), "Natural source identity")
        pool = load_npz(source); require(np.array_equal(pool["states"], states), "Source state support")
        host = row["host_indices"]; donor = row["donor_indices"]; senders = row["senders"]
        natural_messages = pool["messages"][host]; donor_packets = pool["messages"][donor, 0, senders, :]
        if rec["live"]:
            networks = core.load_networks(source_path(seed, condition, step, "checkpoint"))
            p_batches = []; m_batches = []; a_batches = []
            for start in range(0, len(host), 1024):
                sl = slice(start, min(start + 1024, len(host)))
                m, p, a, route1, route2 = replay(networks, x_all[host[sl]], natural_messages[sl], senders[sl], donor_packets[sl])
                p_batches.append(p); m_batches.append(m); a_batches.append(a)
                # Self channel is unchanged; only the specified sender's outward
                # slots are replaced.  Visibility remains live for all viewers.
                expected_route = route_window(natural_messages[sl, 0], True)
                donor_onehot = np.eye(8, dtype=np.float64)[donor_packets[sl]].reshape(len(m), 32)
                for viewer in range(3):
                    rows = np.flatnonzero(senders[sl] != viewer)
                    slots = 32 * senders[sl][rows, None] + np.arange(32)
                    expected_route[rows[:, None], viewer, slots] = donor_onehot[rows]
                require(np.array_equal(route1, expected_route), "Independent W1 route")
            p = np.concatenate(p_batches); m = np.concatenate(m_batches); a = np.concatenate(a_batches)
            require(rec["path"] is not None and rec["data_sha256"] == sha(rec["path"]), "Saved live probe hash")
            saved = load_npz(rec["path"])
            compare(saved["action_probabilities"], p, "saved_vs_independent_probabilities", errors)
            require(np.array_equal(saved["action_indices"], a), "Saved greedy action indices")
            require(np.array_equal(saved["messages"], m), "Saved recomputed messages")
            for field, expected in (("host_indices", host), ("donor_indices", donor), ("group_index", row["group_index"]),
                                    ("background_index", row["background_index"]), ("host_endpoint", row["host_endpoint"]),
                                    ("donor_endpoint", row["donor_endpoint"]), ("senders", senders),
                                    ("listeners", row["listeners"]), ("candidate_receiver_actions", row["candidate_receiver_actions"]),
                                    ("target_receiver_actions", row["target_receiver_actions"]), ("donor_packets", donor_packets)):
                require(np.array_equal(saved[field], expected), "Saved row metadata " + field)
            diagonal = row["host_endpoint"] == row["donor_endpoint"]
            compare(p[diagonal], pool["action_probabilities"][host[diagonal]], "natural_diagonal_probabilities", errors)
            calc = independent_metrics(p, row, static_arrays_group_layers(static))
            for field in ("M", "layer_means", "group_means", "off_diagonal_cell_means", "off_diagonal_margin_mean", "margin_mean"):
                compare(np.asarray(rec["metrics"][field]), np.asarray(calc[field]), "metric_" + field, errors)
            counts["live_files"] += 1; counts["live_rows"] += len(host); counts["independent_module_samples"] += 6 * len(host)
        else:
            p = pool["action_probabilities"][host]
            calc = independent_metrics(p, row, static_arrays_group_layers(static))
            for field in ("M", "layer_means", "group_means", "off_diagonal_cell_means", "off_diagonal_margin_mean", "margin_mean"):
                compare(np.asarray(rec["metrics"][field]), np.asarray(calc[field]), "silent_metric_" + field, errors)
            require(rec["path"] is None and rec["alias_of_source"] and rec["neural_forward_samples"] == 0, "Silent alias contract")
            counts["silent_aliases"] += 1; counts["silent_rows"] += len(host)
        counts["records"] += 1; counts["diagonal_rows"] += 48 * 36 * 4
    calculated = protocol.primary(_audit_policy_records(result["records"]))
    require(calculated == result["primary"], "Independent primary reconstruction")
    expected_counts = dict(records=384, live_files=192, silent_aliases=192, live_rows=192 * 27648,
                            silent_rows=192 * 27648, independent_module_samples=31850496, diagonal_rows=384 * 48 * 36 * 4)
    require(counts == expected_counts, "Independent audit scope")
    return dict(status="passed", plan_sha256=expected_plan_sha, scope=counts, max_errors=errors,
                primary=calculated, audit_source_sha256=sha(__file__),
                limits=["All 192 live NPZ files and all 192 silent aliases replayed; every live row uses an independent six-forward recomputation.",
                        "The probe is first-window-only and measures four researcher-defined full-action candidates; no semantic decoder, compositionality, lexicon, or language-origin claim follows."])


def static_arrays_group_layers(static):
    # The audit receives the array dictionary from dataset.verify; this helper
    # is overwritten by the caller below through the module-local binding.
    return np.asarray(static["group_layer_index"], dtype=np.int64) if isinstance(static, dict) else np.arange(48) // 4


def _audit_policy_records(records):
    out = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            rows = [r for r in records if r["seed"] == seed and r["condition"] == condition]
            rows.sort(key=lambda r: r["update"])
            rule, visibility = condition.split("_PL_")
            out.append(dict(seed=seed, condition=condition, rule=rule, live=visibility == "live", trajectory=rows))
    return out


def freeze(run, expected_plan_sha):
    run = Path(run).resolve(); path = HERE / "audit_freeze_001.json"
    require(not path.exists(), "Never overwrite audit freeze")
    require(not (run / "execution").exists(), "Audit freeze must precede execution")
    require(sha(run / "plan.json") == expected_plan_sha, "Frozen candidate plan")
    value = dict(status="frozen_before_probe_execution", at=time.time(), plan_sha256=expected_plan_sha,
                 source_sha256={str(p): sha(p) for p in (HERE / "audit.py",)},
                 policy_output_files_read=0, neural_forward_samples=0,
                 independent_replay_module_samples=31850496)
    dataset.write_json(path, value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--out")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze(args.run, args.plan_sha), ensure_ascii=False))
    else:
        out = Path(args.out).resolve(); require(not out.exists(), "Never overwrite audit output"); out.mkdir(parents=True)
        started = time.perf_counter()
        try:
            value = audit(args.run, args.plan_sha); value["elapsed_seconds"] = time.perf_counter() - started
            dataset.write_json(out / "verification.json", value)
            print(json.dumps({k: value[k] for k in ("status", "scope", "max_errors")}, ensure_ascii=False))
        except BaseException as error:
            dataset.write_json(out / "failure.json", dict(status="failed", error=repr(error), elapsed_seconds=time.perf_counter() - started)); raise
