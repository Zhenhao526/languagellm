"""Independent replay audit for the public/private codebook study.

This module deliberately does not import the study producer (``runner``) or
its wire wrapper (``wire``).  It rebuilds the discrete map, causal rollout,
greedy action choice, and settlement comparison from the frozen dependencies,
then compares every saved evaluation artifact with that replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from research_program.triadic_rule_formation_study import runner as formation
from research_program.triadic_reciprocal_execution_study import environment, kernel
from . import design


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return design.sha(Path(path))


def read(path):
    return design.read(Path(path))


def write(path, value):
    design.write(Path(path), value)


def _recode(tokens, maps):
    tokens = np.asarray(tokens)
    maps = tuple(np.asarray(row, dtype=np.int8) for row in maps)
    require(tokens.ndim == 3 and tokens.shape[1:] == (3, 4), "Invalid internal token tensor")
    require(len(maps) == 3 and all(row.shape == (8,) for row in maps), "Invalid replay maps")
    out = np.empty_like(tokens)
    for sender, mapping in enumerate(maps):
        require(np.array_equal(np.sort(mapping), np.arange(8)), "Replay map is not a bijection")
        out[:, sender] = mapping[tokens[:, sender]]
    return out


def _routed(tokens, live):
    tokens = np.asarray(tokens)
    require(tokens.ndim == 3 and tokens.shape[1:] == (3, 4), "Invalid wire token tensor")
    require(tokens.dtype.kind in "iu" and ((tokens >= 0) & (tokens < 8)).all(), "Invalid wire symbols")
    visibility = np.ones((3, 3), dtype=np.float64) if live else np.eye(3, dtype=np.float64)
    onehot = np.eye(8, dtype=np.float64)[tokens]
    visible = onehot[:, None, :, :, :] * visibility[None, :, :, None, None]
    bits = np.broadcast_to(visibility[None], (len(tokens), 3, 3))
    return np.concatenate((visible.reshape(len(tokens), 3, 96), bits), axis=-1)


def _categorical(probabilities):
    require(probabilities.shape[-2:] == (4, 8), "Invalid sender probabilities")
    return np.argmax(probabilities, axis=-1).astype(np.int8)


def replay(networks, observations, live, maps):
    """Independent greedy causal replay, retaining internal and wire tokens."""
    x = np.asarray(observations, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1:] == (3, 54), "Invalid replay observations")
    require(len(networks) == 9 and np.isfinite(x).all(), "Invalid replay input")
    caches = [None] * 9
    sender_probabilities = []
    sender_log_probabilities = []
    messages = []
    wire_messages = []
    inputs = x
    for window in range(2):
        logits = []
        for actor in range(3):
            z, caches[3 * actor + window] = formation.core.base.actor_forward(
                networks[3 * actor + window], inputs[:, actor]
            )
            logits.append(z.reshape(len(x), 4, 8))
        z = np.stack(logits, axis=1)
        probabilities, log_probabilities = formation.core.base.policy_distribution(z)
        internal = _categorical(probabilities)
        on_wire = _recode(internal, maps)
        sender_probabilities.append(probabilities)
        sender_log_probabilities.append(log_probabilities)
        messages.append(internal)
        wire_messages.append(on_wire)
        if window == 0:
            inputs = np.concatenate((x, _routed(on_wire, live)), axis=-1)
    action_inputs = np.concatenate((x, _routed(wire_messages[0], live), _routed(wire_messages[1], live)), axis=-1)
    action_logits = []
    for actor in range(3):
        z, caches[3 * actor + 2] = formation.core.base.actor_forward(
            networks[3 * actor + 2], action_inputs[:, actor]
        )
        action_logits.append(z)
    return dict(
        messages=np.stack(messages, axis=1),
        wire_messages=np.stack(wire_messages, axis=1),
        sender_probabilities=np.stack(sender_probabilities, axis=1),
        sender_log_probabilities=np.stack(sender_log_probabilities, axis=1),
        action_logits=np.stack(action_logits, axis=1),
        action_inputs=action_inputs,
        caches=caches,
    )


def _same(expected, actual, label):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    require(expected.shape == actual.shape, f"{label} shape mismatch: {expected.shape} != {actual.shape}")
    require(expected.dtype == actual.dtype, f"{label} dtype mismatch: {expected.dtype} != {actual.dtype}")
    if expected.dtype.kind in "fc":
        delta = float(np.max(np.abs(expected - actual))) if expected.size else 0.0
        require(delta == 0.0, f"{label} changed in replay: max_abs_error={delta}")
        return delta
    require(np.array_equal(expected, actual), f"{label} changed in replay")
    return 0.0


def _arrays(static):
    return {part: formation.make_arrays(static["partitions"][part]) for part in design.PARTS}


def _evaluation_paths(run, static):
    rows = []
    for step in design.STEPS:
        row = next(item for item in run["trajectory"] if item["update"] == step)
        rows.append((f"trajectory_{step:04d}", design.TARGET, row["evaluation"], row["checkpoint_path"]))
    for part in design.PARTS:
        if part != design.TARGET:
            rows.append((f"final_{part}", part, run["final"][part], run["final_checkpoint_sha256"]))
    require(len(rows) == len(design.STEPS) + len(design.PARTS) - 1, "Incomplete evaluation set")
    return rows


def _checkpoint_path(run, label, execution):
    if label.startswith("trajectory_"):
        step = int(label.rsplit("_", 1)[-1])
        return Path(execution) / f"seed_{run['seed']}_{run['condition']}" / f"checkpoint_{step:04d}.npz"
    return Path(execution) / f"seed_{run['seed']}_{run['condition']}" / "checkpoint_6000.npz"


def replay_evaluation(run, label, part, record, checkpoint, arrays, static):
    path = Path(record["path"])
    require(path.is_file() and sha(path) == record["data_sha256"], f"Evaluation artifact hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        stored = {key: saved[key].copy() for key in saved.files}
    ids = stored["state_indices"]
    require(ids.dtype == np.int64 and ids.ndim == 1, f"Invalid state indices: {path}")
    expected_ids = np.arange(len(arrays[part]["packed_states"]), dtype=np.int64)
    require(np.array_equal(ids, expected_ids), f"Evaluation is not the complete partition: {path}")
    networks = formation.core.load_networks(checkpoint)
    map_name, live, maps = design.parse_condition(run["condition"])
    require(record["map_name"] == map_name and record["live"] is live, f"Map/channel metadata mismatch: {path}")
    observed = arrays[part]["x_PL"]
    actions = np.empty((len(ids), 3), dtype=np.int16)
    probabilities = np.empty((len(ids), 3, 17), dtype=np.float64)
    messages = np.empty((len(ids), 2, 3, 4), dtype=np.int8)
    wire_messages = np.empty((len(ids), 2, 3, 4), dtype=np.int8)
    expected_reward = np.empty(len(ids), dtype=np.float64)
    full_probability = np.empty(len(ids), dtype=np.float64)
    execution_probability = np.empty(len(ids), dtype=np.float64)
    full_posterior = np.empty(len(ids), dtype=np.float64)
    for start in range(0, len(ids), 1024):
        stop = min(start + 1024, len(ids)); sl = slice(start, stop)
        trace = replay(networks, observed[sl], live, maps)
        terms = kernel.objective_terms(trace["action_logits"], arrays[part]["rewards"][sl], design.RULE)
        messages[sl] = trace["messages"]
        wire_messages[sl] = trace["wire_messages"]
        probabilities[sl] = terms["probabilities"]
        actions[sl] = probabilities[sl].argmax(-1)
        expected_reward[sl] = terms["native_expected_reward"]
        full_probability[sl] = terms["full_success_probability"]
        execution_probability[sl] = terms["execution_probability"]
        full_posterior[sl] = terms["full_success_posterior_mass"]
    states = arrays[part]["packed_states"][ids]
    native = environment.settle(states, actions, design.RULE)
    strict = environment.settle(states, actions, "strict")
    common = environment.settle(states, actions, "reciprocal")
    expected_data = dict(
        states=states, state_indices=ids, messages=messages, wire_messages=wire_messages,
        action_indices=actions, action_probabilities=probabilities,
        conditional_exact_expected_reward=expected_reward,
        conditional_exact_full_success_probability=full_probability,
        conditional_exact_execution_probability=execution_probability,
        conditional_full_posterior_mass=full_posterior,
    )
    expected_data.update(native)
    expected_data.update({f"strict__{key}": value for key, value in strict.items()})
    expected_data.update({f"common_reciprocal__{key}": value for key, value in common.items()})
    require(set(stored) == set(expected_data), f"Evaluation key set mismatch: {path}")
    errors = {}
    for key in sorted(expected_data):
        errors[key] = _same(expected_data[key], stored[key], f"{path.name}:{key}")
    require(sha(checkpoint) == (record["checkpoint_sha256"] if label.startswith("trajectory_") else run["final_checkpoint_sha256"]),
            f"Checkpoint hash mismatch: {checkpoint}")
    return dict(label=label, partition=part, path=str(path), worlds=len(ids), checkpoint=str(checkpoint), max_abs_error=max(errors.values(), default=0.0))


def audit(out):
    out = Path(out).resolve()
    plan, static = _load_verified_static(out)
    execution = out / "execution"
    results_path = execution / "results.json"
    require(results_path.is_file(), "Missing completed execution results")
    results = read(results_path)
    require(results["status"] == "completed" and sha(results_path) == read(execution / "status.json")["results_sha256"], "Execution result is not frozen")
    arrays = _arrays(static)
    reports = []
    started = time.perf_counter()
    for run in results["runs"]:
        run_dir = execution / f"seed_{run['seed']}_{run['condition']}"
        require(run_dir.is_dir(), f"Missing run directory: {run_dir}")
        result_path = run_dir / "result.json"
        require(result_path.is_file() and sha(result_path) == read(run_dir / "status.json")["result_sha256"], f"Run result is not frozen: {run_dir}")
        require(read(result_path) == run, f"Aggregate/run result mismatch: {run_dir}")
        for label, part, record, _ in _evaluation_paths(run, static):
            checkpoint = _checkpoint_path(run, label, execution)
            require(checkpoint.is_file(), f"Missing checkpoint: {checkpoint}")
            reports.append(replay_evaluation(run, label, part, record, checkpoint, arrays, static))
    require(len(reports) == len(results["runs"]) * (len(design.STEPS) + len(design.PARTS) - 1), "Incomplete replay audit")
    verification = dict(
        status="verified", audited_at=time.time(), output=str(out), plan_sha256=sha(out / "plan.json"),
        results_sha256=sha(results_path), evaluations=len(reports), reports=reports,
        max_abs_error=max(row["max_abs_error"] for row in reports),
        imports_producer_runner=False, imports_producer_wire=False,
        elapsed_seconds=time.perf_counter() - started,
    )
    write(execution / "verification.json", verification)
    write(execution / "receipt.json", dict(status="verified", verification_sha256=sha(execution / "verification.json"), evaluations=len(reports)))
    return verification


def _load_verified_static(out):
    plan = read(out / "plan.json")
    static = read(out / "prepared.json")
    freeze = read(out / "freeze.json")
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Plan changed before audit")
    require(sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "Prepared changed before audit")
    require(static == design.make_prepared() and plan["source_sha256"] == design.sources(), "Frozen design changed before audit")
    for relative, digest in plan["source_sha256"].items():
        require(sha(out / "source_snapshot" / relative) == digest, f"Source snapshot changed: {relative}")
    return plan, static


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    answer = audit(args.out)
    print(json.dumps({k: answer[k] for k in ("status", "evaluations", "max_abs_error", "elapsed_seconds")}, ensure_ascii=False))
