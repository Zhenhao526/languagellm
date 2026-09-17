"""Read-only locality and recoding probes for the audited FI policies."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from research_program.triadic_factorized_neutral_altpartner_information_control_study import runner as fi
from research_program.triadic_reciprocal_execution_study import environment
from research_program.triadic_message_study import runner as core

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "research_program/triadic_factorized_neutral_altpartner_information_control_study/results/fi_001"
SEEDS = tuple(range(66701, 66717))
SCHEDULES = ("static", "rematched")
CHANNELS = ("live", "own")
MODES = ("natural", "cross_closed", "symbol_permutation", "position_rotation", "position_reverse",
         "mask_sender_A", "mask_sender_B", "mask_sender_C")
METRICS = ("q_rate", "conditional_q_rate", "target_pair_legal_rate", "proposal_legal_rate", "physical_execution_rate")
SYMBOL_PERMUTATION = np.asarray((3, 7, 1, 6, 0, 4, 2, 5), dtype=np.int8)
POSITION_ORDERS = {
    "position_rotation": np.asarray((1, 2, 3, 0), dtype=np.int8),
    "position_reverse": np.asarray((3, 2, 1, 0), dtype=np.int8),
}
T15 = 2.1314495455597715


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), "Refuse to overwrite " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def condition(schedule, channel):
    require(schedule in SCHEDULES and channel in CHANNELS, "Unknown schedule/channel")
    return f"{schedule}_altpair_factorized_strict_FI_{'live' if channel == 'live' else 'silent'}"


def source_files():
    return (HERE / "__init__.py", HERE / "runner.py", HERE / "audit.py", HERE / "plan.md",
            HERE / "README.md", HERE / "tests/test_probe.py")


def input_files():
    return (SOURCE / "plan.json", SOURCE / "prepared.json", SOURCE / "freeze.json",
            SOURCE / "execution/results.json", SOURCE / "execution/status.json")


def verify_source():
    require(SOURCE.is_dir(), "FI source directory missing")
    for path in input_files():
        require(path.is_file(), "Missing FI source: " + str(path))
    plan = read(SOURCE / "plan.json"); prepared = read(SOURCE / "prepared.json"); freeze = read(SOURCE / "freeze.json")
    require(sha(SOURCE / "plan.json") == freeze["plan_sha256"], "FI plan hash mismatch")
    require(sha(SOURCE / "prepared.json") == freeze["prepared_sha256"], "FI prepared hash mismatch")
    require(plan.get("status") == "prepared_without_training", "FI source is not prepared")
    require(read(SOURCE / "execution/status.json").get("status") == "completed", "FI source execution incomplete")
    result = read(SOURCE / "execution/results.json")
    require(result.get("status") == "completed" and len(result.get("runs", [])) == 64, "FI source results incomplete")
    for run in result["runs"]:
        run_path = SOURCE / "execution" / f"seed_{run['seed']}_{run['condition']}"
        require((run_path / "result.json").is_file(), "FI per-run result missing")
        require(sha(run_path / "checkpoint_6000.npz") == run["final_checkpoint_sha256"], "FI checkpoint hash mismatch")
    return plan, prepared, result


def config():
    return dict(
        schema="triadic_factorized_neutral_altpartner_message_locality_v1",
        source_experiment="triadic_factorized_neutral_altpartner_information_control_study/fi_001",
        seeds=list(SEEDS), schedules=list(SCHEDULES), trained_channels=list(CHANNELS), modes=list(MODES),
        metrics=list(METRICS), new_layout_worlds=56160, alphabet_size=8, tokens_per_window=4, windows=2,
        symbol_permutation=SYMBOL_PERMUTATION.tolist(), position_orders={k: v.tolist() for k, v in POSITION_ORDERS.items()},
        route_definition="self payload is retained; transforms and masks affect only cross-agent payload; cross_closed zeros all visibility bits",
        rollout_definition="greedy messages; transformed first-window route feeds second-window senders; transformed second-window route feeds action head",
        training_updates=0, optimizer_updates=0, model_calls=0,
        interpretation_boundary="post-hoc task-policy mechanism probe; no word, lexicon, compositionality or language-origin claim",
    )


def prepare_metadata(source_hashes, input_hashes):
    return dict(schema=config()["schema"], config=config(), source_sha256=source_hashes, input_sha256=input_hashes,
                prepared_at=core.base.now(), no_training=True, no_model_calls=True)


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), "Never overwrite preparation")
    verify_source()
    source_hashes = {str(p.resolve().relative_to(ROOT)): sha(p) for p in source_files()}
    input_hashes = {str(p.resolve().relative_to(ROOT)): sha(p) for p in input_files()}
    prepared = prepare_metadata(source_hashes, input_hashes)
    out.mkdir(parents=True)
    for path in source_files():
        target = out / "source_snapshot" / path.resolve().relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    write_new(out / "prepared.json", prepared)
    plan = dict(status="prepared_without_policy_reads", created_at=core.base.now(), runtime=dict(python=platform.python_version(), numpy=np.__version__),
                source_sha256=source_hashes, input_sha256=input_hashes, prepared_sha256=sha(out / "prepared.json"), config=config(),
                no_training=True, no_model_calls=True, budget=budget())
    write_new(out / "plan.json", plan)
    write_new(out / "freeze.json", dict(plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json")))
    verify(out)
    write_new(out / "receipt.json", dict(status="prepared_static_only", plan_sha256=sha(out / "plan.json"),
                                         prepared_sha256=sha(out / "prepared.json"), no_training=True, no_model_calls=True))
    return read(out / "receipt.json")


def budget():
    policy_rows = len(SEEDS) * len(SCHEDULES) * len(CHANNELS)
    rows = policy_rows * len(MODES)
    worlds = rows * 56160
    return dict(policy_rows=policy_rows, modes=len(MODES), rows=rows, worlds=worlds,
                model_forwards=worlds * 9, optimizer_updates=0, new_training=0)


def verify(out):
    out = Path(out).resolve(); plan = read(out / "plan.json"); prepared = read(out / "prepared.json"); freeze = read(out / "freeze.json")
    require(plan["config"] == config() and prepared["config"] == config(), "Locality config changed")
    require(sha(out / "prepared.json") == plan["prepared_sha256"] == freeze["prepared_sha256"], "Prepared hash chain mismatch")
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Plan hash chain mismatch")
    verify_source()
    for rel, digest in plan["source_sha256"].items():
        current = ROOT / rel; snap = out / "source_snapshot" / rel
        require(sha(current) == digest and sha(snap) == digest, "Source snapshot changed: " + rel)
    for rel, digest in plan["input_sha256"].items():
        require(sha(ROOT / rel) == digest, "Input changed: " + rel)
    return plan, prepared


def transform_tokens(tokens, mode):
    tokens = np.asarray(tokens)
    require(tokens.ndim == 3 and tokens.shape[1:] == (3, 4) and tokens.dtype.kind in "iu" and ((tokens >= 0) & (tokens < 8)).all(),
            "Invalid token array")
    if mode == "symbol_permutation":
        return SYMBOL_PERMUTATION[tokens]
    if mode in POSITION_ORDERS:
        return tokens[..., POSITION_ORDERS[mode]]
    return tokens.copy()


def _base_visibility(channel):
    require(channel in CHANNELS, "Unknown base channel")
    return np.ones((3, 3), dtype=np.float64) if channel == "live" else np.eye(3, dtype=np.float64)


def routed_window(tokens, channel, mode):
    """Route [B,3,4] tokens to [B,3,99] with mode-specific cross edges."""
    tokens = np.asarray(tokens)
    require(tokens.ndim == 3 and tokens.shape[1:] == (3, 4) and tokens.dtype.kind in "iu" and ((tokens >= 0) & (tokens < 8)).all(),
            "Invalid message tokens")
    base = _base_visibility(channel)
    payload = base.copy(); bits = base.copy()
    if mode == "cross_closed":
        payload = np.eye(3, dtype=np.float64); bits = np.zeros((3, 3), dtype=np.float64)
    elif mode.startswith("mask_sender_"):
        sender = "ABC".index(mode[-1]);
        for viewer in range(3):
            if viewer != sender:
                payload[viewer, sender] = 0.0; bits[viewer, sender] = 0.0
    transformed = transform_tokens(tokens, mode)
    original_onehot = np.eye(8, dtype=np.float64)[tokens]
    transformed_onehot = np.eye(8, dtype=np.float64)[transformed]
    visible = np.empty((len(tokens), 3, 3, 4, 8), dtype=np.float64)
    transform_mode = mode in ("symbol_permutation", "position_rotation", "position_reverse")
    for viewer in range(3):
        for sender in range(3):
            use = transformed_onehot[:, sender] if transform_mode and viewer != sender and payload[viewer, sender] else original_onehot[:, sender]
            visible[:, viewer, sender] = use * payload[viewer, sender]
    return np.concatenate((visible.reshape(len(tokens), 3, 96), np.broadcast_to(bits[None], (len(tokens), 3, 3))), axis=-1)


def rollout_mode(networks, observations, channel, mode):
    x = np.asarray(observations, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1:] == (3, 54) and len(networks) == 9, "Invalid FI rollout input")
    core.base.finite(x, "FI observations")
    inputs = x; messages = []
    for window in range(2):
        logits = []
        for actor in range(3):
            z, _ = core.base.actor_forward(networks[3 * actor + window], inputs[:, actor]); logits.append(z.reshape(len(x), 4, 8))
        p, _ = core.base.policy_distribution(np.stack(logits, axis=1)); msg = fi.core.categorical_tokens(p, None)
        messages.append(msg)
        if window == 0:
            inputs = np.concatenate((x, routed_window(msg, channel, mode)), axis=-1)
    routes = [routed_window(messages[i], channel, mode) for i in range(2)]
    action_input = np.concatenate((x, routes[0], routes[1]), axis=-1)
    action_logits = np.stack([core.base.actor_forward(networks[3 * actor + 2], action_input[:, actor])[0]
                              for actor in range(3)], axis=1)
    return dict(messages=np.stack(messages, axis=1), action_logits=action_logits)


def _event_pair_mask(full):
    return full.reshape(len(full), 3, 8).any(axis=2)


def evaluate_mode(networks, arrays, spec, channel, mode):
    n = int(spec["world_count"]); ids = np.arange(n, dtype=np.int64)
    engaged = physical_count = q_count = pair_count = proposal_count = proposal_den = 0
    third_count = third_den = 0; actor_hits = np.zeros(3, dtype=np.int64)
    plan_counts = np.zeros(24, dtype=np.int64); pair_counts = np.zeros(3, dtype=np.int64)
    expected = full_prob = partial_prob = 0.0
    for start in range(0, n, fi.CONFIG["evaluation_batch_size"]):
        stop = min(start + fi.CONFIG["evaluation_batch_size"], n); ix = ids[start:stop]
        states = arrays["packed_states"][ix]; full = arrays["native_rewards"][ix] == 1.
        trace = rollout_mode(networks, arrays["x_FI"][ix], channel, mode)
        terms = fi.kernel.objective_terms(trace["action_logits"], arrays["native_rewards"][ix])
        expected += float(terms["J"].sum()); full_prob += float(terms["full_success_probability"].sum()); partial_prob += float(terms["partial_success_probability"].sum())
        intent_p, _, proposal_p, _ = fi.kernel.factorized_distribution(trace["action_logits"])
        intent = intent_p.argmax(axis=-1).astype(np.int16); proposal = proposal_p.argmax(axis=-1).astype(np.int16); actions = np.where(intent == 1, proposal + 1, 0).astype(np.int16)
        engaged += int((intent == 1).sum()); actor_hits += (intent == 1).sum(axis=0, dtype=np.int64)
        legal = np.zeros((len(ix), 3, 17), dtype=bool)
        for event in range(24):
            for actor in range(3): legal[:, actor, core.base.JOINT_ACTIONS[event, actor]] |= full[:, event]
        selected = legal[np.arange(len(ix))[:, None], np.arange(3)[None, :], actions]
        proposal_count += int(selected[intent == 1].sum()); proposal_den += int((intent == 1).sum())
        settled = environment.settle(states, actions, "strict"); executed = settled["executed"]; ok = np.any(executed, axis=1)
        actual = settled["actual_pair_index"].astype(np.int64); safe = np.maximum(actual, 0)
        event = safe * 8 + np.maximum(settled["executed_site"], 0).astype(np.int64) * 2 + np.maximum(settled["executed_destination"], 0).astype(np.int64)
        physical_count += int(ok.sum()); pair_targets = _event_pair_mask(full)
        pair_count += int((ok & pair_targets[np.arange(len(ix)), safe]).sum()); q_count += int((ok & full[np.arange(len(ix)), event]).sum())
        if ok.any():
            pair_counts += np.bincount(actual[ok], minlength=3).astype(np.int64); plan_counts += np.bincount(event[ok], minlength=24).astype(np.int64)
            third = np.asarray([2, 1, 0], dtype=np.int64); third_count += int((intent[np.flatnonzero(ok), third[actual[ok]]] == 0).sum()); third_den += int(ok.sum())
    return dict(mode=mode, channel=channel, q_rate=float(q_count / n), conditional_q_rate=float(q_count / physical_count) if physical_count else 0.0,
                target_pair_legal_rate=float(pair_count / physical_count) if physical_count else 0.0,
                proposal_legal_rate=float(proposal_count / proposal_den) if proposal_den else 0.0, physical_execution_rate=float(physical_count / n),
                engagement_rate=float(engaged / (3 * n)), neutral_rate=float(1 - engaged / (3 * n)), actor_engagement_rates=(actor_hits / n).tolist(),
                third_agent_neutral_rate=float(third_count / third_den) if third_den else 0.0,
                legal_plan_selection_counts=plan_counts.tolist(), legal_pair_selection_counts=pair_counts.tolist(), worlds=n, physical_worlds=int(physical_count),
                proposal_legal_denominator=int(proposal_den), conditional_q_denominator_worlds=int(physical_count), conditional_q_numerator_worlds=int(q_count),
                target_pair_denominator_worlds=int(physical_count), target_pair_numerator_worlds=int(pair_count),
                exact_expected_reward_mean=float(expected / n), exact_full_success_probability_mean=float(full_prob / n), exact_partial_success_probability_mean=float(partial_prob / n),
                legal_plan_count=int(full.sum(axis=1).min()), legal_pair_count=2, action_factorization="intent:neutral/engage; proposal:16-way conditional on engage")


def delta_metrics(natural, current):
    return {key: float(current[key] - natural[key]) for key in METRICS}


def _policy_rows(seed, prepared, arrays):
    rows = []
    spec = prepared["partitions"]["new_layouts"]
    for schedule in SCHEDULES:
        for channel in CHANNELS:
            cond = condition(schedule, channel); path = SOURCE / "execution" / f"seed_{seed}_{cond}"; checkpoint = path / "checkpoint_6000.npz"
            nets = fi.load_networks(checkpoint); checkpoint_hash = sha(checkpoint)
            natural = evaluate_mode(nets, arrays, spec, channel, "natural")
            for mode in MODES:
                current = natural if mode == "natural" else evaluate_mode(nets, arrays, spec, channel, mode)
                rows.append(dict(seed=seed, schedule=schedule, trained_channel=channel, condition=cond, mode=mode,
                                 checkpoint_sha256=checkpoint_hash, metrics=current, delta_from_natural=delta_metrics(natural, current),
                                 source_policy_result_sha256=sha(path / "result.json"), intervention={
                                     "mode": mode, "definition": "natural route" if mode == "natural" else config()["route_definition"],
                                     "rollout": config()["rollout_definition"]}, no_training_updates=True))
    require(len(rows) == len(SCHEDULES) * len(CHANNELS) * len(MODES), "Per-seed row count")
    return rows


def worker(payload):
    seed, prepared, output = payload; output = Path(output)
    arrays = fi.make_arrays(prepared["partitions"]["new_layouts"])
    before = {k: core.base.array_sha(arrays[k]) for k in ("packed_states", "native_rewards", "x_FI")}
    rows = _policy_rows(seed, prepared, arrays)
    after = {k: core.base.array_sha(arrays[k]) for k in ("packed_states", "native_rewards", "x_FI")}; require(before == after, "Probe arrays mutated")
    write_new(output / f"seed_{seed}_result.json", dict(seed=seed, rows=rows, array_hashes=before))
    return rows


def _stats(values):
    x = np.asarray(values, dtype=np.float64); require(x.shape == (16,) and np.isfinite(x).all(), "Expected sixteen paired values")
    mean = float(x.mean()); sd = float(x.std(ddof=1)); half = T15 * sd / math.sqrt(16)
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=sd / 4, df=15, t_critical=T15,
                ci95_lower=mean - half, ci95_upper=mean + half, positive_count=int((x > 0).sum()), zero_count=int((x == 0).sum()), negative_count=int((x < 0).sum()))


def summarize(rows):
    by = {(int(r["seed"]), r["schedule"], r["trained_channel"], r["mode"]): r for r in rows}
    require(len(by) == 512, "Expected 512 unique rows")
    summaries = {}
    for schedule in SCHEDULES:
        for mode in MODES:
            for channel in CHANNELS:
                values = {key: [by[seed, schedule, channel, mode]["delta_from_natural"][key] for seed in SEEDS] for key in METRICS}
                summaries[f"{schedule}_{channel}_{mode}"] = {key: _stats(value) for key, value in values.items()}
            if mode != "natural":
                for key in METRICS:
                    values = [by[seed, schedule, "live", mode]["delta_from_natural"][key] - by[seed, schedule, "own", mode]["delta_from_natural"][key] for seed in SEEDS]
                    summaries[f"{schedule}_live_minus_own_{mode}"] = summaries.get(f"{schedule}_live_minus_own_{mode}", {})
                    summaries[f"{schedule}_live_minus_own_{mode}"][key] = _stats(values)
    return summaries


def execute(out):
    out = Path(out).resolve(); plan, prepared = verify(out); execution = out / "execution"; require(not execution.exists(), "Never overwrite execution"); execution.mkdir()
    started = time.perf_counter(); write_new(execution / "started.json", dict(started_at=core.base.now(), plan_sha256=sha(out / "plan.json")))
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        groups = pool.map(worker, [(seed, prepared, str(execution)) for seed in SEEDS])
    rows = [row for group in groups for row in group]; require(len(rows) == budget()["rows"], "Probe row count")
    rows.sort(key=lambda r: (r["seed"], SCHEDULES.index(r["schedule"]), CHANNELS.index(r["trained_channel"]), MODES.index(r["mode"])))
    result = dict(status="completed_json_only_locality_probe", completed_at=core.base.now(), elapsed_seconds=time.perf_counter() - started,
                  plan_sha256=sha(out / "plan.json"), budget=budget(), rows=rows, summaries=summarize(rows),
                  interpretation_boundary=config()["interpretation_boundary"], no_training_updates=True, no_model_calls=True)
    write_new(execution / "results.json", result)
    write_new(execution / "status.json", dict(status="completed", completed_at=core.base.now(), results_sha256=sha(execution / "results.json")))
    write_new(out / "receipt.json", dict(status="passed", output_results_sha256=sha(execution / "results.json"), rows=len(rows), worlds=budget()["worlds"],
                                         model_forwards=budget()["model_forwards"], optimizer_updates=0, new_training=0, elapsed_seconds=result["elapsed_seconds"]))
    return dict(status=result["status"], elapsed_seconds=result["elapsed_seconds"], budget=budget())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "verify", "execute")); parser.add_argument("--out", required=True); args = parser.parse_args()
    value = prepare(args.out) if args.command == "prepare" else verify(args.out)[0] if args.command == "verify" else execute(args.out)
    print(json.dumps(value, ensure_ascii=False))
