"""Post-hoc local message-code and field-visibility probe for FI-001.

The FI-001 checkpoints are kept fixed.  For each live policy, the first-window
packet of one selected sender is changed only for the two other viewers.  The
second window and the action are then greedily replayed.  The probe contains
six fixed symbol bijections, two packet-position permutations, and two field
masks.  Silent/own policies are analytic no-op controls because their
cross-agent route is already absent; their rows alias the audited natural
baseline and are never used to claim a neural effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from research_program.triadic_factorized_neutral_altpartner_information_control_study import design, metrics, runner
from research_program.triadic_reciprocal_execution_study import environment
from research_program.triadic_message_study import runner as core


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRANSFORM_NAMES = tuple([f"perm_{i:02d}" for i in range(6)] + ["position_rotation", "position_reverse", "cross_payload_zero", "cross_visibility_zero"])
PERMUTATIONS = {
    "perm_00": (6, 7, 1, 0, 4, 5, 3, 2),
    "perm_01": (4, 1, 3, 2, 6, 5, 7, 0),
    "perm_02": (4, 3, 2, 6, 1, 5, 7, 0),
    "perm_03": (2, 0, 4, 7, 3, 5, 1, 6),
    "perm_04": (7, 2, 6, 4, 0, 5, 3, 1),
    "perm_05": (2, 3, 1, 7, 0, 6, 5, 4),
}
POSITION_ORDERS = {"position_rotation": (1, 2, 3, 0), "position_reverse": (3, 2, 1, 0)}
METRICS = ("q_rate", "conditional_q_rate", "target_pair_legal_rate", "proposal_legal_rate", "physical_execution_rate")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), "Refuse to overwrite " + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


def transformation_manifest():
    return dict(
        schema="triadic_fi_recode_mask_manifest_v1",
        transforms=list(TRANSFORM_NAMES),
        symbol_permutations={key: list(value) for key, value in PERMUTATIONS.items()},
        position_orders={key: list(value) for key, value in POSITION_ORDERS.items()},
        definitions={
            **{key: "replace selected sender packet [x0,x1,x2,x3] by pi[x0],pi[x1],pi[x2],pi[x3] for cross-viewers; self-view remains natural" for key in PERMUTATIONS},
            "position_rotation": "replace selected sender packet [x0,x1,x2,x3] by [x1,x2,x3,x0] for cross-viewers; self-view remains natural",
            "position_reverse": "replace selected sender packet [x0,x1,x2,x3] by [x3,x2,x1,x0] for cross-viewers; self-view remains natural",
            "cross_payload_zero": "set selected sender's four one-hot payloads to zero for cross-viewers while retaining cross visibility bits",
            "cross_visibility_zero": "retain selected sender payload for cross-viewers while clearing its three cross visibility bits",
        },
        intervention_scope="first-window selected-sender packet only; recompute second-window messages and action greedily",
        baseline_scope="same final checkpoint and same new-layout worlds; natural baseline is the audited FI-001 final result",
        controls="own/silent rows are route-invariant no-op aliases because cross-agent payloads and visibility bits are absent by construction",
        no_training=True,
        no_model_calls=True,
    )


def route(tokens, mode, sender=None, transform=None):
    """Build a viewer-specific [payload, visibility] route for one window."""
    tokens = np.asarray(tokens)
    require(mode in ("live", "own"), "Unknown route mode")
    require(tokens.ndim == 3 and tokens.shape[1:] == (3, 4) and tokens.dtype.kind in "iu"
            and ((tokens >= 0) & (tokens < 8)).all(), "Invalid token array")
    require((sender is None) == (transform is None), "sender/transform must be paired")
    visibility = np.ones((3, 3), dtype=np.float64) if mode == "live" else np.eye(3, dtype=np.float64)
    onehot = np.eye(8, dtype=np.float64)[tokens]
    visible = onehot[:, None, :, :, :] * visibility[None, :, :, None, None]
    bits = np.broadcast_to(visibility[None], (len(tokens), 3, 3)).copy()
    if sender is not None:
        require(int(sender) in (0, 1, 2) and transform in TRANSFORM_NAMES, "Invalid intervention")
        viewers = [v for v in range(3) if v != int(sender)]
        # In own/silent routing the two cross-viewer blocks are zero by
        # construction; a cross-viewer intervention therefore has no route
        # to overwrite.  Keep this branch explicit so the control is an
        # analytic no-op rather than an accidental symbol substitution.
        if mode != "live":
            return np.concatenate((visible.reshape(len(tokens), 3, 96), bits), axis=-1)
        if transform in PERMUTATIONS:
            changed = np.asarray(PERMUTATIONS[transform], dtype=np.int8)[tokens[:, int(sender), :]]
            changed_onehot = np.eye(8, dtype=np.float64)[changed]
            for viewer in viewers:
                visible[:, viewer, int(sender), :, :] = changed_onehot
        elif transform in POSITION_ORDERS:
            changed = tokens[:, int(sender), :][:, np.asarray(POSITION_ORDERS[transform], dtype=np.int8)]
            changed_onehot = np.eye(8, dtype=np.float64)[changed]
            for viewer in viewers:
                visible[:, viewer, int(sender), :, :] = changed_onehot
        elif transform == "cross_payload_zero":
            for viewer in viewers:
                visible[:, viewer, int(sender), :, :] = 0.0
        elif transform == "cross_visibility_zero":
            for viewer in viewers:
                bits[:, viewer, int(sender)] = 0.0
        else:
            raise ValueError("Unknown transformation")
    return np.concatenate((visible.reshape(len(tokens), 3, 96), bits), axis=-1)


def first_messages(networks, observations):
    x = np.asarray(observations, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1:] == (3, 54), "Invalid FI observations")
    logits = []
    for actor in range(3):
        z, _ = core.base.actor_forward(networks[3 * actor], x[:, actor])
        logits.append(z.reshape(len(x), 4, 8))
    probabilities, _ = core.base.policy_distribution(np.stack(logits, axis=1))
    return core.categorical_tokens(probabilities)


def action_logits_after_first(networks, x, first, mode, sender=None, transform=None):
    routed_first = route(first, mode, sender, transform)
    second_input = np.concatenate((x, routed_first), axis=-1)
    second_logits = []
    for actor in range(3):
        z, _ = core.base.actor_forward(networks[3 * actor + 1], second_input[:, actor])
        second_logits.append(z.reshape(len(x), 4, 8))
    second_probabilities, _ = core.base.policy_distribution(np.stack(second_logits, axis=1))
    second = core.categorical_tokens(second_probabilities)
    routed_second = route(second, mode)
    action_input = np.concatenate((x, routed_first, routed_second), axis=-1)
    action_logits = []
    for actor in range(3):
        z, _ = core.base.actor_forward(networks[3 * actor + 2], action_input[:, actor])
        action_logits.append(z)
    return np.stack(action_logits, axis=1)


def _event_pair_mask(full):
    return full.reshape(len(full), 3, 8).any(axis=2)


def _greedy(trace_logits):
    intent_p, _, proposal_p, _ = runner.kernel.factorized_distribution(trace_logits)
    intent = intent_p.argmax(axis=-1).astype(np.int16)
    proposal = proposal_p.argmax(axis=-1).astype(np.int16)
    return intent, proposal, np.where(intent == 1, proposal + 1, 0).astype(np.int16)


def _empty_counter():
    return dict(engagement=0, neutral=0, physical=0, q=0, pair=0, proposal=0, proposal_den=0, third=0, third_den=0,
                expected=0.0, full_prob=0.0, partial_prob=0.0, pair_counts=np.zeros(3, dtype=np.int64),
                plan_counts=np.zeros(24, dtype=np.int64), actor_hits=np.zeros(3, dtype=np.int64))


def _accumulate(counter, action_logits, arrays, ix):
    states = arrays["packed_states"][ix]; full = arrays["native_rewards"][ix] == 1.0
    terms = runner.kernel.objective_terms(action_logits, arrays["native_rewards"][ix])
    counter["expected"] += float(terms["J"].sum()); counter["full_prob"] += float(terms["full_success_probability"].sum()); counter["partial_prob"] += float(terms["partial_success_probability"].sum())
    intent, proposal, actions = _greedy(action_logits)
    settled = environment.settle(states, actions, "strict")
    pair_targets = _event_pair_mask(full)
    counter["engagement"] += int((intent == 1).sum()); counter["neutral"] += int((intent == 0).sum()); counter["actor_hits"] += (intent == 1).sum(axis=0, dtype=np.int64)
    legal_actions = np.zeros((len(ix), 3, 17), dtype=bool)
    for event in range(24):
        for actor in range(3):
            legal_actions[:, actor, core.base.JOINT_ACTIONS[event, actor]] |= full[:, event]
    selected = legal_actions[np.arange(len(ix))[:, None], np.arange(3)[None, :], actions]
    counter["proposal"] += int(selected[intent == 1].sum()); counter["proposal_den"] += int((intent == 1).sum())
    executed = settled["executed"]; physical_ok = np.any(executed, axis=1); counter["physical"] += int(physical_ok.sum())
    actual_pair = settled["actual_pair_index"].astype(np.int64); safe_pair = np.maximum(actual_pair, 0)
    event = safe_pair * 8 + np.maximum(settled["executed_site"], 0).astype(np.int64) * 2 + np.maximum(settled["executed_destination"], 0).astype(np.int64)
    pair_legal = physical_ok & pair_targets[np.arange(len(ix)), safe_pair]; q_hits = physical_ok & full[np.arange(len(ix)), event]
    counter["pair"] += int(pair_legal.sum()); counter["q"] += int(q_hits.sum())
    if physical_ok.any():
        counter["pair_counts"] += np.bincount(actual_pair[physical_ok], minlength=3).astype(np.int64); counter["plan_counts"] += np.bincount(event[physical_ok], minlength=24).astype(np.int64)
        third = np.asarray([2, 1, 0], dtype=np.int64)
        counter["third"] += int((intent[np.flatnonzero(physical_ok), third[actual_pair[physical_ok]]] == 0).sum()); counter["third_den"] += int(physical_ok.sum())


def finalize(counter, n, spec, mode, sender, transform):
    physical = counter["physical"]; engaged = counter["engagement"]
    return dict(mode=mode, sender=sender, transform=transform, value=float(counter["q"] / n), q_rate=float(counter["q"] / n),
                conditional_q_rate=float(counter["q"] / physical) if physical else 0.0,
                target_pair_legal_rate=float(counter["pair"] / physical) if physical else 0.0,
                proposal_legal_rate=float(counter["proposal"] / counter["proposal_den"]) if counter["proposal_den"] else 0.0,
                physical_execution_rate=float(physical / n), engagement_rate=float(engaged / (3 * n)), neutral_rate=float(counter["neutral"] / (3 * n)),
                actor_engagement_rates=[float(v / n) for v in counter["actor_hits"]],
                third_agent_neutral_rate=float(counter["third"] / counter["third_den"]) if counter["third_den"] else 0.0,
                legal_plan_selection_counts=counter["plan_counts"].tolist(), legal_pair_selection_counts=counter["pair_counts"].tolist(),
                worlds=n, physical_worlds=int(physical), proposal_legal_denominator=int(counter["proposal_den"]),
                conditional_q_denominator_worlds=int(physical), conditional_q_numerator_worlds=int(counter["q"]),
                target_pair_denominator_worlds=int(physical), target_pair_numerator_worlds=int(counter["pair"]),
                exact_expected_reward_mean=float(counter["expected"] / n), exact_full_success_probability_mean=float(counter["full_prob"] / n),
                exact_partial_success_probability_mean=float(counter["partial_prob"] / n), legal_plan_count=2, legal_pair_count=2,
                partition=spec["partition"], action_factorization="intent:neutral/engage; proposal:16-way conditional on engage")


def evaluate_transform(networks, arrays, spec, first, mode, sender=None, transform=None):
    n = int(spec["world_count"]); counter = _empty_counter(); ids = np.arange(n, dtype=np.int64)
    for start in range(0, n, runner.CONFIG["evaluation_batch_size"]):
        stop = min(start + runner.CONFIG["evaluation_batch_size"], n); ix = ids[start:stop]
        logits = action_logits_after_first(networks, arrays["x_FI"][ix], first[ix], mode, sender, transform)
        _accumulate(counter, logits, arrays, ix)
    return finalize(counter, n, spec, mode, sender, transform)


def delta(natural, intervened):
    return {key: float(natural[key] - intervened[key]) for key in METRICS}


def run_policy(source, prepared, arrays, seed, schedule, trained_channel):
    condition = f"{schedule}_altpair_factorized_strict_FI_{'live' if trained_channel == 'live' else 'silent'}"
    run_path = source / "execution" / f"seed_{seed}_{condition}"
    checkpoint = run_path / "checkpoint_6000.npz"; result_path = run_path / "result.json"
    require(checkpoint.is_file() and result_path.is_file(), "Missing FI checkpoint or result")
    result = json.loads(result_path.read_text(encoding="utf8")); natural = result["final"]["new_layouts"]
    mode = "live" if trained_channel == "live" else "own"
    networks = runner.load_networks(checkpoint)
    first = None if mode == "own" else first_messages(networks, arrays["x_FI"])
    rows = []
    for sender in range(3):
        for transform in TRANSFORM_NAMES:
            if mode == "own":
                intervened = dict(natural); alias = True; forwards = 0
            else:
                intervened = evaluate_transform(networks, arrays, prepared["partitions"]["new_layouts"], first, mode, sender, transform)
                alias = False; forwards = 6 * int(prepared["partitions"]["new_layouts"]["world_count"])
            rows.append(dict(seed=seed, schedule=schedule, trained_channel=trained_channel, condition=condition, sender=sender,
                             transform=transform, checkpoint_sha256=sha(checkpoint), source_result_sha256=sha(result_path),
                             natural=natural, intervened=intervened, natural_minus_intervened=delta(natural, intervened),
                             alias_of_natural=alias, neural_forward_samples=forwards, optimizer_updates=0,
                             intervention="selected sender first-window packet transformed for cross-viewers only; W2 and action recomputed greedily"))
    return dict(seed=seed, schedule=schedule, trained_channel=trained_channel, condition=condition, natural_mode=mode,
                checkpoint_sha256=sha(checkpoint), source_result_sha256=sha(result_path), natural=natural,
                first_messages_sha256=None if first is None else core.base.array_sha(first), rows=rows)


def _worker(payload):
    source, output, probe_prepared, seed, schedule, trained_channel = payload
    source = Path(source); output = Path(output)
    # The probe preparation only records hashes.  Load the immutable FI-001
    # static specification separately so that the probe cannot silently alter
    # the world generator.
    source_static = json.loads((source / "prepared.json").read_text(encoding="utf8"))
    arrays = runner.make_arrays(source_static["partitions"]["new_layouts"])
    return run_policy(source, source_static, arrays, int(seed), schedule, trained_channel)


def _stats(values):
    x = np.asarray(values, dtype=np.float64); require(x.shape == (16,), "Sixteen seed values required")
    return metrics.stats(x)


def summarize(policies):
    by = {(int(p["seed"]), p["schedule"], p["trained_channel"]): p for p in policies}; require(len(by) == 64, "Expected 64 policy blocks")
    summaries = {}
    for channel in ("live", "own"):
        for schedule in design.SCHEDULES:
            selected = [by[seed, schedule, channel] for seed in design.SEEDS]
            for transform in TRANSFORM_NAMES:
                label = f"{schedule}_{channel}_{transform}"
                values = []
                for policy in selected:
                    rows = [r for r in policy["rows"] if r["transform"] == transform]
                    values.append(float(np.mean([r["natural_minus_intervened"]["target_pair_legal_rate"] for r in rows])))
                summaries[label] = dict(transform=transform, schedule=schedule, trained_channel=channel,
                                        target_pair_legal_rate=_stats(values), by_seed=values,
                                        q_rate=_stats([float(np.mean([r["natural_minus_intervened"]["q_rate"] for r in policy["rows"] if r["transform"] == transform])) for policy in selected]),
                                        physical_execution_rate=_stats([float(np.mean([r["natural_minus_intervened"]["physical_execution_rate"] for r in policy["rows"] if r["transform"] == transform])) for policy in selected]))
            for family, transforms in (("symbol_ensemble", [f"perm_{i:02d}" for i in range(6)]), ("position", ["position_rotation", "position_reverse"]), ("field", ["cross_payload_zero", "cross_visibility_zero"])):
                label = f"{schedule}_{channel}_{family}"
                values = []
                q_values = []; physical_values = []
                for policy in selected:
                    chosen = [r for r in policy["rows"] if r["transform"] in transforms]
                    values.append(float(np.mean([r["natural_minus_intervened"]["target_pair_legal_rate"] for r in chosen])))
                    q_values.append(float(np.mean([r["natural_minus_intervened"]["q_rate"] for r in chosen])))
                    physical_values.append(float(np.mean([r["natural_minus_intervened"]["physical_execution_rate"] for r in chosen])))
                summaries[label] = dict(family=family, transforms=transforms, schedule=schedule, trained_channel=channel,
                                        target_pair_legal_rate=_stats(values), q_rate=_stats(q_values), physical_execution_rate=_stats(physical_values), by_seed=values)
    return summaries


def prepare(out, source):
    out = Path(out).resolve(); source = Path(source).resolve(); require(not out.exists(), "Never overwrite preparation")
    require((source / "prepared.json").is_file() and (source / "freeze.json").is_file(), "FI source is incomplete")
    manifest = transformation_manifest(); out.mkdir(parents=True)
    write_new(out / "transformation_manifest.json", manifest)
    prepared = dict(schema="triadic_fi_recode_mask_probe_v1", source=str(source), source_prepared_sha256=sha(source / "prepared.json"),
                    source_freeze_sha256=sha(source / "freeze.json"), source_execution_status_sha256=sha(source / "execution/status.json"),
                    manifest_sha256=sha(out / "transformation_manifest.json"), transforms=list(TRANSFORM_NAMES),
                    policy_grid="16 seeds × 2 schedules × 2 trained channels × 3 selected senders × 10 transforms",
                    endpoint_worlds=56160, no_training=True, no_optimizer_updates=True, no_model_calls=True)
    write_new(out / "prepared.json", prepared)
    plan = dict(status="prepared_without_policy_reads", created_at=core.base.now(), runtime=dict(python=platform.python_version(), numpy=np.__version__),
                source_sha256={"prepared.json": prepared["source_prepared_sha256"], "freeze.json": prepared["source_freeze_sha256"], "execution/status.json": prepared["source_execution_status_sha256"]},
                manifest_sha256=prepared["manifest_sha256"], prepared_sha256=sha(out / "prepared.json"), transforms=list(TRANSFORM_NAMES),
                intervention_scope=manifest["intervention_scope"], no_training=True, no_optimizer_updates=True)
    write_new(out / "plan.json", plan)
    write_new(out / "freeze.json", dict(plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"), manifest_sha256=sha(out / "transformation_manifest.json")))
    write_new(out / "receipt.json", dict(status="prepared_static_only", plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"), manifest_sha256=sha(out / "transformation_manifest.json"), neural_forward_samples=0, optimizer_updates=0))
    return plan


def verify_preparation(out):
    out = Path(out).resolve(); plan = json.loads((out / "plan.json").read_text()); prepared = json.loads((out / "prepared.json").read_text()); freeze = json.loads((out / "freeze.json").read_text())
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Probe plan hash mismatch")
    require(sha(out / "prepared.json") == freeze["prepared_sha256"], "Probe prepared hash mismatch")
    require(sha(out / "transformation_manifest.json") == freeze["manifest_sha256"] == prepared["manifest_sha256"], "Manifest hash mismatch")
    require(plan["prepared_sha256"] == sha(out / "prepared.json"), "Prepared chain mismatch")
    return plan, prepared


def execute(out, source, workers=4):
    out = Path(out).resolve(); source = Path(source).resolve(); _plan, prepared = verify_preparation(out)
    require(prepared["source"] == str(source), "Source path differs from prepared probe")
    execution = out / "execution"; require(not execution.exists(), "Never overwrite execution"); execution.mkdir()
    tasks = [(str(source), str(execution), str(out / "prepared.json"), seed, schedule, channel)
             for seed in design.SEEDS for schedule in design.SCHEDULES for channel in ("live", "own")]
    started = time.perf_counter();
    with multiprocessing.get_context("spawn").Pool(int(workers)) as pool:
        policies = pool.map(_worker, tasks)
    require(len(policies) == 64 and all(len(p["rows"]) == 30 for p in policies), "Complete recode grid required")
    rows = [r for p in policies for r in p["rows"]]
    actual_forward = int(sum(r["neural_forward_samples"] for r in rows) + sum(3 * prepared["endpoint_worlds"] for p in policies if p["natural_mode"] == "live"))
    result = dict(status="completed_json_only_recode_mask_probe", completed_at=core.base.now(), elapsed_seconds=time.perf_counter() - started,
                  source=str(source), source_prepared_sha256=prepared["source_prepared_sha256"], source_freeze_sha256=prepared["source_freeze_sha256"],
                  manifest_sha256=prepared["manifest_sha256"], transforms=list(TRANSFORM_NAMES), policy_blocks=len(policies), intervention_rows=len(rows),
                  live_intervention_rows=sum(not r["alias_of_natural"] for r in rows), alias_rows=sum(r["alias_of_natural"] for r in rows),
                  endpoint_worlds=prepared["endpoint_worlds"], model_forward_samples=actual_forward, optimizer_updates=0, no_new_training=True,
                  summaries=summarize(policies), policies=policies,
                  interpretation_boundary="This is a post-hoc route/code sensitivity probe for a task-coordination policy. It does not demonstrate words, compositionality, or human language origin.")
    write_new(execution / "results.json", result)
    status = dict(status="completed", results_sha256=sha(execution / "results.json"), completed_at=core.base.now(), elapsed_seconds=result["elapsed_seconds"])
    write_new(execution / "status.json", status)
    receipt = dict(status="passed", policy_blocks=64, intervention_rows=len(rows), live_intervention_rows=result["live_intervention_rows"], alias_rows=result["alias_rows"], endpoint_worlds=prepared["endpoint_worlds"], model_forward_samples=actual_forward, optimizer_updates=0, results_sha256=sha(execution / "results.json"))
    write_new(execution / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "verify", "execute")); parser.add_argument("--source", required=True); parser.add_argument("--out", required=True); parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.out, args.source)
    elif args.command == "verify":
        verify_preparation(args.out)
        value = {"status": "verified", "plan_sha256": sha(Path(args.out).resolve() / "plan.json")}
    else:
        value = execute(args.out, args.source, args.workers)
    print(json.dumps(value, ensure_ascii=False))
