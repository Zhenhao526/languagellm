"""Independent replay audit for the FI single-slot probe.

This file deliberately duplicates the routing, rollout, settlement and metric
accumulation code instead of importing ``slot_probe``.  The probe is post-hoc,
so the audit checks both the live intervention rows and the route-invariant
FI-silent aliases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from research_program.triadic_factorized_neutral_altpartner_information_control_study import runner
from research_program.triadic_reciprocal_execution_study import environment
from research_program.triadic_message_study import runner as core


TRANSFORMS = tuple(
    f"{family}_{slot}"
    for family in ("slot_payload_zero", "slot_symbol_shift", "slot_symbol_xor4")
    for slot in range(4)
)
METRICS = (
    "q_rate",
    "conditional_q_rate",
    "target_pair_legal_rate",
    "proposal_legal_rate",
    "physical_execution_rate",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def route(tokens, mode, sender=None, transform=None):
    """Reconstruct the FI route and apply one selected-slot intervention."""
    tokens = np.asarray(tokens)
    require(mode in ("live", "own"), "Invalid route mode")
    require(
        tokens.ndim == 3
        and tokens.shape[1:] == (3, 4)
        and tokens.dtype.kind in "iu"
        and ((tokens >= 0) & (tokens < 8)).all(),
        "Invalid token array",
    )
    require((sender is None) == (transform is None), "Invalid intervention pair")
    visibility = np.ones((3, 3), dtype=np.float64) if mode == "live" else np.eye(3, dtype=np.float64)
    onehot = np.eye(8, dtype=np.float64)[tokens]
    visible = onehot[:, None, :, :, :] * visibility[None, :, :, None, None]
    bits = np.broadcast_to(visibility[None], (len(tokens), 3, 3)).copy()
    if sender is not None and mode == "live":
        family, slot_text = transform.rsplit("_", 1)
        require(family in {"slot_payload_zero", "slot_symbol_shift", "slot_symbol_xor4"}, "Unknown transform")
        slot = int(slot_text)
        require(0 <= slot < 4, "Invalid slot")
        viewers = [viewer for viewer in range(3) if viewer != int(sender)]
        if family == "slot_payload_zero":
            for viewer in viewers:
                visible[:, viewer, int(sender), slot, :] = 0.0
        else:
            changed = tokens[:, int(sender), slot].copy()
            changed = (changed + 1) % 8 if family == "slot_symbol_shift" else (changed ^ 4)
            changed_onehot = np.eye(8, dtype=np.float64)[changed]
            for viewer in viewers:
                visible[:, viewer, int(sender), slot, :] = changed_onehot
    return np.concatenate((visible.reshape(len(tokens), 3, 96), bits), axis=-1)


def first_messages(networks, x):
    logits = []
    for actor in range(3):
        z, _ = core.base.actor_forward(networks[3 * actor], x[:, actor])
        logits.append(z.reshape(len(x), 4, 8))
    probs, _ = core.base.policy_distribution(np.stack(logits, axis=1))
    return core.categorical_tokens(probs)


def action_logits_after_first(networks, x, first, sender, transform):
    routed_first = route(first, "live", sender, transform)
    second_input = np.concatenate((x, routed_first), axis=-1)
    second_logits = []
    for actor in range(3):
        z, _ = core.base.actor_forward(networks[3 * actor + 1], second_input[:, actor])
        second_logits.append(z.reshape(len(x), 4, 8))
    probs, _ = core.base.policy_distribution(np.stack(second_logits, axis=1))
    second = core.categorical_tokens(probs)
    action_input = np.concatenate((x, routed_first, route(second, "live")), axis=-1)
    action_logits = [
        core.base.actor_forward(networks[3 * actor + 2], action_input[:, actor])[0]
        for actor in range(3)
    ]
    return np.stack(action_logits, axis=1)


def pair_mask(full):
    return full.reshape(len(full), 3, 8).any(axis=2)


def empty_counter():
    return dict(
        engagement=0,
        neutral=0,
        physical=0,
        q=0,
        pair=0,
        proposal=0,
        proposal_den=0,
        third=0,
        third_den=0,
        expected=0.0,
        full_prob=0.0,
        partial_prob=0.0,
        pair_counts=np.zeros(3, dtype=np.int64),
        plan_counts=np.zeros(24, dtype=np.int64),
        actor_hits=np.zeros(3, dtype=np.int64),
    )


def accumulate(counter, logits, arrays, ix):
    states = arrays["packed_states"][ix]
    full = arrays["native_rewards"][ix] == 1.0
    terms = runner.kernel.objective_terms(logits, arrays["native_rewards"][ix])
    counter["expected"] += float(terms["J"].sum())
    counter["full_prob"] += float(terms["full_success_probability"].sum())
    counter["partial_prob"] += float(terms["partial_success_probability"].sum())
    intent_p, _, proposal_p, _ = runner.kernel.factorized_distribution(logits)
    intent = intent_p.argmax(-1).astype(np.int16)
    proposal = proposal_p.argmax(-1).astype(np.int16)
    actions = np.where(intent == 1, proposal + 1, 0).astype(np.int16)
    settled = environment.settle(states, actions, "strict")
    counter["engagement"] += int((intent == 1).sum())
    counter["neutral"] += int((intent == 0).sum())
    counter["actor_hits"] += (intent == 1).sum(axis=0, dtype=np.int64)
    legal = np.zeros((len(ix), 3, 17), dtype=bool)
    for event in range(24):
        for actor in range(3):
            legal[:, actor, core.base.JOINT_ACTIONS[event, actor]] |= full[:, event]
    selected = legal[np.arange(len(ix))[:, None], np.arange(3)[None, :], actions]
    counter["proposal"] += int(selected[intent == 1].sum())
    counter["proposal_den"] += int((intent == 1).sum())
    executed = settled["executed"]
    ok = np.any(executed, axis=1)
    counter["physical"] += int(ok.sum())
    actual = settled["actual_pair_index"].astype(np.int64)
    safe = np.maximum(actual, 0)
    event = (
        safe * 8
        + np.maximum(settled["executed_site"], 0).astype(np.int64) * 2
        + np.maximum(settled["executed_destination"], 0).astype(np.int64)
    )
    counter["pair"] += int((ok & pair_mask(full)[np.arange(len(ix)), safe]).sum())
    counter["q"] += int((ok & full[np.arange(len(ix)), event]).sum())
    if ok.any():
        counter["pair_counts"] += np.bincount(actual[ok], minlength=3).astype(np.int64)
        counter["plan_counts"] += np.bincount(event[ok], minlength=24).astype(np.int64)
        third = np.asarray([2, 1, 0])
        counter["third"] += int((intent[np.flatnonzero(ok), third[actual[ok]]] == 0).sum())
        counter["third_den"] += int(ok.sum())


def finish(counter, n, spec, sender, transform):
    physical = counter["physical"]
    engaged = counter["engagement"]
    return dict(
        mode="live",
        sender=sender,
        transform=transform,
        value=float(counter["q"] / n),
        q_rate=float(counter["q"] / n),
        conditional_q_rate=float(counter["q"] / physical) if physical else 0.0,
        target_pair_legal_rate=float(counter["pair"] / physical) if physical else 0.0,
        proposal_legal_rate=float(counter["proposal"] / counter["proposal_den"])
        if counter["proposal_den"]
        else 0.0,
        physical_execution_rate=float(physical / n),
        engagement_rate=float(engaged / (3 * n)),
        neutral_rate=float(counter["neutral"] / (3 * n)),
        actor_engagement_rates=[float(v / n) for v in counter["actor_hits"]],
        third_agent_neutral_rate=float(counter["third"] / counter["third_den"])
        if counter["third_den"]
        else 0.0,
        legal_plan_selection_counts=counter["plan_counts"].tolist(),
        legal_pair_selection_counts=counter["pair_counts"].tolist(),
        worlds=n,
        physical_worlds=int(physical),
        proposal_legal_denominator=int(counter["proposal_den"]),
        conditional_q_denominator_worlds=int(physical),
        conditional_q_numerator_worlds=int(counter["q"]),
        target_pair_denominator_worlds=int(physical),
        target_pair_numerator_worlds=int(counter["pair"]),
        exact_expected_reward_mean=float(counter["expected"] / n),
        exact_full_success_probability_mean=float(counter["full_prob"] / n),
        exact_partial_success_probability_mean=float(counter["partial_prob"] / n),
        legal_plan_count=2,
        legal_pair_count=2,
        partition=spec["partition"],
        action_factorization="intent:neutral/engage; proposal:16-way conditional on engage",
    )


def evaluate(networks, arrays, spec, first, sender, transform):
    n = int(spec["world_count"])
    counter = empty_counter()
    ids = np.arange(n, dtype=np.int64)
    for start in range(0, n, runner.CONFIG["evaluation_batch_size"]):
        ix = ids[start : min(start + runner.CONFIG["evaluation_batch_size"], n)]
        logits = action_logits_after_first(networks, arrays["x_FI"][ix], first[ix], sender, transform)
        accumulate(counter, logits, arrays, ix)
    return finish(counter, n, spec, sender, transform)


def compare(a, b, path="", tol=3e-13):
    require(set(a) == set(b), path + " keys differ")
    for key in a:
        x, y = a[key], b[key]
        current = path + str(key) + "/"
        if isinstance(x, dict):
            compare(x, y, current, tol)
        elif isinstance(x, list):
            require(len(x) == len(y), current + " length differs")
            for i, (u, v) in enumerate(zip(x, y)):
                if isinstance(u, (int, float)) and isinstance(v, (int, float)):
                    require(np.isclose(float(u), float(v), atol=tol, rtol=0), current + str(i) + " differs")
                else:
                    require(u == v, current + str(i) + " differs")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            require(np.isclose(float(x), float(y), atol=tol, rtol=0), current + " differs")
        else:
            require(x == y, current + " differs")


def policy_audit(payload):
    source, saved_policy, static = payload
    source = Path(source)
    policy = saved_policy
    seed = int(policy["seed"])
    schedule = policy["schedule"]
    channel = policy["trained_channel"]
    condition = policy["condition"]
    checkpoint = source / "execution" / f"seed_{seed}_{condition}" / "checkpoint_6000.npz"
    result_path = source / "execution" / f"seed_{seed}_{condition}" / "result.json"
    require(sha(checkpoint) == policy["checkpoint_sha256"], "Checkpoint hash mismatch")
    require(sha(result_path) == policy["source_result_sha256"], "Source result hash mismatch")
    source_natural = json.loads(result_path.read_text())["final"]["new_layouts"]
    compare(source_natural, policy["natural"], f"{seed}/{schedule}/{channel}/natural/")
    rows = policy["rows"]
    require(len(rows) == 36, "Policy row count")
    require({row["transform"] for row in rows} == set(TRANSFORMS), "Transform grid mismatch")
    if channel == "own":
        sample = np.arange(3 * 3 * 4, dtype=np.int8).reshape(3, 3, 4) % 8
        base = route(sample, "own")
        checks = 0
        for sender in range(3):
            for transform in TRANSFORMS:
                require(np.array_equal(base, route(sample, "own", sender, transform)), "Own route is not invariant")
                checks += 1
        for row in rows:
            require(row["alias_of_natural"] is True, "Own row must be an alias")
            compare(row["natural"], row["intervened"], "own/alias/")
            require(all(float(row["natural_minus_intervened"][key]) == 0.0 for key in METRICS), "Own alias delta")
        return dict(rows=36, live_rows=0, aliases=36, evaluations=0, forwards=0, max_error=0.0, route_checks=checks)
    networks = runner.load_networks(checkpoint)
    arrays = runner.make_arrays(static["partitions"]["new_layouts"])
    first = first_messages(networks, arrays["x_FI"])
    require(core.base.array_sha(first) == policy["first_messages_sha256"], "First message hash mismatch")
    checked = 0
    max_error = 0.0
    for row in rows:
        require(row["alias_of_natural"] is False, "Live row cannot be an alias")
        actual = evaluate(networks, arrays, static["partitions"]["new_layouts"], first, int(row["sender"]), row["transform"])
        compare(actual, row["intervened"], f"{seed}/{schedule}/{channel}/{row['sender']}/{row['transform']}/")
        expected_delta = {key: float(row["natural"][key] - actual[key]) for key in METRICS}
        compare(expected_delta, row["natural_minus_intervened"], "delta/")
        max_error = max(max_error, *(abs(float(actual[key]) - float(row["intervened"][key])) for key in METRICS))
        checked += 1
    n = int(static["partitions"]["new_layouts"]["world_count"])
    return dict(rows=36, live_rows=36, aliases=0, evaluations=36, forwards=3 * n + 36 * 6 * n, max_error=float(max_error), route_checks=0)


def main(source, probe, output, workers=4):
    source = Path(source).resolve()
    probe = Path(probe).resolve()
    output = Path(output).resolve()
    require(not output.exists(), "Never overwrite audit output")
    output.mkdir(parents=True)
    saved = json.loads((probe / "execution/results.json").read_text())
    static = json.loads((source / "prepared.json").read_text())
    probe_prep = json.loads((probe / "prepared.json").read_text())
    require(
        saved.get("status") == "completed_json_only_single_slot_probe"
        and saved.get("policy_blocks") == 64
        and saved.get("intervention_rows") == 2304,
        "Invalid single-slot result grid",
    )
    require(saved["source_prepared_sha256"] == sha(source / "prepared.json") == probe_prep["source_prepared_sha256"], "FI prepared hash mismatch")
    require(saved["source_freeze_sha256"] == sha(source / "freeze.json") == probe_prep["source_freeze_sha256"], "FI freeze hash mismatch")
    policies = saved["policies"]
    require(len(policies) == 64, "Expected 64 policies")
    payloads = [(str(source), policy, static) for policy in policies]
    with multiprocessing.get_context("spawn").Pool(int(workers)) as pool:
        checked = pool.map(policy_audit, payloads)
    require(len(checked) == 64, "Audit policy count")
    rows = sum(item["rows"] for item in checked)
    live_rows = sum(item["live_rows"] for item in checked)
    aliases = sum(item["aliases"] for item in checked)
    evaluations = sum(item["evaluations"] for item in checked)
    forwards = sum(item["forwards"] for item in checked)
    route_checks = sum(item["route_checks"] for item in checked)
    max_error = max(item["max_error"] for item in checked)
    n = int(static["partitions"]["new_layouts"]["world_count"])
    verification = dict(
        status="passed",
        source=str(source),
        probe=str(probe),
        policy_blocks=64,
        rows_replayed=rows,
        live_rows_replayed=live_rows,
        alias_rows_checked=aliases,
        evaluations=evaluations,
        endpoint_worlds=n,
        worlds=int(rows * n),
        model_forward_samples=forwards,
        optimizer_updates=0,
        max_abs_error=float(max_error),
        checkpoint_hashes_checked=64,
        source_result_hashes_checked=64,
        first_message_hashes_checked=32,
        route_invariance_checks=route_checks,
        transforms=list(TRANSFORMS),
        independent_replay="Duplicate route, W2/action rollout, settlement and metric accumulation; no probe implementation imported",
    )
    verification_path = output / "verification.json"
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    receipt = dict(
        status="passed",
        verification_sha256=sha(verification_path),
        model_forward_samples=forwards,
        optimizer_updates=0,
        max_abs_error=float(max_error),
    )
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    main(args.source, args.probe, args.output, args.workers)
