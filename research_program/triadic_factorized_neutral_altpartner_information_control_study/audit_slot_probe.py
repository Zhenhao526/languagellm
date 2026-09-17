"""Independent replay audit for the FI single-slot probe."""
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


TRANSFORMS = tuple(f"{family}_{slot}" for family in ("slot_payload_zero", "slot_symbol_shift", "slot_symbol_xor4") for slot in range(4))
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


def route(tokens, mode, sender=None, transform=None):
    tokens = np.asarray(tokens); require(mode in ("live", "own"), "Invalid route mode"); require(tokens.ndim == 3 and tokens.shape[1:] == (3, 4) and tokens.dtype.kind in "iu" and ((tokens >= 0) & (tokens < 8)).all(), "Invalid tokens"); require((sender is None) == (transform is None), "Invalid sender/transform")
    visibility = np.ones((3, 3), dtype=np.float64) if mode == "live" else np.eye(3, dtype=np.float64); onehot = np.eye(8, dtype=np.float64)[tokens]; visible = onehot[:, None] * visibility[None, :, :, None, None]; bits = np.broadcast_to(visibility[None], (len(tokens), 3, 3)).copy()
    if sender is not None and mode == "live":
        family, slot_text = transform.rsplit("_", 1); require(family in ("slot_payload_zero", "slot_symbol_shift", "slot_symbol_xor4") and slot_text in {str(i) for i in range(4)}, "Unknown slot transform"); slot = int(slot_text); viewers = [v for v in range(3) if v != int(sender)]
        if family == "slot_payload_zero":
            for viewer in viewers: visible[:, viewer, int(sender), slot, :] = 0.0
        else:
            changed = tokens[:, int(sender), slot].copy(); changed = (changed + 1) % 8 if family == "slot_symbol_shift" else (changed ^ 4); changed_onehot = np.eye(8, dtype=np.float64)[changed]
            for viewer in viewers: visible[:, viewer, int(sender), slot, :] = changed_onehot
    return np.concatenate((visible.reshape(len(tokens), 3, 96), bits), axis=-1)


def first_messages(networks, x):
    logits = []
    for actor in range(3):
        z, _ = core.base.actor_forward(networks[3 * actor], x[:, actor]); logits.append(z.reshape(len(x), 4, 8))
    p, _ = core.base.policy_distribution(np.stack(logits, axis=1)); return core.categorical_tokens(p)


def intervention_logits(networks, x, first, sender, transform):
    routed = route(first, "live", sender, transform); second_input = np.concatenate((x, routed), axis=-1); second = []
    for actor in range(3):
        z, _ = core.base.actor_forward(networks[3 * actor + 1], second_input[:, actor]); second.append(z.reshape(len(x), 4, 8))
    p, _ = core.base.policy_distribution(np.stack(second, axis=1)); second_tokens = core.categorical_tokens(p); action_input = np.concatenate((x, routed, route(second_tokens, "live")), axis=-1)
    return np.stack([core.base.actor_forward(networks[3 * actor + 2], action_input[:, actor])[0] for actor in range(3)], axis=1)


def pair_mask(full):
    return full.reshape(len(full), 3, 8).any(axis=2)


def empty():
    return dict(engagement=0, neutral=0, physical=0, q=0, pair=0, proposal=0, proposal_den=0, third=0, third_den=0, expected=0.0, full_prob=0.0, partial_prob=0.0, pair_counts=np.zeros(3, dtype=np.int64), plan_counts=np.zeros(24, dtype=np.int64), actor_hits=np.zeros(3, dtype=np.int64))


def add(c, logits, arrays, ix):
    states = arrays["packed_states"][ix]; full = arrays["native_rewards"][ix] == 1.0; terms = runner.kernel.objective_terms(logits, arrays["native_rewards"][ix]); c["expected"] += float(terms["J"].sum()); c["full_prob"] += float(terms["full_success_probability"].sum()); c["partial_prob"] += float(terms["partial_success_probability"].sum())
    intent_p, _, proposal_p, _ = runner.kernel.factorized_distribution(logits); intent = intent_p.argmax(-1).astype(np.int16); proposal = proposal_p.argmax(-1).astype(np.int16); actions = np.where(intent == 1, proposal + 1, 0).astype(np.int16); settled = environment.settle(states, actions, "strict"); c["engagement"] += int((intent == 1).sum()); c["neutral"] += int((intent == 0).sum()); c["actor_hits"] += (intent == 1).sum(axis=0, dtype=np.int64)
    legal = np.zeros((len(ix), 3, 17), dtype=bool)
    for event in range(24):
        for actor in range(3): legal[:, actor, core.base.JOINT_ACTIONS[event, actor]] |= full[:, event]
    selected = legal[np.arange(len(ix))[:, None], np.arange(3)[None, :], actions]; c["proposal"] += int(selected[intent == 1].sum()); c["proposal_den"] += int((intent == 1).sum()); executed = settled["executed"]; ok = np.any(executed, axis=1); c["physical"] += int(ok.sum()); actual = settled["actual_pair_index"].astype(np.int64); safe = np.maximum(actual, 0); event = safe * 8 + np.maximum(settled["executed_site"], 0).astype(np.int64) * 2 + np.maximum(settled["executed_destination"], 0).astype(np.int64); c["pair"] += int((ok & pair_mask(full)[np.arange(len(ix)), safe]).sum()); c["q"] += int((ok & full[np.arange(len(ix)), event]).sum())
    if ok.any():
        c["pair_counts"] += np.bincount(actual[ok], minlength=3).astype(np.int64); c["plan_counts"] += np.bincount(event[ok], minlength=24).astype(np.int64); third = np.asarray([2, 1, 0]); c["third"] += int((intent[np.flatnonzero(ok), third[actual[ok]]] == 0).sum()); c["third_den"] += int(ok.sum())


def finish(c, n, spec, sender, transform):
    physical = c["physical"]; return dict(mode="live", sender=sender, transform=transform, value=float(c["q"] / n), q_rate=float(c["q"] / n), conditional_q_rate=float(c["q"] / physical) if physical else 0.0, target_pair_legal_rate=float(c["pair"] / physical) if physical else 0.0, proposal_legal_rate=float(c["proposal"] / c["proposal_den"]) if c["proposal_den"] else 0.0, physical_execution_rate=float(physical / n), engagement_rate=float(c["engagement"] / (3 * n)), neutral_rate=float(c["neutral"] / (3 * n)), actor_engagement_rates=[float(v / n) for v in c["actor_hits"]], third_agent_neutral_rate=float(c["third"] / c["third_den"]) if c["third_den"] else 0.0, legal_plan_selection_counts=c["plan_counts"].tolist(), legal_pair_selection_counts=c["pair_counts"].tolist(), worlds=n, physical_worlds=int(physical), proposal_legal_denominator=int(c["proposal_den"]), conditional_q_denominator_worlds=int(physical), conditional_q_numerator_worlds=int(c["q"]), target_pair_denominator_worlds=int(physical), target_pair_numerator_worlds=int(c["pair"]), exact_expected_reward_mean=float(c["expected"] / n), exact_full_success_probability_mean=float(c["full_prob"] / n), exact_partial_success_probability_mean=float(c["partial_prob"] / n), legal_plan_count=2, legal_pair_count=2, partition=spec["partition"], action_factorization="intent:neutral/engage; proposal:16-way conditional on engage")


def evaluate(networks, arrays, spec, first, sender, transform):
    n = int(spec["world_count"]); c = empty(); ids = np.arange(n, dtype=np.int64)
    for start in range(0, n, runner.CONFIG["evaluation_batch_size"]):
        ix = ids[start:min(start + runner.CONFIG["evaluation_batch_size"], n)]; add(c, intervention_logits(networks, arrays["x_FI"][ix], first[ix], sender, transform), arrays, ix)
    return finish(c, n, spec, sender, transform)


def compare(a, b, path="", tol=3e-13):
    require(set(a) == set(b), path + " keys differ")
    for key in a:
        x, y = a[key], b[key]; p = path + str(key) + "/"
        if isinstance(x, dict): compare(x, y, p, tol)
        elif isinstance(x, list):
            require(len(x) == len(y), p + " length differs")
            for i, (u, v) in enumerate(zip(x, y)):
                if isinstance(u, (int, float)) and isinstance(v, (int, float)): require(np.isclose(float(u), float(v), atol=tol, rtol=0), p + str(i) + " differs")
                else: require(u == v, p + str(i) + " differs")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)): require(np.isclose(float(x), float(y), atol=tol, rtol=0), p + " differs")
        else: require(x == y, p + " differs")


def audit_policy(payload):
    source, policy, static = payload; source = Path(source); seed = int(policy["seed"]); schedule = policy["schedule"]; channel = policy["trained_channel"]; condition = policy["condition"]; folder = source / "execution" / f"seed_{seed}_{condition}"; checkpoint = folder / "checkpoint_6000.npz"; result_path = folder / "result.json"; require(sha(checkpoint) == policy["checkpoint_sha256"], "Checkpoint hash mismatch"); require(sha(result_path) == policy["source_result_sha256"], "Source result hash mismatch"); source_natural = json.loads(result_path.read_text())["final"]["new_layouts"]; compare(source_natural, policy["natural"], f"{seed}/{schedule}/{channel}/natural/"); rows = policy["rows"]; require(len(rows) == 36, "Policy rows")
    if channel == "own":
        sample = np.arange(3 * 3 * 4, dtype=np.int8).reshape(3, 3, 4) % 8; base = route(sample, "own");
        for sender in range(3):
            for transform in TRANSFORMS: require(np.array_equal(base, route(sample, "own", sender, transform)), "Own route is not invariant")
        for row in rows:
            require(row["alias_of_natural"] is True, "Own alias flag"); compare(row["natural"], row["intervened"], "own/alias/"); require(all(float(row["natural_minus_intervened"][key]) == 0.0 for key in METRICS), "Own alias delta")
        return dict(rows=36, live_rows=0, aliases=36, evaluations=0, forwards=0, max_error=0.0, route_checks=36)
    networks = runner.load_networks(checkpoint); arrays = runner.make_arrays(static["partitions"]["new_layouts"]); first = first_messages(networks, arrays["x_FI"]); require(core.base.array_sha(first) == policy["first_messages_sha256"], "First message hash mismatch"); max_error = 0.0
    for row in rows:
        require(row["alias_of_natural"] is False, "Live alias flag"); actual = evaluate(networks, arrays, static["partitions"]["new_layouts"], first, int(row["sender"]), row["transform"]); compare(actual, row["intervened"], f"{seed}/{schedule}/{channel}/{row['sender']}/{row['transform']}/"); expected_delta = {key: float(row["natural"][key] - actual[key]) for key in METRICS}; compare(expected_delta, row["natural_minus_intervened"], "delta/"); max_error = max(max_error, *(abs(float(actual[key]) - float(row["intervened"][key])) for key in METRICS))
    n = int(static["partitions"]["new_layouts"]["world_count"]); return dict(rows=36, live_rows=36, aliases=0, evaluations=36, forwards=3 * n + 36 * 6 * n, max_error=float(max_error), route_checks=0)


def main(source, probe, output, workers=4):
    source = Path(source).resolve(); probe = Path(probe).resolve(); output = Path(output).resolve(); require(not output.exists(), "Never overwrite audit output"); output.mkdir(parents=True); saved = json.loads((probe / "execution/results.json").read_text()); static = json.loads((source / "prepared.json").read_text()); prep = json.loads((probe / "prepared.json").read_text()); require(saved["status"] == "completed_json_only_single_slot_probe" and saved["policy_blocks"] == 64 and saved["intervention_rows"] == 2304, "Invalid slot probe result"); require(saved["source_prepared_sha256"] == sha(source / "prepared.json") == prep["source_prepared_sha256"], "Source prepared hash"); require(saved["source_freeze_sha256"] == sha(source / "freeze.json") == prep["source_freeze_sha256"], "Source freeze hash"); policies = saved["policies"]; require(len(policies) == 64, "Policy blocks");
    with multiprocessing.get_context("spawn").Pool(int(workers)) as pool: checked = pool.map(audit_policy, [(str(source), p, static) for p in policies])
    rows = sum(c["rows"] for c in checked); live = sum(c["live_rows"] for c in checked); aliases = sum(c["aliases"] for c in checked); evals = sum(c["evaluations"] for c in checked); forwards = sum(c["forwards"] for c in checked); route_checks = sum(c["route_checks"] for c in checked); max_error = max(c["max_error"] for c in checked); n = int(static["partitions"]["new_layouts"]["world_count"])
    verification = dict(status="passed", source=str(source), probe=str(probe), policy_blocks=64, rows_replayed=rows, live_rows_replayed=live, alias_rows_checked=aliases, evaluations=evals, endpoint_worlds=n, worlds=rows * n, model_forward_samples=forwards, optimizer_updates=0, max_abs_error=float(max_error), checkpoint_hashes_checked=64, source_result_hashes_checked=64, first_message_hashes_checked=32, route_invariance_checks=route_checks, transforms=list(TRANSFORMS), independent_replay="Duplicate single-slot route, W2/action rollout, settlement and metric accumulation; no slot probe implementation imported")
    path = output / "verification.json"; path.write_text(json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2) + "\n"); receipt = dict(status="passed", verification_sha256=sha(path), model_forward_samples=forwards, optimizer_updates=0, max_abs_error=float(max_error)); (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n"); print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--source", required=True); parser.add_argument("--probe", required=True); parser.add_argument("--output", required=True); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args(); main(args.source, args.probe, args.output, args.workers)
