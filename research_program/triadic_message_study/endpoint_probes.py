"""Predeclared endpoint message probes; prepare is model-free, execute is explicit.

Does not train, initialize networks, select pairs from outcomes, or alter sources.
Actual weights are read only by execute after all sixteen original runs finish.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from research_program.triadic_message_study import private_fact_pairs as pairs

ROOT, HERE = pairs.ROOT, pairs.HERE
require, sha, json_bytes = pairs.require, pairs.sha, pairs.json_bytes
CONTRACT = {
    "schema": "triadic_message_endpoint_probes_v1", "endpoint_update": 6000,
    "conditions": ["FI_silent", "FI_live", "PI_silent", "PI_live"],
    "seeds": [47101, 47102, 47103, 47104],
    "partitions": ["train", "new_needs", "new_layouts", "new_needs_and_layouts"],
    "worlds_per_run": 143424, "natural_message_and_action_mode": "greedy_first_argmax",
    "main_deletion": "cross_channels_closed_from_window1_regenerate_all_window2_and_actions",
    "secondary_deletion": "action_only_cross_drop_keep_natural_own_window2",
    "constant_full": "from_W1_replace_other_people_with_four_category0_symbols_keep_original_visibility_regenerate_W2_actions",
    "deletion_scope": "all four conditions, all official worlds, same endpoint parameters",
    "content_scope": "PI_live and PI_silent only; all eligible pairs and both transplant directions",
    "pair_rule": "same partition, needs, owners, public material, listener private material; other private sites swapped; disjoint R1 action projections",
    "transplant": "replace other people in both received windows; preserve own W1; regenerate own W2 after donor W1; action sees donor other W2 and regenerated own W2",
    "local_readouts": ["natural both endpoints locally suitable", "natural action differs",
        "natural suitable and differs", "transplant action changes", "transplant suitable for donor world",
        "transplant suitable for recipient world", "action_probability_total_variation"],
    "identity_checks": ["listener PI input and own W1 equal", "transplanted inputs exactly equal donor inputs",
        "regenerated own W2 exactly equals donor own W2", "donor probability/action reproduced"],
    "probability_roundoff": {"rtol": 0, "atol": 2e-12,
        "scope": "saved endpoint vs recomputation only; exact routed inputs, tokens, argmax and same-batch donor probabilities are never relaxed"},
    "silent_invariants": "both deletion variants unchanged; same-observation pair has same natural action and cannot be suitable at both endpoints",
    "independent_units": "four paired training initializations; states, actors, directions and pairs are repeated measurements",
    "limits": "endpoint route/content interventions are local-policy dependence; PI training contrast is total learning effect; neither proves compositionality nor successful new team policy",
}


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(json_bytes(value))


def observe_packed(packed, full=False):
    from research_program.triadic_learning_baseline import runner as base
    return base.encode_observations([{a: pairs.env.observe(pairs.unpack(row), a,
        shared_needs=full, full_information=full) for a in pairs.env.AGENTS} for row in packed])


def distribution(logits):
    require(np.isfinite(logits).all(), "Nonfinite logits")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def action_only_drop(networks, observations, natural, runner):
    """Preserves natural own W2, so it is deliberately the weaker intervention."""
    x = np.concatenate((observations, runner.routed_window(natural["messages"][:, 0], False),
        runner.routed_window(natural["messages"][:, 1], False)), axis=-1)
    logits = np.stack([runner.base.actor_forward(networks[3*a+2], x[:, a])[0]
                       for a in range(3)], axis=1)
    return dict(messages=natural["messages"].copy(), action_logits=logits, action_inputs=x)


def constant_route(tokens, live, runner):
    route = runner.routed_window(tokens, live)
    if live:
        body = route[:, :, :96].reshape(-1, 3, 3, 4, 8)
        for viewer in range(3):
            for sender in range(3):
                if viewer != sender:
                    body[:, viewer, sender] = 0
                    body[:, viewer, sender, :, 0] = 1
    return route


def regenerate_from_first(networks, observations, natural, live, runner, constant=False):
    """Reuse natural W1; rebuild every W2 before building any action input."""
    route = constant_route if constant else lambda m, live, runner: runner.routed_window(m, False)
    first = natural["messages"][:, 0]
    route1 = route(first, live, runner)
    sender_inputs = np.concatenate((observations, route1), axis=-1)
    second = np.stack([np.argmax(distribution(runner.base.actor_forward(networks[3*a+1],
        sender_inputs[:, a])[0].reshape(-1, 4, 8)), axis=-1) for a in range(3)], axis=1).astype(np.int8)
    inputs = np.concatenate((observations, route1, route(second, live, runner)), axis=-1)
    logits = np.stack([runner.base.actor_forward(networks[3*a+2], inputs[:, a])[0]
                       for a in range(3)], axis=1)
    return dict(messages=np.stack((first, second), axis=1), action_logits=logits, action_inputs=inputs)


def transplant_listener(networks, own_x, source_messages, donor_messages, listener, live, runner):
    """Pure neural evaluation helper. Does not settle or change anyone else's action.

    Caller must independently establish source/donor listener observation equality.
    A same-batch reconstruction of the donor gives an exact arithmetic identity.
    """
    require(own_x.ndim == 2 and own_x.shape[1] == 54, "Expected listener's 54 observed features")
    require(source_messages.shape == donor_messages.shape == (len(own_x), 2, 3, 4), "Wrong message tensor")
    require(np.array_equal(source_messages[:, 0, listener], donor_messages[:, 0, listener]),
            "Own first-window messages differ despite the stipulated same observation")
    first = donor_messages[:, 0].copy()
    first[:, listener] = source_messages[:, 0, listener]
    route1 = runner.routed_window(first, live)[:, listener]
    donor_route1 = runner.routed_window(donor_messages[:, 0], live)[:, listener]
    require(np.array_equal(route1, donor_route1), "First-window transplant identity failed")
    sender_input = np.concatenate((own_x, route1), axis=-1)
    logits2 = runner.base.actor_forward(networks[3*listener+1], sender_input)[0].reshape(-1, 4, 8)
    own_second = np.argmax(distribution(logits2), axis=-1).astype(np.int8)
    require(np.array_equal(own_second, donor_messages[:, 1, listener]), "Regenerated own W2 differs from donor")
    second = donor_messages[:, 1].copy()
    second[:, listener] = own_second
    inputs = np.concatenate((own_x, route1, runner.routed_window(second, live)[:, listener]), axis=-1)
    donor_inputs = np.concatenate((own_x, donor_route1,
        runner.routed_window(donor_messages[:, 1], live)[:, listener]), axis=-1)
    require(np.array_equal(inputs, donor_inputs), "Complete donor-listener input identity failed")
    logits = runner.base.actor_forward(networks[3*listener+2], inputs)[0]
    donor_logits = runner.base.actor_forward(networks[3*listener+2], donor_inputs)[0]
    probabilities, donor_probabilities = distribution(logits), distribution(donor_logits)
    require(np.array_equal(probabilities, donor_probabilities), "Same-batch donor probability identity failed")
    return dict(action_probabilities=probabilities, action_indices=np.argmax(probabilities, axis=-1),
                own_second_messages=own_second, action_inputs=inputs,
                same_batch_donor_probability_exact=True)


def settle_choices(states, actions):
    """Independent vectorized physical matching + semantic score, no policy labels."""
    states, actions = np.asarray(states), np.asarray(actions)
    require(states.shape == (len(actions), 10) and actions.shape[1:] == (3,), "Wrong state/action shape")
    require(actions.dtype.kind in "iu" and ((actions >= 0) & (actions < 17)).all(), "Invalid action")
    reward = np.zeros(len(states), dtype=np.float64)
    executed = np.zeros((len(states), 3), dtype=bool)
    for profile in pairs.PROFILES:
        mask = (actions == profile).all(axis=1)
        if not mask.any():
            continue
        for actor, index in enumerate(profile):
            if index == 0:
                continue
            action = pairs.MENUS[actor][index]
            site = pairs.env.SITES.index(action["site"])
            destination = pairs.env.DESTINATIONS.index(action["destination"])
            need = states[mask, actor]
            material = states[mask, 3+site]
            accepted = np.array([pairs.independent_accepts(int(n), int(m), destination)
                                 for n, m in zip(need, material)], dtype=bool)
            reward[mask] += accepted / 2
            executed[mask, actor] = True
    return reward, executed


def verify_pairs(pair_dir):
    pair_dir = Path(pair_dir).resolve()
    manifest = read(pair_dir/"manifest.json")
    for name, digest in manifest["files_sha256"].items():
        require(sha(pair_dir/name) == digest, "Pair file changed")
    for relative, digest in manifest["source_sha256"].items():
        require(sha(ROOT/relative) == digest, "Pair source changed")
    return manifest, read(pair_dir/"pairs.json"), read(pair_dir/"partitions.json")


def prepare(output, source_run, pair_dir):
    from research_program.triadic_message_study import runner
    output, source_run, pair_dir = [Path(p).resolve() for p in (output, source_run, pair_dir)]
    require(not output.exists(), "Refuse to overwrite probe preparation")
    _, prepared = runner.verify(source_run)
    pm, rows, parts = verify_pairs(pair_dir)
    require(json_bytes(parts) == json_bytes(prepared["partitions"]), "Pair/training partitions differ")
    require(len(rows) == pm["total_pairs"] == 4320, "Unexpected fixed pair denominator")
    require(list(runner.SEEDS) == CONTRACT["seeds"] and list(runner.CONDITIONS) == CONTRACT["conditions"], "Study matrix changed")
    paths = [Path(__file__).resolve(), HERE/"private_fact_pairs.py", HERE/"信息与因果测量审查.md",
             HERE/"tests/test_private_fact_pairs.py", HERE/"tests/test_endpoint_probes.py"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    output.mkdir(parents=True)
    for relative in hashes:
        target = output/"source_snapshot"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
    plan = dict(contract=CONTRACT, prepared_at=runner.now(), source_run=str(source_run), pair_dir=str(pair_dir),
        training_freeze_sha256=sha(source_run/"freeze.json"), pair_manifest_sha256=sha(pair_dir/"manifest.json"),
        sources=hashes, runtime=dict(python=runner.platform.python_version(), numpy=np.__version__),
        neural_forward_calls=0, result_hashes="bound and saved after all runs complete, before loading any checkpoint")
    write_new(output/"plan.json", plan)
    write_new(output/"freeze.json", dict(plan_sha256=sha(output/"plan.json")))
    return dict(status="prepared_not_measured", output=str(output), pairs=len(rows))


def verify(output):
    from research_program.triadic_message_study import runner
    output = Path(output).resolve()
    plan = read(output/"plan.json")
    require(sha(output/"plan.json") == read(output/"freeze.json")["plan_sha256"], "Probe plan changed")
    require(plan["contract"] == CONTRACT, "Probe definitions changed")
    require(plan["runtime"] == dict(python=runner.platform.python_version(), numpy=np.__version__), "Runtime changed")
    for relative, digest in plan["sources"].items():
        require(sha(ROOT/relative) == sha(output/"source_snapshot"/relative) == digest, "Probe source changed")
    source = Path(plan["source_run"])
    require(sha(source/"freeze.json") == plan["training_freeze_sha256"], "Training freeze changed")
    _, prepared = runner.verify(source)
    require(sha(Path(plan["pair_dir"])/"manifest.json") == plan["pair_manifest_sha256"], "Pair manifest changed")
    _, rows, parts = verify_pairs(plan["pair_dir"])
    require(json_bytes(parts) == json_bytes(prepared["partitions"]), "Partition changed")
    return plan, prepared, rows


def compare_saved(actual, expected, label):
    require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(), label+" invalid")
    error = float(np.max(np.abs(actual-expected))) if actual.size else 0.0
    require(np.allclose(actual, expected, rtol=0, atol=2e-12), label+" exceeds fixed roundoff tolerance")
    return dict(exact=bool(np.array_equal(actual, expected)), max_abs_difference=error)


def run_partition(networks, saved, pair_rows, condition, runner, output):
    states, natural_messages = saved["states"], saved["messages"]
    n = len(states)
    full, live = runner.condition_settings(condition)
    buffers = {mode: dict(messages=np.empty_like(natural_messages),
        probabilities=np.empty_like(saved["action_probabilities"]), actions=np.empty_like(saved["action_indices"]))
        for mode in ("cross_channels_closed", "action_only_drop", "constant_full")}
    buffers["constant_full"]["actual_action_inputs_uint8"] = np.empty((n, 3, 252), dtype=np.uint8)
    checks = dict(natural_saved_probability_exact=True, natural_max_abs_probability_difference=0.0)
    for start in range(0, n, 1024):
        sl = slice(start, min(start+1024, n))
        x = observe_packed(states[sl], full)
        natural = runner.rollout(networks, x, live)
        natural_probs = distribution(natural["action_logits"])
        require(np.array_equal(natural["messages"], natural_messages[sl]), "Natural message replay changed")
        require(np.array_equal(natural_probs.argmax(-1), saved["action_indices"][sl]), "Natural action replay changed")
        chk = compare_saved(natural_probs, saved["action_probabilities"][sl], "Natural probability replay")
        checks["natural_saved_probability_exact"] &= chk["exact"]
        checks["natural_max_abs_probability_difference"] = max(checks["natural_max_abs_probability_difference"], chk["max_abs_difference"])
        closed = regenerate_from_first(networks, x, natural, live, runner)
        dropped = action_only_drop(networks, x, natural, runner)
        constant = regenerate_from_first(networks, x, natural, live, runner, constant=True)
        for mode, trace in (("cross_channels_closed", closed), ("action_only_drop", dropped), ("constant_full", constant)):
            p = distribution(trace["action_logits"])
            buffers[mode]["messages"][sl] = trace["messages"]
            buffers[mode]["probabilities"][sl] = p
            buffers[mode]["actions"][sl] = p.argmax(-1)
            if mode == "constant_full":
                require(np.isin(trace["action_inputs"], (0, 1)).all(), "Actual input cannot be stored losslessly as uint8")
                buffers[mode]["actual_action_inputs_uint8"][sl] = trace["action_inputs"].astype(np.uint8)
            if not live:
                require(np.array_equal(trace["messages"], natural["messages"])
                    and np.array_equal(p, natural_probs), "Silent deletion changed own computation")
    natural_r, natural_e = settle_choices(states, saved["action_indices"])
    require(np.array_equal(natural_r, saved["greedy_reward"])
            and np.array_equal(natural_e, saved["executed"]), "Independent natural physical replay differs")
    deletion = {}
    payload = {"states": states, "natural_action_indices": saved["action_indices"]}
    for mode, b in buffers.items():
        reward, executed = settle_choices(states, b["actions"])
        deletion[mode] = dict(worlds=n, reward_sum=float(reward.sum()), full_successes=int((reward == 1).sum()),
            full_success_rate=float((reward == 1).mean()), reward_mean=float(reward.mean()),
            full_success_rate_change_from_natural=float((reward == 1).mean()-(natural_r == 1).mean()),
            worlds_with_any_action_change=int((b["actions"] != saved["action_indices"]).any(1).sum()),
            agent_action_changes=(b["actions"] != saved["action_indices"]).sum(0).tolist(),
            fullsuccess_lost=int(((natural_r == 1) & (reward != 1)).sum()),
            fullsuccess_gained=int(((natural_r != 1) & (reward == 1)).sum()),
            executed_transports=int(executed.sum()))
        payload.update({mode+"_"+k: v for k, v in b.items()})
        payload[mode+"_reward"] = reward
    np.savez_compressed(output/"deletions.npz", **payload)
    content_rows, transplant_arrays = [], []
    if not full:
        for listener in range(3):
            selected = [r for r in pair_rows if r["listener_index"] == listener]
            for start in range(0, len(selected), 1024):
                group = selected[start:start+1024]
                low = np.array([r["row_index_low"] for r in group])
                high = np.array([r["row_index_high"] for r in group])
                require(np.array_equal(states[low], [r["state_low"] for r in group])
                    and np.array_equal(states[high], [r["state_high"] for r in group]), "Pair row/state mismatch")
                low_x, high_x = observe_packed(states[low])[:, listener], observe_packed(states[high])[:, listener]
                require(np.array_equal(low_x, high_x), "Listener private observation changed")
                outcomes = []
                for source_ids, donor_ids, own_x in ((low, high, low_x), (high, low, high_x)):
                    result = transplant_listener(networks, own_x, natural_messages[source_ids],
                        natural_messages[donor_ids], listener, live, runner)
                    cmp = compare_saved(result["action_probabilities"], saved["action_probabilities"][donor_ids, listener], "Saved donor probabilities")
                    require(np.array_equal(result["action_indices"], saved["action_indices"][donor_ids, listener]), "Donor argmax changed")
                    outcomes.append((result, cmp))
                for k, row in enumerate(group):
                    ai, aj = [int(saved["action_indices"][ids[k], listener]) for ids in (low, high)]
                    suitable_low = ai in row["fullsuccess_actions_low"]
                    suitable_high = aj in row["fullsuccess_actions_high"]
                    if not live:
                        require(ai == aj and not (suitable_low and suitable_high), "Silent indistinguishability invariant failed")
                    record = dict(pair_id=row["pair_id"], listener=pairs.env.AGENTS[listener],
                        natural_action_low=ai, natural_action_high=aj,
                        natural_suitable_low=suitable_low, natural_suitable_high=suitable_high,
                        natural_both_suitable=suitable_low and suitable_high, natural_action_changed=ai != aj,
                        directions=[])
                    for d, (result, cmp) in enumerate(outcomes):
                        own, donor = ((low[k], high[k]) if d == 0 else (high[k], low[k]))
                        action = int(result["action_indices"][k])
                        own_set, donor_set = ((row["fullsuccess_actions_low"], row["fullsuccess_actions_high"])
                            if d == 0 else (row["fullsuccess_actions_high"], row["fullsuccess_actions_low"]))
                        probability = result["action_probabilities"][k]
                        record["directions"].append(dict(direction="low_from_high" if d == 0 else "high_from_low",
                            transplanted_action=action, action_changed=action != int(saved["action_indices"][own, listener]),
                            suitable_for_recipient_world=action in own_set, suitable_for_donor_world=action in donor_set,
                            probability_total_variation=float(np.abs(probability-saved["action_probabilities"][own, listener]).sum()/2),
                            donor_probability_saved_exact=bool(np.array_equal(probability, saved["action_probabilities"][donor, listener])),
                            donor_probability_saved_max_abs=float(np.abs(probability-saved["action_probabilities"][donor, listener]).max()),
                            same_batch_donor_probability_exact=True))
                        transplant_arrays.append(probability)
                    content_rows.append(record)
        np.savez_compressed(output/"transplanted_listener_probabilities.npz",
                            probabilities=np.array(transplant_arrays),
                            pair_id=np.array([r["pair_id"] for r in content_rows]))
        write_new(output/"content_pairs.json", content_rows)
    content_summary = {a: dict(pairs=sum(r["listener"] == a for r in content_rows),
        both_naturally_suitable=sum(r["listener"] == a and r["natural_both_suitable"] for r in content_rows),
        natural_action_changed=sum(r["listener"] == a and r["natural_action_changed"] for r in content_rows)) for a in pairs.env.AGENTS}
    return dict(worlds=n, natural_reward_mean=float(natural_r.mean()), natural_full_success_rate=float((natural_r == 1).mean()),
        checks=checks, deletions=deletion, content=content_summary,
        computation=dict(natural_replay_network_row_forwards=n*9,
            cross_closed_network_row_forwards=n*6, action_only_network_row_forwards=n*3,
            constant_full_network_row_forwards=n*6,
            transplant_network_row_forwards=len(content_rows)*2*3,
            meaning="one row passed through one module; W1 reused for all interventions; donor equality includes an additional action forward"),
        files_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()})


def execute(output):
    from research_program.triadic_message_study import runner
    output = Path(output).resolve()
    plan, prepared, rows = verify(output)
    source_execution = Path(plan["source_run"])/"execution"
    source_result = read(source_execution/"results.json")
    require(read(source_execution/"status.json")["status"] == "completed"
            and source_result["status"] == "completed" and source_result["completed_run_count"] == 16,
            "All sixteen source runs must finish before measuring")
    actual_runs = {(r["seed"], r["condition"]): r for r in source_result["runs"]}
    require(len(source_result["runs"]) == len(actual_runs) == 16
        and set(actual_runs) == {(r["seed"], r["condition"]) for r in prepared["runs"]}, "Source run matrix differs")
    execution = output/"execution"
    require(not execution.exists(), "Never overwrite or resume a started endpoint probe")
    execution.mkdir()
    inputs = [source_execution/"results.json", source_execution/"status.json"]
    for run in prepared["runs"]:
        folder = source_execution/run["directory"]
        record = actual_runs[(run["seed"], run["condition"])]
        require(sha(folder/"checkpoint_6000.npz") == record["final_checkpoint_sha256"], "Checkpoint differs from source result")
        for part in CONTRACT["partitions"]:
            require(sha(folder/f"final_{part}.npz") == record["final"][part]["data_sha256"], "Endpoint data differ from source result")
        inputs.append(folder/"checkpoint_6000.npz")
        inputs.extend(folder/f"final_{part}.npz" for part in CONTRACT["partitions"])
    input_hashes = {str(p): sha(p) for p in inputs}
    write_new(execution/"input_freeze.json", dict(inputs_sha256=input_hashes,
        frozen_before_parameter_load_or_forward=True, probe_plan_sha256=sha(output/"plan.json")))
    started = time.perf_counter()
    write_new(execution/"started.json", dict(started_at=runner.now(), explicit_endpoint_only=True))
    result = []
    try:
        for run in prepared["runs"]:
            source_folder = source_execution/run["directory"]
            with np.load(source_folder/"checkpoint_6000.npz", allow_pickle=False) as checkpoint:
                require(int(checkpoint["update"]) == 6000, "Wrong endpoint")
                networks = [{key: checkpoint[f"agent{a}_{module}_{key}"].copy()
                    for key in ("W1", "b1", "W2", "b2", "W3", "b3")}
                    for a in range(3) for module in runner.MODULES]
            for part in CONTRACT["partitions"]:
                target = execution/run["directory"]/part
                target.mkdir(parents=True)
                with np.load(source_folder/f"final_{part}.npz", allow_pickle=False) as archive:
                    saved = {key: archive[key] for key in archive.files}
                require(len(saved["states"]) == prepared["partitions"][part]["world_count"]
                    and np.array_equal(saved["state_indices"], np.arange(len(saved["states"]))), "Wrong endpoint denominator/order")
                from itertools import product
                spec = prepared["partitions"][part]
                expected_states = np.array([tuple(n)+tuple(l)+tuple(p)
                    for n, l, p in product(spec["needs"], spec["layouts"], spec["private_sites"])], dtype=np.int16)
                require(np.array_equal(saved["states"], expected_states), "Endpoint states differ from official partition")
                row = run_partition(networks, saved, [r for r in rows if r["partition"] == part],
                    run["condition"], runner, target)
                row.update(seed=run["seed"], condition=run["condition"], partition=part)
                write_new(target/"result.json", row)
                result.append(row)
        require(all(sha(path) == digest for path, digest in input_hashes.items()), "Source inputs changed during measurement")
        verify(output)
        write_new(execution/"results.json", dict(status="completed", completed_at=runner.now(),
            rows=result, run_partitions=len(result), elapsed_seconds=time.perf_counter()-started,
            local_pair_scope_not_complete_team_policy=True, no_training=True))
        write_new(execution/"status.json", dict(status="completed"))
    except BaseException as error:
        write_new(execution/"failure.json", dict(error_type=type(error).__name__, error=str(error)))
        write_new(execution/"status.json", dict(status="failed"))
        raise
    return dict(status="completed", run_partitions=len(result), output=str(execution))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--pairs", type=Path, default=HERE/"private_fact_pairs_001")
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.source_run is not None, "Preparation needs the frozen training source run")
        result = prepare(args.out, args.source_run, args.pairs)
    elif args.command == "verify":
        verify(args.out)
        result = dict(status="verified_not_measured")
    else:
        result = execute(args.out)
    print(json.dumps(result, ensure_ascii=False))
