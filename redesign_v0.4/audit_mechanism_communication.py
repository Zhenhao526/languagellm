"""Independent saved-endpoint and fixed-probe audit of mechanism_001.

Never trains. Uses the already independent partner audit's scene/sampling helpers
for task evaluation and separate arithmetic for all protocol-drift metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from audit_partner_communication import (PAIRS, MODES, ROOT, SavedBank, array_rows,
    audit_alignment, audit_intervention, equal_number, read_json, replay_draw,
    seed_stream, sha256)


CONTEXTS = ("food_food", "mixed", "water_water")
ORIGINAL_PAIRS = ((0, 1), (2, 3))


def _probe_hash(probe):
    h = hashlib.sha256()
    for name in sorted(key for key in probe if key != "sha256"):
        value = probe[name]
        h.update(name.encode())
        if isinstance(value, torch.Tensor):
            h.update(str(value.dtype).encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(value.numpy().tobytes())
        else:
            h.update(str(value).encode())
    return h.hexdigest()


def _verify_reference_cases(reference, bank):
    """Recreate both probes from their seeds and the saved photo bank."""
    main = reference["probe"]
    n, horizon = main["cases"], main["horizon"]
    rng = np.random.default_rng(reference["seed"])
    remaining = horizon - np.arange(n, dtype=np.int64) % horizon
    local = np.empty(n, dtype=np.int64)
    for time in np.unique(remaining):
        indices = np.flatnonzero(remaining == time)
        values = (np.arange(len(indices), dtype=np.int64) % 3 + int(time)) % 3
        local[indices] = values[rng.permutation(len(values))]
    kinds = np.column_stack((local == 2, local >= 1)).astype(np.int64)
    reverse = (local == 1) & (rng.random(n) < .5)
    kinds[reverse] = kinds[reverse, ::-1]
    features, ids = bank.sample(kinds, rng)
    public = np.column_stack((np.zeros((n, 2), np.float32), remaining.astype(np.float32) / horizon))
    for key, value in {"features": features, "public": public, "image_ids": ids,
                       "kinds": kinds, "local_context": local, "remaining": remaining}.items():
        actual = value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
        assert torch.equal(actual, main[key]), f"main reference reconstruction: {key}"
    assert main["sha256"] == _probe_hash(main)
    support = reference["support_probe"]
    support_seed = int(np.random.SeedSequence([reference["seed"], 991]).generate_state(1)[0])
    assert support_seed == support["seed"]
    support_case = bank.cases(support_seed, support["cases"], "full", horizon)
    assert torch.equal(support["features"], support_case["features"][:, 0])
    for key, value in {"image_ids": support_case["image_ids"][:, 0],
                       "full_image_ids": support_case["image_ids"],
                       "full_kinds": support_case["kinds"], "kinds": support_case["kinds"][:, 0],
                       "sender_context": support_case["kinds"][:, 0].sum(axis=1),
                       "receiver_context": support_case["kinds"][:, 1].sum(axis=1),
                       "remaining": support_case["remaining"]}.items():
        assert torch.equal(support[key], torch.from_numpy(value)), f"support reference reconstruction: {key}"
    # np.float32 division is the original public-state construction.
    expected_public = np.column_stack((np.zeros((support["cases"], 2), np.float32),
                                      support_case["remaining"].astype(np.float32) / horizon))
    assert torch.equal(support["public"], torch.from_numpy(expected_public))
    assert support["sha256"] == _probe_hash(support)


@torch.no_grad()
def _probabilities(agents, probe, listeners=False):
    sent, heard = [], []
    n = probe["cases"]
    for agent in agents:
        rep = agent.observe(probe["features"], probe["public"])
        sent.append(agent.send(rep[1]).softmax(-1))
        if listeners:
            heard.append(torch.stack([agent.act(*rep, torch.full((n,), symbol, dtype=torch.int64)).softmax(-1)
                                      for symbol in range(5)], dim=1))
    return torch.stack(sent), torch.stack(heard) if listeners else None


def _validate_support(probabilities, probe, pairs, saved, minimum_count, minimum_fraction):
    """Independently count endpoint use, union only declared partners' support."""
    symbols = probabilities.argmax(-1).numpy()
    contexts = probe["receiver_context"].numpy()
    usage = {}
    for sender in range(4):
        usage[sender] = {}
        for context in ("overall",) + CONTEXTS:
            mask = np.ones(len(contexts), dtype=bool) if context == "overall" else contexts == CONTEXTS.index(context)
            counts = np.bincount(symbols[sender, mask], minlength=5)
            threshold = max(minimum_count, math.ceil(minimum_fraction * int(mask.sum())))
            supported = np.flatnonzero(counts >= threshold).tolist()
            record = saved["senders"][sender]["overall"] if context == "overall" else saved["senders"][sender]["by_receiver_own_context"][context]
            assert record["counts"] == counts.tolist()
            assert record["n"] == int(mask.sum()) and record["threshold"] == threshold
            assert record["symbols"] == supported
            usage[sender][context] = {"counts": counts, "symbols": supported, "n": int(mask.sum())}
        np.testing.assert_allclose(saved["senders"][sender]["mean_stochastic_symbol_probabilities"],
                                   probabilities[sender].mean(0).numpy(), rtol=0, atol=1e-6)
    assert saved["trained_pairs"] == [list(pair) for pair in pairs]
    result = []
    for receiver in range(4):
        partners = sorted(pair[1] if pair[0] == receiver else pair[0] for pair in pairs if receiver in pair)
        assert saved["receivers"][receiver]["declared_partners"] == partners
        by_context = {}
        for context in ("overall",) + CONTEXTS:
            counts = np.sum([usage[sender][context]["counts"] for sender in partners], axis=0)
            supported = sorted(set(itertools.chain.from_iterable(usage[sender][context]["symbols"] for sender in partners)))
            total = int(counts[supported].sum())
            weights = [float(counts[symbol]) / total if symbol in supported and total else 0. for symbol in range(5)]
            record = saved["receivers"][receiver]["overall"] if context == "overall" else saved["receivers"][receiver]["by_own_context"][context]
            assert record["symbols"] == supported and record["pooled_greedy_counts"] == counts.tolist()
            assert record["reference_partner_cases"] == sum(usage[sender][context]["n"] for sender in partners)
            np.testing.assert_allclose(record["weights_on_supported_symbols"], weights, rtol=0, atol=1e-12)
            by_context[context] = {"symbols": supported, "weights": weights}
        result.append(by_context)
    return result


def _verify_drift(agents, reference, saved, original_support, current_pairs):
    probe, support_probe = reference["probe"], reference["support_probe"]
    current_senders, current_listeners = _probabilities(agents, probe, listeners=True)
    support_probabilities, _ = _probabilities(agents, support_probe)
    current_support = _validate_support(support_probabilities, support_probe, current_pairs,
                                        saved["current_received_support"], reference["minimum_count"], reference["minimum_fraction"])
    assert saved["probe_sha256"] == probe["sha256"] and saved["support_probe_sha256"] == support_probe["sha256"]
    assert saved["initial_received_support"] == reference["initial_received_support"]
    assert saved["current_trained_pairs"] == [list(pair) for pair in current_pairs]
    kinds = probe["kinds"].numpy()
    categories = probe["local_context"].numpy()
    before_senders = reference["initial_sender_probabilities"].numpy()
    before_listeners = reference["initial_listener_action_probabilities"].numpy()
    current_senders, current_listeners = current_senders.numpy(), current_listeners.numpy()
    sender_overall, listener_mixed = [], []
    for who in range(4):
        for context in ("overall",) + CONTEXTS:
            mask = np.ones(len(categories), dtype=bool) if context == "overall" else categories == CONTEXTS.index(context)
            before, after = before_senders[who, mask], current_senders[who, mask]
            tv = float((.5 * np.abs(before - after).sum(axis=1)).mean())
            changed = float((before.argmax(-1) != after.argmax(-1)).mean())
            record = saved["senders"][who]["overall"] if context == "overall" else saved["senders"][who]["by_own_context"][context]
            assert record["n"] == int(mask.sum())
            equal_number(tv, record["mean_probability_total_variation"], "sender TV")
            equal_number(changed, record["greedy_raw_symbol_change_rate"], "sender argmax")
            if context == "overall":
                sender_overall.append(changed)
        for category, context in enumerate(CONTEXTS):
            mask = categories == category
            record = saved["listeners"][who]["by_own_context"][context]
            initial = original_support[who][context]
            current = current_support[who][context]
            effects = []
            for symbol in range(5):
                before, after = before_listeners[who, :, symbol], current_listeners[who, :, symbol]
                old_action, new_action = before.argmax(-1), after.argmax(-1)
                old_resource = kinds[np.arange(len(kinds)), old_action]
                new_resource = kinds[np.arange(len(kinds)), new_action]
                metrics = {"mean_action_probability_total_variation": float((.5 * np.abs(before[mask] - after[mask]).sum(axis=1)).mean()),
                           "greedy_option_change_rate": float((old_action[mask] != new_action[mask]).mean()),
                           "greedy_resource_change_rate": float((old_resource[mask] != new_resource[mask]).mean())}
                item = record["all_five_symbols"][symbol]
                assert item["received_symbol"] == symbol and item["n"] == int(mask.sum())
                assert item["in_initial_received_support"] == (symbol in initial["symbols"])
                assert item["in_current_received_support"] == (symbol in current["symbols"])
                for name, value in metrics.items():
                    equal_number(value, item[name], f"listener {who} {context} {symbol} {name}")
                effects.append(metrics)
            summary = record["initial_received_support_summary"]
            symbols = initial["symbols"]
            assert summary["symbols"] == symbols and summary["symbol_count"] == len(symbols)
            for name in effects[0]:
                mean = float(np.mean([effects[symbol][name] for symbol in symbols])) if symbols else None
                weighted = float(sum(initial["weights"][symbol] * effects[symbol][name] for symbol in symbols)) if symbols else None
                equal_number(mean, summary["equal_symbol_mean"][name], "fixed support mean")
                equal_number(weighted, summary["initial_usage_weighted_mean"][name], "fixed support weighted mean")
                if context == "mixed" and name == "greedy_resource_change_rate":
                    listener_mixed.append(mean)
            initial_set, current_set = set(symbols), set(current["symbols"])
            change = record["received_support_change"]
            assert change["initial_symbols"] == sorted(initial_set) and change["current_symbols"] == sorted(current_set)
            assert change["added_symbols"] == sorted(current_set - initial_set)
            assert change["removed_symbols"] == sorted(initial_set - current_set)
            equal_number(len(initial_set & current_set) / len(initial_set | current_set) if initial_set | current_set else None,
                         change["jaccard"], "support Jaccard")
    return {"sender_raw_symbol_change_rates": sender_overall,
            "mixed_listener_resource_change_on_initial_support": listener_mixed,
            "all_contexts_all_five_symbols_and_initial_support_summaries_reproduced": True}


def audit_protocol_run(run, bank, module, config):
    reference_path = run / "protocol_reference.pt"
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    origin = read_json(run / "origin.json")
    assert sha256(reference_path) == origin["protocol_reference_sha256"]
    assert reference["seed"] == origin["seed"] + config["protocol_seed_offset"]
    assert reference["probe"]["cases"] == config["protocol_probe_n"]
    assert reference["support_probe"]["cases"] == config["protocol_support_n"]
    _verify_reference_cases(reference, bank)
    condition = origin["condition"]
    old_pairs = PAIRS if condition == "rotating_to_rotating" else ORIGINAL_PAIRS
    assert reference["trained_pairs"] == [list(pair) for pair in old_pairs]
    agents = [module.ResourceAgent() for _ in range(4)]
    states = torch.load(run / "initial.pt", map_location="cpu", weights_only=True)
    for agent, state in zip(agents, states):
        agent.load_state_dict(state, strict=True)
        agent.eval()
    p, q = _probabilities(agents, reference["probe"], listeners=True)
    support_p, _ = _probabilities(agents, reference["support_probe"])
    assert torch.equal(p, reference["initial_sender_probabilities"])
    assert torch.equal(q, reference["initial_listener_action_probabilities"])
    assert torch.equal(support_p, reference["initial_support_sender_probabilities"])
    original_support = _validate_support(support_p, reference["support_probe"], old_pairs,
                                         reference["initial_received_support"], reference["minimum_count"], reference["minimum_fraction"])
    curve = read_json(run / "learning_curve.json")
    assert [entry["update"] for entry in curve] == config["checkpoints"]
    records = []
    for entry in curve:
        update = entry["update"]
        cumulative_edges = np.asarray(entry["cumulative_edge_training_updates"])
        current_pairs = tuple(pair for pair in PAIRS if cumulative_edges[pair[0], pair[1]] > 0)
        states = torch.load(run / f"checkpoint_{update:04d}.pt", map_location="cpu", weights_only=True)
        for agent, state in zip(agents, states):
            agent.load_state_dict(state, strict=True)
        drift = _verify_drift(agents, reference, entry["protocol_drift"], original_support, current_pairs)
        if update == 0:
            assert all(value == 0 for value in drift["sender_raw_symbol_change_rates"])
            assert all(value == 0 for value in drift["mixed_listener_resource_change_on_initial_support"])
        records.append({"update": update, **drift})
    assert read_json(run / "result.json")["protocol_drift"] == curve[-1]["protocol_drift"]
    return {"reference_data_and_source_policy_probabilities_independently_reconstructed": True,
            "weights_only_reference_reload_succeeded": True,
            "initial_received_support_uses_only_original_partners": True,
            "primary_listener_support_not_reselected_at_endpoint": True,
            "checkpoints_replayed": len(records), "checkpoints": records}


# Terminal trace audit below is an adapted copy of the independent partner audit.
# The original audit_partner_communication.py is left unchanged.

@torch.no_grad()
def audit(folder):
    folder = Path(folder).resolve()
    completed = read_json(folder / "completed.json")
    assert completed["status"] == "completed", "only audit completed experiments"
    config = read_json(folder / "config.json")
    for name, digest in read_json(folder / "source_hashes.json").items():
        assert sha256(folder / "source" / name) == digest, f"source hash mismatch: {name}"
    spec = importlib.util.spec_from_file_location("audit_saved_resource_agents", folder / "source/agents.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bank = SavedBank(folder)
    n, horizon = config["evaluation_n"], config["horizon"]
    total_updates = config["additional_updates"]
    output, external_by_run = [], {}
    trace_count = case_count = 0
    for seed in config["seeds"]:
        for condition in config["conditions"] + (["all_plastic"] if seed in config["confirmatory_seeds"] else []):
            run = folder / f"{condition}_s{seed}"
            saved = read_json(run / "result.json")
            origin_metadata = read_json(run / "origin.json")
            origin_checkpoint = Path(origin_metadata["origin_path"])
            assert sha256(run / "initial.pt") == origin_metadata["initial_sha256"] == origin_metadata["origin_sha256"] == sha256(origin_checkpoint)
            assert sha256(run / "checkpoint_0000.pt") == sha256(run / "initial.pt")
            origin_result = read_json(run / "origin_result.json")
            assert sha256(run / "origin_result.json") == origin_metadata["origin_result_sha256"]
            curve = read_json(run / "learning_curve.json")
            origin_curve = read_json(origin_checkpoint.parent / "learning_curve.json")
            assert curve[0]["evaluation"] == origin_curve[-1]["evaluation"]
            assert read_json(run / "zero_update_validation.json")["passed"]
            assert saved["updates"] == total_updates
            evaluation = saved["evaluation"]
            assert evaluation["master_seed"] == seed + 800000
            states = torch.load(run / f"checkpoint_{total_updates:04d}.pt", map_location="cpu", weights_only=True)
            assert len(states) == 4
            agents = [module.ResourceAgent() for _ in range(4)]
            for agent, state in zip(agents, states):
                agent.load_state_dict(state, strict=True)
                agent.eval()
            run_record = {"seed": seed, "condition": condition,
                          "initial_checkpoint_matches_origin_endpoint": True, "zero_update_checkpoint_evaluation_equals_origin": True, "pairs": []}
            hashes = {}
            for pair_index, pair in enumerate(PAIRS):
                pair_saved = next(p for p in evaluation["pairs"] if p["agents"] == list(pair))
                origin_pair = next(p for p in origin_result["evaluation"]["pairs"] if p["agents"] == list(pair))
                selected_agents = [agents[i] for i in pair]
                pair_record = {"agents": list(pair), "tasks": {}}
                for task in ("full", "curriculum"):
                    task_seed = pair_saved["evaluation_seeds"][task]
                    assert task_seed == seed_stream(evaluation["master_seed"], 100 if task == "full" else 101, pair_index)
                    case = bank.cases(task_seed, n, task, horizon)
                    rep = [a.observe(case["features"][:, i], case["public"]) for i, a in enumerate(selected_agents)]
                    send_logits = [a.send(rep[i][1]) for i, a in enumerate(selected_agents)]
                    decoded = {}
                    task_record = {}
                    for mode in MODES:
                        path = run / "traces" / f"pair_{pair[0]}{pair[1]}_{task}_{mode}_trace.jsonl"
                        trace_meta = pair_saved["traces"][f"{task}_{mode}"]
                        assert sha256(path) == trace_meta["sha256"]
                        rows = [json.loads(line) for line in path.read_text().splitlines()]
                        assert len(rows) == n == trace_meta["cases"]
                        a = array_rows(rows)
                        assert np.array_equal(a["case"], np.arange(n))
                        for key in ("kinds", "image_ids", "inventory", "remaining"):
                            assert np.array_equal(a[key], case[key]), f"scene reconstruction {run.name} {pair} {task} {mode} {key}"
                        assert np.all(bank.labels[a["image_ids"]] == a["kinds"])
                        assert np.all(bank.splits[a["image_ids"]] == "test")
                        policy_rngs = [np.random.default_rng(task_seed + 11001 + i) for i in range(2)]
                        greedy = mode != "stochastic"
                        sent = np.column_stack([replay_draw(send_logits[i], policy_rngs[i], greedy).numpy() for i in range(2)])
                        assert np.array_equal(sent, a["sent"]), f"sender replay {run.name} {pair} {task} {mode}"
                        delivered = sent.copy()
                        if mode == "blank":
                            delivered.fill(0)
                        elif mode == "shuffle":
                            intervention_rng = np.random.default_rng(task_seed + 22001)
                            for time in np.unique(case["remaining"]):
                                indices = np.flatnonzero(case["remaining"] == time)
                                for sender in range(2):
                                    delivered[indices, sender] = sent[intervention_rng.permutation(indices), sender]
                        assert np.array_equal(delivered, a["delivered"]), f"delivery replay {run.name} {pair} {task} {mode}"
                        actions = np.column_stack([replay_draw(selected_agents[i].act(*rep[i], torch.from_numpy(delivered[:, 1 - i])),
                                                              policy_rngs[i], greedy).numpy() for i in range(2)])
                        assert np.array_equal(actions, a["actions"]), f"action replay {run.name} {pair} {task} {mode}"
                        chosen = np.take_along_axis(a["kinds"], actions[..., None], axis=-1)[..., 0]
                        success = chosen[:, 0] != chosen[:, 1]
                        assert np.array_equal(chosen, a["selected_kinds"])
                        assert np.array_equal(success, a["success"])
                        assert np.array_equal(success.astype(float), a["reward"])
                        stats = pair_saved["tasks"][task][mode]
                        assert stats["external_cases_sha256"] == case["hash"]
                        assert origin_pair["tasks"][task][mode]["external_cases_sha256"] == case["hash"]
                        equal_number(float(success.mean()), stats["balanced_gathering"], "success", 0)
                        equal_number(float(success.mean()), stats["mean_reward_per_step"], "reward", 0)
                        table = np.zeros((2, 3, 5), dtype=np.int64)
                        for sender in range(2):
                            np.add.at(table[sender], (a["kinds"][:, sender].sum(axis=1), sent[:, sender]), 1)
                        assert table.tolist() == stats["symbols_by_local_resource_set"]
                        same = a["kinds"][..., 0] == a["kinds"][..., 1]
                        directions = []
                        for sender in range(2):
                            mask = same[:, sender] & ~same[:, 1 - sender]
                            direction = next(d for d in stats["by_direction"] if d["restricted_sender"] == sender)
                            assert direction["population_sender"] == pair[sender]
                            assert direction["population_receiver"] == pair[1 - sender]
                            assert direction["n"] == int(mask.sum())
                            equal_number(float(success[mask].mean()), direction["success"], "direction", 0)
                            directions.append({"sender": pair[sender], "receiver": pair[1 - sender],
                                               "n": int(mask.sum()), "success": float(success[mask].mean())})
                        task_record[mode] = {"success": float(success.mean()), "directions": directions,
                                             "checkpoint_and_rng_replay_match": True,
                                             "scores_independently_recomputed": True}
                        decoded[mode] = a
                        hashes[(pair, task, mode)] = case["hash"]
                        trace_count += 1
                        case_count += n
                    for mode in ("shuffle", "blank"):
                        assert np.array_equal(decoded["normal"]["sent"], decoded[mode]["sent"])
                    task_record["greedy_conditions_have_identical_original_sent_messages"] = True
                    pair_record["tasks"][task] = task_record
                intervention_seed = pair_saved["intervention_seed"]
                assert intervention_seed == seed_stream(evaluation["master_seed"], 200, pair_index)
                pair_record["same_observation_interventions"] = audit_intervention(
                    selected_agents, pair, bank, intervention_seed, pair_saved["intervention"], config["intervention_n"], horizon)
                run_record["pairs"].append(pair_record)
            # Independently verify equal-pair aggregates; pair counts are not replicates.
            for group in ("all", "original", "cross"):
                rows = [p for p in run_record["pairs"] if group == "all" or
                        ((tuple(p["agents"]) in ((0, 1), (2, 3))) == (group == "original"))]
                assert len(rows) == evaluation["aggregates"][group]["pair_count"]
                for task in ("full", "curriculum"):
                    for mode in MODES:
                        mean = float(np.mean([p["tasks"][task][mode]["success"] for p in rows]))
                        equal_number(mean, evaluation["aggregates"][group]["tasks"][task][mode]["balanced_gathering"], "aggregate", 0)
            run_record["raw_sender_alignment"] = audit_alignment(agents, bank, evaluation["sender_alignment"])
            run_record["fixed_probe_protocol_drift"] = audit_protocol_run(run, bank, module, config)
            output.append(run_record)
            external_by_run[(seed, condition)] = hashes
            print(json.dumps({"audited": run.name, "pairs": 6, "trace_files": 48}), flush=True)
    for seed in config["seeds"]:
        conditions = config["conditions"] + (["all_plastic"] if seed in config["confirmatory_seeds"] else [])
        first = external_by_run[(seed, "both_learn")]
        assert all(external_by_run[(seed, condition)] == first for condition in conditions)
        assert len({sha256(folder / f"{condition}_s{seed}/initial.pt") for condition in conditions}) == 1
        assert len({sha256(folder / f"{condition}_s{seed}/protocol_reference.pt") for condition in conditions}) == 1
    assert len(output) == completed["runs"] == 47
    report = {"status": "passed", "runs": len(output), "trace_files": trace_count, "trace_cases": case_count,
              "source_snapshot_hashes_verified": True,
              "all_conditions_and_origins_share_endpoint_evaluation_cases": True,
              "protocol_checkpoints_replayed": sum(r["fixed_probe_protocol_drift"]["checkpoints_replayed"] for r in output),
              "scope": "Independent saved-policy replay, scene generation, rewards, directional scores, shuffle interventions, raw alignment, and fixed probes. No training or adaptation. The three reused all-plastic references retain their earlier independent audit.",
              "runs_audited": output}
    (folder / "独立机制通信核查.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (folder / "独立机制通信核查.md").write_text(
        f"# 独立机制通信核查\n\n核查通过：{len(output)} 个新运行、{trace_count} 份终点轨迹、{case_count:,} 个案例，"
        f"以及 {report['protocol_checkpoints_replayed']} 个固定探针检查点。没有训练或修改参数。\n\n"
        "从保存参数独立重放正常、打乱、恒符号与按策略采样的发送和行动；独立重建外部照片案例、消息投递、资源回报、两个受限方向、固定观察逐符号干预及发送一致性。"
        "探针与起点常用符号支持范围均从保存数据和独立实现重建。全部吻合。\n\n"
        "10个群体是重复单位；配对和案例不是独立群体。3个既有全可变参照沿用上一批独立核查。\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=ROOT / "results/mechanism_001")
    args = parser.parse_args()
    torch.set_num_threads(4)
    summary = audit(args.directory)
    print(json.dumps({k: summary[k] for k in ("status", "runs", "trace_files", "trace_cases", "protocol_checkpoints_replayed")}))
