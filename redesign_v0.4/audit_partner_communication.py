"""Independently replay completed partner endpoints; never train or adapt.

Reads the experiment's saved source, features, scaling, seeds, and checkpoints.
Scene scoring, RNG reconstruction, and message interventions are implemented here
rather than calling the experiment's evaluation functions. The source agent class
is needed only to execute the saved neural policy. All four sampling modes are
replayed, including the original categorical sampler's float32 arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
PAIRS = tuple(itertools.combinations(range(4), 2))
MODES = ("normal", "shuffle", "blank", "stochastic")
FULL_SCENES = np.array([s for s in itertools.product((0, 1), repeat=4)
                        if len(set(s)) == 2], dtype=np.int64).reshape(14, 2, 2)
_same = FULL_SCENES[..., 0] == FULL_SCENES[..., 1]
COURSE_SCENES = FULL_SCENES[_same.sum(axis=1) == 1]


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_stream(master, stream, index=0):
    return int(np.random.SeedSequence([int(master), int(stream), int(index)]).generate_state(1)[0])


def equal_number(actual, expected, name, atol=1e-6):
    if actual is None or expected is None:
        assert actual is expected, f"{name}: {actual} != {expected}"
    else:
        assert np.isfinite(actual) and np.isfinite(expected), f"{name}: nonfinite"
        assert abs(actual - expected) <= atol, f"{name}: {actual} != {expected}"


def replay_draw(logits, rng, greedy):
    # Match the stored policy sampler exactly, not torch.multinomial or a new RNG.
    probabilities = F.log_softmax(logits, -1).exp()
    if greedy:
        return probabilities.argmax(-1)
    uniforms = torch.from_numpy(rng.random((len(probabilities), 1)).astype("float32"))
    return (probabilities.cumsum(-1) < uniforms).sum(-1).clamp(max=probabilities.shape[-1] - 1)


class SavedBank:
    def __init__(self, folder):
        self.manifest = read_json(folder / "data_manifest.json")["images"]
        raw = np.load(folder / "data_features.npz")["features"].astype(np.float32)
        scaling = np.load(folder / "feature_scaling.npz")
        self.features = torch.from_numpy((raw - scaling["center"]) / max(float(scaling["scale"]), 1e-6))
        assert all(e["category"] in ("food", "water") for e in self.manifest)
        self.labels = np.array([int(e["category"] == "water") for e in self.manifest])
        self.splits = np.array([e["split"] for e in self.manifest])
        assert not ({e["sha256"] for e in self.manifest if e["split"] == "train"}
                    & {e["sha256"] for e in self.manifest if e["split"] == "test"})

    def sample(self, kinds, rng):
        ids = np.empty(kinds.shape, dtype=np.int64)
        for kind in (0, 1):
            mask = kinds == kind
            pool = np.flatnonzero((self.splits == "test") & (self.labels == kind))
            ids[mask] = rng.choice(pool, int(mask.sum()))
        return self.features[torch.from_numpy(ids)], ids

    def cases(self, seed, n, task, horizon):
        rng = np.random.default_rng(seed)
        scenes = FULL_SCENES if task == "full" else COURSE_SCENES
        kinds = scenes[rng.integers(len(scenes), size=n)].copy()
        features, ids = self.sample(kinds, rng)
        inventory = np.zeros((n, 2), dtype=np.int64)
        remaining = horizon - np.arange(n, dtype=np.int64) % horizon
        public = torch.from_numpy(np.column_stack((inventory, remaining / horizon)).astype(np.float32))
        digest = hashlib.sha256()
        digest.update(task.encode())
        digest.update(b"test")
        for value in (kinds, ids, inventory, remaining):
            digest.update(np.ascontiguousarray(value, dtype=np.int64).tobytes())
        return {"kinds": kinds, "features": features, "image_ids": ids,
                "inventory": inventory, "remaining": remaining, "public": public,
                "hash": digest.hexdigest()}


def array_rows(rows):
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def audit_intervention(agents, pair, bank, seed, saved, n, horizon):
    case = bank.cases(seed, n, "full", horizon)
    kinds, features, public = case["kinds"], case["features"], case["public"]
    assert saved["external_cases_sha256"] == case["hash"]
    rep = [agents[i].observe(features[:, i], public) for i in range(2)]
    sent = [agents[i].send(rep[i][1]).argmax(-1) for i in range(2)]
    original_p = [agents[i].act(*rep[i], sent[1 - i]).softmax(-1).numpy() for i in range(2)]
    actions = np.column_stack([p.argmax(-1) for p in original_p])
    selected = np.take_along_axis(kinds, actions[..., None], axis=-1)[..., 0]
    success = selected[:, 0] != selected[:, 1]
    reference = bank.cases(seed + 33001, n, "full", horizon)
    same = kinds[..., 0] == kinds[..., 1]
    output = []
    for receiver in range(2):
        sender = 1 - receiver
        direction = next(d for d in saved["directions"] if d["receiver"] == receiver)
        assert direction["population_receiver"] == pair[receiver]
        assert direction["population_sender"] == pair[sender]
        _, ref_local = agents[sender].observe(reference["features"][:, sender], reference["public"])
        ref_sent = agents[sender].send(ref_local).argmax(-1).numpy()
        counts = np.bincount(ref_sent, minlength=5)
        assert counts.tolist() == direction["reference_counts"]
        minimum = max(5, n // 100)
        assert direction["reference_min_count"] == minimum
        used = np.flatnonzero(counts >= minimum)
        relevant = same[:, sender] & ~same[:, receiver]
        assert int(relevant.sum()) == direction["restricted_sender_mixed_receiver_cases"]
        resources, probabilities = [], []
        for symbol in range(5):
            p = agents[receiver].act(*rep[receiver], torch.full((n,), symbol, dtype=torch.int64)).softmax(-1).numpy()
            new_action = p.argmax(-1)
            new_resource = kinds[np.arange(n), receiver, new_action]
            new_success = new_resource != selected[:, sender]
            food_p = (p * (kinds[:, receiver] == 0)).sum(axis=1)
            resources.append(new_resource)
            probabilities.append(food_p)
            mask = relevant & (sent[sender].numpy() != symbol)
            mean = lambda values: float(values[mask].mean()) if mask.any() else None
            actual = {"choice_flip_rate": mean(new_action != actions[:, receiver]),
                      "resource_flip_rate": mean(new_resource != selected[:, receiver]),
                      "mean_action_total_variation": mean(.5 * np.abs(p - original_p[receiver]).sum(axis=1)),
                      "baseline_success": mean(success), "intervened_success": mean(new_success),
                      "success_change": mean(new_success.astype(float) - success.astype(float)),
                      "mixed_recipient_mean_food_probability": float(food_p[~same[:, receiver]].mean())}
            effect = next(e for e in direction["interventions"] if e["replacement_symbol"] == symbol)
            assert effect["n_changed_relevant_cases"] == int(mask.sum())
            assert effect["in_observed_sender_support"] == bool(counts[symbol] >= minimum)
            for name, value in actual.items():
                equal_number(value, effect[name], f"intervention {pair} {receiver} {symbol} {name}")
        sensitivity = direction["observed_symbol_sensitivity"]
        assert sensitivity["symbols"] == used.tolist()
        if len(used) >= 2 and relevant.any():
            changes = float((np.ptp(np.stack(resources)[used][:, relevant], axis=0) > 0).mean())
            probability_range = float(np.ptp(np.stack(probabilities)[used][:, relevant], axis=0).mean())
        else:
            changes = probability_range = None
        equal_number(changes, sensitivity["fraction_cases_resource_changes_for_some_used_symbol"], "resource sensitivity")
        equal_number(probability_range, sensitivity["mean_food_probability_range"], "probability sensitivity")
        output.append({"sender": pair[sender], "receiver": pair[receiver], "used_symbols": used.tolist(),
                       "cases": int(relevant.sum()), "resource_response_fraction": changes,
                       "all_five_symbol_interventions_reproduced": True})
    return output


def audit_alignment(agents, bank, saved):
    """Reconstruct raw agreement descriptively; no alignment or word meanings."""
    n, horizon = saved["cases"], saved["horizon_clock"]
    rng = np.random.default_rng(saved["seed"])
    remaining = horizon - np.arange(n, dtype=np.int64) % horizon
    local = np.empty(n, dtype=np.int64)
    for time in np.unique(remaining):
        positions = np.flatnonzero(remaining == time)
        values = (np.arange(len(positions), dtype=np.int64) % 3 + int(time)) % 3
        local[positions] = values[rng.permutation(len(values))]
    kinds = np.column_stack((local == 2, local >= 1)).astype(np.int64)
    reverse = (local == 1) & (rng.random(n) < .5)
    kinds[reverse] = kinds[reverse, ::-1]
    features, ids = bank.sample(kinds, rng)
    public_array = np.column_stack((np.zeros((n, 2), np.float32), remaining.astype(np.float32) / horizon))
    public = torch.from_numpy(public_array)
    h = hashlib.sha256(b"test")
    for value in (kinds, ids, remaining, public_array):
        h.update(np.ascontiguousarray(value).tobytes())
    assert h.hexdigest() == saved["external_cases_sha256"]
    probabilities = np.stack([a.send(a.observe(features, public)[1]).softmax(-1).numpy() for a in agents])
    symbols = probabilities.argmax(-1)
    equal_number(float(np.all(symbols == symbols[0:1], axis=0).mean()), saved["all_four_greedy_agreement"], "all-four alignment")
    for left, right in PAIRS:
        item = next(p for p in saved["pairs"] if p["agents"] == [left, right])
        values = {"greedy_agreement": symbols[left] == symbols[right],
                  "mean_total_variation": .5 * np.abs(probabilities[left] - probabilities[right]).sum(axis=1),
                  "independent_sample_expected_agreement": (probabilities[left] * probabilities[right]).sum(axis=1)}
        for name, value in values.items():
            equal_number(float(value.mean()), item[name], f"alignment {name}")
            for category, subset in enumerate(item["by_local_resource_set"]):
                equal_number(float(value[local == category].mean()), subset[name], f"alignment subset {name}")
    return {"raw_metrics_reproduced": True, "symbols_remapped": False,
            "interpretation": "Raw symbol agreement only; constant signals can agree, and this is not shared semantics."}


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
    total_updates = config["course_updates"] + config["exploratory_full_updates"] + config["pure_reward_updates"]
    output, external_by_run = [], {}
    trace_count = case_count = 0
    for seed in config["seeds"]:
        prepared_hash = sha256(folder / f"prepared_s{seed}.pt")
        for condition in config["conditions"]:
            run = folder / f"{condition}_s{seed}"
            saved = read_json(run / "result.json")
            assert sha256(run / "initial.pt") == prepared_hash
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
                          "initial_checkpoint_matches_population_preparation": True, "pairs": []}
            hashes = {}
            for pair_index, pair in enumerate(PAIRS):
                pair_saved = next(p for p in evaluation["pairs"] if p["agents"] == list(pair))
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
            output.append(run_record)
            external_by_run[(seed, condition)] = hashes
            print(json.dumps({"audited": run.name, "pairs": 6, "trace_files": 48}), flush=True)
    for seed in config["seeds"]:
        assert external_by_run[(seed, "fixed")] == external_by_run[(seed, "rotating")]
    assert len(output) == completed["runs"]
    report = {"status": "passed", "runs": len(output), "trace_files": trace_count, "trace_cases": case_count,
              "source_snapshot_hashes_verified": True, "no_identical_image_bytes_across_train_test": True,
              "fixed_rotating_external_cases_identical": True,
              "paired_condition_sent_messages_not_required_identical": True,
              "stochastic_actions_replayed_with_saved_pair_task_seeds": True,
              "scope": "Saved terminal policies only; no training or adaptation. Scores, input generation, RNGs and interventions independently reconstructed.",
              "runs_audited": output,
              "limits": ["Six pair scores within one population share agents and are not independent replicates.",
                         "Rotating populations met all six pairs; fixed cross-pairs were never trained together. This is not equal-exposure zero-shot transfer.",
                         "Raw symbol agreement can arise from constant signals; it does not establish common semantics.",
                         "Constant symbol 0 can carry learned meaning; it is not a neutral silence token.",
                         "Held-out cases recombine a finite photo bank and do not establish general-world communication."]}
    (folder / "独立伙伴通信核查.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lines = ["# 独立伙伴通信核查", "", f"核查通过：{len(output)} 个群体运行、{trace_count} 份轨迹、{case_count:,} 个最终评估案例。未训练或修改参数。", "",
             "从保存的图片特征、缩放值和四主体最终检查点重新推理全部发送与动作。四种评估模式均通过，包括按每对、每任务保存的种子重建发送和行动随机流的随机采样模式。资源选择、配齐成功和奖励逐行独立重算一致。", "",
             "固定与轮换条件的外部案例相同，但它们学到的发信不要求相同。同一运行内，正常、打乱和恒符号0三个条件的原始发信相同；随机采样的发信可以不同。打乱置换使用独立随机流，并逐公共时间、逐发送方重新生成后核对。", "",
             "按种子重新生成同观察干预案例，保持接收端照片、公共状态与伙伴原行动不变，仅替换一个离散符号。五种替换的行为和收益变化均与保存结果一致。原始符号一致性另行重建，也与保存指标一致。", "",
             "解释边界：这些核查验证终点评估的实现和消息作用证据；原始编号一致不等于共同语义或共同词典。轮换组训练过所有伙伴组合，固定组的交叉配对没有共同训练经历，两者不是相同接触量的新伙伴零样本泛化。一个群体内六条边和两个方向也不是独立重复。", "",
             "详细逐运行、逐边、逐方向结果见同目录 `独立伙伴通信核查.json`。"]
    (folder / "独立伙伴通信核查.md").write_text("\n".join(lines) + "\n")
    return report


def self_test():
    # Test exact sampling arithmetic independently of any experiment or data.
    from agents import draw
    generator = torch.Generator().manual_seed(826)
    for categories in (2, 5):
        logits = torch.randn((4096, categories), generator=generator)
        for greedy in (False, True):
            expected = draw(logits, np.random.default_rng(993), greedy)[0]
            actual = replay_draw(logits, np.random.default_rng(993), greedy)
            assert torch.equal(actual, expected)
    assert FULL_SCENES.shape == (14, 2, 2)
    assert COURSE_SCENES.shape == (8, 2, 2)
    print(json.dumps({"self_test": "passed", "categorical_draw_matches_original": True,
                      "training_performed": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=ROOT / "results/partners_001")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.self_test:
        self_test()
    else:
        summary = audit(args.directory)
        print(json.dumps({k: summary[k] for k in ("status", "runs", "trace_files", "trace_cases")}))
