"""Read-only evaluation of four independently trained resource communicators.

All six unordered pairs are evaluated in the unchanged two-agent task. Pair and
task seeds are deterministic functions of one master seed, so training conditions
can use identical holdout cases without sharing their learned parameters.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from curriculum_eval import LOCAL_SET_ORDER, evaluate, message_intervention


PAIRS = tuple(itertools.combinations(range(4), 2))
ORIGINAL_PAIRS = frozenset(((0, 1), (2, 3)))
FULL_MODES = (("normal", "normal", True), ("shuffle", "shuffle", True),
              ("blank", "blank", True), ("stochastic", "normal", False))
CHECKPOINT_MODES = tuple(item for item in FULL_MODES if item[0] != "blank")


def _validate(agents, seed, n, horizon):
    if len(agents) != 4 or len({id(a) for a in agents}) != 4:
        raise ValueError("exactly four distinct agent objects are required")
    for name, value, minimum in (("seed", seed, 0), ("n", n, 1), ("horizon", horizon, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")


def _seed(master, stream, index=0):
    """Stable seed derivation, independent of Python hash randomization."""
    return int(np.random.SeedSequence([int(master), int(stream), int(index)]).generate_state(1)[0])


def _summary(values):
    values = [float(v) for v in values]
    return {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}


def _population_directions(stats, pair, intervention=False):
    entries = stats["directions"] if intervention else stats["by_direction"]
    for entry in entries:
        sender = entry["sender"] if intervention else entry["restricted_sender"]
        receiver = entry["receiver"] if intervention else entry["mixed_receiver"]
        entry["population_sender"] = pair[sender]
        entry["population_receiver"] = pair[receiver]
    stats["population_agents_in_local_order"] = list(pair)
    return stats


def _aggregates(pairs):
    result = {}
    for group in ("original", "cross", "all"):
        chosen = [p for p in pairs if group == "all" or p["group"] == group]
        tasks = {}
        for task in chosen[0]["tasks"]:
            modes = tuple(chosen[0]["tasks"][task])
            task_result = {}
            for mode in modes:
                rates = [p["tasks"][task][mode]["mean_reward_per_step"] for p in chosen]
                cases = [p["tasks"][task][mode]["cases"] for p in chosen]
                task_result[mode] = {
                    "mean_reward_per_step": float(np.mean(rates)),
                    "balanced_gathering": float(np.mean([p["tasks"][task][mode]["balanced_gathering"] for p in chosen])),
                    "min_pair_success": min(rates), "max_pair_success": max(rates),
                    "total_cases": int(sum(cases)),
                    "case_weighted_success": float(np.average(rates, weights=cases)),
                    "by_pair": [{"agents": p["agents"], "success": rate} for p, rate in zip(chosen, rates)]}
            for replacement in ("shuffle", "blank"):
                if replacement not in modes:
                    continue
                gaps = [p["tasks"][task]["normal"]["mean_reward_per_step"] -
                        p["tasks"][task][replacement]["mean_reward_per_step"] for p in chosen]
                task_result[f"normal_minus_{replacement}"] = {
                    **_summary(gaps), "by_pair": [{"agents": p["agents"], "difference": gap}
                                                   for p, gap in zip(chosen, gaps)]}
            tasks[task] = task_result
        result[group] = {"pair_count": len(chosen), "agents": [p["agents"] for p in chosen],
                         "tasks": tasks,
                         "interpretation": "Equal-weight mean over pairs sharing trained agents; these pairs are not independent training replicates."}
    return result


def _write_trace(folder, pair, task, mode, rows):
    path = folder / f"pair_{pair[0]}{pair[1]}_{task}_{mode}_trace.jsonl"
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    return {"path": str(path), "cases": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _evaluate_pairs(agents, bank, seed, n, tasks, modes, split, horizon,
                    trace_dir=None, intervention_n=None):
    pairs = []
    folder = Path(trace_dir).resolve() if trace_dir is not None else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    for pair_index, pair in enumerate(PAIRS):
        selected = [agents[i] for i in pair]
        row = {"agents": list(pair), "group": "original" if pair in ORIGINAL_PAIRS else "cross",
               "tasks": {}, "evaluation_seeds": {}}
        if folder is not None:
            row["traces"] = {}
        for task in tasks:
            task_seed = _seed(seed, 100 if task == "full" else 101, pair_index)
            row["evaluation_seeds"][task] = task_seed
            row["tasks"][task] = {}
            for name, mode, greedy in modes:
                stats, traces = evaluate(selected, bank, task_seed, n=n, task=task, mode=mode,
                                         greedy=greedy, split=split, horizon=horizon, trace=folder is not None)
                row["tasks"][task][name] = _population_directions(stats, pair)
                if folder is not None:
                    row["traces"][f"{task}_{name}"] = _write_trace(folder, pair, task, name, traces)
            hashes = {m["external_cases_sha256"] for m in row["tasks"][task].values()}
            if len(hashes) != 1:
                raise RuntimeError(f"Unpaired external cases for pair {pair}, task {task}")
        if intervention_n is not None:
            intervention_seed = _seed(seed, 200, pair_index)
            intervention = message_intervention(selected, bank, intervention_seed, n=intervention_n,
                                                task="full", split=split, horizon=horizon)
            row["intervention_seed"] = intervention_seed
            row["intervention"] = _population_directions(intervention, pair, intervention=True)
        pairs.append(row)
    return pairs


@torch.no_grad()
def sender_alignment(agents, bank, seed, n=2048, split="test", horizon=16):
    """Raw symbol agreement on identical inputs; no symbol remapping or labels.

    FF, mixed FW/WF, and WW are balanced within each public time group up to one
    case. Simulator category IDs select photographs and score tables only; the
    four senders receive the exact same feature and public tensors.
    """
    _validate(agents, seed, n, horizon)
    rng = np.random.default_rng(seed)
    remaining = horizon - np.arange(n, dtype=np.int64) % horizon
    local_set = np.empty(n, dtype=np.int64)
    for time in np.unique(remaining):
        positions = np.flatnonzero(remaining == time)
        labels = np.arange(len(positions), dtype=np.int64) % 3
        # Rotate the remainder between groups instead of always favoring FF.
        labels = (labels + int(time)) % 3
        local_set[positions] = labels[rng.permutation(len(labels))]
    kinds = np.column_stack((local_set == 2, local_set >= 1)).astype(np.int64)
    mixed = local_set == 1
    reverse = mixed & (rng.random(n) < .5)
    kinds[reverse] = kinds[reverse, ::-1]
    features, image_ids = bank.sample(kinds, split, rng)
    public_array = np.column_stack((np.zeros((n, 2), np.float32), remaining.astype(np.float32) / horizon))
    public = torch.from_numpy(public_array)
    digest = hashlib.sha256()
    digest.update(split.encode())
    for value in (kinds, image_ids, remaining, public_array):
        digest.update(np.ascontiguousarray(value).tobytes())
    probabilities = []
    for agent in agents:
        _, local = agent.observe(features, public)
        probability = agent.send(local).softmax(-1).cpu().numpy()
        if probability.shape != (n, 5) or not np.isfinite(probability).all():
            raise ValueError("Expected five finite symbol probabilities per input")
        probabilities.append(probability)
    probabilities = np.stack(probabilities)
    argmax = probabilities.argmax(axis=-1)
    result = {"seed": seed, "split": split, "cases": n, "horizon_clock": horizon,
              "external_cases_sha256": digest.hexdigest(), "local_set_order": LOCAL_SET_ORDER,
              "same_exact_inputs_for_all_agents": True, "symbol_permutation_applied": False,
              "agents": [], "pairs": [],
              "interpretation": "Descriptive agreement of raw symbol IDs and probabilities on identical observations. Shared constant signals or public-time codes can also agree; this is not proof of common semantics, resource understanding, or a population language. Never align raw IDs across independently trained seeds."}
    for agent in range(4):
        by_set = []
        for category, name in enumerate(LOCAL_SET_ORDER):
            mask = local_set == category
            by_set.append({"local_set": name, "n": int(mask.sum()),
                           "mean_symbol_probabilities": probabilities[agent, mask].mean(axis=0).tolist() if mask.any() else None,
                           "greedy_symbol_counts": np.bincount(argmax[agent, mask], minlength=5).tolist()})
        result["agents"].append({"agent": agent, "by_local_resource_set": by_set,
                                 "mean_symbol_probabilities": probabilities[agent].mean(axis=0).tolist(),
                                 "greedy_symbol_counts": np.bincount(argmax[agent], minlength=5).tolist()})
    for left, right in PAIRS:
        agreement = argmax[left] == argmax[right]
        tv = .5 * np.abs(probabilities[left] - probabilities[right]).sum(axis=1)
        independent_agreement = (probabilities[left] * probabilities[right]).sum(axis=1)
        pair_result = {"agents": [left, right], "group": "original" if (left, right) in ORIGINAL_PAIRS else "cross",
                       "greedy_agreement": float(agreement.mean()), "mean_total_variation": float(tv.mean()),
                       "independent_sample_expected_agreement": float(independent_agreement.mean()),
                       "by_local_resource_set": []}
        for category, name in enumerate(LOCAL_SET_ORDER):
            mask = local_set == category
            pair_result["by_local_resource_set"].append({
                "local_set": name, "n": int(mask.sum()),
                "greedy_agreement": float(agreement[mask].mean()) if mask.any() else None,
                "mean_total_variation": float(tv[mask].mean()) if mask.any() else None,
                "independent_sample_expected_agreement": float(independent_agreement[mask].mean()) if mask.any() else None})
        result["pairs"].append(pair_result)
    all_four = np.all(argmax == argmax[0:1], axis=0)
    result["all_four_greedy_agreement"] = float(all_four.mean())
    result["all_four_by_local_resource_set"] = [
        {"local_set": name, "n": int((local_set == category).sum()),
         "greedy_agreement": float(all_four[local_set == category].mean()) if (local_set == category).any() else None}
        for category, name in enumerate(LOCAL_SET_ORDER)]
    result["aggregates"] = {}
    for group in ("original", "cross", "all"):
        chosen = [p for p in result["pairs"] if group == "all" or p["group"] == group]
        result["aggregates"][group] = {"pair_count": len(chosen),
            **{name: _summary(p[name] for p in chosen) for name in
               ("greedy_agreement", "mean_total_variation", "independent_sample_expected_agreement")}}
    return result


@torch.no_grad()
def evaluate_population(agents, bank, seed, n=8192, trace_dir=None, intervention_n=2048,
                        include_curriculum=True, alignment_n=2048, split="test", horizon=16):
    """Final population evaluation; does not update weights or adapt to partners."""
    _validate(agents, seed, n, horizon)
    for name, count in (("intervention_n", intervention_n), ("alignment_n", alignment_n)):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError(f"{name} must be a positive integer")
    tasks = ("full", "curriculum") if include_curriculum else ("full",)
    pairs = _evaluate_pairs(agents, bank, seed, n, tasks, FULL_MODES, split, horizon,
                            trace_dir=trace_dir, intervention_n=intervention_n)
    return {"master_seed": int(seed), "evaluation": "final", "population_size": 4,
            "cases_per_pair_task_mode": int(n), "split": split, "pairs": pairs,
            "aggregates": _aggregates(pairs),
            "sender_alignment": sender_alignment(agents, bank, _seed(seed, 300), alignment_n, split, horizon),
            "seed_scheme": "SeedSequence([master_seed, stream, pair_index]); streams full=100, curriculum=101, intervention=200, sender_alignment=300; pairs in lexicographic order. Reuse master seed across training conditions for paired holdout cases.",
            "interpretation": "Original pairs are (0,1) and (2,3); cross means the four other pairs. These labels describe the fixed-partner reference design, not whether rotating populations have met. All evaluations are zero-update, with no partner identity passed to policies."}


@torch.no_grad()
def checkpoint_population(agents, bank, seed, n=1024, split="test", horizon=16):
    """Light checkpoint: six pairs, full task, normal/shuffle/stochastic only."""
    _validate(agents, seed, n, horizon)
    pairs = _evaluate_pairs(agents, bank, seed, n, ("full",), CHECKPOINT_MODES, split, horizon)
    return {"master_seed": int(seed), "evaluation": "checkpoint", "population_size": 4,
            "cases_per_pair_task_mode": int(n), "split": split, "pairs": pairs,
            "aggregates": _aggregates(pairs)}
