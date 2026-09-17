"""Read-only endpoint evaluation and protocol drift after partner contact.

All comparisons hold photo features, public time, and received symbol fixed.
Reference objects contain only tensors and primitives and support
torch.load(..., weights_only=True). Category IDs are probe metadata for sampling
and scoring; they never enter a sender or listener policy.
"""
from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np
import torch

from partner_eval import checkpoint_population, evaluate_population
from resource_env import FEASIBLE_SCENES


CONTEXTS = ("food_food", "mixed", "water_water")
ORIGINAL_PAIRS = ((0, 1), (2, 3))
ALL_PAIRS = tuple(itertools.combinations(range(4), 2))


def _pairs(pairs, population_size):
    pairs = [tuple(sorted(map(int, pair))) for pair in pairs]
    if any(len(pair) != 2 or pair[0] == pair[1] or min(pair) < 0 or max(pair) >= population_size for pair in pairs):
        raise ValueError("invalid trained pair")
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate trained pair")
    if set(itertools.chain.from_iterable(pairs)) != set(range(population_size)):
        raise ValueError("every receiver needs at least one declared trained partner")
    return sorted(pairs)


def _validate_agents(agents):
    if len(agents) != 4 or len({id(agent) for agent in agents}) != 4:
        raise ValueError("four distinct agents required")


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    return torch.from_numpy(np.ascontiguousarray(value)).clone()


def _hash_probe(probe):
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


def _freeze_probe(**values):
    probe = {key: _tensor(value) if isinstance(value, (np.ndarray, torch.Tensor)) else value
             for key, value in values.items()}
    probe["sha256"] = _hash_probe(probe)
    return probe


def _balanced_probe(bank, seed, n, horizon):
    rng = np.random.default_rng(seed)
    remaining = horizon - np.arange(n, dtype=np.int64) % horizon
    context = np.empty(n, dtype=np.int64)
    for time in np.unique(remaining):
        indices = np.flatnonzero(remaining == time)
        values = (np.arange(len(indices), dtype=np.int64) % 3 + int(time)) % 3
        context[indices] = values[rng.permutation(len(values))]
    kinds = np.column_stack((context == 2, context >= 1)).astype(np.int64)
    reverse = (context == 1) & (rng.random(n) < .5)
    kinds[reverse] = kinds[reverse, ::-1]
    features, ids = bank.sample(kinds, "test", rng)
    public = np.column_stack((np.zeros((n, 2), np.float32), remaining.astype(np.float32) / horizon))
    return _freeze_probe(seed=int(seed), cases=int(n), horizon=int(horizon), split="test",
                         features=features, public=public, image_ids=ids, kinds=kinds,
                         local_context=context, remaining=remaining)


def _natural_support_probe(bank, seed, n, horizon):
    # Full-task scenes provide the sender/receiver context correlations. All
    # senders see the same role-0 photos; role 1 determines receiver strata only.
    rng = np.random.default_rng(seed)
    kinds = FEASIBLE_SCENES[rng.integers(len(FEASIBLE_SCENES), size=n)].copy()
    features, ids = bank.sample(kinds, "test", rng)
    remaining = horizon - np.arange(n, dtype=np.int64) % horizon
    public = np.column_stack((np.zeros((n, 2), np.float32), remaining.astype(np.float32) / horizon))
    return _freeze_probe(seed=int(seed), cases=int(n), horizon=int(horizon), split="test",
                         features=features[:, 0], public=public, image_ids=ids[:, 0],
                         full_image_ids=ids, full_kinds=kinds, kinds=kinds[:, 0],
                         sender_context=kinds[:, 0].sum(axis=1),
                         receiver_context=kinds[:, 1].sum(axis=1), remaining=remaining)


def _sender_probabilities(agents, probe):
    values = []
    for agent in agents:
        _, local = agent.observe(probe["features"], probe["public"])
        values.append(agent.send(local).softmax(-1).detach().cpu())
    probabilities = torch.stack(values)
    assert probabilities.shape == (4, probe["cases"], 5)
    assert torch.isfinite(probabilities).all()
    return probabilities


def _listener_probabilities(agents, probe):
    values = []
    for agent in agents:
        rep = agent.observe(probe["features"], probe["public"])
        per_symbol = [agent.act(*rep, torch.full((probe["cases"],), symbol, dtype=torch.int64)).softmax(-1)
                      for symbol in range(5)]
        values.append(torch.stack(per_symbol, dim=1).detach().cpu())
    probabilities = torch.stack(values)
    assert probabilities.shape == (4, probe["cases"], 5, 2)
    assert torch.isfinite(probabilities).all()
    return probabilities


def _supported(counts, minimum_count, minimum_fraction):
    n = int(sum(counts))
    threshold = max(minimum_count, int(math.ceil(minimum_fraction * n)))
    return {"n": n, "counts": list(map(int, counts)), "threshold": threshold,
            "symbols": [symbol for symbol, count in enumerate(counts) if count >= threshold]}


def _support(probabilities, probe, pairs, minimum_count, minimum_fraction):
    symbols = probabilities.argmax(-1).numpy()
    receiver_context = probe["receiver_context"].numpy()
    senders = []
    for sender in range(4):
        contexts = {}
        for category, name in enumerate(CONTEXTS):
            mask = receiver_context == category
            counts = np.bincount(symbols[sender, mask], minlength=5)
            contexts[name] = _supported(counts, minimum_count, minimum_fraction)
        senders.append({"agent": sender,
                        "overall": _supported(np.bincount(symbols[sender], minlength=5), minimum_count, minimum_fraction),
                        "by_receiver_own_context": contexts,
                        "mean_stochastic_symbol_probabilities": probabilities[sender].mean(0).tolist()})
    receivers = []
    for receiver in range(4):
        partners = sorted(pair[1] if pair[0] == receiver else pair[0] for pair in pairs if receiver in pair)
        contexts = {}
        for context in ("overall",) + CONTEXTS:
            usage = [senders[sender]["overall"] if context == "overall" else senders[sender]["by_receiver_own_context"][context]
                     for sender in partners]
            union = sorted(set(itertools.chain.from_iterable(item["symbols"] for item in usage)))
            counts = np.sum([item["counts"] for item in usage], axis=0)
            total_retained = int(counts[union].sum())
            contexts[context] = {"symbols": union, "pooled_greedy_counts": counts.tolist(),
                                 "reference_partner_cases": sum(item["n"] for item in usage),
                                 "weights_on_supported_symbols": [(float(counts[symbol]) / total_retained if symbol in union and total_retained else 0.)
                                                                  for symbol in range(5)]}
        receivers.append({"agent": receiver, "declared_partners": partners,
                          "overall": contexts.pop("overall"), "by_own_context": contexts})
    return {"trained_pairs": [list(pair) for pair in pairs], "senders": senders, "receivers": receivers,
            "minimum_greedy_count": minimum_count, "minimum_greedy_fraction": minimum_fraction,
            "support_definition": "Union of each declared partner's endpoint greedy-symbol support on independent full-task reference scenes; per-partner threshold max(minimum_count, ceil(minimum_fraction * stratum_n)).",
            "interpretation": "Estimated endpoint use, not a historical log of actually received tokens. Stochastic sampling may have used additional symbols. Receiver-context strata respect full-scene feasibility."}


@torch.no_grad()
def protocol_reference(agents, bank, seed, n=1024, trained_pairs=ORIGINAL_PAIRS,
                       support_n=2048, horizon=16, minimum_count=10, minimum_fraction=.01):
    """Freeze identical observations and each policy's pre-contact responses."""
    _validate_agents(agents)
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1
           for value in (n, support_n, horizon, minimum_count)):
        raise ValueError("probe sizes, horizon and minimum_count must be positive integers")
    if not 0 < minimum_fraction <= 1:
        raise ValueError("minimum_fraction must be in (0,1]")
    pairs = _pairs(trained_pairs, len(agents))
    probe = _balanced_probe(bank, seed, n, horizon)
    support_seed = int(np.random.SeedSequence([int(seed), 991]).generate_state(1)[0])
    support_probe = _natural_support_probe(bank, support_seed, support_n, horizon)
    initial_senders = _sender_probabilities(agents, probe).clone()
    initial_listeners = _listener_probabilities(agents, probe).clone()
    initial_support_probabilities = _sender_probabilities(agents, support_probe).clone()
    return {"schema_version": 1, "seed": int(seed), "population_size": 4,
            "context_order": list(CONTEXTS), "trained_pairs": [list(pair) for pair in pairs],
            "probe": probe, "support_probe": support_probe,
            "initial_sender_probabilities": initial_senders,
            "initial_listener_action_probabilities": initial_listeners,
            "initial_support_sender_probabilities": initial_support_probabilities,
            "initial_received_support": _support(initial_support_probabilities, support_probe, pairs, minimum_count, minimum_fraction),
            "minimum_count": int(minimum_count), "minimum_fraction": float(minimum_fraction),
            "policy_reference_scope": "Pre-contact endpoint policies; no training, adaptation, symbol remapping, or semantic labels supplied to policies."}


def _sender_change(before, after, mask):
    if not mask.any():
        return {"n": 0, "mean_probability_total_variation": None, "greedy_raw_symbol_change_rate": None}
    return {"n": int(mask.sum()),
            "mean_probability_total_variation": float((.5 * (before[mask] - after[mask]).abs().sum(-1)).mean()),
            "greedy_raw_symbol_change_rate": float((before[mask].argmax(-1) != after[mask].argmax(-1)).float().mean())}


def _listener_change(before, after, kinds, mask):
    if not mask.any():
        return {"n": 0, "mean_action_probability_total_variation": None,
                "greedy_option_change_rate": None, "greedy_resource_change_rate": None}
    old_action, new_action = before.argmax(-1), after.argmax(-1)
    old_resource = kinds.gather(1, old_action[:, None]).squeeze(1)
    new_resource = kinds.gather(1, new_action[:, None]).squeeze(1)
    return {"n": int(mask.sum()),
            "mean_action_probability_total_variation": float((.5 * (before[mask] - after[mask]).abs().sum(-1)).mean()),
            "greedy_option_change_rate": float((old_action[mask] != new_action[mask]).float().mean()),
            "greedy_resource_change_rate": float((old_resource[mask] != new_resource[mask]).float().mean())}


def _summarize_symbols(effects, support):
    symbols = support["symbols"]
    keys = ("mean_action_probability_total_variation", "greedy_option_change_rate", "greedy_resource_change_rate")
    result = {"symbols": symbols, "symbol_count": len(symbols),
              "n_own_observations": effects[0]["n"], "symbol_observation_combinations": effects[0]["n"] * len(symbols),
              "equal_symbol_mean": {}, "initial_usage_weighted_mean": {}}
    weights = support["weights_on_supported_symbols"]
    for key in keys:
        valid = bool(symbols) and effects[0]["n"] > 0
        result["equal_symbol_mean"][key] = float(np.mean([effects[symbol][key] for symbol in symbols])) if valid else None
        result["initial_usage_weighted_mean"][key] = float(sum(weights[symbol] * effects[symbol][key] for symbol in symbols)) if valid else None
    return result


@torch.no_grad()
def protocol_drift(agents, reference, current_trained_pairs=None):
    """Compare current policies against one unchanged pre-contact reference.

    Primary listener summaries use the *initial* partners' estimated symbol
    support; newly used or newly encountered symbols do not alter that mask.
    Current support is descriptive, and all five symbols remain reported.
    """
    _validate_agents(agents)
    if reference["schema_version"] != 1:
        raise ValueError("unsupported protocol reference schema")
    probe, support_probe = reference["probe"], reference["support_probe"]
    assert _hash_probe(probe) == probe["sha256"], "probe changed after reference creation"
    assert _hash_probe(support_probe) == support_probe["sha256"], "support probe changed after reference creation"
    pairs = _pairs(current_trained_pairs if current_trained_pairs is not None else reference["trained_pairs"], 4)
    current_senders = _sender_probabilities(agents, probe)
    current_listeners = _listener_probabilities(agents, probe)
    current_support_probabilities = _sender_probabilities(agents, support_probe)
    current_support = _support(current_support_probabilities, support_probe, pairs,
                               reference["minimum_count"], reference["minimum_fraction"])
    old_support = reference["initial_received_support"]
    categories, kinds = probe["local_context"], probe["kinds"]
    all_cases = torch.ones(probe["cases"], dtype=torch.bool)
    senders, listeners = [], []
    for who in range(4):
        before_sender = reference["initial_sender_probabilities"][who]
        senders.append({"agent": who, "overall": _sender_change(before_sender, current_senders[who], all_cases),
                        "by_own_context": {name: _sender_change(before_sender, current_senders[who], categories == category)
                                           for category, name in enumerate(CONTEXTS)}})
        by_context = {}
        for category, name in enumerate(CONTEXTS):
            mask = categories == category
            effects = [_listener_change(reference["initial_listener_action_probabilities"][who, :, symbol],
                                         current_listeners[who, :, symbol], kinds, mask)
                       for symbol in range(5)]
            initial = old_support["receivers"][who]["by_own_context"][name]
            current = current_support["receivers"][who]["by_own_context"][name]
            before_set, after_set = set(initial["symbols"]), set(current["symbols"])
            by_context[name] = {"n": int(mask.sum()),
                                "all_five_symbols": [{"received_symbol": symbol, **effect,
                                                       "in_initial_received_support": symbol in before_set,
                                                       "in_current_received_support": symbol in after_set}
                                                      for symbol, effect in enumerate(effects)],
                                "initial_received_support_summary": _summarize_symbols(effects, initial),
                                "received_support_change": {"initial_symbols": sorted(before_set),
                                                            "current_symbols": sorted(after_set),
                                                            "added_symbols": sorted(after_set - before_set),
                                                            "removed_symbols": sorted(before_set - after_set),
                                                            "jaccard": len(before_set & after_set) / len(before_set | after_set) if before_set | after_set else None}}
        listeners.append({"agent": who,
                          "initial_declared_partners": old_support["receivers"][who]["declared_partners"],
                          "current_declared_partners": current_support["receivers"][who]["declared_partners"],
                          "by_own_context": by_context})
    return {"schema_version": 1, "reference_seed": reference["seed"],
            "probe_cases": probe["cases"], "support_cases": support_probe["cases"],
            "probe_sha256": probe["sha256"], "support_probe_sha256": support_probe["sha256"],
            "initial_trained_pairs": reference["trained_pairs"], "current_trained_pairs": [list(pair) for pair in pairs],
            "senders": senders, "listeners": listeners,
            "initial_received_support": old_support, "current_received_support": current_support,
            "comparison": "Same own photos, public time, and received symbol; only policy endpoint changes. No symbol permutation.",
            "primary_listener_summary": "Per own-context drift restricted to the initial declared partners' estimated support. Support masks and usage weights remain fixed at the initial endpoint.",
            "interpretation": "Raw policy drift, not semantic innovation. FF/WW option probabilities can drift without changing the resource. Endpoint support estimates are not histories of tokens received during training."}
