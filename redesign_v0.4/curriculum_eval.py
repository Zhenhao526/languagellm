"""Held-out evaluation for the two-resource communication curriculum.

Simulator category IDs are used only to sample photographs and score behavior.
The policy receives its own frozen image features, public empty inventory and
remaining time, and exactly one discrete symbol. This module never trains a
policy. Capacity one makes all cases IID after consumption; the public clock
cycles through the same 16 values as the original episodic task.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
import torch

from agents import draw
from resource_env import FEASIBLE_SCENES, transition
from run_pilot import public_tensor


_same = FEASIBLE_SCENES[..., 0] == FEASIBLE_SCENES[..., 1]
CURRICULUM_SCENES = FEASIBLE_SCENES[_same.sum(axis=1) == 1].copy()
CURRICULUM_SCENES.flags.writeable = False
LOCAL_SET_ORDER = ["food_food", "food_water_either_order", "water_water"]


def sample_curriculum(rng, n=None):
    """Sample one or n of the eight equally likely asymmetric cases."""
    if n is not None and (not isinstance(n, (int, np.integer)) or n < 0):
        raise ValueError("n must be a nonnegative integer or None")
    return CURRICULUM_SCENES[rng.integers(len(CURRICULUM_SCENES), size=n)].copy()


def _scenes(task):
    if task == "curriculum":
        return CURRICULUM_SCENES
    if task == "full":
        return FEASIBLE_SCENES
    raise ValueError("task must be 'curriculum' or 'full'")


def coordination_bounds(task="curriculum"):
    """Exhaust all decentralized local-category policies, a generous bound.

    IID images within category provide no additional knowledge of a partner.
    Independent private randomness and shared clocks only mix deterministic
    policies, so they cannot improve this expectation. Finite samples can.
    """
    scenes = _scenes(task)
    local_indices = scenes[..., 0] * 2 + scenes[..., 1]
    best = 0
    for pa, pb in itertools.product(itertools.product((0, 1), repeat=4), repeat=2):
        actions = np.column_stack((np.asarray(pa)[local_indices[:, 0]],
                                   np.asarray(pb)[local_indices[:, 1]]))
        selected = np.take_along_axis(scenes, actions[..., None], axis=-1)[..., 0]
        best = max(best, int((selected[:, 0] != selected[:, 1]).sum()))
    return {"task": task, "number_of_scenes": len(scenes),
            "best_decentralized_balanced_scenes": best,
            "no_message_expected_success_upper_bound": best / len(scenes),
            "full_information_expected_success": 1.0,
            "scope": "uniform IID scenes, capacity one, initially empty inventory"}


def _cases(bank, seed, n, task, split, horizon):
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    rng = np.random.default_rng(seed)
    scenes = _scenes(task)
    kinds = scenes[rng.integers(len(scenes), size=n)].copy()
    features, ids = bank.sample(kinds, split, rng)
    inventory = np.zeros((n, 2), dtype=np.int64)
    remaining = horizon - np.arange(n, dtype=np.int64) % horizon
    public = public_tensor(inventory, horizon, horizon)
    public[:, 2] = torch.from_numpy((remaining / horizon).astype(np.float32))
    digest = hashlib.sha256()
    digest.update(task.encode())
    digest.update(split.encode())
    for value in (kinds, ids, inventory, remaining):
        digest.update(np.ascontiguousarray(value, dtype=np.int64).tobytes())
    return kinds, features, ids, inventory, remaining, public, digest.hexdigest()


def _shuffle(sent, remaining, rng):
    """Reassign each sender independently within identical public inputs."""
    delivered = sent.copy()
    for time in np.unique(remaining):
        indices = np.flatnonzero(remaining == time)
        for sender in range(2):
            delivered[indices, sender] = sent[rng.permutation(indices), sender]
    return delivered


def _directions(kinds, success):
    same = kinds[..., 0] == kinds[..., 1]
    result = []
    for sender in range(2):
        receiver = 1 - sender
        mask = same[:, sender] & ~same[:, receiver]
        entry = {"restricted_sender": sender, "mixed_receiver": receiver,
                 "n": int(mask.sum()),
                 "success": float(success[mask].mean()) if mask.any() else None,
                 "by_restricted_resource": []}
        for resource in range(2):
            subset = mask & (kinds[:, sender, 0] == resource)
            entry["by_restricted_resource"].append({
                "resource": resource, "n": int(subset.sum()),
                "success": float(success[subset].mean()) if subset.any() else None})
        result.append(entry)
    return result


@torch.no_grad()
def evaluate(agents, bank, seed, n=8192, task="full", mode="normal", greedy=True,
             split="test", horizon=16, trace=False):
    """Return (metrics, optional per-case rows) for paired IID holdout cases.

    Reusing seed/n/task/split/horizon gives identical external cases in all
    message conditions. Environment, policy, and intervention RNGs are separate;
    message changes cannot perturb later scenes or photos. Stochastic conditions
    also share the same action-sampling uniforms. ``blank`` always delivers symbol
    0: it removes current partner information, but is not assumed behaviorally
    neutral because 0 can have acquired a convention during training.
    """
    if mode not in ("normal", "shuffle", "blank"):
        raise ValueError("mode must be normal, shuffle, or blank")
    if len(agents) != 2:
        raise ValueError("exactly two independent agents are required")
    kinds, features, ids, inventory, remaining, public, case_hash = _cases(
        bank, seed, n, task, split, horizon)
    rep = [agents[i].observe(features[:, i], public) for i in range(2)]
    policy_rng = [np.random.default_rng(seed + 11001 + i) for i in range(2)]
    send_logits = [agents[i].send(rep[i][1]) for i in range(2)]
    send_p = [value.softmax(-1).numpy() for value in send_logits]
    sent = np.column_stack([draw(send_logits[i], policy_rng[i], greedy)[0].numpy()
                            for i in range(2)])
    if mode == "shuffle":
        delivered = _shuffle(sent, remaining, np.random.default_rng(seed + 22001))
    elif mode == "blank":
        delivered = np.zeros_like(sent)
    else:
        delivered = sent.copy()
    action_logits = [agents[i].act(*rep[i], torch.from_numpy(delivered[:, 1 - i]))
                     for i in range(2)]
    action_p = [value.softmax(-1).numpy() for value in action_logits]
    actions = np.column_stack([draw(action_logits[i], policy_rng[i], greedy)[0].numpy()
                               for i in range(2)])
    nxt, rewards, info = transition(inventory, kinds, actions, capacity=1)
    assert np.all(nxt == 0), "IID evaluation assumes empty post-consumption storage"
    success = info["balanced_gathering"]
    table = np.zeros((2, 3, 5), dtype=np.int64)
    probability_table = np.zeros((2, 3, 5), dtype=np.float64)
    diagnostics = []
    for agent in range(2):
        local_set = kinds[:, agent].sum(axis=1)
        np.add.at(table[agent], (local_set, sent[:, agent]), 1)
        for value in range(3):
            mask = local_set == value
            if mask.any():
                probability_table[agent, value] = send_p[agent][mask].mean(axis=0)
        mixed = local_set == 1
        entropy = -(action_p[agent] * np.log(np.clip(action_p[agent], 1e-12, 1))).sum(axis=1)
        sender_entropy = -(send_p[agent] * np.log(np.clip(send_p[agent], 1e-12, 1))).sum(axis=1)
        diagnostics.append({"agent": agent,
                            "sender_entropy_nats": float(sender_entropy.mean()),
                            "mixed_action_entropy_nats": float(entropy[mixed].mean()),
                            "mixed_max_action_probability": float(action_p[agent][mixed].max(axis=1).mean())})
    result = {"task": task, "mode": mode, "split": split, "greedy": greedy,
              "cases": n, "horizon_clock": horizon, "external_cases_sha256": case_hash,
              "balanced_gathering": float(success.mean()),
              "mean_reward_per_step": float(rewards.mean()),
              "shortage_per_step": float(info["shortage"].sum(axis=1).mean()),
              "overflow_per_step": float(info["overflow"].sum(axis=1).mean()),
              "by_direction": _directions(kinds, success),
              "symbols_by_local_resource_set": table.tolist(),
              "mean_symbol_probabilities_by_local_resource_set": probability_table.tolist(),
              "local_set_order": LOCAL_SET_ORDER,
              "policy_diagnostics": diagnostics,
              "communication_channel": "one integer in [0,4]; no partner features or gradients",
              "symbol_table_interpretation": "conditional usage frequencies only; not assigned meanings or evidence of a shared lexicon",
              "bounds": coordination_bounds(task)}
    rows = []
    if trace:
        for i in range(n):
            rows.append({"case": i, "remaining": int(remaining[i]),
                         "inventory": inventory[i].tolist(), "kinds": kinds[i].tolist(),
                         "image_ids": ids[i].tolist(), "sent": sent[i].tolist(),
                         "delivered": delivered[i].tolist(), "actions": actions[i].tolist(),
                         "selected_kinds": info["selected_kinds"][i].tolist(),
                         "reward": float(rewards[i]), "success": bool(success[i])})
    return result, rows


@torch.no_grad()
def message_intervention(agents, bank, seed, n=4096, task="full", split="test", horizon=16):
    """Hold recipient observation and partner action fixed; replace one symbol.

    Report all five replacements and independently observed symbol support.
    Only the restricted-sender/mixed-receiver cases enter the primary effects:
    changing the action on duplicate resources cannot change resource choice.
    These functional responses do not establish identical conventions in the two
    directions. Results use greedy decisions and also retain probability effects.
    """
    kinds, features, ids, inventory, remaining, public, case_hash = _cases(
        bank, seed, n, task, split, horizon)
    rep = [agents[i].observe(features[:, i], public) for i in range(2)]
    sent = [agents[i].send(rep[i][1]).argmax(-1) for i in range(2)]
    original_p = [agents[i].act(*rep[i], sent[1 - i]).softmax(-1).numpy() for i in range(2)]
    original_action = np.column_stack([value.argmax(axis=1) for value in original_p])
    original_selected = np.take_along_axis(kinds, original_action[..., None], axis=-1)[..., 0]
    original_success = original_selected[:, 0] != original_selected[:, 1]
    _, ref_features, _, _, _, ref_public, _ = _cases(bank, seed + 33001, n, task, split, horizon)
    same = kinds[..., 0] == kinds[..., 1]
    directions = []
    for receiver in range(2):
        sender = 1 - receiver
        _, ref_local = agents[sender].observe(ref_features[:, sender], ref_public)
        reference = agents[sender].send(ref_local).argmax(-1).numpy()
        reference_counts = np.bincount(reference, minlength=5)
        minimum = max(5, n // 100)
        relevant = same[:, sender] & ~same[:, receiver]
        effects = []
        all_food_probabilities = []
        all_selected_resources = []
        for symbol in range(5):
            received = torch.full((n,), symbol, dtype=torch.int64)
            changed_p = agents[receiver].act(*rep[receiver], received).softmax(-1).numpy()
            changed_action = changed_p.argmax(axis=1)
            changed_resource = kinds[np.arange(n), receiver, changed_action]
            changed_success = changed_resource != original_selected[:, sender]
            food_probability = (changed_p * (kinds[:, receiver] == 0)).sum(axis=1)
            all_food_probabilities.append(food_probability)
            all_selected_resources.append(changed_resource)
            mask = relevant & (sent[sender].numpy() != symbol)
            def mean(value):
                return float(value[mask].mean()) if mask.any() else None
            effects.append({"replacement_symbol": symbol,
                            "reference_count": int(reference_counts[symbol]),
                            "in_observed_sender_support": bool(reference_counts[symbol] >= minimum),
                            "n_changed_relevant_cases": int(mask.sum()),
                            "choice_flip_rate": mean(changed_action != original_action[:, receiver]),
                            "resource_flip_rate": mean(changed_resource != original_selected[:, receiver]),
                            "mean_action_total_variation": mean(.5 * np.abs(changed_p - original_p[receiver]).sum(axis=1)),
                            "baseline_success": mean(original_success),
                            "intervened_success": mean(changed_success),
                            "success_change": mean(changed_success.astype(float) - original_success.astype(float)),
                            "mixed_recipient_mean_food_probability": float(food_probability[~same[:, receiver]].mean())})
        used = np.flatnonzero(reference_counts >= minimum)
        if len(used) >= 2 and relevant.any():
            probability_range = np.ptp(np.stack(all_food_probabilities)[used][:, relevant], axis=0)
            resource_range = np.ptp(np.stack(all_selected_resources)[used][:, relevant], axis=0)
            sensitivity = {"n": int(relevant.sum()), "symbols": used.tolist(),
                           "mean_food_probability_range": float(probability_range.mean()),
                           "fraction_cases_resource_changes_for_some_used_symbol": float((resource_range > 0).mean())}
        else:
            sensitivity = {"n": int(relevant.sum()), "symbols": used.tolist(),
                           "mean_food_probability_range": None,
                           "fraction_cases_resource_changes_for_some_used_symbol": None}
        directions.append({"sender": sender, "receiver": receiver,
                           "restricted_sender_mixed_receiver_cases": int(relevant.sum()),
                           "reference_counts": reference_counts.tolist(), "reference_min_count": minimum,
                           "interventions": effects, "observed_symbol_sensitivity": sensitivity})
    return {"task": task, "split": split, "cases": n, "greedy": True,
            "external_cases_sha256": case_hash, "directions": directions,
            "interpretation": "causal effect of replacing the recipient's discrete input at identical observations; not a translation dictionary or proof of a shared language"}
