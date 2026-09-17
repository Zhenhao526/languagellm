"""Exact optimistic bounds for frozen communication functions on finite photos.

These diagnostics never train a policy. True resource labels enter the oracle
only, not any deployed agent. The 8 of 14 one-restricted scenes supply the bound;
the remaining 6 scenes are optimistically granted perfect success.
"""
import itertools

import numpy as np
import torch

from contact_eval import ORIGINAL_PAIRS, ALL_PAIRS


GROUPS = {'original': ORIGINAL_PAIRS,
          'cross': tuple(p for p in ALL_PAIRS if p not in ORIGINAL_PAIRS),
          'all': ALL_PAIRS}


def graph_partners(pairs):
    return [[j if i == who else i for i, j in pairs if who in (i, j)] for who in range(4)]


def sender_information_bound(distributions, pairs):
    """Input agent,time,true_resource,symbol probabilities; equal graph edges."""
    p = np.asarray(distributions, dtype=np.float64)
    assert p.ndim == 4 and p.shape[0] == 4 and p.shape[2:] == (2, 5)
    values, weights = [], []
    for receiver, partners in enumerate(graph_partners(pairs)):
        mixture = p[partners].mean(0)
        values.append(.5 * np.maximum(mixture[:, 0], mixture[:, 1]).sum(-1))
        weights.append(len(partners))
    per_receiver = np.asarray(values)
    restricted = float(np.average(per_receiver.mean(-1), weights=weights))
    return {'one_restricted_success_upper_bound': restricted,
            'full_success_upper_bound': 3 / 7 + 4 / 7 * restricted,
            'by_receiver_time': per_receiver.tolist(),
            'partner_sets': graph_partners(pairs)}


def listener_compatibility_bound(resource_probabilities, pairs):
    """Input receiver,time,symbol,selected_resource on mixed private pictures."""
    p = np.asarray(resource_probabilities, dtype=np.float64)
    assert p.ndim == 4 and p.shape[0] == 4 and p.shape[2:] == (5, 2)
    values, best, weights = [], [], []
    for sender, partners in enumerate(graph_partners(pairs)):
        mixed = p[partners].mean(0)
        # q is the sender's restricted resource; receiver should choose 1-q.
        desired = mixed[:, :, [1, 0]]
        values.append(desired.max(1))
        best.append(desired.argmax(1))
        weights.append(len(partners))
    per_sender = np.asarray(values)
    restricted = float(np.average(per_sender.mean((1, 2)), weights=weights))
    return {'one_restricted_success_upper_bound': restricted,
            'full_success_upper_bound': 3 / 7 + 4 / 7 * restricted,
            'by_sender_time_resource': per_sender.tolist(),
            'best_symbol_by_sender_time_resource': np.asarray(best).tolist(),
            'partner_sets': graph_partners(pairs)}


@torch.no_grad()
def frozen_function_bounds(agents, bank, split='test', horizon=16):
    pools = [bank.pools[split, q] for q in range(2)]
    restricted_ids = [np.asarray(list(itertools.product(pool, repeat=2)), dtype=np.int64) for pool in pools]
    mixed = np.asarray(list(itertools.product(pools[0], pools[1])), dtype=np.int64)
    mixed_ids = np.concatenate((mixed, mixed[:, ::-1]), axis=0)
    mixed_kinds = torch.from_numpy(np.concatenate((np.tile([0, 1], (len(mixed), 1)),
                                                  np.tile([1, 0], (len(mixed), 1))), axis=0))
    sender = {mode: np.empty((4, horizon, 2, 5), dtype=np.float64) for mode in ('normal', 'stochastic')}
    listener = {mode: np.empty((4, horizon, 5, 2), dtype=np.float64) for mode in ('normal', 'stochastic')}
    for who, agent in enumerate(agents):
        for tick in range(horizon):
            remaining = (tick + 1) / horizon
            for q, ids in enumerate(restricted_ids):
                public = torch.zeros(len(ids), 3)
                public[:, 2] = remaining
                _, local = agent.observe(bank.features[torch.from_numpy(ids)], public)
                probability = agent.send(local).softmax(-1)
                sender['stochastic'][who, tick, q] = probability.double().mean(0).numpy()
                sender['normal'][who, tick, q] = torch.nn.functional.one_hot(probability.argmax(-1), 5).double().mean(0).numpy()
            public = torch.zeros(len(mixed_ids), 3)
            public[:, 2] = remaining
            reps = agent.observe(bank.features[torch.from_numpy(mixed_ids)], public)
            for symbol in range(5):
                p = agent.act(*reps, torch.full((len(mixed_ids),), symbol, dtype=torch.int64)).softmax(-1)
                action = p.argmax(-1)
                selected = mixed_kinds.gather(1, action[:, None]).squeeze(1)
                listener['normal'][who, tick, symbol] = torch.nn.functional.one_hot(selected, 2).double().mean(0).numpy()
                listener['stochastic'][who, tick, symbol] = torch.stack([
                    (p * (mixed_kinds == q)).sum(-1).double().mean() for q in range(2)]).numpy()
    # Normalise softmax roundoff; these distributions represent the mathematical
    # policy probabilities, not float32 inverse-CDF boundary artefacts.
    for bank_of_probabilities in (sender, listener):
        for mode, p in bank_of_probabilities.items():
            bank_of_probabilities[mode] = p / p.sum(-1, keepdims=True)
    return {'schema_version': 1, 'split': split, 'horizon': horizon,
            'photo_ids_by_resource': [pool.tolist() for pool in pools],
            'restricted_ordered_photo_pairs_by_resource': [len(ids) for ids in restricted_ids],
            'mixed_ordered_photo_pairs': len(mixed_ids),
            'enumeration': 'All ordered image pairs with replacement and every public time; no Monte Carlo sampling.',
            'scope': 'Upper bounds for uniform feasible scenes on this finite image distribution, no partner ID, one shared conditional policy per agent. Other six of fourteen scenes granted perfect success. Optimistic resource oracle is analysis only; bounds need not be attainable by this architecture or jointly across graph groups.',
            'fixed_sender': {mode: {group: sender_information_bound(p, pairs) for group, pairs in GROUPS.items()}
                             for mode, p in sender.items()},
            'fixed_listener': {mode: {group: listener_compatibility_bound(p, pairs) for group, pairs in GROUPS.items()}
                               for mode, p in listener.items()},
            'sender_distributions_agent_time_resource_symbol': {m: p.tolist() for m, p in sender.items()},
            'listener_resource_probabilities_agent_time_symbol_resource': {m: p.tolist() for m, p in listener.items()}}
