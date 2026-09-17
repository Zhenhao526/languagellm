"""Two-window message rollout with an 18-way action head.

The first 17 actions are inherited unchanged (wait plus sixteen transport
proposals); index 17 is the explicit cancel action.  Sender modules and the
causal routing are byte-for-byte conceptually the same as the audited message
runner, while checkpoint loading is local so a 17-way checkpoint can never be
silently accepted.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from research_program.triadic_message_study import runner as message_core

base = message_core.base
MODULES = ('sender1', 'sender2', 'action')
DIMENSIONS = {'sender1': (54, 64, 64, 32), 'sender2': (153, 64, 64, 32), 'action': (252, 64, 64, 18)}

require = base.require
finite = base.finite
policy_distribution = base.policy_distribution
actor_forward = base.actor_forward
actor_backward = base.actor_backward
make_adam = base.make_adam
adam_step = base.adam_step
entropy_and_logit_gradient = base.entropy_and_logit_gradient
entropy_coefficient = base.entropy_coefficient
draw_uniforms = message_core.draw_uniforms
categorical_tokens = message_core.categorical_tokens
routed_window = message_core.routed_window
paired_sender_derivative = message_core.paired_sender_derivative


def now():
    return base.now()


def sha(path):
    return base.sha(Path(path))


def json_bytes(value):
    return base.json_bytes(value)


def array_sha(value):
    return base.array_sha(value)


def json_hash(value):
    return base.json_hash(value)


def make_network(seed, dimensions):
    rng = np.random.default_rng(seed)
    network = {}
    for layer, (left, right) in enumerate(zip(dimensions, dimensions[1:]), 1):
        network[f'W{layer}'] = rng.normal(0, math.sqrt(2 / (left + right)), (left, right))
        network[f'b{layer}'] = np.zeros(right, dtype=np.float64)
    return network


def make_networks(seed):
    networks = [make_network(np.random.SeedSequence([seed, a, m, 180]), DIMENSIONS[module])
                for a in range(3) for m, module in enumerate(MODULES)]
    for i, j in __import__('itertools').combinations(range(9), 2):
        require(all(not np.shares_memory(networks[i][key], networks[j][key]) for key in networks[i]),
                'Network parameters are shared')
    return networks


def rollout(networks, observations, live, uniforms=None):
    x = np.asarray(observations, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1:] == (3, 54) and len(networks) == 9, 'Invalid rollout inputs')
    finite(x, 'observations')
    if uniforms is not None:
        require(np.shape(uniforms) == (len(x), 2, 3, 4), 'Wrong full message trajectory uniforms')
    caches = [None] * 9
    sender_logits, sender_probabilities, sender_log_probabilities, messages = [], [], [], []
    inputs = x
    for window in range(2):
        logits = []
        for a in range(3):
            z, caches[3 * a + window] = actor_forward(networks[3 * a + window], inputs[:, a])
            logits.append(z.reshape(len(x), 4, 8))
        z = np.stack(logits, axis=1)
        p, lp = policy_distribution(z)
        m = categorical_tokens(p, None if uniforms is None else uniforms[:, window])
        sender_logits.append(z); sender_probabilities.append(p); sender_log_probabilities.append(lp); messages.append(m)
        if window == 0:
            inputs = np.concatenate((x, routed_window(m, live)), axis=-1)
    action_inputs = np.concatenate((x, routed_window(messages[0], live), routed_window(messages[1], live)), axis=-1)
    action_logits = []
    for a in range(3):
        z, caches[3 * a + 2] = actor_forward(networks[3 * a + 2], action_inputs[:, a])
        require(z.shape == (len(x), 18), 'Cancel-enabled action head must have 18 logits')
        action_logits.append(z)
    return {'messages': np.stack(messages, axis=1), 'sender_logits': np.stack(sender_logits, axis=1),
            'sender_probabilities': np.stack(sender_probabilities, axis=1),
            'sender_log_probabilities': np.stack(sender_log_probabilities, axis=1),
            'action_logits': np.stack(action_logits, axis=1), 'action_inputs': action_inputs, 'caches': caches}


def save_checkpoint(path, networks, optimizer, update, batch_rng, message_rngs):
    require(not Path(path).exists(), 'Checkpoint exists')
    payload = {f'agent{a}_{module}_{key}': value for a in range(3) for m, module in enumerate(MODULES)
               for key, value in networks[3 * a + m].items()}
    for a in range(3):
        for m, module in enumerate(MODULES):
            payload.update({f'adam_agent{a}_{module}_{moment}_{key}': value
                            for moment in ('m', 'v') for key, value in optimizer[3 * a + m][moment].items()})
    payload['update'] = np.array(update, dtype=np.int64)
    payload['batch_rng_json'] = np.array(json.dumps(batch_rng.bit_generator.state, sort_keys=True))
    payload['message_rngs_json'] = np.array(json.dumps({key: rng.bit_generator.state for key, rng in message_rngs.items()}, sort_keys=True))
    np.savez_compressed(path, **payload)
    return sha(path)


def load_networks(path):
    networks = []
    with np.load(path, allow_pickle=False) as saved:
        for a in range(3):
            for module in MODULES:
                dimensions = DIMENSIONS[module]; network = {}
                for layer, (left, right) in enumerate(zip(dimensions, dimensions[1:]), 1):
                    for key, shape in ((f'W{layer}', (left, right)), (f'b{layer}', (right,))):
                        name = f'agent{a}_{module}_{key}'
                        require(name in saved, 'Missing checkpoint parameter ' + name)
                        value = saved[name].copy()
                        require(value.shape == shape and value.dtype == np.float64, 'Saved network shape/dtype mismatch')
                        finite(value, 'loaded parameter'); network[key] = value
                networks.append(network)
    return networks


def parameter_hash(networks):
    return json_hash({f'agent{a}_{module}_{key}': array_sha(value)
                      for a in range(3) for m, module in enumerate(MODULES)
                      for key, value in networks[3 * a + m].items()})
