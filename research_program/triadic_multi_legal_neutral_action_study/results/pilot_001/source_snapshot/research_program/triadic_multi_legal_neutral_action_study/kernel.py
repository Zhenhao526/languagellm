"""Exact receiver and sampled-sender gradients with an all-cancel event."""
from __future__ import annotations

import numbers

import numpy as np

from . import core18

base = core18.base
JOINT_ACTIONS = np.vstack((base.JOINT_ACTIONS, np.asarray([[17, 17, 17]], dtype=np.int64)))
require = base.require
finite = base.finite


def _native_table(rewards):
    native = np.asarray(rewards, dtype=np.float64)
    require(native.ndim == 2 and native.shape[1] == 24 and len(native) > 0, 'Native reward table must be [B,24]')
    finite(native, 'Native rewards')
    require(np.isin(native, (0., .5, 1.)).all() and (native == 1).sum(axis=1).min() >= 1,
            'Native rewards must contain at least one full plan per world')
    require((native == 1).sum(axis=1).max() == 2, 'Neutral study requires exactly two full plans per world')
    return native


def objective_terms(logits, rewards, cancel_reward):
    logits = np.asarray(logits, dtype=np.float64); native = _native_table(rewards)
    cancel_reward = float(cancel_reward)
    require(logits.shape == (len(native), 3, 18), 'Incorrect cancel-enabled logit shape')
    require(np.isfinite(cancel_reward) and 0 <= cancel_reward <= 1, 'Invalid cancel reward')
    finite(logits, 'Logits')
    p, lp = base.policy_distribution(logits)
    joint = np.ones((len(native), 25), dtype=np.float64)
    for actor in range(3):
        joint *= p[:, actor, JOINT_ACTIONS[:, actor]]
    event_rewards = np.concatenate((native, np.full((len(native), 1), cancel_reward, dtype=np.float64)), axis=1)
    weighted = joint * event_rewards
    J = weighted.sum(axis=1)
    mean_gradient = -p * J[:, None, None]
    for event in range(25):
        for actor in range(3):
            mean_gradient[:, actor, JOINT_ACTIONS[event, actor]] += weighted[:, event]
    positive = event_rewards > 0
    log_mass = np.full(event_rewards.shape, -np.inf, dtype=np.float64)
    log_mass[positive] = np.log(event_rewards[positive])
    for actor in range(3):
        log_mass += lp[:, actor, JOINT_ACTIONS[:, actor]]
    maximum = log_mass.max(axis=1, keepdims=True); finite(maximum, 'Largest log reward mass')
    shifted = np.exp(log_mass - maximum); normalizer = shifted.sum(axis=1, keepdims=True)
    posterior = shifted / normalizer; log_J = (maximum + np.log(normalizer))[:, 0]
    finite(log_J, 'Stable log reward'); finite(posterior, 'Reward posterior')
    require((log_J <= 1e-12).all() and (posterior[~positive] == 0).all(), 'Invalid reward posterior')
    log_gradient = -p.copy()
    for event in range(25):
        for actor in range(3):
            log_gradient[:, actor, JOINT_ACTIONS[event, actor]] += posterior[:, event]
    finite(mean_gradient, 'Mean reward gradient'); finite(log_gradient, 'Log reward gradient')
    return dict(probabilities=p, log_probabilities=lp, J=J, log_J=log_J,
                mean_J_logit_gradient=mean_gradient, log_J_logit_gradient=log_gradient,
                posterior_weights=posterior, native_expected_reward=(joint[:, :24] * native).sum(axis=1),
                cancel_expected_reward=cancel_reward * joint[:, 24],
                full_success_probability=(joint[:, :24] * (native == 1)).sum(axis=1),
                partial_success_probability=(joint[:, :24] * (native == .5)).sum(axis=1),
                cancel_protocol_probability=joint[:, 24], utility_values=event_rewards)


def loss_and_derivative(terms, update):
    entropy, entropy_gradient = base.entropy_and_logit_gradient(terms['probabilities'], terms['log_probabilities'])
    beta = base.entropy_coefficient(update)
    loss = -float(terms['log_J'].mean() + beta * entropy.mean())
    derivative = -(terms['log_J_logit_gradient'] + beta * entropy_gradient) / len(terms['log_J'])
    finite(derivative, 'Loss derivative'); require(np.isfinite(loss), 'Nonfinite loss')
    return dict(loss=loss, derivative=derivative, mean_actor_entropy=float(entropy.mean()),
                entropy_coefficient=beta, selected_objective_mean=float(terms['log_J'].mean()))


def training_gradients(networks, observations, rewards, cancel_reward, live, uniforms, update):
    x = np.asarray(observations, dtype=np.float64); native = _native_table(rewards)
    require(x.shape == (len(native), 3, 54) and isinstance(live, (bool, np.bool_)), 'Invalid training inputs')
    require(isinstance(update, numbers.Integral) and not isinstance(update, (bool, np.bool_)) and 1 <= int(update) <= 6000,
            'Invalid update')
    B = len(x); uniforms = np.asarray(uniforms, dtype=np.float64)
    require(uniforms.shape == (2, B, 2, 3, 4) and np.isfinite(uniforms).all() and ((uniforms >= 0) & (uniforms < 1)).all(),
            'Invalid trajectory uniforms')
    trace = core18.rollout(networks, np.concatenate((x, x)), live, uniforms.reshape(2 * B, 2, 3, 4))
    terms = objective_terms(trace['action_logits'], np.concatenate((native, native)), cancel_reward)
    receiver = loss_and_derivative(terms, update)
    entropy, _ = base.entropy_and_logit_gradient(terms['probabilities'], terms['log_probabilities'])
    F = terms['log_J'] + receiver['entropy_coefficient'] * entropy
    sender = core18.paired_sender_derivative(trace['sender_probabilities'].reshape(2, B, 2, 3, 4, 8),
        trace['sender_log_probabilities'].reshape(2, B, 2, 3, 4, 8), trace['messages'].reshape(2, B, 2, 3, 4), F.reshape(2, B))
    ds = sender['derivative'].reshape(2 * B, 2, 3, 4, 8); gradients = []
    for actor in range(3):
        for module in range(3):
            derivative = receiver['derivative'][:, actor] if module == 2 else ds[:, module, actor].reshape(2 * B, 32)
            gradients.append(base.actor_backward(networks[3 * actor + module], trace['caches'][3 * actor + module], derivative))
    row = dict(partial_success_utility=.5, cancel_reward=cancel_reward,
        mean_expected_utility=float(terms['J'].mean()), mean_log_expected_utility=float(terms['log_J'].mean()),
        min_log_expected_utility=float(terms['log_J'].min()), max_log_expected_utility=float(terms['log_J'].max()),
        zero_float_expected_utility_states=int((terms['J'] == 0).sum()), mean_native_expected_reward=float(terms['native_expected_reward'].mean()),
        mean_cancel_expected_reward=float(terms['cancel_expected_reward'].mean()), mean_full_success_probability=float(terms['full_success_probability'].mean()),
        mean_partial_success_probability=float(terms['partial_success_probability'].mean()), mean_cancel_protocol_probability=float(terms['cancel_protocol_probability'].mean()),
        mean_F=float(F.mean()), receiver_loss=receiver['loss'], mean_actor_entropy=receiver['mean_actor_entropy'],
        entropy_coefficient=receiver['entropy_coefficient'], sender_surrogate_loss=sender['surrogate_loss'],
        sender_advantage_mean=float(sender['advantage'].mean()), sender_advantage_abs_mean=float(np.abs(sender['advantage']).mean()),
        sender_advantage_squared_mean=float(np.square(sender['advantage']).mean()), sender_advantage_max_abs=float(np.abs(sender['advantage']).max()),
        sender_mean_complete_log_score=float(sender['log_scores'].mean()), sampled_messages_sha256=core18.array_sha(trace['messages']))
    return gradients, row
