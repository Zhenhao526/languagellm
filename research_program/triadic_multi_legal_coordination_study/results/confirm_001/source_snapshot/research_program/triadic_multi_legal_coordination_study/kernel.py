"""Strict objective that permits multiple full-success plans per world."""
from __future__ import annotations

import numbers

import numpy as np

from research_program.triadic_message_study import runner as core

base = core.base
JOINT = base.JOINT_ACTIONS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value, message):
    require(np.isfinite(value).all(), message)


def _native_table(rewards):
    native = np.asarray(rewards, dtype=np.float64)
    require(native.ndim == 2 and native.shape[1] == 24 and len(native) > 0, 'Native reward table must be [B,24]')
    finite(native, 'Native rewards')
    require(np.isin(native, (0., .5, 1.)).all(), 'Native rewards must be 0, .5 or 1')
    require((native == 1).sum(axis=1).min() >= 1, 'Every world needs a full-success plan')
    return native


def action_conditioned_rewards(probabilities, rewards):
    p = np.asarray(probabilities, dtype=np.float64); native = _native_table(rewards)
    require(p.shape == (len(native), 3, 17), 'Invalid probability shape')
    value = np.zeros_like(p)
    for actor in range(3):
        remainder = np.ones((len(native), 24), dtype=np.float64)
        for other in range(3):
            if other != actor:
                remainder *= p[:, other, JOINT[:, other]]
        weighted = remainder * native
        for event in range(24):
            value[:, actor, JOINT[event, actor]] += weighted[:, event]
    finite(value, 'Action-conditioned rewards'); require(((value >= 0) & (value <= 1 + 1e-12)).all(), 'Reward range')
    return value


def objective_terms(logits, rewards):
    logits = np.asarray(logits, dtype=np.float64); native = _native_table(rewards)
    require(logits.shape == (len(native), 3, 17), 'Incorrect logit/reward shape'); finite(logits, 'Logits')
    p, lp = base.policy_distribution(logits)
    joint = np.ones((len(native), 24), dtype=np.float64)
    for actor in range(3):
        joint *= p[:, actor, JOINT[:, actor]]
    weighted = joint * native; J = weighted.sum(axis=1)
    mean_gradient = -p * J[:, None, None]
    for event in range(24):
        for actor in range(3):
            mean_gradient[:, actor, JOINT[event, actor]] += weighted[:, event]
    positive = native > 0
    log_mass = np.full(native.shape, -np.inf, dtype=np.float64); log_mass[positive] = np.log(native[positive])
    for actor in range(3):
        log_mass += lp[:, actor, JOINT[:, actor]]
    maximum = log_mass.max(axis=1, keepdims=True); finite(maximum, 'Largest rewarding-event log mass')
    shifted = np.exp(log_mass - maximum); normalizer = shifted.sum(axis=1, keepdims=True)
    posterior = shifted / normalizer; log_J = (maximum + np.log(normalizer))[:, 0]
    finite(log_J, 'Stable log reward'); finite(posterior, 'Reward posterior')
    require((log_J <= 1e-12).all() and (posterior[~positive] == 0).all(), 'Invalid posterior support')
    log_gradient = -p.copy()
    for event in range(24):
        for actor in range(3):
            log_gradient[:, actor, JOINT[event, actor]] += posterior[:, event]
    finite(mean_gradient, 'Mean reward gradient'); finite(log_gradient, 'Log reward gradient')
    return dict(probabilities=p, log_probabilities=lp, J=J, log_J=log_J,
                mean_J_logit_gradient=mean_gradient, log_J_logit_gradient=log_gradient,
                posterior_weights=posterior, native_expected_reward=(joint * native).sum(axis=1),
                full_success_probability=(joint * (native == 1)).sum(axis=1),
                partial_success_probability=(joint * (native == .5)).sum(axis=1),
                full_success_posterior_mass=(posterior * (native == 1)).sum(axis=1),
                partial_success_posterior_mass=(posterior * (native == .5)).sum(axis=1),
                utility_values=native, action_conditioned_expected_reward=action_conditioned_rewards(p, native))


def training_gradients(networks, observations, rewards, live, uniforms, update):
    x = np.asarray(observations, dtype=np.float64); native = _native_table(rewards)
    require(x.shape == (len(native), 3, 54), 'Incorrect observations'); finite(x, 'Observations')
    require(isinstance(live, (bool, np.bool_)), 'live must be boolean')
    require(isinstance(update, numbers.Integral) and not isinstance(update, (bool, np.bool_)) and 1 <= int(update) <= 6000, 'Invalid update')
    B = len(x); uniforms = np.asarray(uniforms, dtype=np.float64)
    require(uniforms.shape == (2, B, 2, 3, 4) and np.isfinite(uniforms).all() and ((uniforms >= 0) & (uniforms < 1)).all(), 'Invalid trajectory uniforms')
    trace = core.rollout(networks, np.concatenate((x, x)), live, uniforms.reshape(2 * B, 2, 3, 4))
    terms = objective_terms(trace['action_logits'], np.concatenate((native, native)))
    receiver = core.coordination.loss_and_derivative(terms, 'mean_log_J', update)
    entropy, _ = base.entropy_and_logit_gradient(terms['probabilities'], terms['log_probabilities'])
    F = terms['log_J'] + receiver['entropy_coefficient'] * entropy
    sender = core.paired_sender_derivative(trace['sender_probabilities'].reshape(2, B, 2, 3, 4, 8),
        trace['sender_log_probabilities'].reshape(2, B, 2, 3, 4, 8), trace['messages'].reshape(2, B, 2, 3, 4), F.reshape(2, B))
    ds = sender['derivative'].reshape(2 * B, 2, 3, 4, 8); gradients = []
    for actor in range(3):
        for module in range(3):
            derivative = receiver['derivative'][:, actor] if module == 2 else ds[:, module, actor].reshape(2 * B, 32)
            gradients.append(base.actor_backward(networks[3 * actor + module], trace['caches'][3 * actor + module], derivative))
    row = dict(partial_success_utility=.5, mean_expected_utility=float(terms['J'].mean()),
        mean_log_expected_utility=float(terms['log_J'].mean()), min_log_expected_utility=float(terms['log_J'].min()),
        max_log_expected_utility=float(terms['log_J'].max()), zero_float_expected_utility_states=int((terms['J'] == 0).sum()),
        mean_native_expected_reward=float(terms['native_expected_reward'].mean()),
        mean_full_success_probability=float(terms['full_success_probability'].mean()),
        mean_partial_success_probability=float(terms['partial_success_probability'].mean()),
        mean_full_success_posterior_mass=float(terms['full_success_posterior_mass'].mean()),
        mean_partial_success_posterior_mass=float(terms['partial_success_posterior_mass'].mean()),
        mean_F=float(F.mean()), receiver_loss=receiver['loss'], mean_actor_entropy=receiver['mean_actor_entropy'],
        entropy_coefficient=receiver['entropy_coefficient'], sender_surrogate_loss=sender['surrogate_loss'],
        sender_advantage_mean=float(sender['advantage'].mean()), sender_advantage_abs_mean=float(np.abs(sender['advantage']).mean()),
        sender_advantage_squared_mean=float(np.square(sender['advantage']).mean()), sender_advantage_max_abs=float(np.abs(sender['advantage']).max()),
        sender_mean_complete_log_score=float(sender['log_scores'].mean()), sampled_messages_sha256=core.base.array_sha(trace['messages']))
    return gradients, row
