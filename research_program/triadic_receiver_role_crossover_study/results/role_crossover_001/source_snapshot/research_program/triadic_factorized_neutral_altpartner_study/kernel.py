"""Exact policy-gradient kernel for a factorized neutral/engage action head."""
from __future__ import annotations

import numbers

import numpy as np

from research_program.triadic_message_study import runner as core

base = core.base
JOINT_ACTIONS = np.asarray(base.JOINT_ACTIONS, dtype=np.int64)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value, message):
    require(np.isfinite(value).all(), message)


def _rewards(rewards):
    native = np.asarray(rewards, dtype=np.float64)
    require(native.ndim == 2 and native.shape[1] == 24 and len(native) > 0,
            'Native reward table must be [B,24]')
    finite(native, 'Native rewards')
    require(np.isin(native, (0., .5, 1.)).all(), 'Native rewards must be 0, .5 or 1')
    require((native == 1).sum(axis=1).min() == 2, 'Every world must have exactly two full plans')
    return native


def factorized_distribution(logits):
    logits = np.asarray(logits, dtype=np.float64)
    require(logits.ndim == 3 and logits.shape[1:] == (3, 18),
            'Factorized action logits must be [B,3,18]')
    finite(logits, 'Factorized action logits')
    intent_p, intent_lp = base.policy_distribution(logits[..., :2])
    proposal_p, proposal_lp = base.policy_distribution(logits[..., 2:])
    return intent_p, intent_lp, proposal_p, proposal_lp


def _event_log_prob(intent_lp, proposal_lp):
    b = len(intent_lp)
    result = np.zeros((b, 24), dtype=np.float64)
    for actor in range(3):
        action = JOINT_ACTIONS[:, actor]
        engaged = action > 0
        result[:, engaged] += intent_lp[:, actor, 1][:, None]
        result[:, ~engaged] += intent_lp[:, actor, 0][:, None]
        if engaged.any():
            result[:, engaged] += proposal_lp[:, actor, action[engaged] - 1]
    return result


def objective_terms(logits, rewards):
    """Stable mean-log expected native reward and factorized gradients."""
    native = _rewards(rewards)
    intent_p, intent_lp, proposal_p, proposal_lp = factorized_distribution(logits)
    log_event = _event_log_prob(intent_lp, proposal_lp)
    log_mass = np.full(native.shape, -np.inf, dtype=np.float64)
    positive = native > 0
    log_mass[positive] = np.log(native[positive])
    log_mass += log_event
    maximum = log_mass.max(axis=1, keepdims=True)
    finite(maximum, 'Largest rewarding event log mass')
    shifted = np.exp(log_mass - maximum)
    normalizer = shifted.sum(axis=1, keepdims=True)
    posterior = shifted / normalizer
    log_J = (maximum + np.log(normalizer))[:, 0]
    event_prob = np.exp(log_event)
    J = (event_prob * native).sum(axis=1)
    finite(J, 'Expected native reward'); finite(log_J, 'Stable log expected reward')
    require((J >= 0).all() and (J <= 1 + 1e-12).all() and (posterior[~positive] == 0).all(),
            'Invalid native reward objective')

    intent_posterior = np.zeros((len(native), 3, 2), dtype=np.float64)
    proposal_posterior = np.zeros((len(native), 3, 16), dtype=np.float64)
    engage_posterior = np.zeros((len(native), 3), dtype=np.float64)
    for event in range(24):
        for actor in range(3):
            action = int(JOINT_ACTIONS[event, actor])
            if action == 0:
                intent_posterior[:, actor, 0] += posterior[:, event]
            else:
                intent_posterior[:, actor, 1] += posterior[:, event]
                proposal_posterior[:, actor, action - 1] += posterior[:, event]
                engage_posterior[:, actor] += posterior[:, event]
    intent_log_gradient = intent_posterior - intent_p
    proposal_log_gradient = proposal_posterior - engage_posterior[:, :, None] * proposal_p
    finite(intent_log_gradient, 'Intent log-gradient'); finite(proposal_log_gradient, 'Proposal log-gradient')
    full_prob = (event_prob * (native == 1)).sum(axis=1)
    partial_prob = (event_prob * (native == .5)).sum(axis=1)
    return dict(intent_probabilities=intent_p, intent_log_probabilities=intent_lp,
                proposal_probabilities=proposal_p, proposal_log_probabilities=proposal_lp,
                event_probabilities=event_prob, J=J, log_J=log_J,
                posterior_weights=posterior, intent_log_gradient=intent_log_gradient,
                proposal_log_gradient=proposal_log_gradient,
                full_success_probability=full_prob, partial_success_probability=partial_prob,
                engagement_probabilities=intent_p[..., 1], native_rewards=native)


def training_gradients(networks, observations, rewards, live, uniforms, update):
    x = np.asarray(observations, dtype=np.float64); native = _rewards(rewards)
    require(x.shape == (len(native), 3, 54), 'Incorrect observations')
    finite(x, 'Observations')
    require(isinstance(live, (bool, np.bool_)), 'live must be boolean')
    require(isinstance(update, numbers.Integral) and not isinstance(update, (bool, np.bool_))
            and 1 <= int(update) <= 6000, 'Invalid update')
    uniforms = np.asarray(uniforms, dtype=np.float64)
    require(uniforms.shape == (2, len(x), 2, 3, 4) and np.isfinite(uniforms).all()
            and ((uniforms >= 0) & (uniforms < 1)).all(), 'Invalid trajectory uniforms')
    trace = core.rollout(networks, np.concatenate((x, x)), bool(live),
                         uniforms.reshape(2 * len(x), 2, 3, 4))
    terms = objective_terms(trace['action_logits'], np.concatenate((native, native)))
    entropy_i, entropy_grad_i = base.entropy_and_logit_gradient(
        terms['intent_probabilities'], terms['intent_log_probabilities'])
    entropy_p, entropy_grad_p = base.entropy_and_logit_gradient(
        terms['proposal_probabilities'], terms['proposal_log_probabilities'])
    beta = base.entropy_coefficient(update)
    entropy = entropy_i + entropy_p
    F = terms['log_J'] + beta * entropy
    sender = core.paired_sender_derivative(
        trace['sender_probabilities'].reshape(2, len(x), 2, 3, 4, 8),
        trace['sender_log_probabilities'].reshape(2, len(x), 2, 3, 4, 8),
        trace['messages'].reshape(2, len(x), 2, 3, 4), F.reshape(2, len(x)))
    ds = sender['derivative'].reshape(2 * len(x), 2, 3, 4, 8)
    d_intent = -(terms['intent_log_gradient'] + beta * entropy_grad_i) / len(terms['J'])
    d_proposal = -(terms['proposal_log_gradient'] + beta * entropy_grad_p) / len(terms['J'])
    gradients = []
    for actor in range(3):
        for module in range(3):
            if module == 2:
                derivative = np.concatenate((d_intent[:, actor], d_proposal[:, actor]), axis=-1)
            else:
                derivative = ds[:, module, actor].reshape(2 * len(x), 32)
            gradients.append(base.actor_backward(networks[3 * actor + module],
                                                   trace['caches'][3 * actor + module], derivative))
    row = dict(mean_expected_utility=float(terms['J'].mean()), mean_log_expected_utility=float(terms['log_J'].mean()),
               min_log_expected_utility=float(terms['log_J'].min()), max_log_expected_utility=float(terms['log_J'].max()),
               zero_expected_utility_states=int((terms['J'] == 0).sum()),
               mean_full_success_probability=float(terms['full_success_probability'].mean()),
               mean_partial_success_probability=float(terms['partial_success_probability'].mean()),
               mean_engagement_probability=float(terms['engagement_probabilities'].mean()),
               mean_F=float(F.mean()), receiver_loss=-float((terms['log_J'] + beta * entropy).mean()),
               mean_actor_entropy=float(entropy.mean()), intent_entropy=float(entropy_i.mean()), proposal_entropy=float(entropy_p.mean()),
               entropy_coefficient=beta, sender_surrogate_loss=sender['surrogate_loss'],
               sender_advantage_mean=float(sender['advantage'].mean()), sender_advantage_abs_mean=float(np.abs(sender['advantage']).mean()),
               sender_advantage_squared_mean=float(np.square(sender['advantage']).mean()), sender_advantage_max_abs=float(np.abs(sender['advantage']).max()),
               sender_mean_complete_log_score=float(sender['log_scores'].mean()), sampled_messages_sha256=core.base.array_sha(trace['messages']))
    return gradients, row


if __name__ == '__main__':
    print('factorized neutral/engage kernel')
