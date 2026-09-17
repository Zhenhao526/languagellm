"""Exact policy-gradient kernel for conflict-specific and global reward controls.

``c0`` and ``c75`` use reciprocal physical execution. When a matching pair
executes, an engaged unmatched third actor multiplies the native reward by
``1 - penalty``. ``global25`` uses the same reciprocal events but multiplies
every native reward by .25. The latter is a global-scale null control for the
log expected-reward objective.
"""
from __future__ import annotations

import numbers

import numpy as np

from research_program.triadic_message_study import runner as core
from . import design

base = core.base
JOINT_ACTIONS = np.asarray(base.JOINT_ACTIONS, dtype=np.int64)
RULES = tuple(design.RULES)
PENALTIES = dict(design.PENALTIES)


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
    require(np.isin(native, (0., .5, 1.)).all(),
            'Native rewards must be 0, .5 or 1')
    require((native == 1).sum(axis=1).min() == 2,
            'Every world must have exactly two full plans')
    return native


def factorized_distribution(logits):
    logits = np.asarray(logits, dtype=np.float64)
    require(logits.ndim == 3 and logits.shape[1:] == (3, 18),
            'Factorized action logits must be [B,3,18]')
    finite(logits, 'Factorized action logits')
    intent_p, intent_lp = base.policy_distribution(logits[..., :2])
    proposal_p, proposal_lp = base.policy_distribution(logits[..., 2:])
    return intent_p, intent_lp, proposal_p, proposal_lp


def _event_log_prob(intent_lp, proposal_lp, rule):
    require(rule in RULES, 'Unknown settlement rule')
    B = len(intent_lp)
    # The first member of every pair is a neutral third branch, the second
    # member an engaged third branch. The third proposal is irrelevant.
    core_log = np.zeros((B, 24), dtype=np.float64)
    for actor in range(3):
        action = JOINT_ACTIONS[:, actor]
        engaged = action > 0
        if engaged.any():
            core_log[:, engaged] += intent_lp[:, actor, 1][:, None]
            core_log[:, engaged] += proposal_lp[:, actor, action[engaged] - 1]
    result = np.repeat(core_log, 2, axis=1)
    third_by_event = np.asarray([2] * 8 + [1] * 8 + [0] * 8, dtype=np.int64)
    for event in range(24):
        third = int(third_by_event[event])
        result[:, 2 * event] += intent_lp[:, third, 0]
        result[:, 2 * event + 1] += intent_lp[:, third, 1]
    return result


def _expanded_native(native, rule):
    if rule == 'global25':
        # Both third-intent branches receive the same global factor.  This
        # shifts log J by a state-independent constant and leaves its exact
        # policy gradient unchanged.
        return np.repeat(native * design.GLOBAL_SCALES[rule], 2, axis=1)
    penalty = PENALTIES[rule]
    expanded = np.empty((len(native), 48), dtype=np.float64)
    expanded[:, 0::2] = native
    expanded[:, 1::2] = native * (1.0 - penalty)
    return expanded


def objective_terms(logits, rewards, rule):
    """Stable log expected reward and exact factorized derivatives."""
    require(rule in RULES, 'Unknown settlement rule')
    native = _rewards(rewards)
    intent_p, intent_lp, proposal_p, proposal_lp = factorized_distribution(logits)
    log_event = _event_log_prob(intent_lp, proposal_lp, rule)
    effective = _expanded_native(native, rule)
    log_mass = np.full(effective.shape, -np.inf, dtype=np.float64)
    positive = effective > 0
    log_mass[positive] = np.log(effective[positive])
    log_mass += log_event
    maximum = log_mass.max(axis=1, keepdims=True)
    finite(maximum, 'Largest rewarding event log mass')
    shifted = np.exp(log_mass - maximum)
    normalizer = shifted.sum(axis=1, keepdims=True)
    posterior = shifted / normalizer
    log_J = (maximum + np.log(normalizer))[:, 0]
    event_prob = np.exp(log_event)
    J = (event_prob * effective).sum(axis=1)
    finite(J, 'Expected penalized reward')
    finite(log_J, 'Stable log expected reward')
    require((J >= 0).all() and (J <= 1 + 1e-10).all()
            and (posterior[~positive] == 0).all(),
            'Invalid penalized reward objective')

    n_events = log_event.shape[1]
    intent_posterior = np.zeros((len(native), 3, 2), dtype=np.float64)
    proposal_posterior = np.zeros((len(native), 3, 16), dtype=np.float64)
    intent_mass = np.zeros((len(native), 3), dtype=np.float64)
    proposal_mass = np.zeros((len(native), 3), dtype=np.float64)
    for event in range(n_events):
        structural = event // 2
        branch = event % 2
        for actor in range(3):
            action = int(JOINT_ACTIONS[structural, actor])
            if action > 0:
                intent_posterior[:, actor, 1] += posterior[:, event]
                proposal_posterior[:, actor, action - 1] += posterior[:, event]
                intent_mass[:, actor] += posterior[:, event]
                proposal_mass[:, actor] += posterior[:, event]
            else:
                intent_posterior[:, actor, int(branch)] += posterior[:, event]
                intent_mass[:, actor] += posterior[:, event]
    intent_log_gradient = intent_posterior - intent_mass[:, :, None] * intent_p
    proposal_log_gradient = proposal_posterior - proposal_mass[:, :, None] * proposal_p
    finite(intent_log_gradient, 'Intent log-gradient')
    finite(proposal_log_gradient, 'Proposal log-gradient')

    # Physical plan probabilities retain both third-intent branches; the cost
    # affects the expected reward, not whether the pair/site plan is native.
    structural_prob = event_prob[:, 0::2] + event_prob[:, 1::2]
    full_prob = (structural_prob * (native == 1)).sum(axis=1)
    partial_prob = (structural_prob * (native == .5)).sum(axis=1)
    return dict(
        intent_probabilities=intent_p, intent_log_probabilities=intent_lp,
        proposal_probabilities=proposal_p, proposal_log_probabilities=proposal_lp,
        event_probabilities=event_prob, event_log_probabilities=log_event,
        J=J, log_J=log_J, posterior_weights=posterior,
        intent_posterior=intent_posterior, proposal_posterior=proposal_posterior,
        intent_mass=intent_mass, proposal_mass=proposal_mass,
        active_posterior=proposal_mass,
        intent_log_gradient=intent_log_gradient,
        proposal_log_gradient=proposal_log_gradient,
        full_success_probability=full_prob, partial_success_probability=partial_prob,
        engagement_probabilities=intent_p[..., 1], native_rewards=native,
        effective_rewards=effective, rule=rule, penalty=PENALTIES[rule],
        global_reward_scale=design.GLOBAL_SCALES[rule])


def training_gradients(networks, observations, rewards, live, uniforms, update, rule):
    x = np.asarray(observations, dtype=np.float64)
    native = _rewards(rewards)
    require(x.shape == (len(native), 3, 54), 'Incorrect observations')
    finite(x, 'Observations')
    require(isinstance(live, (bool, np.bool_)), 'live must be boolean')
    require(rule in RULES, 'Unknown settlement rule')
    require(isinstance(update, numbers.Integral) and not isinstance(update, (bool, np.bool_))
            and 1 <= int(update) <= 3000, 'Invalid update')
    uniforms = np.asarray(uniforms, dtype=np.float64)
    require(uniforms.shape == (2, len(x), 2, 3, 4) and np.isfinite(uniforms).all()
            and ((uniforms >= 0) & (uniforms < 1)).all(), 'Invalid trajectory uniforms')
    trace = core.rollout(networks, np.concatenate((x, x)), bool(live),
                         uniforms.reshape(2 * len(x), 2, 3, 4))
    terms = objective_terms(trace['action_logits'], np.concatenate((native, native)), rule)
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
    row = dict(
        mean_expected_utility=float(terms['J'].mean()),
        mean_log_expected_utility=float(terms['log_J'].mean()),
        min_log_expected_utility=float(terms['log_J'].min()),
        max_log_expected_utility=float(terms['log_J'].max()),
        zero_expected_utility_states=int((terms['J'] == 0).sum()),
        mean_full_success_probability=float(terms['full_success_probability'].mean()),
        mean_partial_success_probability=float(terms['partial_success_probability'].mean()),
        mean_engagement_probability=float(terms['engagement_probabilities'].mean()),
        mean_active_posterior=float(terms['active_posterior'].mean()),
        mean_F=float(F.mean()), receiver_loss=-float((terms['log_J'] + beta * entropy).mean()),
        mean_actor_entropy=float(entropy.mean()), intent_entropy=float(entropy_i.mean()),
        proposal_entropy=float(entropy_p.mean()), entropy_coefficient=beta,
        sender_surrogate_loss=sender['surrogate_loss'],
        sender_advantage_mean=float(sender['advantage'].mean()),
        sender_advantage_abs_mean=float(np.abs(sender['advantage']).mean()),
        sender_advantage_squared_mean=float(np.square(sender['advantage']).mean()),
        sender_advantage_max_abs=float(np.abs(sender['advantage']).max()),
        sender_mean_complete_log_score=float(sender['log_scores'].mean()),
        sampled_messages_sha256=core.base.array_sha(trace['messages']),
        rule=rule, penalty=PENALTIES[rule])
    return gradients, row


if __name__ == '__main__':
    print('continuous third-agent conflict-penalty kernel')
