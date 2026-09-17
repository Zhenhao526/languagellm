"""Partial-success utility intervention; no preparation or model calls on import.

The environment's native settlement remains R in {0, .5, 1}. Training uses
U_alpha(R) in {0, alpha, 1}. J below denotes expected *utility*, retained as an
internal key for the frozen receiver-loss function. Public training diagnostics
name expected utility and native expected reward separately.
"""
from __future__ import annotations

import numbers

import numpy as np

from research_program.triadic_message_study import runner as core

base = core.base
require, finite = base.require, base.finite
ALPHAS = (0.5, 0.1)


def _native_table(native_rewards):
    native = np.asarray(native_rewards, dtype=np.float64)
    require(native.ndim == 2 and native.shape[1] == 24 and len(native) > 0,
            "Native rewards must be a nonempty [B,24] table")
    finite(native, "native rewards")
    require(np.isin(native, (0.0, 0.5, 1.0)).all(),
            "Native rewards must be exactly 0, .5 or 1")
    require(((native == 1).sum(axis=1) == 1).all(),
            "Every study world must have exactly one full-success plan")
    return native


def utility_table(native_rewards, alpha):
    """Map native outcomes without changing positive support or its full plan."""
    require(isinstance(alpha, numbers.Real) and not isinstance(alpha, (bool, np.bool_))
            and np.isfinite(alpha) and float(alpha) in ALPHAS,
            "Alpha must be exactly one of the fixed positive values .5 or .1")
    native = _native_table(native_rewards)
    return np.where(native == 0.5, float(alpha), native)


def objective_terms(logits, native_rewards, alpha):
    """Exact expected utility and stable log expectation over full action support.

All 17 actions per agent remain in softmax. The 24 structural joint plans are
the only potentially rewarding plans; every other joint plan contributes zero.
The log calculation has no epsilon, floor, probability clipping or sampled
action approximation. Ordinary expected values may underflow to zero, while
the log expectation and utility-weighted posterior remain finite.
"""
    logits = np.asarray(logits, dtype=np.float64)
    native = _native_table(native_rewards)
    utility = utility_table(native, alpha)
    require(logits.ndim == 3 and logits.shape == (len(native), 3, 17),
            "Incorrect logit/native reward shape")
    finite(logits, "logits")
    probabilities, log_probabilities = base.policy_distribution(logits)
    joint = np.ones((len(native), 24), dtype=np.float64)
    for actor in range(3):
        joint *= probabilities[:, actor, base.JOINT_ACTIONS[:, actor]]
    weighted = joint * utility
    J = weighted.sum(axis=1)
    utility_gradient = -probabilities * J[:, None, None]
    for plan in range(24):
        for actor in range(3):
            utility_gradient[:, actor, base.JOINT_ACTIONS[plan, actor]] += weighted[:, plan]

    positive = utility > 0
    log_mass = np.full(utility.shape, -np.inf, dtype=np.float64)
    log_mass[positive] = np.log(utility[positive])
    for actor in range(3):
        log_mass += log_probabilities[:, actor, base.JOINT_ACTIONS[:, actor]]
    maximum = log_mass.max(axis=1, keepdims=True)
    finite(maximum, "largest log utility mass")
    shifted_mass = np.exp(log_mass - maximum)
    normalization = shifted_mass.sum(axis=1, keepdims=True)
    posterior = shifted_mass / normalization
    log_J = (maximum + np.log(normalization))[:, 0]
    finite(log_J, "log expected utility")
    finite(posterior, "utility posterior")
    require((log_J <= 1e-12).all(), "Expected utility cannot exceed 1")
    require((posterior[~positive] == 0).all(), "Zero-utility plan has posterior mass")
    require(np.allclose(posterior.sum(axis=1), 1, rtol=0, atol=1e-12),
            "Utility posterior does not normalize")
    log_gradient = -probabilities.copy()
    for plan in range(24):
        for actor in range(3):
            log_gradient[:, actor, base.JOINT_ACTIONS[plan, actor]] += posterior[:, plan]
    finite(utility_gradient, "expected utility logit gradient")
    finite(log_gradient, "log expected utility logit gradient")
    native_expected_reward = (joint * native).sum(axis=1)
    full_probability = (joint * (native == 1)).sum(axis=1)
    partial_probability = (joint * (native == 0.5)).sum(axis=1)
    full_posterior = (posterior * (native == 1)).sum(axis=1)
    partial_posterior = (posterior * (native == 0.5)).sum(axis=1)
    return {
        "probabilities": probabilities, "log_probabilities": log_probabilities,
        "J": J, "log_J": log_J, "mean_J_logit_gradient": utility_gradient,
        "log_J_logit_gradient": log_gradient, "posterior_weights": posterior,
        "native_expected_reward": native_expected_reward,
        "full_success_probability": full_probability,
        "partial_success_probability": partial_probability,
        "full_success_posterior_mass": full_posterior,
        "partial_success_posterior_mass": partial_posterior,
        "utility_values": utility,
    }


def training_gradients(networks, observations, native_rewards, live, uniforms, update, alpha):
    """Frozen two-trajectory LOO training, with only the utility table changed.

This function never updates parameters or optimizer state. Reward tables and
the unique full plan are used after rollout, and never supplied to a network.
    """
    x = np.asarray(observations, dtype=np.float64)
    native = _native_table(native_rewards)
    utility_table(native, alpha)  # Reject invalid utility before any model call.
    require(x.shape == (len(native), 3, 54), "Incorrect training observations")
    finite(x, "training observations")
    require(isinstance(live, (bool, np.bool_)), "Live channel flag must be boolean")
    require(isinstance(update, numbers.Integral) and not isinstance(update, (bool, np.bool_))
            and 1 <= int(update) <= base.CONFIG["updates"], "Invalid training update")
    B = len(x)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    require(uniforms.shape == (2, B, 2, 3, 4) and np.isfinite(uniforms).all()
            and ((uniforms >= 0) & (uniforms < 1)).all(),
            "Two valid independent full trajectories required")
    trace = core.rollout(networks, np.concatenate((x, x)), live,
                         uniforms.reshape(2 * B, 2, 3, 4))
    terms = objective_terms(trace["action_logits"], np.concatenate((native, native)), alpha)
    receiver = core.coordination.loss_and_derivative(terms, "mean_log_J", update)
    entropy, _ = base.entropy_and_logit_gradient(terms["probabilities"], terms["log_probabilities"])
    F = terms["log_J"] + receiver["entropy_coefficient"] * entropy
    sender = core.paired_sender_derivative(
        trace["sender_probabilities"].reshape(2, B, 2, 3, 4, 8),
        trace["sender_log_probabilities"].reshape(2, B, 2, 3, 4, 8),
        trace["messages"].reshape(2, B, 2, 3, 4), F.reshape(2, B))
    gradients = []
    ds = sender["derivative"].reshape(2 * B, 2, 3, 4, 8)
    for actor in range(3):
        for module in range(3):
            d = receiver["derivative"][:, actor] if module == 2 else ds[:, module, actor].reshape(2 * B, 32)
            gradients.append(base.actor_backward(networks[3 * actor + module],
                trace["caches"][3 * actor + module], d))
    row = {
        "partial_success_utility": float(alpha),
        "mean_expected_utility": float(terms["J"].mean()),
        "mean_log_expected_utility": float(terms["log_J"].mean()),
        "min_log_expected_utility": float(terms["log_J"].min()),
        "max_log_expected_utility": float(terms["log_J"].max()),
        "zero_float_expected_utility_states": int((terms["J"] == 0).sum()),
        "mean_native_expected_reward": float(terms["native_expected_reward"].mean()),
        "mean_full_success_probability": float(terms["full_success_probability"].mean()),
        "mean_partial_success_probability": float(terms["partial_success_probability"].mean()),
        "mean_full_success_posterior_mass": float(terms["full_success_posterior_mass"].mean()),
        "mean_partial_success_posterior_mass": float(terms["partial_success_posterior_mass"].mean()),
        "mean_F": float(F.mean()), "receiver_loss": receiver["loss"],
        "mean_actor_entropy": receiver["mean_actor_entropy"],
        "entropy_coefficient": receiver["entropy_coefficient"],
        "sender_surrogate_loss": sender["surrogate_loss"],
        "sender_advantage_mean": float(sender["advantage"].mean()),
        "sender_advantage_abs_mean": float(np.abs(sender["advantage"]).mean()),
        "sender_advantage_squared_mean": float(np.square(sender["advantage"]).mean()),
        "sender_advantage_max_abs": float(np.abs(sender["advantage"]).max()),
        "sender_mean_complete_log_score": float(sender["log_scores"].mean()),
        "sampled_messages_sha256": core.array_sha(trace["messages"]),
    }
    return gradients, row
