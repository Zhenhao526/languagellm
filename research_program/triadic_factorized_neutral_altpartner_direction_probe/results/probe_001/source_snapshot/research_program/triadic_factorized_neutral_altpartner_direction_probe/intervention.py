"""Causal replacement of one sender's outward W1 packet for factorized heads."""
from __future__ import annotations

import numpy as np

from research_program.triadic_message_study import runner as core
from research_program.triadic_factorized_neutral_altpartner_study import kernel


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _replace(tokens, persons, donor):
    tokens = np.asarray(tokens); persons = np.asarray(persons); donor = np.asarray(donor)
    n = len(tokens)
    require(tokens.shape == (n, 3, 4) and persons.shape == (n,) and donor.shape == (n, 4), 'Invalid packet shapes')
    out = np.broadcast_to(tokens[:, None], (n, 3, 3, 4)).copy()
    for viewer in range(3):
        rows = np.flatnonzero(persons != viewer)
        out[rows, viewer, persons[rows]] = donor[rows]
    visibility = np.ones((n, 3, 3), dtype=np.float64)
    routes = np.concatenate((np.eye(8, dtype=np.float64)[out].reshape(n, 3, 96), visibility), axis=-1)
    return out, routes


def intervene(networks, receiver_x, receiver_messages, changed_person, donor_packets):
    """Replace selected sender W1 and recompute W2 plus factorized actions.

    Receiver observations remain the receiver-world PL features.  Only the
    selected sender's first-window outward packet comes from the donor world;
    all W2 heads and action heads are recomputed causally.
    """
    x = np.asarray(receiver_x, dtype=np.float64); native = np.asarray(receiver_messages, dtype=np.int8)
    persons = np.asarray(changed_person, dtype=np.int64); donor = np.asarray(donor_packets, dtype=np.int8)
    n = len(x)
    require(x.shape == (n, 3, 54) and np.all(x[:, :, 53] == 0), 'Invalid PL receiver features')
    require(native.shape == (n, 2, 3, 4) and donor.shape == (n, 4), 'Invalid native/donor messages')
    require(persons.shape == (n,) and np.all((persons >= 0) & (persons < 3)), 'Invalid sender indices')
    first_tokens, first_routes = _replace(native[:, 0], persons, donor)
    second_inputs = np.concatenate((x, first_routes), axis=-1)
    second_logits = np.stack([core.base.actor_forward(networks[3 * actor + 1], second_inputs[:, actor])[0].reshape(n, 4, 8)
                              for actor in range(3)], axis=1)
    second_probabilities, second_log_probabilities = core.base.policy_distribution(second_logits)
    second_tokens = core.categorical_tokens(second_probabilities)
    second_routes = np.concatenate((np.eye(8, dtype=np.float64)[second_tokens].reshape(n, 3, 96),
                                    np.ones((n, 3, 3), dtype=np.float64)), axis=-1)
    action_inputs = np.concatenate((x, first_routes, second_routes), axis=-1)
    action_logits = np.stack([core.base.actor_forward(networks[3 * actor + 2], action_inputs[:, actor])[0]
                              for actor in range(3)], axis=1)
    intent_p, intent_lp, proposal_p, proposal_lp = kernel.factorized_distribution(action_logits)
    # Convert the factorized heads to the joint 17-action distribution used
    # by the researcher-side event and plan-mass diagnostics.  Action 0 is
    # neutral; transport actions are engage probability times proposal mass.
    action_probabilities = np.zeros((n, 3, 17), dtype=np.float64)
    action_probabilities[:, :, 0] = intent_p[:, :, 0]
    action_probabilities[:, :, 1:] = intent_p[:, :, 1, None] * proposal_p
    intent = intent_p.argmax(axis=-1).astype(np.int16)
    proposal = proposal_p.argmax(axis=-1).astype(np.int16)
    actions = np.where(intent == 1, proposal + 1, 0).astype(np.int16)
    return dict(generated_messages=np.stack((native[:, 0], second_tokens), axis=1),
                delivered_tokens=np.stack((first_tokens, second_tokens), axis=1),
                action_inputs=action_inputs, action_logits=action_logits,
                intent_probabilities=intent_p, proposal_probabilities=proposal_p,
                action_probabilities=action_probabilities,
                intent_indices=intent, proposal_indices=proposal, action_indices=actions,
                neural_forward_calls=6, neural_forward_samples=6 * n,
                reused_native_first_window=True)
