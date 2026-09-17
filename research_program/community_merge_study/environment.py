"""Role-symmetric referential selection with community-specific surface codes."""
from __future__ import annotations

import numpy as np

from . import design, policy


def _permute_messages(messages, groups):
    out = np.asarray(messages, dtype=np.int64).copy()
    groups = np.asarray(groups, dtype=np.int64)
    for key in np.unique(groups):
        idx = np.flatnonzero(groups == key)
        if len(idx) > 1:
            out[idx] = messages[np.roll(idx, 1)]
    return out


def _surface_messages(messages, community_id, population):
    out = np.asarray(messages, dtype=np.int64).copy()
    for community in range(design.COMMUNITIES):
        idx = np.flatnonzero(np.asarray(community_id) == community)
        if len(idx):
            valid = out[idx] >= 0
            if np.any(valid):
                mapped = out[idx].copy()
                mapped[valid] = design.surface_permutation(population, community)[mapped[valid]]
                out[idx] = mapped
    return out


def _canonical_messages(messages, community_id, population):
    out = np.asarray(messages, dtype=np.int64).copy()
    for community in range(design.COMMUNITIES):
        idx = np.flatnonzero(np.asarray(community_id) == community)
        if len(idx):
            inv = design.inverse_permutation(design.surface_permutation(population, community))
            valid = out[idx] >= 0
            if np.any(valid):
                mapped = out[idx].copy()
                mapped[valid] = inv[mapped[valid]]
                out[idx] = mapped
    return out


def _sender_arrays(params, meanings, partner_id, visibility):
    n = len(meanings)
    probs = np.zeros((n, design.MESSAGE_LENGTH, design.ALPHABET_SIZE), dtype=np.float64)
    context = np.asarray(meanings, dtype=np.int64).copy()
    for partner in range(design.PARTNERS):
        idx = np.flatnonzero(np.asarray(partner_id) == partner)
        if not len(idx):
            continue
        if visibility == "hidden":
            logits = params["sender_hidden"][meanings[idx]]
        else:
            context[idx] = partner * design.MEANINGS + meanings[idx]
            logits = params["sender_visible"][partner, meanings[idx]]
        probs[idx] = policy.softmax(logits, axis=-1)
    return probs, context


def _receiver_arrays(params, message_state, scene_meanings, partner_id, visibility):
    n = len(message_state)
    scores = np.zeros((n, design.SCENE_SIZE), dtype=np.float64)
    context = np.asarray(message_state, dtype=np.int64).copy()
    for partner in range(design.PARTNERS):
        idx = np.flatnonzero(np.asarray(partner_id) == partner)
        if not len(idx):
            continue
        if visibility == "hidden":
            logits = params["receiver_hidden"][message_state[idx]]
        else:
            context[idx] = partner * design.MESSAGE_STATE_COUNT + message_state[idx]
            logits = params["receiver_visible"][partner, message_state[idx]]
        scores[idx] = logits[np.arange(len(idx))[:, None], scene_meanings[idx]]
    return policy.softmax(scores, axis=-1), context


def rollout(
    communities,
    fresh,
    episode,
    population,
    visibility,
    role,
    *,
    channel="live",
    message_mode="natural",
    sample=True,
    message_override=None,
):
    """Run a parent or child episode batch.

    ``fresh is None`` denotes parent training, where one community policy acts
    in both roles.  In child training the fresh agent alternates between
    sender and receiver (or remains sender under ``sender_only``), while the
    other role is filled by the selected incumbent community.
    """
    design.require(population in design.POPULATIONS and visibility in design.VISIBILITIES, "bad rollout factors")
    design.require(role in design.ROLES and channel in design.CHANNELS, "bad rollout factors")
    design.require(message_mode in ("natural", "permuted", "closed"), "bad message mode")
    n = len(episode["goal"])
    partner_id = np.asarray(episode["partner_id"], dtype=np.int64)
    community_id = np.asarray(episode["community_id"], dtype=np.int64)
    meanings = np.asarray(episode["goal_meaning"], dtype=np.int64)
    parent = fresh is None
    if parent:
        # Parent streams set one fixed community id; retaining it here applies
        # the intended external surface code while the internal policy stays
        # in its canonical alphabet.
        sender_fresh = np.zeros(n, dtype=bool)
        receiver_fresh = np.zeros(n, dtype=bool)
        sender_owner = np.zeros(n, dtype=np.int64)
        receiver_owner = np.zeros(n, dtype=np.int64)
        sender_params = communities[0]
        sender_probs, sender_context = _sender_arrays(sender_params, meanings, np.zeros(n, dtype=np.int64), "hidden")
        selected_canonical = np.zeros((n, design.MESSAGE_LENGTH), dtype=np.int64)
        for slot in range(design.MESSAGE_LENGTH):
            selected_canonical[:, slot] = policy.sample(sender_probs[:, slot], episode["message_uniforms"][:, slot]) if sample else sender_probs[:, slot].argmax(axis=-1)
        selected_external = _surface_messages(selected_canonical, community_id, population)
    else:
        sender_fresh = np.asarray(episode["fresh_role"], dtype=np.int8) == 0
        receiver_fresh = ~sender_fresh
        sender_owner = np.where(sender_fresh, -1, community_id).astype(np.int64)
        receiver_owner = np.where(receiver_fresh, -1, community_id).astype(np.int64)
        selected_canonical = np.zeros((n, design.MESSAGE_LENGTH), dtype=np.int64)
        selected_external = np.zeros((n, design.MESSAGE_LENGTH), dtype=np.int64)
        sender_probs = np.zeros((n, design.MESSAGE_LENGTH, design.ALPHABET_SIZE), dtype=np.float64)
        sender_context = meanings.copy()
        for mask, params, is_fresh in (
            (sender_fresh, fresh, True),
            (~sender_fresh, None, False),
        ):
            idx = np.flatnonzero(mask)
            if not len(idx):
                continue
            if is_fresh:
                sp, ctx = _sender_arrays(params, meanings[idx], partner_id[idx], visibility)
                sender_probs[idx] = sp
                sender_context[idx] = ctx
                for slot in range(design.MESSAGE_LENGTH):
                    selected_external[idx, slot] = policy.sample(sp[:, slot], episode["message_uniforms"][idx, slot]) if sample else sp[:, slot].argmax(axis=-1)
                selected_canonical[idx] = selected_external[idx]
            else:
                for community in range(design.COMMUNITIES):
                    sub = idx[community_id[idx] == community]
                    if not len(sub):
                        continue
                    cp = communities[community]
                    sp, ctx = _sender_arrays(cp, meanings[sub], np.zeros(len(sub), dtype=np.int64), "hidden")
                    sender_probs[sub] = sp
                    sender_context[sub] = ctx
                    for slot in range(design.MESSAGE_LENGTH):
                        selected_canonical[sub, slot] = policy.sample(sp[:, slot], episode["message_uniforms"][sub, slot]) if sample else sp[:, slot].argmax(axis=-1)
                    selected_external[sub] = _surface_messages(selected_canonical[sub], community_id[sub], population)

    if message_override is not None:
        selected_external = np.asarray(message_override, dtype=np.int64).copy()
    delivered_external = selected_external.copy()
    if channel == "silent" or message_mode == "closed":
        delivered_external.fill(design.NULL_MESSAGE)
    elif message_mode == "permuted":
        if parent:
            groups = community_id
        else:
            groups = partner_id * 2 + np.asarray(episode["fresh_role"], dtype=np.int64)
        delivered_external = _permute_messages(delivered_external, groups)

    # Incumbent receivers undo their community surface transform.  A fresh
    # receiver learns directly from the external channel and has no inverse
    # mapping supplied by the environment.
    delivered_canonical = delivered_external.copy()
    if parent:
        delivered_canonical = _canonical_messages(delivered_external, community_id, population)
        receiver_params = communities[0]
        message_state = design.message_index(delivered_canonical)
        receiver_probs, receiver_context = _receiver_arrays(receiver_params, message_state, np.asarray(episode["scene_meanings"]), np.zeros(n, dtype=np.int64), "hidden")
    else:
        incumbent_receiver = ~receiver_fresh
        if np.any(incumbent_receiver):
            delivered_canonical[incumbent_receiver] = _canonical_messages(delivered_external[incumbent_receiver], community_id[incumbent_receiver], population)
        message_state = design.message_index(delivered_canonical)
        receiver_probs = np.zeros((n, design.SCENE_SIZE), dtype=np.float64)
        receiver_context = message_state.copy()
        if np.any(receiver_fresh):
            idx = np.flatnonzero(receiver_fresh)
            rp, ctx = _receiver_arrays(fresh, message_state[idx], np.asarray(episode["scene_meanings"])[idx], partner_id[idx], visibility)
            receiver_probs[idx] = rp
            receiver_context[idx] = ctx
        if np.any(incumbent_receiver):
            idx = np.flatnonzero(incumbent_receiver)
            for community in range(design.COMMUNITIES):
                sub = idx[community_id[idx] == community]
                if not len(sub):
                    continue
                rp, ctx = _receiver_arrays(communities[community], message_state[sub], np.asarray(episode["scene_meanings"])[sub], np.zeros(len(sub), dtype=np.int64), "hidden")
                receiver_probs[sub] = rp
                receiver_context[sub] = ctx

    actions = policy.sample(receiver_probs, episode["action_uniforms"]) if sample else receiver_probs.argmax(axis=-1)
    target_index = np.asarray(episode["target_index"], dtype=np.int64)
    rewards = np.where(actions == target_index, design.CORRECT_REWARD, design.WRONG_REWARD).astype(np.float64)
    return {
        "selected_canonical": selected_canonical,
        "selected_messages": selected_external,
        "delivered_messages": delivered_external,
        "delivered_canonical": delivered_canonical,
        "message_state": message_state,
        "sender_probs": sender_probs,
        "sender_context": sender_context,
        "receiver_probs": receiver_probs,
        "receiver_context": receiver_context,
        "actions": actions,
        "rewards": rewards,
        "team_return": rewards.copy(),
        "sender_fresh": sender_fresh,
        "receiver_fresh": receiver_fresh,
        "sender_owner": sender_owner,
        "receiver_owner": receiver_owner,
        "episode": episode,
        "message_mode": message_mode,
        "channel": channel,
        "population": population,
        "visibility": visibility,
        "role": role,
    }


def recombination_messages(communities, fresh, episode, population, visibility):
    """Construct slot-wise donor messages for a compositionality probe."""
    n = len(episode["goal"])
    out = np.zeros((n, design.MESSAGE_LENGTH), dtype=np.int64)
    meanings = np.asarray(episode["goal_meaning"], dtype=np.int64)
    partner = np.asarray(episode["partner_id"], dtype=np.int64)
    community = np.asarray(episode["community_id"], dtype=np.int64)
    if fresh is None:
        sender_fresh = np.zeros(n, dtype=bool)
    else:
        sender_fresh = np.asarray(episode["fresh_role"], dtype=np.int8) == 0
    for i, meaning in enumerate(meanings):
        attrs = design.attrs_from_meaning(meaning)
        donor0 = int(attrs[0]) * design.VALUES  # second factor fixed to zero
        donor1 = int(attrs[1])  # first factor fixed to zero
        donors = (donor0, donor1)
        if sender_fresh[i]:
            for slot, donor in enumerate(donors):
                out[i, slot] = policy.sender_sequence(fresh, donor, partner_id=partner[i], visibility=visibility, external=False)[slot]
        else:
            cp = communities[int(community[i])]
            perm = design.surface_permutation(population, community[i])
            for slot, donor in enumerate(donors):
                out[i, slot] = policy.sender_sequence(cp, donor, visibility="hidden", external=True, permutation=perm)[slot]
    return out


def oracle_return():
    return float(design.CORRECT_REWARD)
