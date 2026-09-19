"""Role-symmetric referential selection with holistic or factor-shared policies."""
from __future__ import annotations

import numpy as np

from . import design, policy


def _permute_messages(messages, groups):
    out = np.asarray(messages, dtype=np.int64).copy(); groups = np.asarray(groups, dtype=np.int64)
    for key in np.unique(groups):
        idx = np.flatnonzero(groups == key)
        if len(idx) > 1: out[idx] = messages[np.roll(idx, 1)]
    return out


def _surface_messages(messages, community_id, population):
    out = np.asarray(messages, dtype=np.int64).copy()
    for community in range(design.COMMUNITIES):
        idx = np.flatnonzero(np.asarray(community_id) == community)
        if len(idx):
            perm = design.surface_permutation(population, community)
            valid = out[idx] >= 0
            for slot in range(design.MESSAGE_LENGTH):
                if np.any(valid[:, slot]): out[idx[valid[:, slot]], slot] = perm[out[idx[valid[:, slot]], slot]]
    return out


def _canonical_messages(messages, community_id, population):
    out = np.asarray(messages, dtype=np.int64).copy()
    for community in range(design.COMMUNITIES):
        idx = np.flatnonzero(np.asarray(community_id) == community)
        if len(idx):
            inv = design.inverse_permutation(design.surface_permutation(population, community))
            valid = out[idx] >= 0
            for slot in range(design.MESSAGE_LENGTH):
                if np.any(valid[:, slot]): out[idx[valid[:, slot]], slot] = inv[out[idx[valid[:, slot]], slot]]
    return out


def rollout(communities, fresh, episode, architecture, population, visibility, *, channel="live", message_mode="natural", sample=True, message_override=None):
    design.require(architecture in design.ARCHITECTURES and population in design.POPULATIONS and visibility in design.VISIBILITIES, "bad rollout factors")
    design.require(channel in design.CHANNELS and message_mode in ("natural", "permuted", "closed"), "bad channel")
    n = len(episode["goal"]); partner_id = np.asarray(episode["partner_id"], dtype=np.int64); community_id = np.asarray(episode["community_id"], dtype=np.int64); meanings = np.asarray(episode["goal_meaning"], dtype=np.int64)
    parent = fresh is None
    # Role 2 is a fresh–fresh expansion episode.  It is used only for
    # meanings containing the new value; role 0/1 retain the incumbent
    # alternating game used by the cultural-transmission controls.
    role = np.asarray(episode["fresh_role"], dtype=np.int8)
    sender_fresh = np.zeros(n, dtype=bool) if parent else np.isin(role, (0, 2))
    receiver_fresh = np.zeros(n, dtype=bool) if parent else np.isin(role, (1, 2))
    sender_owner = np.zeros(n, dtype=np.int64) if parent else np.where(sender_fresh, -1, community_id).astype(np.int64)
    receiver_owner = np.zeros(n, dtype=np.int64) if parent else np.where(receiver_fresh, -1, community_id).astype(np.int64)
    selected_canonical = np.zeros((n, design.MESSAGE_LENGTH), dtype=np.int64); selected_external = np.zeros_like(selected_canonical)
    sender_probs = np.zeros((n, design.MESSAGE_LENGTH, design.ALPHABET_SIZE), dtype=np.float64); sender_context = meanings.copy()
    if parent:
        sp, ctx = policy.sender_arrays(communities[0], meanings, np.zeros(n, dtype=np.int64), "hidden"); sender_probs = sp; sender_context = ctx
        for slot in range(design.MESSAGE_LENGTH): selected_canonical[:, slot] = policy.sample(sp[:, slot], episode["message_uniforms"][:, slot]) if sample else sp[:, slot].argmax(axis=-1)
        selected_external = _surface_messages(selected_canonical, community_id, population)
    else:
        for mask, params, is_fresh in ((sender_fresh, fresh, True), (~sender_fresh, None, False)):
            idx = np.flatnonzero(mask)
            if not len(idx): continue
            if is_fresh:
                sp, ctx = policy.sender_arrays(params, meanings[idx], partner_id[idx], visibility); sender_probs[idx] = sp; sender_context[idx] = ctx
                for slot in range(design.MESSAGE_LENGTH): selected_external[idx, slot] = policy.sample(sp[:, slot], episode["message_uniforms"][idx, slot]) if sample else sp[:, slot].argmax(axis=-1)
                selected_canonical[idx] = selected_external[idx]
            else:
                for community in range(design.COMMUNITIES):
                    sub = idx[community_id[idx] == community]
                    if not len(sub): continue
                    sp, ctx = policy.sender_arrays(communities[community], meanings[sub], np.zeros(len(sub), dtype=np.int64), "hidden")
                    for slot in range(design.MESSAGE_LENGTH): selected_canonical[sub, slot] = policy.sample(sp[:, slot], episode["message_uniforms"][sub, slot]) if sample else sp[:, slot].argmax(axis=-1)
                    selected_external[sub] = _surface_messages(selected_canonical[sub], community_id[sub], population)
    if message_override is not None: selected_external = np.asarray(message_override, dtype=np.int64).copy()
    delivered_external = selected_external.copy()
    if channel == "silent" or message_mode == "closed": delivered_external.fill(design.NULL_MESSAGE)
    elif message_mode == "permuted": delivered_external = _permute_messages(delivered_external, community_id if parent else partner_id * 2 + np.asarray(episode["fresh_role"], dtype=np.int64))
    delivered_canonical = delivered_external.copy()
    if parent:
        delivered_canonical = _canonical_messages(delivered_external, community_id, population)
        message_state = design.message_index(delivered_canonical)
        receiver_probs, receiver_context = policy.receiver_arrays(communities[0], message_state, np.asarray(episode["scene_meanings"]), np.zeros(n, dtype=np.int64), "hidden")
    else:
        incumbent_receiver = ~receiver_fresh
        if np.any(incumbent_receiver): delivered_canonical[incumbent_receiver] = _canonical_messages(delivered_external[incumbent_receiver], community_id[incumbent_receiver], population)
        message_state = design.message_index(delivered_canonical); receiver_probs = np.zeros((n, design.SCENE_SIZE), dtype=np.float64); receiver_context = message_state.copy()
        if np.any(receiver_fresh):
            idx = np.flatnonzero(receiver_fresh)
            fresh_scene = np.asarray(episode.get("fresh_scene_meanings", episode["scene_meanings"]))
            rp, ctx = policy.receiver_arrays(fresh, message_state[idx], fresh_scene[idx], partner_id[idx], visibility)
            receiver_probs[idx] = rp; receiver_context[idx] = ctx
        if np.any(incumbent_receiver):
            idx = np.flatnonzero(incumbent_receiver)
            for community in range(design.COMMUNITIES):
                sub = idx[community_id[idx] == community]
                if len(sub): receiver_probs[sub], receiver_context[sub] = policy.receiver_arrays(communities[community], message_state[sub], np.asarray(episode["scene_meanings"])[sub], np.zeros(len(sub), dtype=np.int64), "hidden")
    actions = policy.sample(receiver_probs, episode["action_uniforms"]) if sample else receiver_probs.argmax(axis=-1)
    target_index = np.asarray(episode["target_index"], dtype=np.int64); rewards = np.where(actions == target_index, design.CORRECT_REWARD, design.WRONG_REWARD).astype(np.float64)
    return {"selected_canonical":selected_canonical,"selected_messages":selected_external,"delivered_messages":delivered_external,"delivered_canonical":delivered_canonical,"message_state":message_state,"sender_probs":sender_probs,"sender_context":sender_context,"receiver_probs":receiver_probs,"receiver_context":receiver_context,"actions":actions,"rewards":rewards,"team_return":rewards.copy(),"sender_fresh":sender_fresh,"receiver_fresh":receiver_fresh,"sender_owner":sender_owner,"receiver_owner":receiver_owner,"episode":episode,"architecture":architecture,"message_mode":message_mode,"channel":channel,"population":population,"visibility":visibility,"surface_mapping":episode.get("surface_mapping", "identity")}


def recombination_messages(communities, fresh, episode, architecture, population, visibility):
    n = len(episode["goal"]); out = np.zeros((n, design.MESSAGE_LENGTH), dtype=np.int64); meanings = np.asarray(episode["goal_meaning"], dtype=np.int64); partner = np.asarray(episode["partner_id"], dtype=np.int64); community = np.asarray(episode["community_id"], dtype=np.int64)
    role = np.asarray(episode["fresh_role"], dtype=np.int8)
    sender_fresh = np.zeros(n, dtype=bool) if fresh is None else np.isin(role, (0, 2))
    for i, meaning in enumerate(meanings):
        if architecture in ("routed", "tied_routed"):
            if sender_fresh[i]:
                out[i] = policy.sender_sequence(fresh, meaning, partner_id=partner[i], visibility=visibility)
            else:
                cp = communities[int(community[i])]
                out[i] = policy.sender_sequence(cp, meaning, visibility="hidden", external=True,
                                                 permutation=design.surface_permutation(population, community[i]))
            continue
        attrs = design.attrs_from_meaning(meaning); donors = (int(attrs[0]) * design.VALUES, int(attrs[1]))
        if sender_fresh[i]:
            for slot, donor in enumerate(donors): out[i, slot] = policy.sender_sequence(fresh, donor, partner_id=partner[i], visibility=visibility, external=False)[slot]
        else:
            cp = communities[int(community[i])]; perm = design.surface_permutation(population, community[i])
            for slot, donor in enumerate(donors): out[i, slot] = policy.sender_sequence(cp, donor, visibility="hidden", external=True, permutation=perm)[slot]
    return out


def oracle_return(): return float(design.CORRECT_REWARD)
