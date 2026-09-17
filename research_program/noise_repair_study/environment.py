"""Staged dual-token environment with receiver representation and channel noise."""
from __future__ import annotations

import numpy as np

from research_program.action_dependent_signaling_study import design, policy

TARGET_WORKER = 0


def _permute_sequences(sequences, partner_id):
    out = sequences.copy()
    for worker in range(design.WORKERS):
        idx = np.flatnonzero(partner_id == worker)
        if len(idx) > 1:
            out[idx] = sequences[np.roll(idx, 1)]
    return out


def _schedule_messages(delivered):
    messages = np.full((len(delivered), design.HORIZON), design.NULL_MESSAGE, dtype=np.int64)
    for slot, arrival in enumerate(design.message_arrival_times("staged", "dual2")):
        messages[:, arrival] = delivered[:, slot]
    return messages


def rollout(params, episode, representation, channel, noise_p, *, message_mode="natural", sample=True,
            partner_filter=None, message_override=None):
    design.require(representation in ("joint_history", "slot_local"), "bad representation")
    design.require(channel in design.CHANNELS and message_mode in ("natural", "closed", "permuted"), "bad channel")
    design.require(0.0 <= float(noise_p) <= 1.0, "bad noise level")
    n = len(episode["goal"])
    active = np.ones(n, dtype=bool) if partner_filter is None else episode["partner_id"] == int(partner_filter)
    remain = episode["capacity"].astype(np.int16).copy()
    actions = np.zeros((n, design.HORIZON), dtype=np.int64)
    rewards = np.zeros((n, design.HORIZON), dtype=np.float64)
    action_probs = [None] * design.HORIZON
    message_probs = [None] * design.MESSAGE_SLOTS
    states = [None] * design.HORIZON
    locals_ = [None] * design.HORIZON
    inventories = [None] * design.HORIZON
    context = design.goal_index(episode["goal"])
    selected = np.zeros((n, design.MESSAGE_SLOTS), dtype=np.int64)
    for slot in range(design.MESSAGE_SLOTS):
        sp = policy.softmax(params["sender_logits_hidden"][context, slot])
        message_probs[slot] = sp
        if message_override is None:
            selected[:, slot] = policy.sample(sp, episode["message_uniforms"][:, slot]) if sample else sp.argmax(axis=-1)
        else:
            selected[:, slot] = np.asarray(message_override)[:, slot]
    delivered = selected.copy()
    if channel == "silent" or message_mode == "closed":
        delivered = np.full_like(delivered, design.NULL_MESSAGE)
    elif message_mode == "permuted":
        delivered = _permute_sequences(selected, episode["partner_id"])
    if float(noise_p) > 0.0:
        flips = (episode["noise_uniforms"] < float(noise_p)) & (delivered >= 0)
        delivered = np.where(flips, 1 - delivered, delivered)
    delivered = np.where(active[:, None], delivered, design.NULL_MESSAGE)
    messages = _schedule_messages(delivered)
    history = np.zeros(n, dtype=np.int64)
    received = np.zeros(n, dtype=np.int8)
    slot_memory = np.full((n, design.MESSAGE_SLOTS), design.NULL_MESSAGE, dtype=np.int64)
    arrivals = design.message_arrival_times("staged", "dual2")
    worker_idx = episode["partner_id"].astype(np.int64)
    for t in range(design.HORIZON):
        incoming = messages[:, t - 1] if t > 0 else np.full(n, design.NULL_MESSAGE, dtype=np.int64)
        seen = incoming != design.NULL_MESSAGE
        if np.any(seen):
            first = seen & (received == 0)
            later = seen & ~first
            history[first] = incoming[first] + 1
            if np.any(later):
                history[later] = 1 + (history[later] - 1) * 2 + incoming[later]
            received[seen] += 1
            for slot, arrival in enumerate(arrivals):
                if t - 1 == arrival:
                    slot_memory[seen, slot] = incoming[seen]
        stage = min(max((t - design.ACTION_START) // design.STEPS_PER_SUBTASK, 0), design.SUBTASKS - 1)
        local = episode["site_type"][:, stage, 0].astype(np.int64)
        inventory = remain[:, stage, :].sum(axis=1).astype(np.int64)
        local_state = slot_memory[:, stage] + 1
        use_local = (representation == "slot_local") & (worker_idx == TARGET_WORKER) & (t >= design.ACTION_START)
        state = np.where(use_local, local_state, history).astype(np.int64)
        ap = policy.softmax(params["worker_logits"][worker_idx, state, t, local, inventory])
        action_probs[t] = ap
        states[t] = state.copy()
        locals_[t] = local.copy()
        inventories[t] = inventory.copy()
        act = policy.sample(ap, episode["action_uniforms"][:, t]) if sample else ap.argmax(axis=-1)
        act = np.where((t < design.ACTION_START) | (~active), 0, act).astype(np.int64)
        actions[:, t] = act
        if t >= design.ACTION_START:
            for i in np.flatnonzero(active):
                action = int(act[i])
                if action == 0:
                    continue
                site = action - 1
                available = int(remain[i, stage, site])
                if available > 0:
                    remain[i, stage, site] -= 1
                target = int(episode["target_bits"][i, stage])
                rewards[i, t] = design.CORRECT_REWARD if available > 0 and int(episode["site_type"][i, stage, site]) == target else design.WRONG_REWARD
    return {
        "messages": messages,
        "selected_messages": selected,
        "delivered_messages": delivered,
        "actions": actions,
        "rewards": rewards,
        "team_return": rewards.mean(axis=1),
        "active": active,
        "message_probs": message_probs,
        "action_probs": action_probs,
        "state": states,
        "local": locals_,
        "inventory": inventories,
        "final_inventory": remain,
        "episodes": episode,
        "message_mode": message_mode,
        "representation": representation,
        "noise_p": float(noise_p),
    }
