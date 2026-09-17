"""Environment for three-valued attributes and staged message timing."""
from __future__ import annotations

import numpy as np

from . import design, policy


def _permute_sequences(sequences, partner_id):
    out = sequences.copy()
    for worker in range(design.WORKERS):
        idx = np.flatnonzero(partner_id == worker)
        if len(idx) > 1:
            out[idx] = sequences[np.roll(idx, 1)]
    return out


def _schedule_messages(delivered, protocol, form):
    messages = np.full((len(delivered), design.HORIZON), design.NULL_MESSAGE, dtype=np.int64) if hasattr(design, "NULL_MESSAGE") else np.full((len(delivered), design.HORIZON), -1, dtype=np.int64)
    for slot, arrival in enumerate(design.message_arrival_times(protocol, form)):
        messages[:, arrival] = delivered[:, slot]
    return messages


def rollout(params, episode, form, task, protocol, channel, *, message_mode="natural", sample=True, partner_filter=None, message_override=None):
    k = design.alphabet_size(form); length = design.message_length(form); null = -1
    design.require(channel in design.CHANNELS and message_mode in ("natural", "closed", "permuted"), "bad channel or mode")
    n = len(episode["goal"])
    active = np.ones(n, dtype=bool) if partner_filter is None else episode["partner_id"] == int(partner_filter)
    remain = episode["capacity"].astype(np.int16).copy()
    actions = np.zeros((n, design.HORIZON), dtype=np.int64)
    rewards = np.zeros((n, design.HORIZON), dtype=np.float64)
    message_probs = [None] * length
    action_probs = [None] * design.HORIZON
    states = [None] * design.HORIZON
    locals_ = [None] * design.HORIZON
    inventories = [None] * design.HORIZON
    context = design.goal_index(episode["goal"])
    selected = np.zeros((n, length), dtype=np.int64)
    for slot in range(length):
        sp = policy.softmax(params["sender_logits_hidden"][context, slot])
        message_probs[slot] = sp
        if message_override is None:
            selected[:, slot] = policy.sample(sp, episode["message_uniforms"][:, slot]) if sample else sp.argmax(axis=-1)
        else:
            selected[:, slot] = np.asarray(message_override)[:, slot]
    delivered = selected.copy()
    if channel == "silent" or message_mode == "closed":
        delivered = np.full_like(delivered, null)
    elif message_mode == "permuted":
        delivered = _permute_sequences(selected, episode["partner_id"])
    delivered = np.where(active[:, None], delivered, null)
    messages = _schedule_messages(delivered, protocol, form)
    history = np.zeros(n, dtype=np.int64)
    received = np.zeros(n, dtype=np.int8)
    worker_idx = episode["partner_id"].astype(np.int64)
    for t in range(design.HORIZON):
        incoming = messages[:, t - 1] if t > 0 else np.full(n, null, dtype=np.int64)
        seen = incoming != null
        if np.any(seen):
            first = seen & (received == 0)
            later = seen & ~first
            history[first] = incoming[first] + 1
            if np.any(later):
                history[later] = 1 + (history[later] - 1) * k + incoming[later]
            received[seen] += 1
        stage = min(max((t - design.ACTION_START) // design.STEPS_PER_SUBTASK, 0), design.SUBTASKS - 1)
        local = episode["site_type"][:, stage, 0].astype(np.int64)
        inventory = remain[:, stage, :].sum(axis=1).astype(np.int64)
        ap = policy.softmax(params["worker_logits"][worker_idx, history, t, local, inventory])
        action_probs[t] = ap
        states[t] = history.copy(); locals_[t] = local.copy(); inventories[t] = inventory.copy()
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
    return {"messages": messages, "selected_messages": selected, "actions": actions, "rewards": rewards, "team_return": rewards.mean(axis=1), "active": active, "message_probs": message_probs, "action_probs": action_probs, "state": states, "local": locals_, "inventory": inventories, "final_inventory": remain, "episodes": episode, "message_mode": message_mode, "form": form, "task": task, "protocol": protocol}


def oracle_team_return():
    return design.SUBTASKS * design.STEPS_PER_SUBTASK * design.CORRECT_REWARD / design.HORIZON
