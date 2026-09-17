"""Two-subtask world with simultaneous versus action-dependent message timing."""
from __future__ import annotations
import numpy as np
from . import design, policy


def require(ok, msg):
    if not ok: raise ValueError(msg)


def _permute_sequences(sequences, partner_id):
    out = sequences.copy()
    for worker in range(design.WORKERS):
        idx = np.flatnonzero(partner_id == worker)
        if len(idx) > 1: out[idx] = sequences[np.roll(idx, 1)]
    return out


def _schedule_messages(selected, delivered, protocol, form):
    n, length = selected.shape
    messages = np.full((n, design.HORIZON), design.NULL_MESSAGE, dtype=np.int64)
    times = design.message_arrival_times(protocol, form)
    for slot, message_index in enumerate(times):
        messages[:, message_index] = delivered[:, slot]
    return messages


def rollout(p, ep, form, partner_mode, partner_visibility, task, channel, protocol, *,
           message_mode="natural", sample=True, partner_filter=None,
           message_override=None):
    k = design.alphabet_size(form); length = design.message_length(form)
    require(channel in design.CHANNELS and message_mode in ("natural", "closed", "permuted"), "bad channel")
    require(protocol in design.PROTOCOLS, "bad protocol")
    n = len(ep["goal"])
    active = np.ones(n, dtype=bool) if partner_filter is None else (ep["partner_id"] == int(partner_filter))
    remain = ep["capacity"].astype(np.int16).copy()
    actions = np.zeros((n, design.HORIZON), dtype=np.int64)
    rewards = np.zeros((n, design.HORIZON), dtype=np.float64)
    msg_probs = [None] * design.MESSAGE_SLOTS
    action_probs = [None] * design.HORIZON
    states = [None] * design.HORIZON
    locals_ = [None] * design.HORIZON
    inventories = [None] * design.HORIZON
    goal_idx = design.goal_index(ep["goal"])
    sender_context = goal_idx
    sender_logits = p["sender_logits_hidden"]
    selected = np.zeros((n, length), dtype=np.int64)
    for slot in range(length):
        sp = policy.softmax(sender_logits[sender_context, slot])
        msg_probs[slot] = sp
        if message_override is None:
            selected[:, slot] = policy.sample(sp, ep["message_uniforms"][:, slot]) if sample else sp.argmax(axis=-1)
        else:
            selected[:, slot] = np.asarray(message_override)[:, slot]
    delivered = selected.copy()
    if channel == "silent" or message_mode == "closed": delivered = np.full((n, length), design.NULL_MESSAGE, dtype=np.int64)
    elif message_mode == "permuted": delivered = _permute_sequences(selected, ep["partner_id"])
    if not np.all(active): delivered = np.where(active[:, None], delivered, design.NULL_MESSAGE)
    messages = _schedule_messages(selected, delivered, protocol, form)
    worker_idx = ep["partner_id"].astype(np.int64)
    memory = np.zeros(n, dtype=np.int64)
    received = np.zeros(n, dtype=np.int8)
    for t in range(design.HORIZON):
        incoming = messages[:, t - 1] if t > 0 else np.full(n, design.NULL_MESSAGE, dtype=np.int64)
        seen = incoming != design.NULL_MESSAGE
        if np.any(seen):
            first = seen & (received == 0)
            memory[first] = incoming[first] + 1
            later = seen & ~first
            memory[later] = 1 + (memory[later] - 1) * k + incoming[later]
            received[seen] += 1
        stage = min(max((t - design.ACTION_START) // design.STEPS_PER_SUBTASK, 0), design.SUBTASKS - 1)
        local = ep["site_type"][:, stage, 0].astype(np.int64)
        inv = remain[:, stage, :].sum(axis=1).astype(np.int64)
        ap = policy.softmax(p["worker_logits"][worker_idx, memory, t, local, inv])
        action_probs[t] = ap
        act = policy.sample(ap, ep["action_uniforms"][:, t]) if sample else ap.argmax(axis=-1)
        act = np.where((t < design.ACTION_START) | (~active), 0, act).astype(np.int64)
        actions[:, t] = act
        states[t] = memory.copy(); locals_[t] = local.copy(); inventories[t] = inv.copy()
        if t >= design.ACTION_START:
            for i in np.flatnonzero(active):
                a = int(act[i])
                if a == 0: continue
                site = a - 1; avail = int(remain[i, stage, site])
                if avail > 0: remain[i, stage, site] -= 1
                target = int(ep["target_bits"][i, stage])
                rewards[i, t] = design.CORRECT_REWARD if avail > 0 and int(ep["site_type"][i, stage, site]) == target else design.WRONG_REWARD
    return {"messages": messages, "selected_messages": selected, "actions": actions,
            "rewards": rewards, "team_return": rewards.mean(axis=1), "active": active,
            "message_probs": msg_probs, "action_probs": action_probs, "state": states,
            "local": locals_, "inventory": inventories, "final_inventory": remain,
            "episodes": ep, "message_mode": message_mode, "form": form, "task": task, "protocol": protocol}


def oracle_team_return(ep):
    return np.full(len(ep["goal"]), design.SUBTASKS * design.STEPS_PER_SUBTASK * design.CORRECT_REWARD / design.HORIZON, dtype=np.float64)
