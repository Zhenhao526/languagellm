"""Staged environment with an optional slot-local receiver and message override."""
from __future__ import annotations

import numpy as np

from research_program.action_dependent_signaling_study import design, policy


def _permute_sequences(sequences, partner_id):
    out = sequences.copy()
    for worker in range(design.WORKERS):
        idx = np.flatnonzero(partner_id == worker)
        if len(idx) > 1:
            out[idx] = sequences[np.roll(idx, 1)]
    return out


def _schedule_messages(delivered, protocol="staged"):
    n, length = delivered.shape
    messages = np.full((n, design.HORIZON), design.NULL_MESSAGE, dtype=np.int64)
    times = design.message_arrival_times(protocol, "dual2")
    for slot, message_index in enumerate(times):
        messages[:, message_index] = delivered[:, slot]
    return messages


def rollout_slot_local(params, ep, channel, *, message_mode="natural", sample=True,
                       partner_filter=None, message_override=None):
    require = design.require
    require(channel in design.CHANNELS and message_mode in ("natural", "closed", "permuted"), "bad channel")
    n = len(ep["goal"])
    active = np.ones(n, dtype=bool) if partner_filter is None else (ep["partner_id"] == int(partner_filter))
    remain = ep["capacity"].astype(np.int16).copy()
    actions = np.zeros((n, design.HORIZON), dtype=np.int64)
    rewards = np.zeros((n, design.HORIZON), dtype=np.float64)
    action_probs = [None] * design.HORIZON
    states = [None] * design.HORIZON
    locals_ = [None] * design.HORIZON
    inventories = [None] * design.HORIZON
    sender_context = design.goal_index(ep["goal"])
    selected = np.zeros((n, design.MESSAGE_SLOTS), dtype=np.int64)
    for slot in range(design.MESSAGE_SLOTS):
        sp = policy.softmax(params["sender_logits_hidden"][sender_context, slot])
        if message_override is None:
            selected[:, slot] = policy.sample(sp, ep["message_uniforms"][:, slot]) if sample else sp.argmax(axis=-1)
        else:
            selected[:, slot] = np.asarray(message_override)[:, slot]
    delivered = selected.copy()
    if channel == "silent" or message_mode == "closed":
        delivered = np.full_like(delivered, design.NULL_MESSAGE)
    elif message_mode == "permuted":
        delivered = _permute_sequences(selected, ep["partner_id"])
    if not np.all(active):
        delivered = np.where(active[:, None], delivered, design.NULL_MESSAGE)
    messages = _schedule_messages(delivered, "staged")
    slot_memory = np.zeros((n, design.SUBTASKS), dtype=np.int64)
    worker_idx = ep["partner_id"].astype(np.int64)
    for t in range(design.HORIZON):
        incoming = messages[:, t - 1] if t > 0 else np.full(n, design.NULL_MESSAGE, dtype=np.int64)
        stage = min(max((t - design.ACTION_START) // design.STEPS_PER_SUBTASK, 0), design.SUBTASKS - 1)
        if t >= design.ACTION_START:
            seen = incoming != design.NULL_MESSAGE
            slot_memory[seen, stage] = incoming[seen] + 1
        state = slot_memory[:, stage]
        local = ep["site_type"][:, stage, 0].astype(np.int64)
        inventory = remain[:, stage, :].sum(axis=1).astype(np.int64)
        ap = policy.softmax(params["worker_logits"][worker_idx, state, t, local, inventory])
        action_probs[t] = ap
        act = policy.sample(ap, ep["action_uniforms"][:, t]) if sample else ap.argmax(axis=-1)
        act = np.where((t < design.ACTION_START) | (~active), 0, act).astype(np.int64)
        actions[:, t] = act
        states[t] = state.copy()
        locals_[t] = local.copy()
        inventories[t] = inventory.copy()
        if t >= design.ACTION_START:
            for i in np.flatnonzero(active):
                action = int(act[i])
                if action == 0:
                    continue
                site = action - 1
                available = int(remain[i, stage, site])
                if available > 0:
                    remain[i, stage, site] -= 1
                target = int(ep["target_bits"][i, stage])
                rewards[i, t] = design.CORRECT_REWARD if available > 0 and int(ep["site_type"][i, stage, site]) == target else design.WRONG_REWARD
    return {
        "messages": messages,
        "selected_messages": selected,
        "actions": actions,
        "rewards": rewards,
        "team_return": rewards.mean(axis=1),
        "active": active,
        "action_probs": action_probs,
        "state": states,
        "local": locals_,
        "inventory": inventories,
        "final_inventory": remain,
        "episodes": ep,
        "message_mode": message_mode,
        "representation": "slot_local",
    }
