"""Eight-stage environment with simultaneous noisy message delivery."""
from __future__ import annotations

import numpy as np

from . import design, policy


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def _permute_sequences(sequences, partner_id):
    out = sequences.copy()
    for worker in range(design.WORKERS):
        idx = np.flatnonzero(partner_id == worker)
        if len(idx) > 1:
            out[idx] = sequences[np.roll(idx, 1)]
    return out


def _corrupt(selected, form, noise_p, noise_uniforms):
    delivered = selected.copy()
    p = float(noise_p)
    if p <= 0.0:
        return delivered
    if form in ("triple2", "quad2"):
        flip = noise_uniforms[:, :, 0] < p
        delivered[flip] = 1 - delivered[flip]
        return delivered
    require(form == "atomic16", "unknown form")
    q = design.atomic_corruption_probability(p)
    replace = noise_uniforms[:, 0, 0] < q
    delivered[replace, 0] = (delivered[replace, 0] + 1 + (noise_uniforms[replace, 0, 1] * 15).astype(np.int64)) % 16
    return delivered

def rollout(params, episode, form, channel, noise_p, *, message_mode="natural", sample=True, partner_filter=None, message_override=None):
    k = design.alphabet_size(form)
    length = design.message_length(form)
    require(form in design.FORMS and channel in ("live", "silent"), "bad form or channel")
    require(message_mode in ("natural", "permuted"), "bad message mode")
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
    if channel == "silent":
        delivered.fill(design.NULL_MESSAGE)
    elif message_mode == "permuted":
        delivered = _permute_sequences(selected, episode["partner_id"])
    else:
        delivered = _corrupt(delivered, form, noise_p, episode["noise_uniforms"])
    delivered = np.where(active[:, None], delivered, design.NULL_MESSAGE)

    # Multiple message slots are all delivered at the same pre-action event.
    # Processing them in slot order creates a deterministic history state.
    memory = np.zeros(n, dtype=np.int64)
    worker_idx = episode["partner_id"].astype(np.int64)
    for t in range(design.HORIZON):
        if t == design.ACTION_START:
            for slot in range(length):
                incoming = delivered[:, slot]
                seen = incoming != design.NULL_MESSAGE
                first = seen & (memory == 0)
                later = seen & ~first
                memory[first] = incoming[first] + 1
                memory[later] = 1 + (memory[later] - 1) * k + incoming[later]
        stage = min(max((t - design.ACTION_START) // design.STEPS_PER_SUBTASK, 0), design.SUBTASKS - 1)
        local = episode["site_type"][:, stage, 0].astype(np.int64)
        inventory = remain[:, stage, :].sum(axis=1).astype(np.int64)
        ap = policy.softmax(params["worker_logits"][worker_idx, memory, t, local, inventory])
        action_probs[t] = ap
        act = policy.sample(ap, episode["action_uniforms"][:, t]) if sample else ap.argmax(axis=-1)
        act = np.where((t < design.ACTION_START) | (~active), 0, act).astype(np.int64)
        actions[:, t] = act
        states[t] = memory.copy()
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
                target = int(episode["target_bits"][i, stage])
                rewards[i, t] = design.CORRECT_REWARD if available > 0 and int(episode["site_type"][i, stage, site]) == target else design.WRONG_REWARD
    return {
        "delivered_messages": delivered,
        "selected_messages": selected,
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
        "noise_p": float(noise_p),
        "channel": channel,
        "message_mode": message_mode,
        "form": form,
    }


def oracle_team_return(episode):
    return np.full(len(episode["goal"]), design.SUBTASKS * design.STEPS_PER_SUBTASK * design.CORRECT_REWARD / design.HORIZON, dtype=np.float64)
