"""Two-site resource world with channel and turnover interventions."""
from __future__ import annotations
import numpy as np
from . import design, policy


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def _permute_within_worker(tokens, partner_id):
    out = tokens.copy()
    for worker in range(design.WORKERS):
        idx = np.flatnonzero(partner_id == worker)
        if len(idx) > 1:
            out[idx] = tokens[np.roll(idx, 1)]
    return out


def rollout(p, ep, channel="live", *, message_mode="natural", sample=True, partner_filter=None):
    require(channel in design.CHANNELS, "bad channel")
    require(message_mode in ("natural", "closed", "silent", "permuted", "scrambled"), "bad message mode")
    n = len(ep["goal"])
    active = np.ones(n, dtype=bool) if partner_filter is None else (ep["partner_id"] == int(partner_filter))
    remain = np.stack([ep["capacity"], ep["capacity"]], axis=1).astype(np.int16)
    messages = np.full((n, design.HORIZON), design.NULL_MESSAGE, dtype=np.int64)
    actions = np.zeros((n, design.HORIZON), dtype=np.int64)
    rewards = np.zeros((n, design.HORIZON), dtype=np.float64)
    message_probs = [None] * design.HORIZON
    action_probs = [None] * design.HORIZON
    states = [None] * design.HORIZON
    local_types = [None] * design.HORIZON
    inventories = [None] * design.HORIZON
    goal = ep["goal"].astype(np.int64)
    sp = policy.softmax(p["sender_logits"][goal])
    selected = policy.sample(sp, ep["message_uniforms"][:, 0]) if sample else sp.argmax(axis=-1)
    selected = selected.astype(np.int64)
    if channel == "silent" or message_mode == "closed":
        delivered = np.full(n, design.NULL_MESSAGE, dtype=np.int64)
    elif message_mode == "permuted":
        delivered = _permute_within_worker(selected, ep["partner_id"])
    elif message_mode == "scrambled":
        offsets = np.floor(ep["scramble_uniforms"] * design.ALPHABET_SIZE).astype(np.int64)
        delivered = (selected + offsets) % design.ALPHABET_SIZE
    else:
        delivered = selected
    delivered = np.where(active, delivered, design.NULL_MESSAGE).astype(np.int64)
    messages[:, 0] = delivered
    message_probs[0] = sp
    memory = np.full(n, design.NULL_MESSAGE, dtype=np.int64)
    for t in range(design.HORIZON):
        incoming = messages[:, t - 1] if t > 0 else np.full(n, design.NULL_MESSAGE, dtype=np.int64)
        seen = incoming != design.NULL_MESSAGE
        memory[seen] = incoming[seen]
        state = memory.copy()
        local = ep["site_type"][:, 0].astype(np.int64)
        inv = np.clip(remain[:, 0], 0, design.HORIZON).astype(np.int64)
        ap = policy.softmax(p["worker_logits"][ep["partner_id"], state, t, local, inv])
        act = policy.sample(ap, ep["action_uniforms"][:, t]) if sample else ap.argmax(axis=-1)
        act = np.where((t < design.ACTION_START) | (~active), 0, act).astype(np.int64)
        actions[:, t] = act
        action_probs[t] = ap
        states[t] = state.copy(); local_types[t] = local.copy(); inventories[t] = inv.copy()
        if t >= design.ACTION_START:
            for i in np.flatnonzero(active):
                a = int(act[i])
                if a == 0:
                    continue
                site = a - 1
                avail = int(remain[i, site])
                if avail > 0:
                    remain[i, site] -= 1
                success = avail > 0 and int(ep["site_type"][i, site]) == int(ep["goal"][i])
                rewards[i, t] = design.CORRECT_REWARD if success else design.WRONG_REWARD
    return {
        "messages": messages, "selected_messages": selected, "actions": actions,
        "rewards": rewards, "team_return": rewards.mean(axis=1), "active": active,
        "message_probs": message_probs, "action_probs": action_probs,
        "state": states, "local": local_types, "inventory": inventories,
        "final_inventory": remain, "episodes": ep, "message_mode": message_mode,
    }


def oracle_team_return(ep):
    return np.full(len(ep["goal"]), (design.HORIZON - design.ACTION_START) * design.CORRECT_REWARD / design.HORIZON, dtype=np.float64)
