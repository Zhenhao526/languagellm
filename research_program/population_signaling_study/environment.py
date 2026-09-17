"""Rollout, controls and causal population codebook diagnostics."""
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


def rollout(p, ep, partner_mode, partner_visibility, scarcity, channel, *, message_mode="natural", sample=True,
           partner_filter=None):
    n = len(ep["goal"])
    require(channel in design.CHANNELS and message_mode in ("natural", "closed", "permuted"), "bad channel")
    active = np.ones(n, dtype=bool) if partner_filter is None else (ep["partner_id"] == int(partner_filter))
    remain = np.stack([ep["capacity"], ep["capacity"]], axis=1).astype(np.int16)
    messages = np.full((n, design.HORIZON), design.NULL_MESSAGE, dtype=np.int64)
    actions = np.zeros((n, design.HORIZON), dtype=np.int64)
    rewards = np.zeros((n, design.HORIZON), dtype=np.float64)
    msg_probs = [None] * design.HORIZON
    action_probs = [None] * design.HORIZON
    states = [None] * design.HORIZON
    local_types = [None] * design.HORIZON
    inventories = [None] * design.HORIZON
    memory = np.full(n, design.NULL_MESSAGE, dtype=np.int64)
    sender_context = ep["goal"].astype(np.int64)
    if partner_visibility == "visible":
        # The sender is allowed to condition on partner ID in this control.
        sender_context = sender_context * design.WORKERS + ep["partner_id"].astype(np.int64)
        sender_logits = p["sender_logits_visible"]
    else:
        sender_logits = p["sender_logits_hidden"]
    sp = policy.softmax(sender_logits[sender_context])
    msg = policy.sample(sp, ep["message_uniforms"][:, 0]) if sample else sp.argmax(axis=-1)
    msg = msg.astype(np.int64)
    if channel == "silent" or message_mode == "closed":
        delivered = np.full(n, design.NULL_MESSAGE, dtype=np.int64)
    elif message_mode == "permuted":
        delivered = _permute_within_worker(msg, ep["partner_id"])
    else:
        delivered = msg
    if not np.all(active):
        delivered = np.where(active, delivered, design.NULL_MESSAGE)
    messages[:, 0] = delivered
    msg_probs[0] = sp
    for t in range(design.HORIZON):
        incoming = messages[:, t - 1] if t > 0 else np.full(n, design.NULL_MESSAGE, dtype=np.int64)
        seen = incoming != design.NULL_MESSAGE
        memory[seen] = incoming[seen]
        state = memory.copy()
        local = ep["site_type"][:, 0].astype(np.int64)  # local site 0's type; site 1 is its complement
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
    return {"messages": messages, "actions": actions, "rewards": rewards,
            "team_return": rewards.mean(axis=1), "active": active,
            "message_probs": msg_probs, "action_probs": action_probs,
            "state": states, "local": local_types, "inventory": inventories,
            "final_inventory": remain, "episodes": ep, "message_mode": message_mode}


def oracle_team_return(ep):
    # Full-information oracle for one active worker in each episode.
    n = len(ep["goal"]); c = int(ep["capacity"][0]); dp = np.zeros((n, c + 1, c + 1), dtype=np.float64)
    for t in range(design.HORIZON - 1, -1, -1):
        nxt = np.full_like(dp, -np.inf)
        acts = [0] if t < design.ACTION_START else range(design.ACTION_COUNT)
        for r0 in range(c + 1):
            for r1 in range(c + 1):
                best = np.full(n, -np.inf)
                for a in acts:
                    if a == 0:
                        cand = dp[:, r0, r1]
                    else:
                        s = a - 1; avail = (r0, r1)[s]
                        rr0 = r0 - int(s == 0 and avail > 0); rr1 = r1 - int(s == 1 and avail > 0)
                        val = np.where((avail > 0) & (ep["site_type"][:, s] == ep["goal"]),
                                       design.CORRECT_REWARD, design.WRONG_REWARD)
                        cand = val + dp[:, rr0, rr1]
                    best = np.maximum(best, cand)
                nxt[:, r0, r1] = best
        dp = nxt
    return dp[:, c, c] / float(design.HORIZON)
