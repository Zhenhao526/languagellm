"""Visible three-object scenes with staged messages."""
from __future__ import annotations

import numpy as np

from . import design, policy


def _permute_sequences(sequences, partner_id):
    out = sequences.copy()
    for worker in range(design.WORKERS):
        idx = np.flatnonzero(partner_id == worker)
        if len(idx) > 1: out[idx] = sequences[np.roll(idx, 1)]
    return out


def _scene_index(site_type):
    x = np.asarray(site_type, dtype=np.int64)
    return x[..., 0] * (design.OBJECT_TYPES ** 2) + x[..., 1] * design.OBJECT_TYPES + x[..., 2]


def _schedule_messages(delivered, form):
    messages = np.full((len(delivered), design.HORIZON), design.NULL_MESSAGE, dtype=np.int64)
    for slot, arrival in enumerate(design.message_arrival_times(form)):
        messages[:, arrival] = delivered[:, slot]
    return messages


def rollout(params, episode, form, task, representation, channel, *, message_mode="natural", sample=True, partner_filter=None, message_override=None):
    design.require(form in design.FORMS and task in design.TASKS and representation in design.REPRESENTATIONS, "bad rollout factors")
    design.require(channel in design.CHANNELS and message_mode in ("natural", "closed", "permuted"), "bad channel")
    if representation == "slot_local": design.require(form == "tri3", "slot-local representation is defined only for tri3")
    n = len(episode["goal"]); active = np.ones(n, dtype=bool) if partner_filter is None else episode["partner_id"] == int(partner_filter)
    remain = episode["capacity"].astype(np.int16).copy(); actions = np.zeros((n, design.HORIZON), dtype=np.int64); rewards = np.zeros((n, design.HORIZON), dtype=np.float64)
    length = design.message_length(form); k = design.alphabet_size(form); message_probs = [None] * length; action_probs = [None] * design.HORIZON; states = [None] * design.HORIZON; scenes = [None] * design.HORIZON; inventories = [None] * design.HORIZON
    context = policy.sender_context(episode["goal"]); selected = np.zeros((n, length), dtype=np.int64)
    for slot in range(length):
        sp = policy.softmax(params["sender_logits"][context, slot]); message_probs[slot] = sp
        selected[:, slot] = policy.sample(sp, episode["message_uniforms"][:, slot]) if message_override is None and sample else (sp.argmax(axis=-1) if message_override is None else np.asarray(message_override)[:, slot])
    delivered = selected.copy()
    if channel == "silent" or message_mode == "closed": delivered = np.full_like(delivered, design.NULL_MESSAGE)
    elif message_mode == "permuted": delivered = _permute_sequences(selected, episode["partner_id"])
    if not np.all(active): delivered = np.where(active[:, None], delivered, design.NULL_MESSAGE)
    messages = _schedule_messages(delivered, form); worker_idx = episode["partner_id"].astype(np.int64); scene = _scene_index(episode["site_type"])
    history = np.zeros(n, dtype=np.int64); received = np.zeros(n, dtype=np.int8); slot_memory = np.zeros((n, design.SUBTASKS), dtype=np.int64)
    for t in range(design.HORIZON):
        incoming = messages[:, t - 1] if t > 0 else np.full(n, design.NULL_MESSAGE, dtype=np.int64)
        stage = min(max((t - design.ACTION_START) // design.STEPS_PER_SUBTASK, 0), design.SUBTASKS - 1)
        seen = incoming != design.NULL_MESSAGE
        if np.any(seen):
            if representation == "slot_local": slot_memory[seen, stage] = incoming[seen] + 1
            else:
                first = seen & (received == 0); later = seen & ~first; history[first] = incoming[first] + 1
                if np.any(later): history[later] = 1 + (history[later] - 1) * k + incoming[later]
            received[seen] += 1
        state = slot_memory[:, stage] if representation == "slot_local" else history; local_scene = scene[:, stage].astype(np.int64); inventory = remain[:, stage, :].sum(axis=1).astype(np.int64)
        ap = policy.softmax(params["worker_logits"][worker_idx, state, t, local_scene, inventory]); action_probs[t] = ap; states[t] = state.copy(); scenes[t] = local_scene.copy(); inventories[t] = inventory.copy()
        act = policy.sample(ap, episode["action_uniforms"][:, t]) if sample else ap.argmax(axis=-1); act = np.where((t < design.ACTION_START) | (~active), 0, act).astype(np.int64); actions[:, t] = act
        if t >= design.ACTION_START:
            for i in np.flatnonzero(active):
                action = int(act[i])
                if action == 0: continue
                site = action - 1; available = int(remain[i, stage, site])
                if available > 0: remain[i, stage, site] -= 1
                target = int(episode["target_bits"][i, stage]); rewards[i, t] = design.CORRECT_REWARD if available > 0 and int(episode["site_type"][i, stage, site]) == target else design.WRONG_REWARD
    return {"messages": messages, "selected_messages": selected, "actions": actions, "rewards": rewards, "team_return": rewards.mean(axis=1), "active": active, "message_probs": message_probs, "action_probs": action_probs, "state": states, "scene": scenes, "inventory": inventories, "final_inventory": remain, "episodes": episode, "message_mode": message_mode, "form": form, "task": task, "representation": representation, "channel": channel}


def oracle_team_return():
    return design.SUBTASKS * design.STEPS_PER_SUBTASK * design.CORRECT_REWARD / design.HORIZON


def deterministic_sequence(params, form, goal):
    ctx = int(design.goal_index(np.asarray(goal, dtype=np.int8)))
    return np.asarray([policy.softmax(params["sender_logits"][ctx, slot]).argmax() for slot in range(design.message_length(form))], dtype=np.int64)


def recombination_override(params, episode, form, task, *, basis="raw"):
    if form != "tri3": return None
    goals = np.asarray(design.GOAL_PAIRS, dtype=np.int8); out = np.zeros((len(episode["goal"]), 2), dtype=np.int64)
    for i, goal in enumerate(episode["goal"]):
        if basis == "raw": target = np.asarray(goal, dtype=np.int8)
        elif basis == "task": target = design.target_bits(np.asarray(goal, dtype=np.int8)[None, :], task)[0]
        else: raise ValueError("unknown recombination basis")
        if basis == "raw":
            donor0 = goals[np.flatnonzero(goals[:, 0] == target[0])[0]]; donor1 = goals[np.flatnonzero(goals[:, 1] == target[1])[0]]
        else:
            mapped = design.target_bits(goals, task); donor0 = goals[np.flatnonzero(mapped[:, 0] == target[0])[0]]; donor1 = goals[np.flatnonzero(mapped[:, 1] == target[1])[0]]
        out[i, 0] = deterministic_sequence(params, form, donor0)[0]; out[i, 1] = deterministic_sequence(params, form, donor1)[1]
    return out
