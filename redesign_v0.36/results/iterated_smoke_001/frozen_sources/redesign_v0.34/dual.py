"""Two-sender/one-receiver policy-gradient loss for v0.33."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "redesign_v0.21"))
from social_model import _draw, _next_logits, _sender_start, receive_messages


def _validate(h, uniforms, positions):
    if not isinstance(h, torch.Tensor) or h.ndim != 2 or h.shape[1] != 96 or h.dtype != torch.float32 or h.device.type != "cpu" or h.requires_grad or not bool(torch.isfinite(h).all()):
        raise ValueError("h must be frozen CPU float32[B,96]")
    if not isinstance(uniforms, np.ndarray) or uniforms.shape != (len(h), 4) or uniforms.dtype != np.float32 or not np.isfinite(uniforms).all() or not ((uniforms >= 0) & (uniforms <= 1)).all():
        raise ValueError("uniforms must be float32[B,4]")
    if not isinstance(positions, np.ndarray) or positions.shape != (len(h), 2) or not np.issubdtype(positions.dtype, np.integer) or not ((positions >= 0) & (positions < 6)).all() or (positions[:, 0] == positions[:, 1]).any():
        raise ValueError("positions must be distinct integer[B,2]")


def dual_loss(food_sender, water_sender, receiver, h_food, h_water, uniforms, positions, entropy_weight):
    """Return separate sender losses and one receiver loss for a dual team.

    The food sender contributes the first token and sees only its designated
    view; the water sender contributes the second token. The receiver sees the
    ordered pair and no visual state. Both sender losses and the receiver loss
    use the same detached two-action reward ``.25*(cF+cW)+.5*cF*cW``.
    """
    if food_sender is water_sender or food_sender is receiver or water_sender is receiver:
        raise ValueError("team agents must be independent")
    if set(map(id, food_sender.parameters())) & set(map(id, water_sender.parameters())):
        raise ValueError("food and water senders must be independent")
    if set(map(id, food_sender.parameters())) & set(map(id, receiver.parameters())) or set(map(id, water_sender.parameters())) & set(map(id, receiver.parameters())):
        raise ValueError("senders and receiver must be independent")
    if len(h_food) != len(h_water):
        raise ValueError("sender views must share batch size")
    _validate(h_food, uniforms, positions)
    _validate(h_water, uniforms, positions)
    if not np.isscalar(entropy_weight) or not np.isfinite(entropy_weight) or entropy_weight < 0:
        raise ValueError("entropy_weight must be finite and nonnegative")

    state_f, logits_f = _sender_start(food_sender, h_food)
    state_w, logits_w = _sender_start(water_sender, h_water)
    u = torch.from_numpy(uniforms)
    token_f, logp_f, entropy_f, prob_f = _draw(logits_f, u[:, 0:1])
    token_w, logp_w, entropy_w, prob_w = _draw(logits_w, u[:, 1:2])
    messages = torch.stack((token_f, token_w), dim=1).detach()
    action_logits = receive_messages(receiver, messages)
    food, logpf, entropy_action_f, prob_action_f = _draw(action_logits[:, 0], u[:, 2:3])
    water, logpw, entropy_action_w, prob_action_w = _draw(action_logits[:, 1], u[:, 3:4])
    actions = torch.stack((food, water), dim=1)
    success = (actions.detach().numpy() == positions).astype(np.float32)
    reward = (.25 * success.sum(1) + .5 * success.prod(1)).astype(np.float32)
    advantage = torch.from_numpy(reward - np.float32(.5))
    ew = float(entropy_weight)
    sender_food_loss = -(logp_f * advantage).mean() - ew * entropy_f.mean()
    sender_water_loss = -(logp_w * advantage).mean() - ew * entropy_w.mean()
    receiver_loss = -((logpf + logpw) * advantage).mean() - ew * (entropy_action_f + entropy_action_w).mean()
    losses = (sender_food_loss, sender_water_loss, receiver_loss)
    if not all(bool(torch.isfinite(x)) for x in losses):
        raise ValueError("non-finite dual communication loss")
    as_np = lambda x: x.detach().numpy().copy()
    trace = {
        "h_food": as_np(h_food), "h_water": as_np(h_water), "uniforms": uniforms.copy(),
        "messages": as_np(messages), "token_food": as_np(token_f), "token_water": as_np(token_w),
        "first_logits_food": as_np(logits_f), "first_logits_water": as_np(logits_w),
        "token_probabilities_food": as_np(prob_f), "token_probabilities_water": as_np(prob_w),
        "action_logits": as_np(action_logits),
        "action_probabilities": as_np(torch.stack((prob_action_f, prob_action_w), 1)),
        "actions": as_np(actions), "positions": positions.copy(), "success": success,
        "reward": reward, "advantage": as_np(advantage), "sender_logp_food": as_np(logp_f),
        "sender_logp_water": as_np(logp_w), "sender_entropy_food": as_np(entropy_f),
        "sender_entropy_water": as_np(entropy_w), "receiver_logp": as_np(logpf + logpw),
        "receiver_entropy": as_np(entropy_action_f + entropy_action_w),
        "sender_food_loss": float(sender_food_loss.detach()), "sender_water_loss": float(sender_water_loss.detach()),
        "receiver_loss": float(receiver_loss.detach()), "entropy_weight": ew, "baseline": .5,
    }
    return sender_food_loss, sender_water_loss, receiver_loss, trace


def joint_sender_tables(first_food, first_water):
    """Return the 49-code log-probability table from two independent tokens."""
    a = F.log_softmax(first_food, -1)
    b = F.log_softmax(first_water, -1)
    return (a[:, :, None] + b[:, None, :]).reshape(len(a), 49)


@torch.no_grad()
def first_token_log_probs(sender, h):
    """Return the seven-way first-token log table for evaluation."""
    if not isinstance(h, torch.Tensor) or h.ndim != 2 or h.shape[1] != 96 or h.dtype != torch.float32 or h.device.type != "cpu" or h.requires_grad or not bool(torch.isfinite(h).all()):
        raise ValueError("h must be frozen CPU float32[B,96]")
    _, logits = _sender_start(sender, h)
    return F.log_softmax(logits, -1)


@torch.no_grad()
def first_token_greedy(sender, h):
    """Return sequential first-token argmax values for evaluation."""
    return first_token_log_probs(sender, h).argmax(-1)
