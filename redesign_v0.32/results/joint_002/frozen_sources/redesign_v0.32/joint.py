"""Joint-round policy-gradient loss for v0.32.

Two directions of one matched pair use the same frozen world batch.  The
reward is the v0.31 two-action score generalized to four actions: each action
contributes .125 and an additional .5 is paid only when all four actions are
correct.  The same detached advantage trains both directions.
"""
from __future__ import annotations
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'redesign_v0.21'))
from social_model import _draw, _next_logits, _sender_start, receive_messages


def joint_reward(success_i, success_j):
    """Compute the detached four-action payoff for one matched pair."""
    a = np.asarray(success_i, dtype=np.float32); b = np.asarray(success_j, dtype=np.float32)
    if a.ndim != 2 or b.shape != a.shape or a.shape[1] != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('success matrices must be finite [B,2]')
    joint = (a.all(1) & b.all(1)).astype(np.float32)
    return (.125 * (a.sum(1) + b.sum(1)) + .5 * joint).astype(np.float32)


def _validate_inputs(h, uniforms, positions):
    if not isinstance(h, torch.Tensor) or h.ndim != 2 or h.shape[1] != 96 or h.dtype != torch.float32 or h.device.type != 'cpu' or h.requires_grad or not bool(torch.isfinite(h).all()):
        raise ValueError('h must be frozen CPU float32[B,96]')
    if not isinstance(uniforms, np.ndarray) or uniforms.shape != (len(h), 4) or uniforms.dtype != np.float32 or not np.isfinite(uniforms).all() or not ((uniforms >= 0) & (uniforms <= 1)).all():
        raise ValueError('uniforms must be float32[B,4]')
    if not isinstance(positions, np.ndarray) or positions.shape != (len(h), 2) or not np.issubdtype(positions.dtype, np.integer) or not ((positions >= 0) & (positions < 6)).all() or (positions[:, 0] == positions[:, 1]).any():
        raise ValueError('positions must be distinct integer[B,2]')


def trajectory(sender, receiver, h, uniforms, positions):
    """Sample one direction while retaining differentiable log probabilities."""
    _validate_inputs(h, uniforms, positions)
    state, first_logits = _sender_start(sender, h)
    ut = torch.from_numpy(uniforms)
    first, logp0, entropy0, prob0 = _draw(first_logits, ut[:, 0:1])
    second_logits = _next_logits(sender, state, first)
    second, logp1, entropy1, prob1 = _draw(second_logits, ut[:, 1:2])
    messages = torch.stack((first, second), dim=1).detach()
    action_logits = receive_messages(receiver, messages)
    food, logpf, entropyf, probf = _draw(action_logits[:, 0], ut[:, 2:3])
    water, logpw, entropyw, probw = _draw(action_logits[:, 1], ut[:, 3:4])
    actions = torch.stack((food, water), dim=1)
    success = (actions.detach().numpy() == positions).astype(np.float32)
    as_np = lambda x: x.detach().numpy().copy()
    return dict(h=as_np(h), uniforms=uniforms.copy(), messages=as_np(messages),
                first_logits=as_np(first_logits), second_logits=as_np(second_logits),
                token_probabilities=as_np(torch.stack((prob0, prob1), 1)),
                action_logits=as_np(action_logits),
                action_probabilities=as_np(torch.stack((probf, probw), 1)),
                actions=as_np(actions), positions=positions.copy(), success=success,
                sender_logp=logp0 + logp1, receiver_logp=logpf + logpw,
                sender_entropy=entropy0 + entropy1, receiver_entropy=entropyf + entropyw)


def pair_loss(agent_i, agent_j, h_i, h_j, uniforms_i, uniforms_j, positions, entropy_weight):
    """Return role losses for i→j and j→i under one joint payoff.

    Returns ``(sender_i, receiver_j, sender_j, receiver_i, traces)``.  The
    four losses share one detached per-world advantage, while entropy is local
    to each direction.  No optimizer step or gradient clipping occurs here.
    """
    if agent_i is agent_j or set(map(id, agent_i.parameters())) & set(map(id, agent_j.parameters())):
        raise ValueError('partner agents must have independent parameters')
    if len(h_i) != len(h_j):
        raise ValueError('directions must share batch size')
    if not np.isscalar(entropy_weight) or not np.isfinite(entropy_weight) or entropy_weight < 0:
        raise ValueError('entropy_weight must be finite and nonnegative')
    left = trajectory(agent_i, agent_j, h_i, uniforms_i, positions)
    right = trajectory(agent_j, agent_i, h_j, uniforms_j, positions)
    reward = joint_reward(left['success'], right['success'])
    joint = (left['success'].all(1) & right['success'].all(1)).astype(np.float32)
    advantage = torch.from_numpy(reward - np.float32(.5))
    ew = float(entropy_weight)
    losses = (
        -(left['sender_logp'] * advantage).mean() - ew * left['sender_entropy'].mean(),
        -(left['receiver_logp'] * advantage).mean() - ew * left['receiver_entropy'].mean(),
        -(right['sender_logp'] * advantage).mean() - ew * right['sender_entropy'].mean(),
        -(right['receiver_logp'] * advantage).mean() - ew * right['receiver_entropy'].mean(),
    )
    if not all(bool(torch.isfinite(x)) for x in losses):
        raise ValueError('non-finite joint communication loss')
    for trace in (left, right):
        trace['reward'] = reward.copy(); trace['advantage'] = advantage.detach().numpy().copy(); trace['joint_success'] = joint.copy(); trace['baseline'] = .5; trace['entropy_weight'] = ew
        trace['sender_loss'] = float(losses[0].detach()) if trace is left else float(losses[2].detach())
        trace['receiver_loss'] = float(losses[1].detach()) if trace is left else float(losses[3].detach())
        trace['sender_logp'] = trace['sender_logp'].detach().numpy().copy(); trace['receiver_logp'] = trace['receiver_logp'].detach().numpy().copy(); trace['sender_entropy'] = trace['sender_entropy'].detach().numpy().copy(); trace['receiver_entropy'] = trace['receiver_entropy'].detach().numpy().copy()
    return (*losses, {'i_to_j': left, 'j_to_i': right})
