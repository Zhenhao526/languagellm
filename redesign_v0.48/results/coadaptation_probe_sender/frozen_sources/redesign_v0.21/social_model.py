"""Discrete communication over a frozen, normalized two-frame visual state.

No private action head is accepted. The sender has no demand input; the
receiver has no visual state. Location truth is used only after both actions
have been sampled. Optimizers, role averaging and clipping belong to caller.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VOCAB, LENGTH, SITES, WIDTH, HISTORY = 7, 2, 6, 96, 18
SENDER_MODULES = ('send_context', 'send_embedding', 'send_recur', 'send_out')
RECEIVER_MODULES = ('receive_embedding', 'actor')
UNUSED_MODULES = ('send_value', 'receive_value')
COMMUNICATION_MODULES = SENDER_MODULES + RECEIVER_MODULES + UNUSED_MODULES


def _validate_agent(agent):
    if (agent.vocab, agent.length, agent.width) != (VOCAB, LENGTH, WIDTH):
        raise ValueError('expected original CampAgent: vocab7, length2, width96')
    if type(agent.send_context) is not nn.Sequential:
        raise ValueError('expected unwrapped CampAgent send_context')
    if agent.send_context[0].in_features != 100 or agent.actor[0].in_features != 52:
        raise ValueError('unexpected sender or receiver interface dimension')
    if hasattr(agent, 'visual_scale') or hasattr(agent, 'policy_scale'):
        raise ValueError('v21 requires plain CampAgent; temporal output is already normalized')
    if any(p.dtype != torch.float32 or p.device.type != 'cpu' for p in agent.parameters()):
        raise ValueError('CPU float32 parameters required')


def trainable_groups(agent):
    """Return disjoint parameter lists, without changing flags or parameters."""
    _validate_agent(agent)
    groups = {role: [p for name in names for p in getattr(agent, name).parameters()]
              for role, names in (('sender', SENDER_MODULES), ('receiver', RECEIVER_MODULES))}
    ids = [id(p) for group in groups.values() for p in group]
    if len(ids) != len(set(ids)):
        raise ValueError('sender and receiver parameter objects must be disjoint')
    return groups


def reset_communication(agent, seed):
    """Fresh communication, in place, with no external Torch RNG mutation.

    All original communication/value modules are reinitialized in the declared
    order. The two unused value modules remain frozen and are never evaluated.
    The visual project, memory, slot_phi and input transform are unchanged.
    The caller must load the trained temporal frontend before this operation.
    """
    _validate_agent(agent)
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**63:
        raise ValueError('seed must be an explicit nonnegative 63-bit integer')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for name in COMMUNICATION_MODULES:
            for module in getattr(agent, name).modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
    agent.requires_grad_(False)
    groups = trainable_groups(agent)
    for params in groups.values():
        for parameter in params:
            parameter.requires_grad_(True)
    # Clear potentially inherited .grad slots; no optimizer state is accepted.
    for parameter in agent.parameters():
        parameter.grad = None
    return {'seed': int(seed), 'reset_module_order': list(COMMUNICATION_MODULES),
            'trainable_parameters': {role: sum(p.numel() for p in params)
                                     for role, params in groups.items()},
            'trainable_tensors': {role: len(params) for role, params in groups.items()}}


def _validate_h(h):
    if (not isinstance(h, torch.Tensor) or h.ndim != 2 or h.shape[1] != WIDTH
            or len(h) == 0 or h.dtype != torch.float32 or h.device.type != 'cpu'
            or h.requires_grad or not bool(torch.isfinite(h).all())):
        raise ValueError('h must be finite frozen CPU float32[B,96], B>0')


def _sender_start(agent, h):
    _validate_h(h)
    # No visible goal and no inventory; both local vectors are identically zero.
    state = agent.send_context(torch.cat((h, h.new_zeros(len(h), 4)), -1))
    return state, agent.send_out(state)


def _next_logits(agent, state, token):
    return agent.send_out(agent.send_recur(agent.send_embedding(token.detach()), state))


def message_log_probs(agent, h):
    """Return differentiable float32[B,49] log p(t0,t1), code=7*t0+t1.

    Each first-token prefix is evaluated as its own B-row recurrent call. This
    retains the original autoregressive parameterization. Do not replace
    sequential greedy decoding by argmax over this joint table.
    """
    state, first_logits = _sender_start(agent, h)
    first = F.log_softmax(first_logits, -1)
    second = torch.stack([F.log_softmax(_next_logits(
        agent, state, torch.full((len(h),), token, dtype=torch.int64)), -1)
        for token in range(VOCAB)], dim=1)
    return (first[:, :, None] + second).reshape(len(h), VOCAB ** LENGTH)


def greedy_messages(agent, h):
    """Sequential first-index argmax, matching the two-token sender policy."""
    state, first_logits = _sender_start(agent, h)
    first = first_logits.argmax(-1).detach()
    second = _next_logits(agent, state, first).argmax(-1).detach()
    return torch.stack((first, second), dim=1)


def receive_messages(agent, messages):
    """Return logits[B,2,6] from only integer messages and zero local context."""
    if (not isinstance(messages, torch.Tensor) or messages.ndim != 2
            or messages.shape[1] != LENGTH or len(messages) == 0
            or messages.dtype != torch.int64 or messages.device.type != 'cpu'
            or messages.requires_grad or not bool(((messages >= 0) & (messages < VOCAB)).all())):
        raise ValueError('messages must be CPU int64[B,2] tokens in0..6')
    embedded = agent.receive_embedding(messages.detach()).flatten(1)
    # The two demand branches share one actor, but receive no other query's
    # action, probability or success. Inventory2 and history18 are constant0.
    context = torch.cat((embedded, embedded.new_zeros(len(messages), 2 + HISTORY)), -1)
    return agent.actor(context).reshape(len(messages), 2, SITES)


def receiver_logits(agent):
    """Return all49 message rows, ordered lexicographically, shape[49,2,6]."""
    codes = torch.arange(VOCAB ** LENGTH, dtype=torch.int64)
    return receive_messages(agent, torch.stack((codes // VOCAB, codes % VOCAB), dim=1))


def _draw(logits, uniform):
    log_probs = F.log_softmax(logits, -1)
    probabilities = log_probs.exp()
    action = (probabilities.detach().cumsum(-1) < uniform).sum(-1).clamp(max=logits.shape[-1] - 1)
    logp = log_probs.gather(1, action[:, None]).squeeze(1)
    entropy = -(probabilities * log_probs).sum(-1)
    return action.detach(), logp, entropy, probabilities


def direction_loss(sender, receiver, h, uniforms, positions, entropy_weight):
    """Return (sender_loss, receiver_loss, trace) for one communication direction.

    uniforms is an external float32[B,4] array: token0, token1, food action,
    water action. Both actions use the SAME message and independently supplied
    draws. Each returned role loss is already a mean over worlds, with summed
    token/goal log probabilities and entropies. Caller forms its stated role
    average; this function does not divide by2, clip, backward or step.

    R=.25*(rF+rW)+.5*rF*rW, and advantage=R-.5. There are no critics. Only the
    observed first-token prefix contributes the second-token entropy, following
    the original sampled-prefix sender loss (not full marginal sequence entropy).
    """
    _validate_h(h)
    if sender is receiver or set(map(id, sender.parameters())) & set(map(id, receiver.parameters())):
        raise ValueError('different agents must not share parameter objects')
    if (not isinstance(uniforms, np.ndarray) or uniforms.shape != (len(h), 4)
            or uniforms.dtype != np.float32 or not np.isfinite(uniforms).all()
            or not ((uniforms >= 0) & (uniforms <= 1)).all()):
        raise ValueError('uniforms must be external numpy float32[B,4] in0..1')
    if (not isinstance(positions, np.ndarray) or positions.shape != (len(h), 2)
            or not np.issubdtype(positions.dtype, np.integer)
            or not ((positions >= 0) & (positions < SITES)).all()
            or (positions[:, 0] == positions[:, 1]).any()):
        raise ValueError('positions must be distinct evaluator-only integer[B,2] locations')
    if not np.isscalar(entropy_weight) or not np.isfinite(entropy_weight) or entropy_weight < 0:
        raise ValueError('entropy_weight must be finite and nonnegative')

    state, first_logits = _sender_start(sender, h)
    uniforms_t = torch.from_numpy(uniforms)
    first, logp0, entropy0, prob0 = _draw(first_logits, uniforms_t[:, 0:1])
    second_logits = _next_logits(sender, state, first)
    second, logp1, entropy1, prob1 = _draw(second_logits, uniforms_t[:, 1:2])
    messages = torch.stack((first, second), dim=1).detach()
    action_logits = receive_messages(receiver, messages)
    food, logpf, entropyf, probf = _draw(action_logits[:, 0], uniforms_t[:, 2:3])
    water, logpw, entropyw, probw = _draw(action_logits[:, 1], uniforms_t[:, 3:4])
    actions = torch.stack((food, water), dim=1)
    success = (actions.numpy() == positions).astype(np.float32)
    reward = (.25 * success.sum(axis=1) + .5 * success.prod(axis=1)).astype(np.float32)
    advantage = torch.from_numpy(reward - np.float32(.5))
    sender_logp, receiver_logp = logp0 + logp1, logpf + logpw
    sender_entropy, receiver_entropy = entropy0 + entropy1, entropyf + entropyw
    sender_loss = -(sender_logp * advantage).mean() - float(entropy_weight) * sender_entropy.mean()
    receiver_loss = -(receiver_logp * advantage).mean() - float(entropy_weight) * receiver_entropy.mean()
    if not bool(torch.isfinite(sender_loss)) or not bool(torch.isfinite(receiver_loss)):
        raise ValueError('non-finite communication loss')
    to_np = lambda tensor: tensor.detach().numpy().copy()
    trace = {
        'h': to_np(h), 'uniforms': uniforms.copy(), 'messages': to_np(messages),
        'first_logits': to_np(first_logits), 'second_logits': to_np(second_logits),
        'token_probabilities': to_np(torch.stack((prob0, prob1), 1)),
        'action_logits': to_np(action_logits), 'action_probabilities': to_np(torch.stack((probf, probw), 1)),
        'actions': to_np(actions), 'positions': positions.copy(), 'success': success, 'reward': reward,
        'advantage': to_np(advantage), 'sender_logp': to_np(sender_logp), 'receiver_logp': to_np(receiver_logp),
        'sender_entropy': to_np(sender_entropy), 'receiver_entropy': to_np(receiver_entropy),
        'sender_loss': float(sender_loss.detach()), 'receiver_loss': float(receiver_loss.detach()),
        'entropy_weight': float(entropy_weight), 'baseline': .5,
    }
    return sender_loss, receiver_loss, trace


def self_test():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'redesign_v0.8'))
    from camp import CampAgent
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99521)
        sender = CampAgent(nn.Sequential(nn.Linear(8, 64)), 7, 2)
        receiver = copy.deepcopy(sender)
        twin = copy.deepcopy(sender)
        original = {key: value.clone() for key, value in sender.state_dict().items()}
        original_ids = {key: id(value) for key, value in sender.named_parameters()}
        rng = torch.random.get_rng_state().clone()
        metadata = reset_communication(sender, 92101)
        reset_communication(twin, 92101)
        reset_communication(receiver, 92102)
        assert torch.equal(rng, torch.random.get_rng_state())
        assert all(torch.equal(value, twin.state_dict()[key]) for key, value in sender.state_dict().items())
        assert original_ids == {key: id(value) for key, value in sender.named_parameters()}
        for key, value in original.items():
            if key.split('.')[0] not in COMMUNICATION_MODULES:
                assert torch.equal(value, sender.state_dict()[key])
        assert not torch.equal(sender.send_out.weight, receiver.send_out.weight)
        for key, value in sender.named_parameters():
            assert value.requires_grad == (key.split('.')[0] in SENDER_MODULES + RECEIVER_MODULES)

        h = torch.randn(8, WIDTH)
        uniforms = np.random.default_rng(99521).random((8, 4), dtype=np.float32)
        positions = np.asarray([(i % 6, (i + 1) % 6) for i in range(8)], dtype=np.int64)
        sender_loss, receiver_loss, trace = direction_loss(sender, receiver, h, uniforms, positions, .02)
        sender_params = trainable_groups(sender)['sender']
        receiver_params = trainable_groups(receiver)['receiver']
        sender_grad = torch.autograd.grad(sender_loss, sender_params + receiver_params, retain_graph=True, allow_unused=True)
        receiver_grad = torch.autograd.grad(receiver_loss, sender_params + receiver_params, allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() for g in sender_grad[:len(sender_params)])
        assert all(g is None for g in sender_grad[len(sender_params):])
        assert all(g is None for g in receiver_grad[:len(sender_params)])
        assert all(g is not None and torch.isfinite(g).all() for g in receiver_grad[len(sender_params):])
        assert all(p.grad is None for agent in (sender, receiver) for p in agent.parameters())

        changed = (positions + 2) % 6
        _, _, truth_trace = direction_loss(sender, receiver, h, uniforms, changed, .02)
        for key in ('messages', 'first_logits', 'second_logits', 'token_probabilities', 'action_logits', 'action_probabilities', 'actions'):
            assert np.array_equal(trace[key], truth_trace[key]), key
        first_rng = torch.random.get_rng_state().clone()
        joint = message_log_probs(sender, h)
        assert torch.equal(first_rng, torch.random.get_rng_state())
        max_probability_error = float((joint.exp().sum(1) - 1).abs().max().detach())
        assert max_probability_error < 1e-6
        codes = trace['messages'][:, 0] * 7 + trace['messages'][:, 1]
        assert np.array_equal(joint.detach().numpy()[np.arange(8), codes], trace['sender_logp'])
        all_receive = receiver_logits(receiver)
        assert all_receive.shape == (49, 2, 6)
        assert np.allclose(all_receive.detach().numpy()[codes], trace['action_logits'], atol=2e-7, rtol=2e-6)
        greedy = greedy_messages(sender, h)
        start, first_logits = _sender_start(sender, h)
        assert torch.equal(greedy[:, 0], first_logits.argmax(-1))
        assert torch.equal(greedy[:, 1], _next_logits(sender, start, greedy[:, 0]).argmax(-1))
        # A distributional counterexample records why sequential and joint MAP
        # are distinct decoding rules even with the same49-message support.
        example = np.zeros((7, 7)); example[0] = .6 / 7; example[1, 0] = .4
        assert (int(example.sum(1).argmax()), int(example[0].argmax())) != tuple(np.unravel_index(example.argmax(), example.shape))

        # Changing h affects only the sender. A fixed integer message produces
        # the same receiver output, with no demand/truth/history API arguments.
        fixed_messages = torch.from_numpy(trace['messages'])
        expected = receive_messages(receiver, fixed_messages).detach().clone()
        message_log_probs(sender, -h)
        assert torch.equal(expected, receive_messages(receiver, fixed_messages).detach())
        try:
            direction_loss(sender, receiver, h.clone().requires_grad_(), uniforms, positions, .02)
        except ValueError:
            pass
        else:
            raise AssertionError('gradient-bearing visual state must be rejected')
        for key in ('send_value', 'receive_value'):
            def unused_forward(*args, **kwargs):
                raise AssertionError('unused value branch was called')
            getattr(sender, key).forward = unused_forward
            getattr(receiver, key).forward = unused_forward
        for entropy in (0., .02):
            sl, rl, tr = direction_loss(sender, receiver, h, uniforms, positions, entropy)
            assert np.isin(tr['reward'], [0., .25, 1.]).all()
            assert np.array_equal(tr['advantage'], tr['reward'] - .5)
            # Explicit formula and 1/2 role average remain caller-reviewable.
            expected_s = -(tr['sender_logp'] * (tr['reward'] - .5)).mean() - entropy * tr['sender_entropy'].mean()
            expected_r = -(tr['receiver_logp'] * (tr['reward'] - .5)).mean() - entropy * tr['receiver_entropy'].mean()
            assert np.isclose(float(sl.detach()), expected_s, atol=1e-6)
            assert np.isclose(float(rl.detach()), expected_r, atol=1e-6)
    return {'passed': True, 'training_run': False,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'metadata': metadata, 'max_joint_probability_error': max_probability_error,
            'checks': ['fresh paired initialization preserves frontend/parameter objects/external RNG',
                       'only declared communication groups require gradients; critics unused',
                       'discrete-only cross-agent path with disjoint loss gradients',
                       'evaluator truth does not affect messages/actions before reward',
                       '49 autoregressive probabilities normalize and match sampled log probability',
                       'receiver enumeration matches direct integer-message policy',
                       'greedy sender is sequential, not complete-message MAP',
                       'visual state gradient rejected; external uniforms only',
                       'constant baseline and entropy/role-loss formulas']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_test(), ensure_ascii=False))
