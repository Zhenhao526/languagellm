"""Post-hoc same-policy closed-channel intervention for the FI study.

The trained policy and first-window messages are held fixed.  ``live`` routes
all messages, ``own`` routes only self messages (the training ``silent``
condition), and ``closed`` keeps each sender's own payload but removes all
cross-agent payloads and visibility bits from the second-window input.  No
optimizer call is made here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research_program.triadic_reciprocal_execution_study import environment
from research_program.triadic_factorized_neutral_altpartner_information_control_study import design, metrics, runner
from research_program.triadic_message_study import runner as core


def require(ok, message):
    if not ok:
        raise ValueError(message)


def routed_window(tokens, mode):
    tokens = np.asarray(tokens)
    require(mode in ('live', 'own', 'closed'), 'Unknown routing mode')
    require(tokens.ndim == 3 and tokens.shape[1:] == (3, 4) and tokens.dtype.kind in 'iu'
            and ((tokens >= 0) & (tokens < 8)).all(), 'Invalid token array')
    if mode == 'live':
        payload_visibility = np.ones((3, 3), dtype=np.float64)
        bit_visibility = payload_visibility
    elif mode == 'own':
        payload_visibility = np.eye(3, dtype=np.float64)
        bit_visibility = payload_visibility
    else:
        # Closed cross-agent channel: keep each sender's own payload, while
        # removing all visibility bits so no cross-agent routing signal leaks.
        payload_visibility = np.eye(3, dtype=np.float64)
        bit_visibility = np.zeros((3, 3), dtype=np.float64)
    onehot = np.eye(8, dtype=np.float64)[tokens]
    visible = onehot[:, None, :, :, :] * payload_visibility[None, :, :, None, None]
    bits = np.broadcast_to(bit_visibility[None], (len(tokens), 3, 3))
    return np.concatenate((visible.reshape(len(tokens), 3, 96), bits), axis=-1)


def rollout_channel(networks, observations, mode, uniforms=None):
    x = np.asarray(observations, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1:] == (3, 54) and len(networks) == 9, 'Invalid rollout input')
    core.base.finite(x, 'observations')
    if uniforms is not None:
        require(np.shape(uniforms) == (len(x), 2, 3, 4), 'Invalid message uniforms')
    caches = [None] * 9
    sender_logits, sender_probabilities, sender_log_probabilities, messages = [], [], [], []
    inputs = x
    for window in range(2):
        logits = []
        for actor in range(3):
            z, caches[3 * actor + window] = core.base.actor_forward(networks[3 * actor + window], inputs[:, actor])
            logits.append(z.reshape(len(x), 4, 8))
        z = np.stack(logits, axis=1); p, lp = core.base.policy_distribution(z)
        m = core.categorical_tokens(p, None if uniforms is None else uniforms[:, window])
        sender_logits.append(z); sender_probabilities.append(p); sender_log_probabilities.append(lp); messages.append(m)
        if window == 0:
            inputs = np.concatenate((x, routed_window(m, mode)), axis=-1)
    action_inputs = np.concatenate((x, routed_window(messages[0], mode), routed_window(messages[1], mode)), axis=-1)
    action_logits = []
    for actor in range(3):
        z, caches[3 * actor + 2] = core.base.actor_forward(networks[3 * actor + 2], action_inputs[:, actor])
        action_logits.append(z)
    return dict(messages=np.stack(messages, axis=1), sender_logits=np.stack(sender_logits, axis=1),
                sender_probabilities=np.stack(sender_probabilities, axis=1),
                sender_log_probabilities=np.stack(sender_log_probabilities, axis=1),
                action_logits=np.stack(action_logits, axis=1), action_inputs=action_inputs, caches=caches)


def _event_pair_mask(full):
    return full.reshape(len(full), 3, 8).any(axis=2)


def _greedy(trace):
    intent_p, _, proposal_p, _ = runner.kernel.factorized_distribution(trace['action_logits'])
    intent = intent_p.argmax(axis=-1).astype(np.int16)
    proposal = proposal_p.argmax(axis=-1).astype(np.int16)
    actions = np.where(intent == 1, proposal + 1, 0).astype(np.int16)
    return intent, proposal, actions


def evaluate_channel(networks, arrays, spec, mode):
    n = int(spec['world_count']); ids = np.arange(n, dtype=np.int64)
    engagement = neutral = physical_count = q_count = pair_legal_count = 0
    proposal_legal_count = proposal_legal_denominator = 0
    third_neutral_count = third_neutral_denominator = 0
    pair_counts = np.zeros(3, dtype=np.int64); plan_counts = np.zeros(24, dtype=np.int64)
    actor_hits = np.zeros(3, dtype=np.int64); actor_total = 0
    expected_sum = full_prob_sum = partial_prob_sum = 0.0
    for start in range(0, n, runner.CONFIG['evaluation_batch_size']):
        stop = min(start + runner.CONFIG['evaluation_batch_size'], n); ix = ids[start:stop]
        states = arrays['packed_states'][ix]; full = arrays['native_rewards'][ix] == 1.
        trace = rollout_channel(networks, arrays['x_FI'][ix], mode)
        terms = runner.kernel.objective_terms(trace['action_logits'], arrays['native_rewards'][ix])
        expected_sum += float(terms['J'].sum()); full_prob_sum += float(terms['full_success_probability'].sum()); partial_prob_sum += float(terms['partial_success_probability'].sum())
        intent, proposal, actions = _greedy(trace)
        physical = environment.settle(states, actions, 'strict')
        pair_targets = _event_pair_mask(full)
        engagement += int((intent == 1).sum()); neutral += int((intent == 0).sum())
        actor_hits += (intent == 1).sum(axis=0, dtype=np.int64); actor_total += len(ix)
        legal_actions = np.zeros((len(ix), 3, 17), dtype=bool)
        for event in range(24):
            for actor in range(3):
                legal_actions[:, actor, core.base.JOINT_ACTIONS[event, actor]] |= full[:, event]
        selected = legal_actions[np.arange(len(ix))[:, None], np.arange(3)[None, :], actions]
        proposal_legal_count += int(selected[intent == 1].sum()); proposal_legal_denominator += int((intent == 1).sum())
        executed = physical['executed']; physical_ok = np.any(executed, axis=1)
        actual_pair = physical['actual_pair_index'].astype(np.int64); physical_count += int(physical_ok.sum())
        safe_pair = np.maximum(actual_pair, 0)
        executed_event = safe_pair * 8 + np.maximum(physical['executed_site'], 0).astype(np.int64) * 2 + np.maximum(physical['executed_destination'], 0).astype(np.int64)
        pair_legal = physical_ok & pair_targets[np.arange(len(ix)), safe_pair]
        q_hits = physical_ok & full[np.arange(len(ix)), executed_event]
        pair_legal_count += int(pair_legal.sum()); q_count += int(q_hits.sum())
        if physical_ok.any():
            pair_counts += np.bincount(actual_pair[physical_ok], minlength=3).astype(np.int64)
            plan_counts += np.bincount(executed_event[physical_ok], minlength=24).astype(np.int64)
            third = np.asarray([2, 1, 0], dtype=np.int64)
            third_neutral_count += int((intent[np.flatnonzero(physical_ok), third[actual_pair[physical_ok]]] == 0).sum())
            third_neutral_denominator += int(physical_ok.sum())
    return dict(
        mode=mode, value=float(q_count / n), q_rate=float(q_count / n),
        conditional_q_rate=float(q_count / physical_count) if physical_count else 0.0,
        target_pair_legal_rate=float(pair_legal_count / physical_count) if physical_count else 0.0,
        proposal_legal_rate=float(proposal_legal_count / proposal_legal_denominator) if proposal_legal_denominator else 0.0,
        physical_execution_rate=float(physical_count / n), engagement_rate=float(engagement / (3 * n)),
        neutral_rate=float(neutral / (3 * n)), actor_engagement_rates=[float(x / max(actor_total, 1)) for x in actor_hits],
        third_agent_neutral_rate=float(third_neutral_count / third_neutral_denominator) if third_neutral_denominator else 0.0,
        legal_plan_selection_counts=plan_counts.tolist(), legal_pair_selection_counts=pair_counts.tolist(), worlds=n,
        physical_worlds=int(physical_count), proposal_legal_denominator=int(proposal_legal_denominator),
        conditional_q_denominator_worlds=int(physical_count), conditional_q_numerator_worlds=int(q_count),
        target_pair_denominator_worlds=int(physical_count), target_pair_numerator_worlds=int(pair_legal_count),
        exact_expected_reward_mean=float(expected_sum / n), exact_full_success_probability_mean=float(full_prob_sum / n),
        exact_partial_success_probability_mean=float(partial_prob_sum / n), legal_plan_count=int(full.sum(axis=1).min()),
        legal_pair_count=2, partition=spec['partition'], action_factorization='intent:neutral/engage; proposal:16-way conditional on engage',
    )


def _stats(values):
    return metrics.stats(np.asarray(values, dtype=np.float64))


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    prepared = json.loads((source / 'prepared.json').read_text()); arrays = runner.make_arrays(prepared['partitions']['new_layouts'])
    rows = []; by = {}
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            for live in (True, False):
                condition = f'{schedule}_altpair_factorized_strict_FI_{"live" if live else "silent"}'
                run_path = source / 'execution' / f'seed_{seed}_{condition}'
                result = json.loads((run_path / 'result.json').read_text()); network = runner.load_networks(run_path / 'checkpoint_6000.npz')
                natural_mode = 'live' if live else 'own'
                natural = evaluate_channel(network, arrays, prepared['partitions']['new_layouts'], natural_mode)
                closed = evaluate_channel(network, arrays, prepared['partitions']['new_layouts'], 'closed')
                row = dict(seed=seed, schedule=schedule, trained_channel=natural_mode, condition=condition,
                           checkpoint_sha256=core.base.sha(run_path / 'checkpoint_6000.npz'),
                           natural=natural, closed=closed,
                           natural_minus_closed={key: float(natural[key] - closed[key]) for key in ('q_rate', 'conditional_q_rate', 'target_pair_legal_rate', 'proposal_legal_rate', 'physical_execution_rate')})
                rows.append(row); by[seed, schedule, natural_mode] = row
    require(len(rows) == 64, 'Expected 64 policy rows')
    summaries = {}
    for trained_channel in ('live', 'own'):
        for schedule in design.SCHEDULES:
            label = f'{schedule}_{trained_channel}_natural_minus_closed'; summaries[label] = {}
            selected = [by[seed, schedule, trained_channel] for seed in design.SEEDS]
            for key in ('q_rate', 'conditional_q_rate', 'target_pair_legal_rate', 'proposal_legal_rate', 'physical_execution_rate'):
                values = [row['natural_minus_closed'][key] for row in selected]
                summaries[label][key] = dict(mean=float(np.mean(values)), statistics=_stats(values))
    result = dict(status='completed_json_only_closed_channel_probe', source=str(source), rows=rows, summaries=summaries,
                  intervention='same-policy final-checkpoint new-layout evaluation; retain self payloads, close cross-agent payloads and all visibility bits from window 1, then recompute window 2 and action',
                  worlds_per_row=int(prepared['partitions']['new_layouts']['world_count']), policy_rows=len(rows),
                  no_optimizer_updates=True, no_new_training=True,
                  interpretation_boundary='Post-hoc channel dependence of a trained task policy; not evidence of words or language origin.')
    result_path = output / 'results.json'; result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', input_policy_rows=len(rows), worlds=int(len(rows) * prepared['partitions']['new_layouts']['world_count'] * 2),
                   model_forwards=int(len(rows) * prepared['partitions']['new_layouts']['world_count'] * 2 * 9), optimizer_updates=0,
                   output_results_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.source, args.output)
