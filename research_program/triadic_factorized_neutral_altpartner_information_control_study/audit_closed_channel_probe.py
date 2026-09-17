"""Independent replay audit for the same-policy closed-channel probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research_program.triadic_factorized_neutral_altpartner_information_control_study import design, runner
from research_program.triadic_reciprocal_execution_study import environment
from research_program.triadic_message_study import runner as core


def require(ok, message):
    if not ok:
        raise ValueError(message)


def route(tokens, mode):
    tokens = np.asarray(tokens)
    require(mode in ('live', 'own', 'closed'), 'Invalid route mode')
    require(tokens.shape[1:] == (3, 4) and tokens.dtype.kind in 'iu' and ((tokens >= 0) & (tokens < 8)).all(), 'Invalid tokens')
    if mode == 'live':
        payload_visibility = np.ones((3, 3)); bit_visibility = payload_visibility
    elif mode == 'own':
        payload_visibility = np.eye(3); bit_visibility = payload_visibility
    else:
        payload_visibility = np.eye(3); bit_visibility = np.zeros((3, 3))
    onehot = np.eye(8)[tokens]
    return np.concatenate(((onehot[:, None] * payload_visibility[None, :, :, None, None]).reshape(len(tokens), 3, 96),
                           np.broadcast_to(bit_visibility[None], (len(tokens), 3, 3))), axis=-1)


def replay_rollout(networks, observations, mode):
    x = np.asarray(observations, dtype=np.float64); require(x.shape[1:] == (3, 54), 'Invalid observations')
    caches = [None] * 9; all_messages = []; sender_p = []; sender_lp = []; inp = x
    for window in range(2):
        logits = []; cs = []
        for actor in range(3):
            z, c = core.base.actor_forward(networks[3 * actor + window], inp[:, actor]); logits.append(z.reshape(len(x), 4, 8)); cs.append(c)
        for actor, c in enumerate(cs): caches[3 * actor + window] = c
        z = np.stack(logits, axis=1); p, lp = core.base.policy_distribution(z)
        m = core.categorical_tokens(p); all_messages.append(m); sender_p.append(p); sender_lp.append(lp)
        if window == 0: inp = np.concatenate((x, route(m, mode)), axis=-1)
    action_input = np.concatenate((x, route(all_messages[0], mode), route(all_messages[1], mode)), axis=-1)
    actions = []
    for actor in range(3):
        z, c = core.base.actor_forward(networks[3 * actor + 2], action_input[:, actor]); actions.append(z); caches[3 * actor + 2] = c
    return dict(messages=np.stack(all_messages, axis=1), action_logits=np.stack(actions, axis=1))


def pair_targets(full):
    return full.reshape(len(full), 3, 8).any(axis=2)


def evaluate(networks, arrays, spec, mode):
    n = int(spec['world_count']); ids = np.arange(n, dtype=np.int64)
    counters = dict(engagement=0, physical=0, q=0, pair=0, proposal=0, proposal_den=0, third=0, third_den=0)
    plan_counts = np.zeros(24, dtype=np.int64); pair_counts = np.zeros(3, dtype=np.int64); actor_hits = np.zeros(3, dtype=np.int64)
    expected = full_prob = partial_prob = 0.0
    for begin in range(0, n, runner.CONFIG['evaluation_batch_size']):
        ix = ids[begin:min(begin + runner.CONFIG['evaluation_batch_size'], n)]
        states = arrays['packed_states'][ix]; full = arrays['native_rewards'][ix] == 1.
        trace = replay_rollout(networks, arrays['x_FI'][ix], mode)
        terms = runner.kernel.objective_terms(trace['action_logits'], arrays['native_rewards'][ix])
        expected += float(terms['J'].sum()); full_prob += float(terms['full_success_probability'].sum()); partial_prob += float(terms['partial_success_probability'].sum())
        intent_p, _, proposal_p, _ = runner.kernel.factorized_distribution(trace['action_logits'])
        intent = intent_p.argmax(-1).astype(np.int16); proposal = proposal_p.argmax(-1).astype(np.int16); actions = np.where(intent == 1, proposal + 1, 0).astype(np.int16)
        settled = environment.settle(states, actions, 'strict'); counters['engagement'] += int((intent == 1).sum()); counters['physical'] += int(np.any(settled['executed'], axis=1).sum()); actor_hits += (intent == 1).sum(axis=0, dtype=np.int64)
        legal_actions = np.zeros((len(ix), 3, 17), dtype=bool)
        for event in range(24):
            for actor in range(3): legal_actions[:, actor, core.base.JOINT_ACTIONS[event, actor]] |= full[:, event]
        selected = legal_actions[np.arange(len(ix))[:, None], np.arange(3)[None, :], actions]
        counters['proposal'] += int(selected[intent == 1].sum()); counters['proposal_den'] += int((intent == 1).sum())
        ok = np.any(settled['executed'], axis=1); actual = settled['actual_pair_index'].astype(np.int64); safe = np.maximum(actual, 0)
        event = safe * 8 + np.maximum(settled['executed_site'], 0).astype(np.int64) * 2 + np.maximum(settled['executed_destination'], 0).astype(np.int64)
        counters['pair'] += int((ok & pair_targets(full)[np.arange(len(ix)), safe]).sum()); counters['q'] += int((ok & full[np.arange(len(ix)), event]).sum())
        if ok.any():
            pair_counts += np.bincount(actual[ok], minlength=3).astype(np.int64); plan_counts += np.bincount(event[ok], minlength=24).astype(np.int64)
            third = np.asarray([2, 1, 0]); counters['third'] += int((intent[np.flatnonzero(ok), third[actual[ok]]] == 0).sum()); counters['third_den'] += int(ok.sum())
    physical = counters['physical']; engaged = counters['engagement'];
    return dict(mode=mode, value=counters['q'] / n, q_rate=counters['q'] / n, conditional_q_rate=counters['q'] / physical if physical else 0.0,
                target_pair_legal_rate=counters['pair'] / physical if physical else 0.0, proposal_legal_rate=counters['proposal'] / counters['proposal_den'] if counters['proposal_den'] else 0.0,
                physical_execution_rate=physical / n, engagement_rate=engaged / (3 * n), neutral_rate=1 - engaged / (3 * n), actor_engagement_rates=(actor_hits / n).tolist(),
                third_agent_neutral_rate=counters['third'] / counters['third_den'] if counters['third_den'] else 0.0, legal_plan_selection_counts=plan_counts.tolist(), legal_pair_selection_counts=pair_counts.tolist(), worlds=n,
                physical_worlds=physical, proposal_legal_denominator=counters['proposal_den'], conditional_q_denominator_worlds=physical, conditional_q_numerator_worlds=counters['q'],
                target_pair_denominator_worlds=physical, target_pair_numerator_worlds=counters['pair'], exact_expected_reward_mean=expected / n,
                exact_full_success_probability_mean=full_prob / n, exact_partial_success_probability_mean=partial_prob / n, legal_plan_count=int(full.sum(axis=1).min()), legal_pair_count=2,
                partition=spec['partition'], action_factorization='intent:neutral/engage; proposal:16-way conditional on engage')


def compare(expected, actual, prefix=''):
    require(set(expected) == set(actual), prefix + 'keys differ')
    for key in expected:
        a, b = expected[key], actual[key]
        if isinstance(a, dict): compare(a, b, prefix + key + '/')
        elif isinstance(a, list):
            require(len(a) == len(b), prefix + key + ' length differs')
            for i, (x, y) in enumerate(zip(a, b)):
                require(np.array_equal(x, y) if isinstance(x, (int, float)) and isinstance(y, (int, float)) and float(x).is_integer() and float(y).is_integer() else np.isclose(x, y, atol=2e-13, rtol=0), prefix + key + f'[{i}] differs')
        elif isinstance(a, (int, float)): require(np.isclose(a, b, atol=2e-13, rtol=0), prefix + key + ' differs')
        else: require(a == b, prefix + key + ' differs')


def main(source, probe, output):
    source = Path(source).resolve(); probe = Path(probe).resolve(); output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    saved = json.loads((probe / 'results.json').read_text()); prepared = json.loads((source / 'prepared.json').read_text()); spec = prepared['partitions']['new_layouts']; arrays = runner.make_arrays(spec)
    require(saved.get('status') == 'completed_json_only_closed_channel_probe' and len(saved.get('rows', [])) == 64, 'Invalid probe result')
    max_error = 0.0; checked = 0
    for row in saved['rows']:
        seed, schedule, trained = int(row['seed']), row['schedule'], row['trained_channel']; condition = row['condition']
        path = source / 'execution' / f'seed_{seed}_{condition}' / 'checkpoint_6000.npz'; require(core.base.sha(path) == row['checkpoint_sha256'], 'Checkpoint hash mismatch')
        nets = runner.load_networks(path); natural_mode = 'live' if trained == 'live' else 'own'
        for mode, key in ((natural_mode, 'natural'), ('closed', 'closed')):
            actual = evaluate(nets, arrays, spec, mode); expected = row[key]; compare(expected, actual, f'{seed}/{schedule}/{trained}/{mode}/')
            for metric in ('q_rate', 'conditional_q_rate', 'target_pair_legal_rate', 'proposal_legal_rate', 'physical_execution_rate'):
                max_error = max(max_error, abs(float(expected[metric]) - float(actual[metric])))
            checked += 1
    verification = dict(status='passed', source=str(source), probe=str(probe), policy_rows=64, evaluations=128,
                        worlds=64 * 2 * int(spec['world_count']), model_forwards=64 * 2 * int(spec['world_count']) * 9,
                        optimizer_updates=0, no_optimizer_updates=True, checkpoint_hashes_checked=64, rows_replayed=checked,
                        max_abs_error=float(max_error), route_modes=['live', 'own', 'closed'], first_window_preserved=True,
                        intervention='closed retains self payloads but removes cross-agent payloads and all visibility bits from second-window routing; policy is replayed without updates')
    path = output / 'verification.json'; path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', verification_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), model_forwards=verification['model_forwards'], optimizer_updates=0)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--probe', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.source, args.probe, args.output)
