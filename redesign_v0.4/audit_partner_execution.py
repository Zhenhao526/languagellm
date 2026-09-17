"""Audit saved partner experiments; reproduce inputs, never retrain policies."""
from pathlib import Path
import hashlib
import itertools
import json
import math
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/partners_001'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def states(path):
    return torch.load(path, weights_only=True, map_location='cpu')


def equal_states(left, right):
    return len(left) == len(right) and all(a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
                                          for a, b in zip(left, right))


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def eval_hashes(evaluation):
    hashes = {}
    for pair in evaluation['pairs']:
        edge = ''.join(str(i) for i in pair['agents'])
        for task, modes in pair['tasks'].items():
            assert len({m['external_cases_sha256'] for m in modes.values()}) == 1
            for mode, metrics in modes.items():
                hashes[f'{edge}/{task}/{mode}'] = metrics['external_cases_sha256']
        if 'intervention' in pair:
            hashes[f'{edge}/intervention'] = pair['intervention']['external_cases_sha256']
    if 'sender_alignment' in evaluation:
        alignment = evaluation['sender_alignment']
        assert alignment['same_exact_inputs_for_all_agents'] and not alignment['symbol_permutation_applied']
        hashes['alignment'] = alignment['external_cases_sha256']
    return hashes


def main():
    completed = read(OUT / 'completed.json')
    assert completed['status'] == 'completed' and completed['runs'] == 6
    config = read(OUT / 'config.json')
    source = {name: {'recorded': expected, 'archived': sha(OUT / 'source' / name), 'current': sha(ROOT / name)}
              for name, expected in read(OUT / 'source_hashes.json').items()}
    assert all(len(set(row.values())) == 1 for row in source.values())
    assert (OUT / 'execution_plan.md').read_bytes() == (ROOT / '对照与伙伴实验_执行方案.md').read_bytes()
    sys.path.insert(0, str(OUT / 'source'))
    import run_partners as saved
    assert all(config[k] == v for k, v in saved.CONFIG.items())
    assert (config['course_updates'], config['exploratory_full_updates'], config['pure_reward_updates']) == (600, 300, 300)
    assert config['seeds'] == [101, 202, 303] and config['conditions'] == ['fixed', 'rotating']
    assert config['batch_size_per_agent'] == 1024 and config['horizon'] == 16
    assert not config['partner_identity_input'] and config['signalling_coefficient'] == 0

    data_checks = {}
    for name in ['manifest.json', 'features.npz', 'encoder_report.json']:
        data_checks[name] = {'archived': sha(OUT / ('data_' + name)), 'current': sha(ROOT / 'data' / name)}
        assert len(set(data_checks[name].values())) == 1
    entries = read(OUT / 'data_manifest.json')['images']
    feature = np.load(OUT / 'data_features.npz')['features'].astype(np.float32)
    train = np.asarray([entry['split'] == 'train' for entry in entries])
    center = feature[train].mean(0)
    scale = float(np.sqrt(((feature[train] - center) ** 2).mean()))
    scaling = np.load(OUT / 'feature_scaling.npz')
    assert np.array_equal(center, scaling['center']) and np.array_equal(scale, scaling['scale'])
    pools = {kind: np.asarray([i for i, entry in enumerate(entries) if entry['split'] == 'train' and entry['category'] == category])
             for kind, category in enumerate(['food', 'water'])}
    full = np.asarray([scene for scene in itertools.product((0, 1), repeat=4) if len(set(scene)) == 2], dtype=np.int64).reshape(14, 2, 2)
    forced = full[..., 0] == full[..., 1]
    course = full[forced.sum(axis=1) == 1]
    matchings = np.asarray([[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]])
    preparations = read(OUT / 'preparations.json')
    preparation_checks = {}
    all_results = read(OUT / 'results.json')
    assert len(all_results) == 6
    runs, paired = {}, {}
    for seed in config['seeds']:
        preparation = preparations[str(seed)]
        assert preparation == read(OUT / f'preparation_s{seed}.json') and preparation['passed']
        original_path = ROOT / f'results/pilot_001/prepared_s{seed}.pt'
        prepared_path = OUT / f'prepared_s{seed}.pt'
        prepared = states(prepared_path)
        assert len(prepared) == 4 and equal_states(prepared[:2], states(original_path))
        assert preparation['old_source_sha256'] == sha(original_path)
        assert preparation['prepared_sha256'] == sha(prepared_path)
        assert preparation['new_preparation_seed'] == seed + 1000
        fresh_extra = saved.make_agents(seed + 1000)
        for who, agent in enumerate(fresh_extra, 2):
            assert all(torch.equal(value, prepared[who]['sender.' + key]) for key, value in agent.sender.state_dict().items())
        practice = preparation['reused_practice'] + preparation['new_practice']
        assert [r['agent'] for r in practice] == [0, 1, 2, 3]
        assert all(r['updates'] == 200 and r['choices'] == 12800 and r['heldout_need_sensitive_choice'] >= .8 for r in practice)
        assert all(r['sender_unchanged_during_preparation'] for r in preparation['new_practice'])
        preparation_checks[str(seed)] = {'four_agents_passed': True, 'original_agents_exactly_reused': True,
                                         'new_sender_weights_match_fresh_seeded_initialization': True,
                                         'prepared_sha256': sha(prepared_path),
                                         'individual_practice_accuracies': [r['heldout_need_sensitive_choice'] for r in practice]}

        # Reconstruct every external batch from the archived image metadata.
        # No image encoder, learned policy, or optimizer runs during this audit.
        world_rng = np.random.default_rng(seed * 10000 + 101)
        layout_rng = np.random.default_rng(seed * 10000 + 302)
        matching_rng = np.random.default_rng(seed * 10000 + 301)
        rotation = np.concatenate([matching_rng.permutation(3) for _ in range(400)])
        rows_by_condition = {condition: [json.loads(line) for line in
                                         (OUT / f'{condition}_s{seed}/training_metrics.jsonl').read_text().splitlines()]
                             for condition in config['conditions']}
        edges = {condition: np.zeros((4, 4), dtype=np.int64) for condition in config['conditions']}
        edge_history = {condition: {0: edge.copy()} for condition, edge in edges.items()}
        unequal_private = 0
        for condition, rows in rows_by_condition.items():
            assert len(rows) == 1200 and [row['update'] for row in rows] == list(range(1, 1201))
            assert all(finite(row) for row in rows)
        for offset in range(1200):
            update = offset + 1
            scenes = course if update <= 600 else full
            kinds = scenes[world_rng.integers(len(scenes), size=2048)].reshape(1024, 2, 2, 2)
            ids = np.empty(kinds.shape, dtype=np.int64)
            for kind in (0, 1):
                mask = kinds == kind
                ids[mask] = world_rng.choice(pools[kind], int(mask.sum()))
            bits = layout_rng.integers(2, size=3)
            for condition, rows in rows_by_condition.items():
                row = rows[offset]
                assert row['stage'] == ('course' if update <= 600 else ('full_exploration' if update <= 900 else 'pure_reward'))
                assert row['task'] == ('curriculum' if update <= 600 else 'full')
                assert row['aux_weight'] == 0 and row['action_entropy_weight'] == (.05 if update <= 900 else 0)
                index = int(rotation[offset]) if condition == 'rotating' else 0
                assert row['matching_index'] == index
                assignment = matchings[index].copy()
                if bits[0]:
                    assignment = assignment[::-1].copy()
                for slot in (0, 1):
                    if bits[slot + 1]:
                        assignment[slot] = assignment[slot, ::-1]
                np.testing.assert_array_equal(row['layout_bits'], bits)
                np.testing.assert_array_equal(row['slot_assignment'], assignment)
                assert row['world_slots_sha256'] == digest(kinds, ids)
                assert [a['agent'] for a in row['agents']] == [0, 1, 2, 3]
                for slot, pair in enumerate(assignment):
                    outcome = row['pair_outcomes'][slot]
                    assert outcome['slot'] == slot and outcome['agents'] == pair.tolist()
                    assert outcome['success'] == outcome['successes'] / 1024
                    assert 0 <= outcome['successes'] <= 1024
                    edges[condition][pair[0], pair[1]] += 1
                    edges[condition][pair[1], pair[0]] += 1
                    for local, who in enumerate(pair):
                        metric = row['agents'][who]
                        partner = int(pair[1 - local])
                        assert metric['partner'] == partner
                        assert metric['received_sha256'] == row['agents'][partner]['sent_sha256']
                        assert metric['private_input_sha256'] == digest(kinds[:, slot, local], ids[:, slot, local])
                        assert metric['sampled_success'] == outcome['success']
                assert row['sampled_training_success'] == np.mean([p['success'] for p in row['pair_outcomes']])
                edge_history[condition][update] = edges[condition].copy()
            unequal_private += sum(a['private_input_sha256'] != b['private_input_sha256']
                                   for a, b in zip(rows_by_condition['fixed'][offset]['agents'], rows_by_condition['rotating'][offset]['agents']))

        curves, results = {}, {}
        for condition, rows in rows_by_condition.items():
            path = OUT / f'{condition}_s{seed}'
            assert sha(path / 'initial.pt') == sha(prepared_path)
            curve = read(path / 'learning_curve.json')
            assert [r['update'] for r in curve] == config['checkpoints']
            for record in curve:
                np.testing.assert_array_equal(record['edge_training_updates'], edge_history[condition][record['update']])
                checkpoint = states(path / f"checkpoint_{record['update']:04d}.pt")
                assert len(checkpoint) == 4
                assert all(torch.isfinite(value).all() for agent in checkpoint for value in agent.values())
                if record['update'] == 0:
                    assert equal_states(checkpoint, prepared)
            final = states(path / 'checkpoint_1200.pt')
            deltas = [{module: sum(float((before[k] - after[k]).square().sum()) for k in before if k.startswith(module + '.')) ** .5
                       for module in ['project', 'sender', 'actor', 'value']} for before, after in zip(prepared, final)]
            assert all(value > 0 for agent in deltas for value in agent.values())
            result = read(path / 'result.json')
            assert result == next(r for r in all_results if r['seed'] == seed and r['condition'] == condition)
            assert result['updates'] == 1200 and result['individual_training_choices_per_agent'] == 1228800
            assert result['joint_dyad_training_steps'] == 2457600
            assert result['final_aux_weight'] == result['final_action_entropy_weight'] == 0
            assert result['trainable_parameters_per_agent'] == [83527] * 4
            expected_edges = ((1 - np.eye(4, dtype=int)) * 400) if condition == 'rotating' else np.asarray([[0, 1200, 0, 0], [1200, 0, 0, 0], [0, 0, 0, 1200], [0, 0, 1200, 0]])
            np.testing.assert_array_equal(edges[condition], expected_edges)
            np.testing.assert_array_equal(result['edge_training_updates'], expected_edges)
            np.testing.assert_array_equal(result['edge_joint_cases'], expected_edges * 1024)
            curves[condition], results[condition] = curve, result
            groups = result['evaluation']['aggregates']
            runs[path.name] = {
                'updates': 1200, 'all_expected_external_inputs_reconstructed': True,
                'all_message_hashes_match_current_partner': True, 'pair_reward_records_consistent': True,
                'all_recorded_numbers_and_checkpoint_parameters_finite': True,
                'all_final_300_updates_auxiliary_free': True, 'initial_bytes_match_shared_prepared': True,
                'edge_training_updates': expected_edges.tolist(), 'parameter_l2_changes_by_agent': deltas,
                'zero_gradient_counts_by_agent': [{module: sum(row['agents'][who]['gradient_norms'][module] == 0 for row in rows)
                                                  for module in ['project', 'sender', 'actor', 'value']} for who in range(4)],
                'final_full_by_group': {group: {mode: groups[group]['tasks']['full'][mode]['mean_reward_per_step']
                                               for mode in ['normal', 'shuffle', 'blank', 'stochastic']}
                                        for group in ['all', 'original', 'cross']}}
        assert eval_hashes(results['fixed']['evaluation']) == eval_hashes(results['rotating']['evaluation'])
        for fixed, rotating in zip(curves['fixed'], curves['rotating']):
            assert eval_hashes(fixed['evaluation']) == eval_hashes(rotating['evaluation'])
        assert unequal_private > 0
        paired[str(seed)] = {'same_prepared_file_bytes': True, 'same_all_1200_world_slot_batches_and_layout_bits': True,
                             'same_all_checkpoint_final_intervention_and_alignment_external_cases': True,
                             'different_individual_private_batches': unequal_private, 'individual_private_batches_compared': 4800}

    audit = {'batch': 'partners_001', 'status': 'passed', 'source_hashes': source,
             'config_and_archived_plan_match_actual_sources': True, 'archived_data_matches_current': data_checks,
             'training_only_feature_scaling_verified': True, 'preparation_checks': preparation_checks,
             'runs': runs, 'paired_checks': paired,
             'scope': 'Inputs and execution records independently reconstructed; no policy training, behavioral reruns, hyperparameter search, or model selection.'}
    target = ROOT / '伙伴实验_正式运行核查.json'
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'status': audit['status'], 'paired_checks': paired, 'audit': str(target)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
