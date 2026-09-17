"""Independently reconstruct contact inputs and check saved continuation states."""
from pathlib import Path
import copy
import itertools
import json
import math
import sys

import numpy as np
import torch

from audit_partner_execution import read, sha, digest, states, finite, equal_states, eval_hashes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/contact_001'
ORIGIN = ROOT / 'results/partners_001'
PAIRS = tuple(itertools.combinations(range(4), 2))
FIXED_PAIRS = ((0, 1), (2, 3))


def prior_edges(condition):
    if condition == 'rotating_to_rotating':
        return (1 - np.eye(4, dtype=np.int64)) * 400
    result = np.zeros((4, 4), dtype=np.int64)
    for left, right in FIXED_PAIRS:
        result[left, right] = result[right, left] = 1200
    return result


def check_artifacts(folder, manifest):
    expected = read(folder / manifest)
    assert all(sha(folder / name) == value for name, value in expected.items())
    return len(expected)


def main():
    completion = read(OUT / 'completed.json')
    assert completion['status'] == 'completed' and completion['runs'] == 9
    config = read(OUT / 'config.json')
    assert config['additional_updates'] == 600 and config['origin_update'] == 1200
    assert config['training_seed_offset'] == 2000 and config['batch_size_per_agent'] == 1024
    assert config['action_entropy_coefficient'] == config['signalling_coefficient'] == 0
    assert config['task'] == 'full' and config['fresh_adam_all_conditions']
    assert config['seeds'] == [101, 202, 303]
    assert config['conditions'] == ['fixed_to_fixed', 'fixed_to_rotating', 'rotating_to_rotating']
    hashes = read(OUT / 'source_hashes.json')
    source_checks = {name: {'recorded': value, 'archived': sha(OUT / 'source' / name), 'current': sha(ROOT / name)}
                     for name, value in hashes.items()}
    assert all(len(set(row.values())) == 1 for row in source_checks.values())
    assert (OUT / 'execution_plan.md').read_bytes() == (ROOT / '已有约定接触实验_执行方案.md').read_bytes()
    sys.path.insert(0, str(OUT / 'source'))
    import run_contact as saved
    import contact_eval as saved_eval
    assert all(config[key] == value for key, value in saved.CONFIG.items())
    source_validation = read(OUT / 'source_validation.json')
    assert source_validation['passed'] and source_validation['source_hashes'] == read(ORIGIN / 'source_hashes.json')
    data_checks = {}
    for name in ('manifest.json', 'features.npz', 'encoder_report.json'):
        data_checks[name] = {'current': sha(ROOT / 'data' / name), 'archived': sha(OUT / ('data_' + name)),
                             'origin': sha(ORIGIN / ('data_' + name))}
        assert len(set(data_checks[name].values())) == 1
    entries = read(OUT / 'data_manifest.json')['images']
    feature = np.load(OUT / 'data_features.npz')['features'].astype(np.float32)
    train_mask = np.asarray([entry['split'] == 'train' for entry in entries])
    center = feature[train_mask].mean(0)
    scale = float(np.sqrt(((feature[train_mask] - center) ** 2).mean()))
    scaled = torch.from_numpy((feature - center) / max(scale, 1e-6))
    for location in (OUT, ORIGIN):
        scaling = np.load(location / 'feature_scaling.npz')
        assert np.array_equal(center, scaling['center']) and np.array_equal(scale, scaling['scale'])
    pools = {kind: np.asarray([i for i, entry in enumerate(entries) if entry['split'] == 'train' and entry['category'] == category])
             for kind, category in enumerate(('food', 'water'))}
    full = np.asarray([scene for scene in itertools.product((0, 1), repeat=4) if len(set(scene)) == 2], dtype=np.int64).reshape(14, 2, 2)
    matchings = np.asarray([[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]])
    aggregate = read(OUT / 'results.json')
    assert len(aggregate) == 9
    runs, pairs = {}, {}
    for seed in config['seeds']:
        conditions = config['conditions']
        rows = {condition: [json.loads(line) for line in (OUT / f'{condition}_s{seed}/training_metrics.jsonl').read_text().splitlines()]
                for condition in conditions}
        for records in rows.values():
            assert len(records) == 600 and [r['update'] for r in records] == list(range(1, 601))
            assert all(finite(r) for r in records)
        rng_seed = seed + 2000
        world = np.random.default_rng(rng_seed * 10000 + 101)
        layout = np.random.default_rng(rng_seed * 10000 + 302)
        match_rng = np.random.default_rng(rng_seed * 10000 + 301)
        rotation = np.concatenate([match_rng.permutation(3) for _ in range(200)])
        edges = {condition: np.zeros((4, 4), dtype=np.int64) for condition in conditions}
        history = {condition: {0: matrix.copy()} for condition, matrix in edges.items()}
        for offset in range(600):
            update = offset + 1
            kinds = full[world.integers(len(full), size=2048)].reshape(1024, 2, 2, 2)
            ids = np.empty(kinds.shape, dtype=np.int64)
            for kind in (0, 1):
                mask = kinds == kind
                ids[mask] = world.choice(pools[kind], int(mask.sum()))
            bits = layout.integers(2, size=3)
            for condition in conditions:
                row = rows[condition][offset]
                assert row['lifetime_update'] == 1200 + update
                assert row['stage'] == 'contact_pure_reward' and row['task'] == 'full'
                assert row['aux_weight'] == row['action_entropy_weight'] == 0
                index = 0 if condition == 'fixed_to_fixed' else int(rotation[offset])
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
                for slot, dyad in enumerate(assignment):
                    outcome = row['pair_outcomes'][slot]
                    assert outcome['agents'] == dyad.tolist() and outcome['slot'] == slot
                    assert outcome['success'] == outcome['successes'] / 1024
                    assert 0 <= outcome['successes'] <= 1024
                    edges[condition][dyad[0], dyad[1]] += 1
                    edges[condition][dyad[1], dyad[0]] += 1
                    for local, who in enumerate(dyad):
                        record = row['agents'][who]
                        partner = int(dyad[1 - local])
                        assert record['partner'] == partner
                        assert record['received_sha256'] == row['agents'][partner]['sent_sha256']
                        assert record['private_input_sha256'] == digest(kinds[:, slot, local], ids[:, slot, local])
                        assert record['sampled_success'] == outcome['success']
                        mixed = record['mixed_resource_diagnostics']
                        count = int((kinds[:, slot, local].sum(-1) == 1).sum())
                        assert mixed['n'] == count
                        if count:
                            assert -.000001 <= mixed['action_entropy'] <= math.log(2) + .000001
                            assert .5 - .000001 <= mixed['maximum_action_probability'] <= 1.000001
                            assert -.000001 <= mixed['sender_entropy'] <= math.log(5) + .000001
                        else:
                            assert mixed['action_entropy'] is None and mixed['maximum_action_probability'] is None and mixed['sender_entropy'] is None
                assert row['sampled_training_success'] == np.mean([p['success'] for p in row['pair_outcomes']])
                history[condition][update] = edges[condition].copy()
        expected_policies = []
        for who in range(4):
            rng = np.random.default_rng(rng_seed * 10000 + 102 + who)
            # draw() consumes two N-by-1 float64 uniforms per agent per update.
            rng.bit_generator.advance(2 * 1024 * 600)
            expected_policies.append(rng.bit_generator.state)
        curves, results, references = {}, {}, {}
        for condition in conditions:
            path = OUT / f'{condition}_s{seed}'
            origin_name = 'rotating' if condition == 'rotating_to_rotating' else 'fixed'
            origin = ORIGIN / f'{origin_name}_s{seed}'
            origin_states = states(origin / 'checkpoint_1200.pt')
            assert sha(path / 'initial.pt') == sha(path / 'checkpoint_0000.pt') == sha(origin / 'checkpoint_1200.pt')
            assert sha(path / 'origin_result.json') == sha(origin / 'result.json')
            old = prior_edges(condition)
            np.testing.assert_array_equal(read(origin / 'result.json')['edge_training_updates'], old)
            metadata = read(path / 'origin.json')
            assert metadata['fresh_adam'] and metadata['initial_optimizer_state_entries'] == [0] * 4
            assert metadata['training_seed'] == rng_seed
            assert metadata['origin_sha256'] == metadata['initial_sha256'] == sha(origin / 'checkpoint_1200.pt')
            assert metadata['origin_result_sha256'] == sha(origin / 'result.json')
            assert metadata['protocol_reference_sha256'] == sha(path / 'protocol_reference.pt')
            reference = torch.load(path / 'protocol_reference.pt', weights_only=True)
            for name in ('probe', 'support_probe'):
                probe = reference[name]
                assert probe['sha256'] == saved_eval._hash_probe(probe)
                torch.testing.assert_close(probe['features'], scaled[probe['image_ids']], rtol=0, atol=0)
                assert torch.all(probe['public'][:, :2] == 0)
                assert probe['cases'] == (1024 if name == 'probe' else 2048)
            expected_pairs = [list(edge) for edge in (PAIRS if condition == 'rotating_to_rotating' else FIXED_PAIRS)]
            assert reference['trained_pairs'] == expected_pairs
            assert metadata['protocol_probe_sha256'] == reference['probe']['sha256']
            assert metadata['protocol_support_probe_sha256'] == reference['support_probe']['sha256']
            zero = read(path / 'zero_update_validation.json')
            assert zero['passed'] and zero['origin_learning_curve_sha256'] == sha(origin / 'learning_curve.json')
            curve = read(path / 'learning_curve.json')
            assert [r['update'] for r in curve] == config['checkpoints']
            assert curve[0]['evaluation'] == read(origin / 'learning_curve.json')[-1]['evaluation']
            for record in curve:
                update = record['update']
                assert record['lifetime_update'] == 1200 + update
                checkpoint = states(path / f'checkpoint_{update:04d}.pt')
                assert len(checkpoint) == 4 and all(torch.isfinite(v).all() for a in checkpoint for v in a.values())
                np.testing.assert_array_equal(record['edge_training_updates'], history[condition][update])
                cumulative = old + history[condition][update]
                np.testing.assert_array_equal(record['cumulative_edge_training_updates'], cumulative)
                drift = record['protocol_drift']
                assert drift['probe_sha256'] == reference['probe']['sha256']
                assert drift['support_probe_sha256'] == reference['support_probe']['sha256']
                assert drift['initial_received_support'] == reference['initial_received_support']
                assert drift['current_trained_pairs'] == [[i, j] for i, j in PAIRS if cumulative[i, j] > 0]
                if update == 0:
                    assert equal_states(checkpoint, origin_states)
                    assert all(s['overall']['mean_probability_total_variation'] == 0 and s['overall']['greedy_raw_symbol_change_rate'] == 0 for s in drift['senders'])
            final = states(path / 'checkpoint_0600.pt')
            final_state = torch.load(path / 'final_training_state.pt', weights_only=True)
            assert equal_states(final_state['agents'], final)
            assert final_state['additional_updates_completed'] == 600
            assert final_state['world_rng'] == world.bit_generator.state
            assert final_state['layout_rng'] == layout.bit_generator.state
            assert final_state['policy_rngs'] == expected_policies
            assert final_state['torch_rng_state'].dtype == torch.uint8 and final_state['torch_rng_state'].shape == torch.get_rng_state().shape
            expected_rotation = np.zeros(600, dtype=np.int64) if condition == 'fixed_to_fixed' else rotation
            np.testing.assert_array_equal(final_state['matching_schedule'].numpy(), expected_rotation)
            np.testing.assert_array_equal(final_state['edge_training_updates'].numpy(), edges[condition])
            assert len(final_state['optimizers']) == 4
            for agent_state, optimizer in zip(final, final_state['optimizers']):
                assert len(optimizer['param_groups']) == 1
                group = optimizer['param_groups'][0]
                assert group['lr'] == .0003 and len(group['params']) == len(agent_state) == 14
                assert len(optimizer['state']) == 14
                for identifier, parameter in zip(group['params'], agent_state.values()):
                    optimizer_parameter = optimizer['state'][identifier]
                    assert int(optimizer_parameter['step']) == 600
                    for moment in ('exp_avg', 'exp_avg_sq'):
                        assert optimizer_parameter[moment].shape == parameter.shape and torch.isfinite(optimizer_parameter[moment]).all()
            result = read(path / 'result.json')
            assert result == next(r for r in aggregate if r['seed'] == seed and r['condition'] == condition)
            assert result['updates'] == 600 and result['lifetime_updates'] == 1800
            assert result['individual_training_choices_per_agent'] == 614400 and result['joint_dyad_training_steps'] == 1228800
            assert result['trainable_parameters_per_agent'] == [83527] * 4
            assert result['final_aux_weight'] == result['final_action_entropy_weight'] == 0
            assert result['protocol_drift'] == curve[-1]['protocol_drift']
            expected_edges = np.zeros((4, 4), dtype=np.int64)
            for i, j in (FIXED_PAIRS if condition == 'fixed_to_fixed' else PAIRS):
                expected_edges[i, j] = expected_edges[j, i] = 600 if condition == 'fixed_to_fixed' else 200
            np.testing.assert_array_equal(edges[condition], expected_edges)
            np.testing.assert_array_equal(result['edge_training_updates'], expected_edges)
            np.testing.assert_array_equal(result['edge_joint_cases'], expected_edges * 1024)
            np.testing.assert_array_equal(result['cumulative_edge_training_updates'], old + expected_edges)
            assert eval_hashes(result['evaluation']) == eval_hashes(read(origin / 'result.json')['evaluation'])
            artifact_count = check_artifacts(path, 'artifact_hashes.json')
            deltas = [{module: sum(float((before[k] - after[k]).square().sum()) for k in before if k.startswith(module + '.')) ** .5
                       for module in ('project', 'sender', 'actor', 'value')} for before, after in zip(origin_states, final)]
            assert all(value > 0 for agent in deltas for value in agent.values())
            curves[condition], results[condition], references[condition] = curve, result, reference
            runs[path.name] = {'updates': 600, 'initial_matches_correct_origin': True, 'all_same_case_zero_update_results_exact': True,
                               'all_world_layout_matching_and_private_inputs_independently_reconstructed': True,
                               'message_route_and_independent_pair_reward_records_verified': True,
                               'all_600_updates_full_task_without_auxiliaries': True, 'mixed_context_counts_reconstructed': True,
                               'all_numbers_and_checkpoints_finite': True, 'saved_optimizer_steps': [600] * 4,
                               'final_saved_parameters_equal_final_checkpoint': True, 'all_saved_numpy_rng_states_exact': True,
                               'torch_rng_state_present_and_weights_only_loadable': True, 'artifact_hashes_verified': artifact_count,
                               'edge_training_updates': expected_edges.tolist(), 'cumulative_edge_training_updates': (old + expected_edges).tolist(),
                               'parameter_l2_changes_by_agent': deltas,
                               'zero_gradient_counts_by_agent': [{module: sum(row['agents'][who]['gradient_norms'][module] == 0 for row in rows[condition])
                                                                 for module in ('project', 'sender', 'actor', 'value')} for who in range(4)],
                               'final_full_by_group': {g: {m: result['evaluation']['aggregates'][g]['tasks']['full'][m]['mean_reward_per_step']
                                                          for m in ('normal', 'shuffle', 'stochastic')}
                                                      for g in ('all', 'original', 'cross')}}
        assert sha(OUT / f'fixed_to_fixed_s{seed}/initial.pt') == sha(OUT / f'fixed_to_rotating_s{seed}/initial.pt')
        assert curves['fixed_to_fixed'][0] == curves['fixed_to_rotating'][0]
        assert torch.equal(references['fixed_to_fixed']['initial_sender_probabilities'], references['fixed_to_rotating']['initial_sender_probabilities'])
        assert torch.equal(references['fixed_to_fixed']['initial_listener_action_probabilities'], references['fixed_to_rotating']['initial_listener_action_probabilities'])
        for condition in conditions[1:]:
            assert eval_hashes(results[condition]['evaluation']) == eval_hashes(results[conditions[0]]['evaluation'])
            for first, alternative in zip(curves[conditions[0]], curves[condition]):
                assert eval_hashes(first['evaluation']) == eval_hashes(alternative['evaluation'])
            assert references[condition]['probe']['sha256'] == references[conditions[0]]['probe']['sha256']
            assert references[condition]['support_probe']['sha256'] == references[conditions[0]]['support_probe']['sha256']
        pairs[str(seed)] = {'FF_and_FR_origin_weights_and_zero_update_record_exact': True,
                            'all_three_conditions_world_layout_evaluation_and_probe_inputs_matched': True,
                            'FR_and_RR_same_matching_schedule': True, 'final_training_rngs_reconstructed': True}
    batch_artifacts = check_artifacts(OUT, 'batch_artifact_hashes.json')
    audit = {'batch': 'contact_001', 'status': 'passed', 'source_hashes': source_checks,
             'source_code_config_plan_data_and_scaling_match': True, 'data_hashes': data_checks,
             'batch_artifacts_verified': batch_artifacts, 'runs': runs, 'paired_checks': pairs,
             'scope': 'Read-only execution/input/state audit. No retraining, optimizer steps, protocol rescue, altered budget, or model selection.'}
    target = ROOT / '接触实验_正式运行核查.json'
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'status': audit['status'], 'paired_checks': pairs, 'audit': str(target)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
